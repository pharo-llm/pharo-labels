"""Pharo test runner: executes reference and candidate implementations in a controlled Pharo environment.

The validation image (Pharo 13 + MuTalk + PharoBench-Support, built by ../../pharo-benchmarks) is never
modified: every job runs in a throw-away copy, deleted afterwards. The harness (pharo/harness.st, from
PharoBench) installs a task's classes, runs its tests (repeated, once reversed), replaces each target
method by a stub (fail-to-pass), runs candidate implementations and MuTalk mutation analysis, and
writes one JSON result.

Code from evaluated models is untrusted. Each image process runs:
* in its own process group, killed (SIGKILL) at the time limit;
* with resource limits: CPU seconds (RLIMIT_CPU), size of written files (RLIMIT_FSIZE);
* on macOS, under ``sandbox-exec`` with network access denied (checked once; ``pharo.sandbox: off``
  disables it, ``auto`` falls back with a warning when it does not work);
* in an empty working directory, with only the job file in it.
Static checks reject code that reaches outside the image (files, network, clock, UI, randomness).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time

from .common import ROOT, file_sha256, resolve

HARNESS = ROOT / "pharo" / "harness.st"

DENIED_GLOBALS = {"FileLocator", "File", "FileStream", "StandardFileStream", "FileSystem", "ZnClient", "ZnServer",
                  "ZnEasy", "Socket", "SocketStream", "NetNameResolver", "OSPlatform", "LibC", "OSSUnixSubprocess",
                  "OSEnvironment", "Smalltalk", "SmalltalkImage", "SystemNavigation", "World", "UIManager", "Clipboard",
                  "Sensor", "Display", "Iceberg", "Metacello", "SessionManager", "UUID", "UUIDGenerator", "Random"}
DENIED_SELECTORS = {"now", "today", "tomorrow", "yesterday", "millisecondClockValue", "microsecondClockValue",
                    "asFileReference", "workingDirectory", "halt", "inspect", "browse", "explore", "openInWorld", "open",
                    "openWithSpec", "compile:", "compile:classified:", "evaluate:", "removeFromSystem", "inform:",
                    "request:", "confirm:", "atRandom", "shuffle", "shuffled", "garbageCollect", "snapshot:andQuit:", "quit"}

SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network*)
(allow network* (local unix))
"""


def method_selector(source: str) -> str | None:
    head = (source or "").strip().splitlines()[0].strip() if (source or "").strip() else ""
    kws = re.findall(r"([a-zA-Z_]\w*:)\s*[a-zA-Z_]\w*", head)
    if kws:
        return "".join(kws)
    m = re.match(r"([-+*/\\<>=~@%|&?,]+)\s+\w+", head)
    if m:
        return m.group(1)
    m = re.match(r"([a-zA-Z_]\w*)", head)
    return m.group(1) if m else None


def method_signature(source: str) -> str:
    """The message pattern of a method source (its first line, without comment)."""
    head = (source or "").strip().splitlines()[0] if (source or "").strip() else ""
    return head.split('"')[0].strip()


def clean_method(text: str) -> str:
    """One method source from a model answer: the first ```smalltalk block (or the text), Tonel wrapper removed."""
    m = re.search(r"```[a-zA-Z]*\n(.*?)(```|$)", text or "", re.S)
    src = (m.group(1) if m else (text or "")).strip("\n")
    lines = src.strip().splitlines()
    if lines and ">>" in lines[0]:
        head = lines[0].split(">>", 1)[1].strip()
        if head.endswith("["):
            body = lines[1:]
            while body and not body[-1].strip():
                body.pop()
            if body and body[-1].strip() == "]":
                body = body[:-1]
            lines = [head[:-1].rstrip()] + body
        else:
            lines = [head] + lines[1:]
    return "\n".join(lines).strip()


def split_methods(text: str) -> list[str]:
    """Several methods in one answer (blocks or Tonel-like `Class >> sel [ ... ]`)."""
    blocks = re.findall(r"```[a-zA-Z]*\n(.*?)```", text or "", re.S) or [text or ""]
    out = []
    for b in blocks:
        parts = re.split(r"\n(?=[A-Z]\w*(?: class)? >> )", "\n" + b.strip())
        for part in parts:
            # methods written one after the other: a new method starts after a blank line, at column 0,
            # with a message pattern (unary, binary or keyword) — not with a statement or a comment
            out += [clean_method(x) for x in re.split(r"\n\s*\n(?=(?:[a-z]\w*:\s*\w+|[a-z]\w*\s*$|[a-z]\w*\s*\n|[-+*/\\<>=~@%|&?,]+\s+\w+)[^\n]*\n)",
                                                  part.strip() + "\n") if x.strip()]
    return [o for o in out if o]


def job(task: dict, *, mutation: bool, candidates: list | None = None, repeat: int = 3, time_limit: int = 10) -> dict:
    """The harness job for a task (PharoBench format: class, test class, target / context methods, tests)."""
    cls, tcls = task["class"], task["test_class"]
    classes = [{"name": cls["name"], "superclass": cls.get("superclass", "Object"),
                "instance_variables": cls.get("instance_variables", []), "class_variables": [],
                "comment": cls.get("comment", "")},
               {"name": tcls["name"], "superclass": tcls.get("superclass", "PBTestCase"), "comment": ""}]
    methods = [{"class": cls["name"], "side": "instance", "source": m["source"], "protocol": "solution", "role": "target"}
               for m in task["target_methods"]]
    methods += [{"class": cls["name"], "side": "instance", "source": m["source"], "protocol": "context", "role": "context"}
                for m in task.get("context_methods") or []]
    methods += [{"class": tcls["name"], "side": "instance", "source": t["source"], "protocol": "tests", "role": "test"}
                for t in task["tests"]]
    out = {"package": task["package"], "classes": classes, "methods": methods, "test_class": tcls["name"],
           "repeat": repeat, "test_time_limit_seconds": time_limit, "mutation": mutation}
    if candidates:
        out["candidates"] = candidates
    return out


def static_problems(facts: list) -> list[str]:
    """Calls outside the image (files, network, clock, UI, randomness) found in installed code."""
    out = []
    for f in facts or []:
        bad_g = sorted(set(f.get("globals") or []) & DENIED_GLOBALS)
        bad_s = sorted(set(f.get("sends") or []) & DENIED_SELECTORS)
        if bad_g or bad_s:
            out.append(f"{f.get('class')}>>{f.get('selector')} uses {', '.join(bad_g + bad_s)}")
    return out


class Runner:
    def __init__(self, cfg: dict, log=print):
        p = cfg["pharo"]
        self.cfg, self.log = cfg, log
        self.home = resolve(p["dir"])
        self.vm = self.home / "pharo"
        self.image = self.home / p["image"]
        self.timeout = int(p["job_timeout_seconds"])
        self.workers = int(p["workers"])
        self.cpu = int(p.get("cpu_seconds_limit", 300))
        self.fsize = int(p.get("file_size_limit_mb", 200)) * 1024 * 1024
        if not self.vm.exists() or not self.image.exists():
            raise RuntimeError(f"No Pharo VM / validation image in {self.home}: build it with "
                               f"`cd ../../pharo-benchmarks && ./run.sh selftest` (or set pharo.dir)")
        info_path = self.home / "image.json"
        self.info = json.loads(info_path.read_text()) if info_path.exists() else {}
        self.info.setdefault("image_sha256", file_sha256(self.image))
        self.info["harness_sha256"] = hashlib.sha256(HARNESS.read_bytes()).hexdigest()[:16]
        self.sandbox = self._sandbox_mode(p.get("sandbox", "auto"))
        self.info["sandbox"] = self.sandbox

    def _sandbox_mode(self, mode: str) -> str:
        if mode == "off" or sys.platform != "darwin" or not shutil.which("sandbox-exec"):
            if mode not in ("off",) and sys.platform == "darwin":
                self.log("sandbox-exec not found: Pharo runs without network isolation")
            return "off"
        prof = self.home.parent / ".pbr_sandbox.sb"
        try:
            prof.write_text(SANDBOX_PROFILE)
            r = subprocess.run(["sandbox-exec", "-f", str(prof), "/usr/bin/true"], capture_output=True, timeout=20)
            if r.returncode == 0:
                self.profile = prof
                return "sandbox-exec (no network)"
        except (OSError, subprocess.SubprocessError):
            pass
        self.log("sandbox-exec does not work here: Pharo runs without network isolation")
        return "off"

    def _limits(self):
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (self.cpu, self.cpu + 5))
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (self.fsize, self.fsize))
        except (ValueError, OSError):
            pass

    def run(self, job_: dict, workdir: Path, timeout: int | None = None) -> dict:
        """One harness job in a private copy of the image. Never raises for a bad task: a crash or a hang is
        a result (stage 'crash' / 'timeout')."""
        timeout = int(timeout or self.timeout)
        workdir = Path(workdir).resolve()
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)
        image = workdir / self.image.name
        shutil.copy2(self.image, image)
        shutil.copy2(self.image.with_suffix(".changes"), image.with_suffix(".changes"))
        for f in self.home.glob("*.sources"):
            os.symlink(f, workdir / f.name)
        out = workdir / "result.json"
        (workdir / "job.json").write_text(json.dumps({**job_, "out": str(out)}, ensure_ascii=False), encoding="utf-8")
        cmd = [str(self.vm), image.name, "st", str(HARNESS)]
        if self.sandbox != "off":
            cmd = ["sandbox-exec", "-f", str(self.profile)] + cmd
        env = {**os.environ, "PB_JOB": str(workdir / "job.json")}
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=workdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", preexec_fn=self._limits)
        result = None
        try:
            output, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, _ = proc.communicate()
            result = {"stage": "timeout", "problems": [], "detail": f"no result after {timeout} s"}
        if result is None:
            try:
                result = json.loads(out.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = {"stage": "crash", "problems": [], "detail": (output or "")[-1500:],
                          "returncode": proc.returncode}
        result["wall_seconds"] = round(time.time() - t0, 2)
        for f in list(workdir.glob(f"{self.image.stem}.*")) + list(workdir.glob("*.sources")) + list(workdir.glob("pharo-local")):
            shutil.rmtree(f) if f.is_dir() and not f.is_symlink() else f.unlink()
        (workdir / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        return result

    def map(self, fn, items: list) -> list:
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as ex:
            return list(ex.map(fn, items))


# ---------------------------------------------------------------------------------------------- gates
GATES = [("installs", "compiles in the clean image: no syntax error, name clash, undeclared name or unknown selector"),
         ("hermetic", "touches nothing outside the image (files, network, clock, UI, randomness)"),
         ("enough_tests", "has at least pharo.min_tests test methods"),
         ("reference_passes", "every test passes on the reference, on every repeated run and in reverse order"),
         ("fail_to_pass", "for every target method, at least one test fails when it is replaced by a stub"),
         ("mutation", "the tests kill at least pharo.min_mutation_score of MuTalk's mutants")]


def gates(result: dict, cfg: dict, mutation: bool = True) -> tuple[dict, str | None]:
    """{gate: True/False/None} and the first failed gate (the rejection reason), from a harness result."""
    p = cfg["pharo"]
    g = {k: None for k, _ in GATES}
    stage = result.get("stage")
    g["installs"] = stage not in ("preflight", "install", "crash", "timeout", "harness_error")
    if g["installs"]:
        g["hermetic"] = not static_problems(result.get("facts"))
        g["enough_tests"] = len(result.get("tests") or []) >= int(p["min_tests"])
        runs = result.get("reference_runs") or []
        g["reference_passes"] = bool(runs) and len(runs) == int(p.get("repeat", 3)) and all(
            all(t.get("outcome") == "pass" for t in r.values()) for r in runs)
        if g["reference_passes"]:
            stubs = result.get("stub_runs") or []
            g["fail_to_pass"] = bool(stubs) and all(any(t.get("outcome") != "pass" for t in s["tests"].values()) for s in stubs)
            if mutation:
                m = result.get("mutation") or {}
                ev = (m.get("evaluated") or 0) - (m.get("terminated") or 0)
                score = (m.get("killed") or 0) / ev if ev > 0 else None
                result["mutation_score"] = score
                g["mutation"] = score is not None and score >= float(p["min_mutation_score"])
    first = next((k for k, _ in GATES if g[k] is False), None)
    return g, first


def candidate_passed(c: dict) -> bool:
    return not c.get("errors") and bool(c.get("tests")) and all(t.get("outcome") == "pass" for t in c["tests"].values())
