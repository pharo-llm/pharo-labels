"""Acceptance gates: turn what the harness observed into pass/fail decisions.

A task is accepted only when every gate passes. The gates are checked in a fixed order and the
first failure is the recorded rejection reason, so the counts read as a filtering funnel. No
gate asks a model for an opinion: each one is a fact observed by running code in the image
(or by the project's own static analysis). The rationale and the literature behind each gate
are in docs/4-benchmark.md.
"""
from __future__ import annotations

import hashlib
import re

GATES = [
    ("well_formed", "The task has the required parts and sizes"),
    ("installs", "It compiles in the clean image: no syntax error, name clash, undeclared name or unknown selector"),
    ("hermetic", "It touches nothing outside the image: no disk, network, clock, UI or unseeded randomness"),
    ("reference_passes", "Every test passes on the reference solution, on every repeated run and in reverse order"),
    ("fail_to_pass", "For every target method, at least one test fails when that method is not implemented"),
    ("mutation", "The tests kill the required share of the mutants of the reference solution (MuTalk)"),
    ("operation_preserved", "The project's rule classifier finds the intended operation in the reference solution"),
    ("solvable", "A model that sees only the task statement writes a solution that passes all tests"),
    ("unique", "Neither the class name nor the solution duplicates an accepted task, and it is not a copy of its seed"),
    ("installs_together", "It loads with all other tasks through the Metacello baseline and its tests still pass"),
]
GATE_IDS = [g for g, _ in GATES]

# Names and messages that make a test depend on something outside the image or on chance.
DENIED_GLOBALS = {
    "FileLocator", "File", "FileStream", "StandardFileStream", "ZnClient", "ZnServer", "ZnEasy", "Socket", "SocketStream",
    "NetNameResolver", "OSPlatform", "LibC", "OSSUnixSubprocess", "OSEnvironment", "Smalltalk", "SmalltalkImage",
    "SystemNavigation", "World", "Morph", "SpPresenter", "UIManager", "Clipboard", "Sensor", "Display", "Iceberg",
    "Metacello", "EpMonitor", "SessionManager", "UUID", "UUIDGenerator",
}
DENIED_SELECTORS = {
    "now", "today", "tomorrow", "yesterday", "millisecondClockValue", "microsecondClockValue", "currentTime",
    "asFileReference", "workingDirectory", "disk", "halt", "inspect", "browse", "explore", "openInWorld", "open",
    "compile:", "compile:classified:", "evaluate:", "removeFromSystem", "subclass:", "inform:", "request:", "confirm:",
    "atRandom", "shuffle", "shuffled", "garbageCollect", "snapshot:andQuit:", "quit",
}


def _norm(source: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r'"[^"]*"', " ", source or "")).strip()


def solution_hash(task: dict) -> str:
    return hashlib.sha256(" || ".join(sorted(_norm(m["source"]) for m in task["target_methods"])).encode()).hexdigest()[:16]


def _shingles(source: str, k: int = 5) -> set:
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*:?|\S", _norm(source))
    return {tuple(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1))}


def similarity(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def well_formed(task: dict, bcfg: dict) -> tuple[bool, str]:
    name = (task.get("class") or {}).get("name") or ""
    tname = (task.get("test_class") or {}).get("name") or ""
    if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", name) or not re.fullmatch(r"[A-Z][A-Za-z0-9]*", tname):
        return False, "class and test class need valid, capitalised names"
    if name == tname:
        return False, "the test class must be a separate class"
    if not task.get("target_methods"):
        return False, "no target method"
    if len(task["target_methods"]) > int(bcfg["max_target_methods"]):
        return False, f"more than {bcfg['max_target_methods']} target methods"
    n_tests = sum(1 for t in task.get("tests") or [] if re.match(r"\s*test", t.get("source") or ""))
    if n_tests < int(bcfg["min_tests"]):
        return False, f"{n_tests} test methods, at least {bcfg['min_tests']} required"
    for m in task["target_methods"] + (task.get("context_methods") or []) + task["tests"]:
        if not (m.get("source") or "").strip():
            return False, "empty method source"
    return True, f"{len(task['target_methods'])} target method(s), {n_tests} tests"


def _hermetic(result: dict) -> tuple[bool, str]:
    bad = []
    for f in result.get("facts") or []:
        g = sorted(set(f.get("globals") or []) & DENIED_GLOBALS)
        s = sorted(set(f.get("sends") or []) & DENIED_SELECTORS)
        if "Random" in (f.get("globals") or []) and "seed:" not in (f.get("sends") or []):
            g.append("Random (without seed:)")
        if g or s:
            bad.append(f"{f['class']}>>{f['selector']}: {', '.join(g + s)}")
    return (not bad), "; ".join(bad) or "no external dependency"


def _not_passing(run: dict) -> list[str]:
    return [t for t, r in (run or {}).items() if r.get("outcome") != "pass"]


def candidate_passes(cand: dict, n_tests: int) -> bool:
    tests = cand.get("tests") or {}
    return not cand.get("errors") and len(tests) == n_tests and n_tests > 0 and not _not_passing(tests)


def mutation_summary(result: dict) -> dict:
    m = result.get("mutation") or {}
    evaluated = int(m.get("evaluated") or 0)
    detected = int(m.get("killed") or 0) + int(m.get("terminated") or 0)
    return {"evaluated": evaluated, "detected": detected, "survived": int(m.get("alive") or 0),
            "score": round(detected / evaluated, 4) if evaluated else None, "error": m.get("error"),
            "survivors": [x for x in m.get("mutants") or [] if x.get("status") == "survived"]}


def evaluate(task: dict, result: dict, bcfg: dict) -> list[dict]:
    """Gates decided by one harness run, in order; evaluation stops at the first failure."""
    out = []

    def gate(gid, passed, detail):
        out.append({"gate": gid, "passed": bool(passed), "detail": detail})
        return bool(passed)

    ok, detail = well_formed(task, bcfg)
    if not gate("well_formed", ok, detail):
        return out
    stage = result.get("stage")
    problems = result.get("problems") or []
    if stage in ("preflight", "install") or problems:
        gate("installs", False, "; ".join(
            f"{p['kind']}: {p.get('class', '')}{'>>' + p['selector'] if p.get('selector') else ''} {p.get('detail', '')}".strip()
            for p in problems[:8]))
        return out
    if stage != "done":
        gate("installs", False, f"{stage}: {result.get('harness_error') or result.get('detail') or ''}"[:600])
        return out
    gate("installs", True, result.get("image", ""))
    ok, detail = _hermetic(result)
    if not gate("hermetic", ok, detail):
        return out
    tests = result.get("tests") or []
    runs = result.get("reference_runs") or []
    failing = sorted({t for r in runs for t in _not_passing(r)})
    if len(tests) < int(bcfg["min_tests"]) or failing or not runs:
        details = [f"{t}: {r[t]['outcome']} {r[t].get('detail', '')}".strip() for r in runs[:1] for t in _not_passing(r)]
        flaky = [t for t in failing if any(r.get(t, {}).get("outcome") == "pass" for r in runs)]
        gate("reference_passes", False,
             (f"not deterministic (passes on one run, not on a repeated or reversed run): {', '.join(flaky)}" if flaky
              else "; ".join(details) or f"{len(tests)} tests found, at least {bcfg['min_tests']} required"))
        return out
    gate("reference_passes", True, f"{len(tests)} tests x {len(runs)} runs")
    weak = [f"{s['class']}>>{s['selector']}" for s in result.get("stub_runs") or [] if not _not_passing(s["tests"])]
    if not gate("fail_to_pass", not weak and bool(result.get("stub_runs")),
                ("no test fails without " + ", ".join(weak)) if weak else
                f"{len(_not_passing(result.get('all_stubbed_run')))}/{len(tests)} tests fail without the solution"):
        return out
    ms = mutation_summary(result)
    after = _not_passing(result.get("after_mutation_run"))
    if ms["error"] or ms["score"] is None:
        gate("mutation", False, f"mutation analysis did not run: {ms['error'] or 'no mutant generated'}")
    elif after:
        gate("mutation", False, "tests no longer pass after the analysis: " + ", ".join(after))
    elif ms["evaluated"] < int(bcfg["min_mutants"]):
        gate("mutation", False, f"only {ms['evaluated']} mutants: the solution is too small to assess its tests")
    else:
        gate("mutation", ms["score"] >= float(bcfg["min_mutation_score"]),
             f"score {ms['score']:.2f} ({ms['detected']}/{ms['evaluated']} mutants detected, "
             f"required {float(bcfg['min_mutation_score']):.2f})")
    return out


def first_failure(gates: list[dict]) -> dict | None:
    return next((g for g in gates if not g["passed"]), None)
