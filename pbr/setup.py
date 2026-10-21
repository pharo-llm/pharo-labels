"""Setup: every dependency of the pipeline, checked (and obtained when missing) before any phase runs.

    ./run.sh setup        only this;   ./run.sh (any phase)   runs it first and stops if something is missing

1. Ollama answers; every model of config.yaml that comes from the Ollama library is pulled when missing
   (annotators, editor, embedder, benchmark author, contract model, oracles, evaluated models).
2. Pharo validation image (Pharo 13 + MuTalk + PharoBench-Support): copied from ``pharo.reuse_from`` when it
   has the same versions, else downloaded and built; a smoke run checks the VM starts. Its global names are listed
   in globals.txt (the method filter "requires dependencies" of phase 3 uses them).
3. Data: every benchmark of phase 2 (into data/existing-benchmarks/) and the Pharo corpus snapshot of phase 3
   (into data/pharo-corpus/). A gated benchmark without a token is a warning, not a failure.

Everything is idempotent: a complete setup takes a few seconds. Log: results/logs/00_setup.log.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess

import requests

from .common import ROOT, data_dir, file_sha256, get_logger, now_iso, resolve, write_json

SUPPORT = ROOT / "pharo" / "support" / "src"


class SetupError(RuntimeError):
    pass


def _run(cmd, log, cwd=None, env=None, timeout=7200, quiet=False) -> str:
    log.info("$ " + " ".join(str(c) for c in cmd)[:220])
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True, errors="replace",
                       timeout=timeout)
    if r.returncode != 0:
        raise SetupError(f"command failed ({r.returncode}): {' '.join(map(str, cmd))[:200]}\n{(r.stderr or r.stdout)[-1500:]}")
    if not quiet and r.stdout.strip():
        for line in r.stdout.strip().splitlines()[-3:]:
            log.info("  " + line[:200])
    return r.stdout


# ---------------------------------------------------------------------------------------------- Ollama
def ollama_url() -> str:
    u = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return u if u.startswith("http") else "http://" + u


def ollama_models() -> set:
    try:
        tags = requests.get(f"{ollama_url()}/api/tags", timeout=15).json().get("models", [])
    except (requests.RequestException, ValueError) as e:
        raise SetupError(f"Ollama does not answer at {ollama_url()} (install it from https://ollama.com, or `ollama serve`)") from e
    names = set()
    for t in tags:
        names.add(t["name"])
        if t["name"].endswith(":latest"):
            names.add(t["name"][:-7])
    return names


def library_models(cfg: dict) -> list[str]:
    b = cfg["benchmark"]
    specs = list(cfg["llm"]["annotators"]) + [cfg["llm"]["editor"], cfg["llm"]["embedder"], b["author"], b["contract_author"]]
    specs += list(b["oracles"]) + list(cfg["evaluation"]["models"])
    names = [s["model"] for s in specs if s.get("provider", "ollama") == "ollama" and s.get("model")]
    return list(dict.fromkeys(names))


def loadable(m: str) -> bool:
    """Listed is not enough: an interrupted or concurrent pull can leave a manifest whose layers are missing."""
    try:
        r = requests.post(f"{ollama_url()}/api/show", json={"model": m}, timeout=60)
        return r.status_code == 200 and "error" not in r.json()
    except (requests.RequestException, ValueError):
        return False


def pull_models(cfg, log):
    have = ollama_models()
    for m in library_models(cfg):
        if m in have and loadable(m):
            log.info(f"✓ Ollama model {m}")
            continue
        if m in have:
            log.warning(f"! Ollama model {m} is listed but cannot be loaded (incomplete download): pulling the missing layers")
        log.info(f"… pulling {m} (Ollama library)")
        _run(["ollama", "pull", m], log, timeout=6 * 3600, quiet=True)
        if not loadable(m):            # still broken: remove it and download it whole
            log.warning(f"! {m} still cannot be loaded: removed and downloaded again")
            subprocess.run(["ollama", "rm", m], capture_output=True)
            _run(["ollama", "pull", m], log, timeout=6 * 3600, quiet=True)
        if not loadable(m):
            raise SetupError(f"{m} was pulled but Ollama still cannot load it")
        log.info(f"✓ Ollama model {m} pulled")


# ---------------------------------------------------------------------------------------------- Pharo image
def support_digest() -> str:
    h = hashlib.sha256()
    for f in sorted(SUPPORT.rglob("*.st")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def pharo_image(cfg, log):
    p = cfg["pharo"]
    home = resolve(p["dir"])
    image = home / p["image"]
    want = {"pharo_version": str(p.get("version", "130")), "mutalk_version": str(p.get("mutalk_version", "v3.0.8"))}
    info_path = home / "image.json"
    if image.exists() and (home / "pharo").exists() and info_path.exists():
        info = json.loads(info_path.read_text())
        if all(info.get(k) == v for k, v in want.items()):
            log.info(f"✓ Pharo validation image {info.get('system_version', '')[:40]} ({home.relative_to(ROOT) if home.is_relative_to(ROOT) else home})")
            return
    src = resolve(p["reuse_from"]) if p.get("reuse_from") else None
    home.mkdir(parents=True, exist_ok=True)
    if src and (src / p["image"]).exists() and (src / "image.json").exists():
        info = json.loads((src / "image.json").read_text())
        if all(info.get(k) == v for k, v in want.items()):
            log.info(f"… copying the validation image from {src}")
            for f in src.iterdir():
                if f.name in ("pharo-local",) or f.suffix == ".log":
                    continue
                dst = home / f.name
                if f.is_dir():
                    shutil.copytree(f, dst, symlinks=True, dirs_exist_ok=True)
                else:
                    shutil.copy2(f, dst)
            _smoke_pharo(home, p, log)
            return
    log.info(f"… downloading Pharo {want['pharo_version']} (image and VM) into {home}")
    _run(["bash", "-c", f"curl -sSL https://get.pharo.org/64/{want['pharo_version']}+vm | bash"], log, cwd=home,
         timeout=1800, quiet=True)
    for f in home.glob("PharoBench.*"):
        f.unlink()
    shutil.copy2(home / "Pharo.image", image)
    shutil.copy2(home / "Pharo.changes", image.with_suffix(".changes"))
    log.info(f"… loading MuTalk {want['mutalk_version']} and PharoBench-Support into the image (several minutes)")
    script = ("Metacello new baseline: 'MuTalk'; "
              f"repository: 'github://pharo-contributions/mutalk:{want['mutalk_version']}/src'; load: 'default'. "
              f"Metacello new baseline: 'PharoBenchSupport'; repository: 'tonel://{SUPPORT}'; load. "
              "(Smalltalk globals includesKey: #MTAnalysis) printString, '|', "
              "(Smalltalk globals includesKey: #PBTestCase) printString, '|', SystemVersion current asString")
    out = _run([home / "pharo", image.name, "eval", "--save", script], log, cwd=home, timeout=1800, quiet=True)
    last = (out.strip().splitlines() or [""])[-1].strip("'")
    if not last.startswith("true|true|"):
        raise SetupError(f"MuTalk or PharoBench-Support could not be loaded: {out[-600:]}")
    write_json(info_path, {**want, "support_sha256": support_digest(), "system_version": last.split("|", 2)[2],
                           "image_sha256": file_sha256(image), "built": now_iso()})
    _smoke_pharo(home, p, log)


def _smoke_pharo(home, p, log):
    out = _run([home / "pharo", p["image"], "eval", "3 + 4"], log, cwd=home, timeout=120, quiet=True)
    if out.strip().splitlines()[-1].strip() != "7":
        raise SetupError(f"the Pharo VM does not evaluate 3 + 4: {out[-300:]}")
    log.info("✓ Pharo validation image ready (VM evaluates 3 + 4 = 7)")


def image_globals(cfg, log):
    """The global names (classes, traits, globals) of the clean Pharo image: the method filter "requires
    dependencies" flags methods that reference a class neither their project nor a standard Pharo defines."""
    p = cfg["pharo"]
    home = resolve(p["dir"])
    out_path = home / "globals.txt"
    if out_path.exists() and out_path.stat().st_size > 1000:
        log.info(f"✓ Pharo image globals ({sum(1 for _ in out_path.open())} names)")
        return
    script = "String streamContents: [ :s | (Smalltalk globals keys asSortedCollection: [ :a :b | a <= b ]) do: [ :k | s nextPutAll: k; nextPut: Character lf ] ]"
    tmp = home / ".globals.st"
    tmp.write_text(f"| f | f := '{out_path}' asFileReference. f ensureDelete. f writeStreamDo: [ :w | w nextPutAll: ({script}) ]. Smalltalk exitSuccess.")
    _run([home / "pharo", p["image"], "st", tmp], log, cwd=home, timeout=300, quiet=True)
    tmp.unlink()
    n = sum(1 for _ in out_path.open()) if out_path.exists() else 0
    if n < 1000:
        raise SetupError(f"the Pharo image listed only {n} globals")
    log.info(f"✓ Pharo image globals: {n} names → {out_path}")


# ---------------------------------------------------------------------------------------------- data
def data(cfg, log):
    import urllib.error
    from . import benchmarks as B
    for key, b in B.BENCHMARKS.items():
        try:
            n = len(B.load_rows(key, 0, int(cfg["project"]["seed"])))
            log.info(f"✓ benchmark {b['name']}: {n} items")
        except urllib.error.HTTPError as e:
            if b.get("gated"):
                log.warning(f"! benchmark {b['name']}: gated (HTTP {e.code}); set HF_TOKEN in .env and accept its terms "
                            "on Hugging Face to include it — phase 2 records it as not available")
            else:
                raise SetupError(f"benchmark {b['name']} cannot be downloaded: HTTP {e.code}") from e
    c = cfg["corpus"]
    snap = data_dir(cfg) / "pharo-corpus" / c["snapshot_revision"]
    parsed = snap / "snapshot" / "data" / "parsed"
    if parsed.is_dir() and len(list(parsed.glob("*.jsonl.gz"))) >= 100:
        log.info(f"✓ Pharo corpus snapshot {c['snapshot_revision']} ({len(list(parsed.glob('*.jsonl.gz')))} repositories)")
        return
    log.info(f"… downloading the Pharo corpus snapshot {c['snapshot_repo']}@{c['snapshot_revision']} (about 1.3 GB)")
    from huggingface_hub import snapshot_download
    snapshot_download(c["snapshot_repo"], repo_type="dataset", revision=c["snapshot_revision"], local_dir=str(snap),
                      max_workers=8, allow_patterns=[
                          "snapshot/data/parsed/*", "snapshot/data/raw/repositories.jsonl", "snapshot/data/processed/sample.json",
                          "snapshot/data/processed/domain_labels.jsonl", "snapshot/output/tables/repository_selection.csv",
                          "snapshot/manifest.json"])
    log.info("✓ Pharo corpus snapshot downloaded")


# ---------------------------------------------------------------------------------------------- main
def main(cfg: dict) -> int:
    log = get_logger(cfg, "00 setup", "00_setup.log")
    log.info("=" * 78)
    log.info("SETUP — dependencies of every phase")
    steps = [("Ollama models", pull_models), ("Pharo image", pharo_image), ("Pharo image globals", image_globals), ("data", data)]
    for name, fn in steps:
        try:
            fn(cfg, log)
        except (SetupError, OSError, subprocess.SubprocessError, requests.RequestException) as e:
            log.error(f"✗ setup failed at '{name}': {e}")
            log.error("The pipeline cannot start without it. Fix the cause above, then ./run.sh again (setup resumes).")
            return 1
    log.info("SETUP complete: every dependency is present")
    return 0
