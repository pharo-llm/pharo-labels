"""The validation image: one pinned Pharo version with MuTalk loaded, and a runner that
executes the harness (pharo/BenchmarkHarness/*.st) on a throw-away copy of it.

Every job runs in its own copy of the image and its changes file and the copy is deleted
afterwards, so no task can see or disturb another one and jobs can run in parallel.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess

HARNESS_DIR = Path("pharo") / "BenchmarkHarness"
IMAGE_NAME = "PTMBench"


class ImageError(RuntimeError):
    """The validation image cannot be built or started."""


def _run(cmd, cwd: Path, timeout: int, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), timeout=timeout, env=env, capture_output=True, text=True, errors="replace")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_image(cfg, log=print) -> dict:
    """Build the validation image once (download Pharo, load MuTalk, save) and describe it.

    Returns {"vm", "image", "dir", "info"}; info is what the reports cite as "the image"."""
    bcfg = cfg["benchmark_generation"]
    home = cfg.resolve(bcfg["dir"]) / "pharo"
    info_path = home / "image.json"
    vm = Path(bcfg["pharo_vm"]) if bcfg.get("pharo_vm") else home / "pharo"
    image = home / f"{IMAGE_NAME}.image"
    wanted = {"pharo_version": str(bcfg["pharo_version"]), "mutalk_version": str(bcfg["mutalk_version"]),
              "base_image": str(bcfg.get("pharo_image") or "")}
    if image.exists() and info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if all(info.get(k) == v for k, v in wanted.items()):
            return {"vm": vm, "image": image, "dir": home, "info": info}
        log("Validation image was built with other settings: rebuilding it.")
    home.mkdir(parents=True, exist_ok=True)
    if bcfg.get("pharo_image"):
        src = Path(bcfg["pharo_image"])
        if not src.exists() or not vm.exists():
            raise ImageError(f"benchmark_generation.pharo_image / pharo_vm not found: {src} / {vm}")
        for f in src.parent.glob("*.sources"):
            if not (home / f.name).exists():
                shutil.copy2(f, home / f.name)
        base = src
    else:
        if not (home / "Pharo.image").exists() or not vm.exists():
            log(f"Downloading Pharo {bcfg['pharo_version']} (image and VM) into {home} …")
            url = f"https://get.pharo.org/64/{bcfg['pharo_version']}+vm"
            p = subprocess.run(f"curl -sSL {url} | bash", shell=True, cwd=str(home), capture_output=True, text=True,
                               timeout=1800)
            if p.returncode != 0 or not (home / "Pharo.image").exists():
                raise ImageError(f"Pharo download failed ({url}): {(p.stderr or p.stdout)[-400:]}")
        base = home / "Pharo.image"
    log(f"Loading MuTalk {bcfg['mutalk_version']} into the validation image …")
    for f in home.glob(f"{IMAGE_NAME}.*"):
        f.unlink()
    shutil.copy2(base, image)
    shutil.copy2(base.with_suffix(".changes"), image.with_suffix(".changes"))
    script = ("Metacello new baseline: 'MuTalk'; "
              f"repository: 'github://pharo-contributions/mutalk:{bcfg['mutalk_version']}/src'; load: 'default'. "
              "(Smalltalk globals includesKey: #MTAnalysis) printString, '|', SystemVersion current asString")
    p = _run([str(vm), image.name, "eval", "--save", script], home, 1800)
    last = (p.stdout.strip().splitlines() or [""])[-1].strip("'")
    if p.returncode != 0 or not last.startswith("true|"):
        raise ImageError(f"MuTalk could not be loaded: {(p.stderr or p.stdout)[-600:]}")
    info = {**wanted, "system_version": last.split("|", 1)[1], "image_sha256": _sha256(image),
            "harness_sha256": harness_digest(cfg)}
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(f"Validation image ready: {info['system_version']}")
    return {"vm": vm, "image": image, "dir": home, "info": info}


def harness_digest(cfg) -> str:
    h = hashlib.sha256()
    for f in sorted((cfg.root / HARNESS_DIR).glob("*.st")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def run_job(cfg, pharo: dict, job: dict, workdir: Path, script: str = "harness.st", timeout: int | None = None) -> dict:
    """Run one harness job in a private copy of the image. Never raises for a bad task: a task
    that crashes or hangs the image gets stage "crash" or "timeout", which the gates reject."""
    bcfg = cfg["benchmark_generation"]
    timeout = int(timeout or bcfg["job_timeout_seconds"])
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    image = workdir / pharo["image"].name
    shutil.copy2(pharo["image"], image)
    shutil.copy2(pharo["image"].with_suffix(".changes"), image.with_suffix(".changes"))
    for f in pharo["dir"].glob("*.sources"):
        os.symlink(f, workdir / f.name)
    out = workdir / "result.json"
    job = {**job, "out": str(out)}
    job_path = workdir / "job.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    env = {**os.environ, "PTM_JOB": str(job_path)}
    proc = subprocess.Popen([str(pharo["vm"]), image.name, "st", str(cfg.root / HARNESS_DIR / script)], cwd=str(workdir),
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                            start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=timeout)
        result = None
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
            result = {"stage": "crash", "problems": [], "detail": (output or "")[-1500:]}
    for f in list(workdir.glob(f"{IMAGE_NAME}.*")) + list(workdir.glob("*.sources")) + list(workdir.glob("pharo-local")):
        shutil.rmtree(f) if f.is_dir() and not f.is_symlink() else f.unlink()
    (workdir / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result
