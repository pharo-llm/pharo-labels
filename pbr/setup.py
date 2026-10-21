"""Setup: every dependency of the pipeline, checked (and obtained when missing) before any phase runs.

    ./run.sh setup        only this;   ./run.sh (any phase)   runs it first and stops if something is missing

1. Ollama answers; every model of config.yaml that comes from the Ollama library is pulled when missing
   (annotators, editor, embedder, benchmark author, evaluated models, bases of custom models).
2. Custom models (``setup.custom_models``): a LoRA adapter published on Hugging Face (the Pharo PT model) is
   downloaded at its pinned revision and merged into its base weights with peft (Ollama >= 0.40 no longer
   applies LoRA adapters); a local base is reused only if its files' sha256 match the Hub. The merged model is
   converted to GGUF with llama.cpp's converter (pinned release, own venv) and registered in Ollama; a smoke
   completion checks it answers. Provenance (revision, hashes, method): data/tools/custom_models.json.
3. Pharo validation image (Pharo 13 + MuTalk + PharoBench-Support): copied from ``pharo.reuse_from`` when it
   has the same versions, else downloaded and built; a smoke run checks the VM starts. Its global names are listed
   in globals.txt (the method filter "requires dependencies" of phase 3 uses them).
4. Data: every benchmark of phase 2 (into data/existing-benchmarks/) and the Pharo corpus snapshot of phase 3
   (into data/pharo-corpus/). A gated benchmark without a token is a warning, not a failure.

Everything is idempotent: a complete setup takes a few seconds. Log: results/logs/00_setup.log.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

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
    specs = list(cfg["llm"]["annotators"]) + [cfg["llm"]["editor"], cfg["llm"]["embedder"], cfg["benchmark"]["author"]]
    specs += list(cfg["evaluation"]["models"])
    custom = {c["ollama_name"] for c in cfg.get("setup", {}).get("custom_models", [])}
    names = [s["model"] for s in specs if s.get("provider", "ollama") == "ollama" and s.get("model") and s["model"] not in custom]
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


# ---------------------------------------------------------------------------------------------- custom models
def tools(cfg) -> Path:
    d = resolve(cfg["setup"].get("tools_dir", "data/tools"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def llama_cpp(cfg, log) -> tuple[Path, Path]:
    """(llama.cpp checkout at the pinned release, python of its conversion venv)."""
    t = tools(cfg)
    lc = cfg["setup"]["llama_cpp"]
    repo = t / f"llama.cpp-{lc['ref']}"
    if not (repo / "convert_hf_to_gguf.py").exists():
        log.info(f"… llama.cpp {lc['ref']} (converter scripts only)")
        if repo.exists():
            shutil.rmtree(repo)
        _run(["git", "clone", "--depth", "1", "--branch", lc["ref"], lc["repo"], repo], log, quiet=True)
    venv = t / "venv-convert"
    py = venv / "bin" / "python"
    marker = venv / f".ok-{lc['ref']}-merge"
    if not marker.exists():
        log.info("… conversion environment (torch CPU, transformers, peft, gguf); one-time, a few minutes")
        if not py.exists():
            _run([sys.executable, "-m", "venv", venv], log)
        _run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], log, quiet=True)
        # what the LoRA converter imports; the release's own requirements file pins torch versions that have no
        # build for recent Pythons, so the packages are installed directly (CPU torch)
        _run([py, "-m", "pip", "install", "-q", "torch", "numpy", "safetensors", "sentencepiece", "transformers", "protobuf",
              "peft", "accelerate"],
             log, timeout=3 * 3600, quiet=True)
        _run([py, "-m", "pip", "install", "-q", "-e", repo / "gguf-py"], log, quiet=True)
        marker.touch()
    log.info(f"✓ llama.cpp {lc['ref']} converter and its environment")
    return repo, py


MERGE = """
import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base, adapter, out = sys.argv[1:4]
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(adapter).save_pretrained(out)       # the tokenizer the adapter was trained with
print("merged", out)
"""


def base_weights(spec: dict, t: Path, log) -> tuple[Path, bool]:
    """(folder of the base model's weights, downloaded by setup?). A local copy is used only if every weight file
    has the sha256 the Hub records for it."""
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(spec["base_hf"], files_metadata=True)
    want = {x.rfilename: x.lfs.sha256 for x in info.siblings if x.rfilename.endswith(".safetensors") and x.lfs}
    for d in spec.get("base_local_dirs") or []:
        d = Path(os.path.expanduser(d))
        if all((d / f).exists() for f in want):
            log.info(f"  checking the local base {d} against {spec['base_hf']} (sha256 of {len(want)} files)")
            if all(file_sha256(d / f) == h for f, h in want.items()):
                log.info(f"  local base {d} is identical to {spec['base_hf']}: reused")
                return d, False
            log.info(f"  local base {d} differs from {spec['base_hf']}: not used")
    out = t / "base" / spec["base_hf"].replace("/", "_")
    log.info(f"… downloading the base {spec['base_hf']} ({sum(x.size or 0 for x in info.siblings) / 1e9:.0f} GB)")
    snapshot_download(spec["base_hf"], local_dir=str(out), allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json"])
    return out, True


def build_custom(cfg, spec: dict, log):
    """A LoRA adapter of the Hub → merged into its base (peft) → GGUF (llama.cpp) → registered in Ollama."""
    name = spec["ollama_name"]
    if name in ollama_models() and loadable(name):
        log.info(f"✓ custom model {name}")
        return
    from huggingface_hub import snapshot_download
    t = tools(cfg)
    slug = name.replace("/", "_").replace(":", "_")
    adir = t / "adapters" / slug
    log.info(f"… custom model {name}: adapter {spec['hf_repo']}@{spec['revision']}")
    snapshot_download(spec["hf_repo"], revision=spec["revision"], local_dir=str(adir),
                      allow_patterns=["adapter_config.json", "adapter_model.safetensors", "*.json", "merges.txt", "vocab.json"],
                      ignore_patterns=["trainer_state.json", "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state*"])
    acfg = json.loads((adir / "adapter_config.json").read_text())
    if acfg.get("base_model_name_or_path") and acfg["base_model_name_or_path"] != spec["base_hf"]:
        raise SetupError(f"{name}: the adapter was trained on {acfg['base_model_name_or_path']}, config says {spec['base_hf']}")
    repo, py = llama_cpp(cfg, log)
    outtype = spec.get("outtype", "q8_0")
    gguf = t / "gguf" / f"{slug}-{outtype}.gguf"
    if not gguf.exists():
        base, downloaded = base_weights(spec, t, log)
        merged = t / "merged" / slug
        if not (merged / "config.json").exists():
            log.info("… merging the adapter into the base weights (fp16, CPU; about 15 GB of memory, several minutes)")
            script = t / "merge_lora.py"
            script.write_text(MERGE)
            _run([py, script, base, adir, merged], log, timeout=3 * 3600)
        log.info(f"… converting the merged model to GGUF {outtype}")
        gguf.parent.mkdir(parents=True, exist_ok=True)
        _run([py, repo / "convert_hf_to_gguf.py", merged, "--outtype", outtype, "--outfile", gguf], log, timeout=3 * 3600,
             quiet=True)
        shutil.rmtree(merged)                       # 15 GB, rebuilt from the adapter if ever needed
        if downloaded and not spec.get("keep_base", False):
            shutil.rmtree(base)
    mf = t / "gguf" / f"Modelfile.{slug}"
    mf.write_text(f"FROM {gguf}\nTEMPLATE \"{{{{ .Prompt }}}}\"\n"
                  "PARAMETER temperature 0\nPARAMETER stop <|endoftext|>\nPARAMETER stop <|file_sep|>\n")
    _run(["ollama", "create", name, "-f", mf], log, timeout=3600, quiet=True)
    smoke = requests.post(f"{ollama_url()}/api/generate", timeout=600, json={
        "model": name, "raw": True, "stream": False, "options": {"temperature": 0, "num_predict": 40},
        "prompt": "Collection >> sumOfSquares\n\t\"Answer the sum of the squares of my elements\"\n"}).json()
    log.info(f"  smoke completion: {smoke.get('response', '')[:120]!r}")
    if not smoke.get("response", "").strip():
        raise SetupError(f"{name} was created but answers nothing")
    prov_path = t / "custom_models.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    prov[name] = {"hf_repo": spec["hf_repo"], "revision": spec["revision"], "base_hf": spec["base_hf"],
                  "adapter_sha256": file_sha256(adir / "adapter_model.safetensors"), "gguf": str(gguf),
                  "gguf_sha256": file_sha256(gguf), "outtype": outtype, "method": "peft merge_and_unload (fp16) + convert_hf_to_gguf",
                  "llama_cpp": cfg["setup"]["llama_cpp"]["ref"], "created": now_iso(),
                  "lora": {k: acfg.get(k) for k in ("r", "lora_alpha", "target_modules")}}
    write_json(prov_path, prov)
    gguf.unlink()          # Ollama keeps its own copy; the GGUF is rebuilt from the adapter if ever needed (sha256 above)
    log.info(f"✓ custom model {name} built and registered in Ollama")


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
    steps = [("Ollama models", pull_models), ("custom models", lambda c, l: [build_custom(c, s, l) for s in
                                                                            c.get("setup", {}).get("custom_models", [])]),
             ("Pharo image", pharo_image), ("Pharo image globals", image_globals), ("data", data)]
    for name, fn in steps:
        try:
            fn(cfg, log)
        except (SetupError, OSError, subprocess.SubprocessError, requests.RequestException) as e:
            log.error(f"✗ setup failed at '{name}': {e}")
            log.error("The pipeline cannot start without it. Fix the cause above, then ./run.sh again (setup resumes).")
            return 1
    log.info("SETUP complete: every dependency is present")
    return 0
