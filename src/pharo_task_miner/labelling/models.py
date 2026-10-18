"""Small language models used as text completers.

The Pharo models of pharo-llm are code-completion models (base or fill-in-the-middle), not chat
models, so every model is driven the same way: a plain few-shot prompt is completed, with
deterministic decoding, until a stop string. Two backends:

* ``ollama``  - any Ollama tag, e.g. ``pharo-llm/Qwen2.5-Coder-SFT:1.5B`` or ``qwen2.5-coder:7b``.
  ``raw`` mode bypasses the model's template (the Pharo models have a FIM template).
* ``hf``      - ``hf:<repo>[@revision]`` loads a Hugging Face model with transformers. A LoRA
  adapter (``adapter_config.json``, e.g. ``pharo-llm/Qwen2.5-Coder-1.5B-PT``) is merged into
  its base model. When ``main`` holds no weights, the last ``checkpoint-N`` branch is used.
"""
from __future__ import annotations

import os
import re
import time


NUM_CTX = 8192   # Ollama context: prompt + answer must fit, or Ollama drops the start of the prompt


class ModelError(RuntimeError):
    """The model cannot be used (server not running, model not installed, missing package)."""


def make_model(spec: str, cfg: dict):
    spec = spec.strip()
    if spec.startswith("hf:"):
        return HFModel(spec[3:], cfg)
    return OllamaModel(spec.removeprefix("ollama:"), cfg)


class OllamaModel:
    def __init__(self, tag: str, cfg: dict):
        import requests
        self.requests = requests
        self.tag = tag
        self.name = tag
        self.url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if not self.url.startswith("http"):
            self.url = "http://" + self.url
        self.timeout = float(cfg.get("timeout_seconds") or 300)
        self.temperature = float(cfg.get("temperature") or 0)
        self.seed = int(cfg.get("random_seed") or 42)
        try:
            names = {m["name"] for m in requests.get(f"{self.url}/api/tags", timeout=10).json().get("models", [])}
        except (requests.RequestException, ValueError) as e:
            raise ModelError(f"No Ollama server at {self.url}: start it with `ollama serve`.") from e
        if tag not in names and f"{tag}:latest" not in names:
            raise ModelError(f"Ollama model {tag!r} is not installed: ollama pull {tag}")

    def describe(self) -> dict:
        return {"backend": "ollama", "model": self.tag, "temperature": self.temperature, "seed": self.seed}

    def complete(self, prompt: str, max_tokens: int, stop: list[str]) -> str:
        payload = {"model": self.tag, "prompt": prompt, "raw": True, "stream": False,
                   "options": {"temperature": self.temperature, "seed": self.seed, "num_predict": max_tokens,
                               "stop": stop, "num_ctx": NUM_CTX}}
        for attempt in range(4):
            try:
                r = self.requests.post(f"{self.url}/api/generate", json=payload, timeout=self.timeout)
                if r.status_code < 500:
                    r.raise_for_status()
                    return r.json().get("response", "")
            except (self.requests.ConnectionError, self.requests.Timeout):
                if attempt == 3:
                    raise
            time.sleep(2 * 2 ** attempt)
        raise ModelError(f"Ollama kept failing for {self.tag}")


class OllamaEmbedder:
    """Sentence embeddings from an Ollama embedding model (e.g. all-minilm, 45 MB), unit length."""

    def __init__(self, tag: str):
        import requests
        self.requests, self.tag = requests, tag
        self.url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if not self.url.startswith("http"):
            self.url = "http://" + self.url
        self([ "probe" ])   # fails early when the model is missing

    def __call__(self, texts: list[str]) -> list[list[float]]:
        try:
            r = self.requests.post(f"{self.url}/api/embed", json={"model": self.tag, "input": list(texts)}, timeout=120)
            r.raise_for_status()
            vecs = r.json()["embeddings"]
        except (self.requests.RequestException, KeyError, ValueError) as e:
            raise ModelError(f"Embedding model {self.tag!r} is not usable: ollama pull {self.tag}") from e
        out = []
        for v in vecs:
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out


def _latest_checkpoint(repo: str) -> str | None:
    from huggingface_hub import list_repo_refs
    best = None
    for b in list_repo_refs(repo).branches:
        m = re.fullmatch(r"checkpoint-(\d+)", b.name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), b.name)
    return best[1] if best else None


def resolve_hf(spec: str) -> tuple[str, str]:
    """``repo[@revision]`` -> (repo, revision); `main` without weights -> last checkpoint branch."""
    from huggingface_hub import list_repo_files
    repo, _, rev = spec.partition("@")
    if rev:
        return repo, rev
    weights = [f for f in list_repo_files(repo) if f.endswith((".safetensors", ".bin"))]
    if weights:
        return repo, "main"
    rev = _latest_checkpoint(repo)
    if not rev:
        raise ModelError(f"{repo} has no weights on main and no checkpoint-N branch.")
    return repo, rev


class HFModel:
    def __init__(self, spec: str, cfg: dict):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ModelError("hf: models need torch, transformers and peft: .venv/bin/pip install -e '.[hf]'") from e
        from huggingface_hub import list_repo_files
        self.torch = torch
        self.repo, self.revision = resolve_hf(spec)
        self.name = f"hf:{self.repo}@{self.revision}"
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        files = set(list_repo_files(self.repo, revision=self.revision))
        self.tok = AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
        if "adapter_config.json" in files:
            from peft import PeftConfig, PeftModel
            self.base = PeftConfig.from_pretrained(self.repo, revision=self.revision).base_model_name_or_path
            base = AutoModelForCausalLM.from_pretrained(self.base, dtype=dtype)
            model = PeftModel.from_pretrained(base, self.repo, revision=self.revision).merge_and_unload()
        else:
            self.base = None
            model = AutoModelForCausalLM.from_pretrained(self.repo, revision=self.revision, dtype=dtype)
        self.model = model.to(self.device).eval()
        self.model.generation_config.max_length = None
        torch.manual_seed(int(cfg.get("random_seed") or 42))

    def describe(self) -> dict:
        return {"backend": "hf", "model": self.repo, "revision": self.revision, "base_model": self.base,
                "device": self.device, "decoding": "greedy"}

    def complete(self, prompt: str, max_tokens: int, stop: list[str]) -> str:
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_tokens, do_sample=False, stop_strings=stop,
                                      tokenizer=self.tok, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        cut = [text.index(s) for s in stop if s in text]
        return text[:min(cut)] if cut else text
