"""The language models that label methods: instruction-tuned models served by Ollama
(qwen3.8:27b, gemma4:12b, mistral-small3.2:24b). Each is asked through the chat API, with thinking off
and deterministic decoding, to continue a few-shot text.
"""
from __future__ import annotations

import os
import time


NUM_CTX = 8192   # Ollama context: prompt + answer must fit, or Ollama drops the start of the prompt
CHAT_INSTRUCTION = ("Continue the text below: write only what comes after its last line, in exactly the "
                    "format of the examples, and nothing else.\n\n")


class ModelError(RuntimeError):
    """The model cannot be used (server not running, model not installed, missing package)."""


def make_model(spec: str, cfg: dict):
    return OllamaModel(spec.strip().removeprefix("ollama:"), cfg)


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
        info = requests.post(f"{self.url}/api/show", json={"model": tag}, timeout=30).json()
        self.thinks = "thinking" in (info.get("capabilities") or [])

    def describe(self) -> dict:
        return {"backend": "ollama", "model": self.tag, "mode": "chat", "temperature": self.temperature, "seed": self.seed}

    def complete(self, prompt: str, max_tokens: int, stop: list[str]) -> str:
        # a chat answer may put a blank line between its lines: keep only the real stops
        payload = {"model": self.tag, "stream": False, "messages": [{"role": "user", "content": CHAT_INSTRUCTION + prompt}],
                   "options": {"temperature": self.temperature, "seed": self.seed, "num_predict": max_tokens,
                               "num_ctx": NUM_CTX, "stop": [s for s in stop if s.strip()]}}
        if self.thinks:
            payload["think"] = False
        for attempt in range(4):
            try:
                r = self.requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
                if r.status_code < 500:
                    r.raise_for_status()
                    return r.json()["message"]["content"]
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
