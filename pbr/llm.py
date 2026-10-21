"""LLM annotation engine: one interface for every model the pipeline asks (annotators, editor, author,
models under evaluation), with what §6 of the brief requires.

* **Structured output, validated.** A JSON schema is sent to the provider (Ollama ``format``, OpenAI
  ``response_format``) and every answer is validated with ``jsonschema``; an invalid answer is asked again
  with the validation error (``max_retries``). Answers that stay invalid are returned with ``valid=False``
  and kept, never silently dropped.
* **Raw outputs preserved.** Every request and answer (all attempts) is stored in ``results/cache/llm/``,
  keyed by a hash of everything that determines the answer; a rerun never asks the same thing twice, so
  every phase resumes item by item after an interruption.
* **Run metadata.** Model digest / quantisation (Ollama ``/api/show``), prompt version, options, tokens,
  latency; one ledger line per call in ``results/logs/llm_calls.jsonl``; cost from ``llm.prices`` and the
  same calls priced on a reference hosted model (``llm.reference_price``).
* **Rate limits.** ``requests_per_minute`` (0 = none) and exponential back-off on transport errors / 429 / 5xx.
* **Providers.** ``ollama`` (default, local), ``openai`` (any OpenAI-compatible endpoint), ``anthropic``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

import jsonschema
import requests

from .common import append_jsonl, now_iso, read_json, results_dir, write_json


class LLMError(RuntimeError):
    pass


@dataclass
class Answer:
    data: dict | None
    text: str
    valid: bool
    errors: list = field(default_factory=list)
    cached: bool = False
    attempts: int = 1
    usage: dict = field(default_factory=dict)
    request_hash: str = ""
    model_info: dict = field(default_factory=dict)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def extract_json(text: str):
    """The JSON object in a model answer (a ```json fence, or the outermost braces)."""
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except ValueError:
            return None
    return None


class _RateLimiter:
    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self):
        if not self.interval:
            return
        with self.lock:
            t = time.time()
            if t < self.next_t:
                time.sleep(self.next_t - t)
            self.next_t = max(t, self.next_t) + self.interval


class Client:
    """A model behind a provider. ``spec``: {name, provider, model, [base_url, api_key_env, family]}."""

    def __init__(self, spec: dict, cfg: dict):
        self.spec = dict(spec)
        self.name = spec["name"]
        self.provider = spec.get("provider", "ollama")
        self.model = spec.get("model", self.name)
        llm = cfg["llm"]
        self.temperature = float(spec.get("temperature", llm.get("temperature", 0)))
        self.num_ctx = int(spec.get("num_ctx", llm.get("num_ctx", 16384)))
        self.timeout = float(llm.get("timeout_seconds", 600))
        self.max_retries = int(llm.get("max_retries", 4))
        self.seed = int(cfg["project"]["seed"])
        self.prices = (llm.get("prices") or {}).get(self.provider) or spec.get("prices") or {"input": 0, "output": 0}
        self.reference_price = llm.get("reference_price") or {}
        self.limiter = _RateLimiter(float(spec.get("requests_per_minute", llm.get("requests_per_minute", 0)) or 0))
        rd = results_dir(cfg)
        self.cache_dir = rd / "cache" / "llm" / _slug(f"{self.provider}_{self.model}")
        self.ledger = rd / "logs" / "llm_calls.jsonl"
        self._info = None
        self._thinks = False
        self._stop = []
        if self.provider == "ollama":
            self.base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            if not self.base.startswith("http"):
                self.base = "http://" + self.base
        elif self.provider == "openai":
            self.base = spec.get("base_url", "https://api.openai.com/v1").rstrip("/")
        elif self.provider == "anthropic":
            self.base = spec.get("base_url", "https://api.anthropic.com/v1").rstrip("/")
        elif self.provider != "reference":
            raise LLMError(f"Unknown provider {self.provider!r} for {self.name}")

    # ------------------------------------------------------------------ model identity
    def info(self) -> dict:
        """Model name, provider and version (Ollama: digest, parameter size, quantisation)."""
        if self._info is not None:
            return self._info
        info = {"name": self.name, "provider": self.provider, "model": self.model, "temperature": self.temperature,
                "num_ctx": self.num_ctx, "seed": self.seed}
        if self.provider == "ollama":
            try:
                tags = requests.get(f"{self.base}/api/tags", timeout=15).json().get("models", [])
            except (requests.RequestException, ValueError) as e:
                raise LLMError(f"No Ollama server at {self.base}: start it with `ollama serve`") from e
            m = next((t for t in tags if t["name"] in (self.model, f"{self.model}:latest")), None)
            if m is None:
                raise LLMError(f"Ollama model {self.model!r} is not installed: ollama pull {self.model}")
            show = requests.post(f"{self.base}/api/show", json={"model": self.model}, timeout=60).json()
            det = show.get("details") or {}
            self._thinks = "thinking" in (show.get("capabilities") or [])
            info.update(digest=m.get("digest", "")[:12], parameter_size=det.get("parameter_size"),
                        quantization=det.get("quantization_level"), family=det.get("family"),
                        modified_at=m.get("modified_at"))
        self._info = info
        return info

    # ------------------------------------------------------------------ one call
    def _post(self, url, payload, headers=None) -> dict:
        last = None
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                r = requests.post(url, json=payload, headers=headers or {}, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    last = LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                else:
                    r.raise_for_status()
                    return r.json()
            except (requests.ConnectionError, requests.Timeout) as e:
                last = e
            except requests.HTTPError as e:
                raise LLMError(f"{self.name}: HTTP error {e.response.status_code}: {e.response.text[:300]}") from e
            time.sleep(min(60, 2 * 2 ** attempt))
        raise LLMError(f"{self.name}: no answer after {self.max_retries + 1} tries ({last})")

    def _raw(self, system: str, messages: list, schema: dict | None, max_tokens: int) -> tuple[str, dict]:
        t0 = time.time()
        if self.provider == "ollama" and self.spec.get("mode") == "completion":
            # a pre-trained (base) model: the text is continued as it is, no chat template, no instruction
            self.info()
            payload = {"model": self.model, "stream": False, "raw": True, "prompt": (system or "") + messages[-1]["content"],
                       "options": {"temperature": self.temperature, "seed": self.seed, "num_predict": max_tokens,
                                   "num_ctx": self.num_ctx, "stop": list(self.spec.get("stop") or []) + list(self._stop)}}
            d = self._post(f"{self.base}/api/generate", payload)
            text = d.get("response", "")
            usage = {"input_tokens": d.get("prompt_eval_count", 0), "output_tokens": d.get("eval_count", 0),
                     "done_reason": d.get("done_reason")}
        elif self.provider == "ollama":
            self.info()
            payload = {"model": self.model, "stream": False,
                       "messages": ([{"role": "system", "content": system}] if system else []) + messages,
                       "options": {"temperature": self.temperature, "seed": self.seed, "num_predict": max_tokens,
                                   "num_ctx": self.num_ctx}}
            if schema:
                payload["format"] = schema
            if self._thinks:
                payload["think"] = False
            d = self._post(f"{self.base}/api/chat", payload)
            text = d.get("message", {}).get("content", "")
            usage = {"input_tokens": d.get("prompt_eval_count", 0), "output_tokens": d.get("eval_count", 0),
                     "done_reason": d.get("done_reason")}
        elif self.provider == "openai":
            key = os.getenv(self.spec.get("api_key_env", "OPENAI_API_KEY"), "")
            payload = {"model": self.model, "temperature": self.temperature, "max_tokens": max_tokens, "seed": self.seed,
                       "messages": ([{"role": "system", "content": system}] if system else []) + messages}
            if schema:
                payload["response_format"] = {"type": "json_schema",
                                              "json_schema": {"name": "answer", "schema": schema, "strict": False}}
            d = self._post(f"{self.base}/chat/completions", payload, {"Authorization": f"Bearer {key}"})
            text = d["choices"][0]["message"]["content"] or ""
            u = d.get("usage") or {}
            usage = {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0),
                     "done_reason": d["choices"][0].get("finish_reason")}
        elif self.provider == "anthropic":
            key = os.getenv(self.spec.get("api_key_env", "ANTHROPIC_API_KEY"), "")
            sys_text = system + (("\n\nAnswer with one JSON object that follows this JSON schema, and nothing else:\n"
                                  + json.dumps(schema)) if schema else "")
            payload = {"model": self.model, "max_tokens": max_tokens, "temperature": self.temperature,
                       "system": sys_text, "messages": messages}
            d = self._post(f"{self.base}/messages", payload,
                           {"x-api-key": key, "anthropic-version": "2023-06-01"})
            text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
            u = d.get("usage") or {}
            usage = {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
                     "done_reason": d.get("stop_reason")}
        else:
            raise LLMError(f"{self.name}: provider {self.provider} cannot be asked")
        usage["latency_s"] = round(time.time() - t0, 2)
        return text, usage

    def _cost(self, usage: dict, prices: dict) -> float:
        return round((usage.get("input_tokens", 0) * float(prices.get("input", 0))
                      + usage.get("output_tokens", 0) * float(prices.get("output", 0))) / 1e6, 6)

    def ask(self, *, system: str, user: str, schema: dict | None = None, max_tokens: int = 1024,
            prompt_version: str = "v1", phase: str = "", item_id: str = "", validator=None, stop=None) -> Answer:
        """Ask once (cached). ``validator(data) -> list[str]`` adds checks the schema cannot express
        (e.g. an evidence quote must occur in the code)."""
        key_obj = {"provider": self.provider, "model": self.model, "system": system, "user": user, "schema": schema,
                   "temperature": self.temperature, "num_ctx": self.num_ctx, "seed": self.seed,
                   "max_tokens": max_tokens, "prompt_version": prompt_version}
        if self.spec.get("mode"):
            key_obj["mode"] = self.spec["mode"]
        if stop:
            key_obj["stop"] = list(stop)
        self._stop = list(stop or [])
        h = hashlib.sha256(json.dumps(key_obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cpath = self.cache_dir / h[:2] / f"{h}.json"
        cached = read_json(cpath)
        if cached is not None:
            return Answer(data=cached.get("data"), text=cached["attempts"][-1]["text"], valid=cached["valid"],
                          errors=cached.get("errors", []), cached=True, attempts=len(cached["attempts"]),
                          usage=cached.get("usage", {}), request_hash=h, model_info=cached.get("model_info", {}))
        messages = [{"role": "user", "content": user}]
        attempts, data, errors, total = [], None, [], {"input_tokens": 0, "output_tokens": 0, "latency_s": 0.0}
        for attempt in range(self.max_retries + 1):
            text, usage = self._raw(system, messages, schema, max_tokens)
            for k in total:
                total[k] = round(total[k] + (usage.get(k) or 0), 2)
            attempts.append({"text": text, "usage": usage, "time": now_iso()})
            data = extract_json(text) if schema else {"text": text}
            errors = []
            if schema:
                if data is None:
                    errors = ["the answer is not a JSON object" + (" (cut at max_tokens)"
                                                                  if usage.get("done_reason") == "length" else "")]
                else:
                    v = jsonschema.Draft202012Validator(schema)
                    errors = [f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}"[:300]
                              for e in sorted(v.iter_errors(data), key=lambda e: list(e.path))][:8]
            if not errors and validator is not None and data is not None:
                errors = list(validator(data))[:8]
            if not errors:
                break
            if attempt < self.max_retries:
                messages = messages[:1] + [{"role": "assistant", "content": text[:6000]},
                                           {"role": "user", "content": "Your answer is not valid:\n- " + "\n- ".join(errors)
                                            + "\nAnswer again with the complete corrected JSON object only."}]
        valid = not errors
        info = self.info() if self.provider != "reference" else {}
        record = {"request": key_obj, "request_hash": h, "attempts": attempts, "data": data, "valid": valid,
                  "errors": errors, "usage": total, "model_info": info, "phase": phase, "item_id": item_id}
        write_json(cpath, record)
        append_jsonl(self.ledger, {"time": now_iso(), "phase": phase, "model": self.name, "item_id": item_id,
                                   "request_hash": h[:16], "valid": valid, "attempts": len(attempts), **total,
                                   "cost_usd": self._cost(total, self.prices),
                                   "reference_cost_usd": self._cost(total, self.reference_price)})
        return Answer(data=data, text=attempts[-1]["text"], valid=valid, errors=errors, cached=False,
                      attempts=len(attempts), usage=total, request_hash=h, model_info=info)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Unit-length sentence embeddings (Ollama /api/embed)."""
        out = []
        for i in range(0, len(texts), 64):
            d = self._post(f"{self.base}/api/embed", {"model": self.model, "input": texts[i:i + 64]})
            for v in d["embeddings"]:
                n = sum(x * x for x in v) ** 0.5 or 1.0
                out.append([x / n for x in v])
        return out


def annotators(cfg: dict) -> list[Client]:
    return [Client(a, cfg) for a in cfg["llm"]["annotators"]]


def ledger_summary(cfg: dict, phase: str | None = None) -> dict:
    """Calls, tokens, time and cost per model (optionally of one phase) from results/logs/llm_calls.jsonl."""
    from .common import read_jsonl
    rows = [r for r in read_jsonl(results_dir(cfg) / "logs" / "llm_calls.jsonl") if phase is None or r.get("phase") == phase]
    out = {}
    for r in rows:
        m = out.setdefault(r["model"], {"calls": 0, "invalid": 0, "retries": 0, "input_tokens": 0, "output_tokens": 0,
                                        "latency_s": 0.0, "cost_usd": 0.0, "reference_cost_usd": 0.0})
        m["calls"] += 1
        m["invalid"] += 0 if r.get("valid") else 1
        m["retries"] += max(0, int(r.get("attempts", 1)) - 1)
        for k in ("input_tokens", "output_tokens"):
            m[k] += int(r.get(k) or 0)
        for k in ("latency_s", "cost_usd", "reference_cost_usd"):
            m[k] = round(m[k] + float(r.get(k) or 0), 4)
    return out
