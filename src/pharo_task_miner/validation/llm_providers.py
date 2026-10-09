"""Structured-output chat clients for LLM annotation (local open-weight models
or hosted APIs).

One interface, three providers, selected per model profile in ``models.yaml``:

* ``ollama``            — local Ollama server; output constrained by a JSON schema
  (``format``); deterministic decoding (temperature 0, fixed seed).
* ``openai_compatible`` — any ``/v1/chat/completions`` server with
  ``response_format: json_schema``: llama.cpp ``llama-server``, LM Studio, vLLM,
  or hosted APIs (OpenAI, OpenRouter, Together, Groq, …) via ``base_url`` +
  ``api_key_env``.
* ``anthropic``         — Claude through the official ``anthropic`` SDK with
  structured outputs (``output_config.format``) and a cached system prompt.

Every call returns a :class:`Completion` that records what a paper needs to
report: the model that actually served the request, token usage, latency, stop
reason and attempts. Transient failures (rate limits, 5xx, timeouts, connection
errors, malformed JSON) are retried with exponential backoff; configuration
errors (bad key, unknown model) raise :class:`ProviderConfigError` so a run
stops instead of silently recording hundreds of failures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import random
import time


class ProviderConfigError(RuntimeError):
    """Unusable configuration (missing key, unknown model, server not running): abort the run."""


class TransientError(RuntimeError):
    """A failure worth retrying."""


@dataclass
class Completion:
    parsed: dict | None
    raw_text: str
    error: str | None = None
    served_model: str | None = None
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)
    latency_seconds: float | None = None
    attempts: int = 1
    request_id: str | None = None
    notes: list[str] = field(default_factory=list)


def _approx_tokens(*texts: str) -> int:
    # ~3 characters per token is conservative for code (Smalltalk keyword syntax, punctuation).
    return sum(len(t) for t in texts) // 3 + 1


class _Base:
    provider = "base"
    default_concurrency = 1

    def __init__(self, profile: dict):
        self.profile = profile
        self.name = profile.get("profile") or profile.get("model")
        self.model = profile["model"]
        self.max_tokens = int(profile.get("max_tokens") or 2000)
        self.timeout = float(profile.get("timeout_seconds") or 300)
        self.max_attempts = max(1, int(profile.get("max_attempts") or 4))
        self.seed = profile.get("seed", 42)
        self.temperature = profile.get("temperature", 0)
        self.concurrency = int(profile.get("concurrency") or self.default_concurrency)

    # -- public ---------------------------------------------------------------------------------
    def complete(self, system: str, user: str, schema: dict, schema_name: str = "annotation") -> Completion:
        """Call the model until it returns schema-valid JSON or attempts run out."""
        t0 = time.monotonic()
        last_err, last_raw, notes = None, "", []
        for attempt in range(1, self.max_attempts + 1):
            try:
                c = self._call(system, user, schema, schema_name)
            except TransientError as e:
                last_err = str(e)
                notes.append(f"attempt {attempt}: {e}")
                self._sleep(attempt)
                continue
            c.attempts = attempt
            c.latency_seconds = round(time.monotonic() - t0, 3)
            c.notes = notes + c.notes
            if c.error is None:
                try:
                    c.parsed = json.loads(_strip_fences(c.raw_text))
                except json.JSONDecodeError as e:
                    c.error = f"invalid JSON: {e}"
            if c.error is None and not isinstance(c.parsed, dict):
                c.error, c.parsed = "JSON output is not an object", None
            if c.error is None:
                return c
            # A refusal or a truncated answer (usually a model stuck repeating itself) will not
            # change on retry with deterministic decoding: record it instead of paying again.
            if c.stop_reason == "refusal" or "truncated" in (c.error or ""):
                return c
            last_err, last_raw = c.error, c.raw_text
            notes.append(f"attempt {attempt}: {c.error}")
            self._sleep(attempt)
        return Completion(parsed=None, raw_text=last_raw, error=last_err or "failed", attempts=self.max_attempts,
                          latency_seconds=round(time.monotonic() - t0, 3), notes=notes)

    def describe(self) -> dict:
        """Provider settings recorded in the run manifest (never secrets)."""
        return {"provider": self.provider, "model": self.model, "max_tokens": self.max_tokens,
                "temperature": self.temperature, "seed": self.seed, "timeout_seconds": self.timeout,
                "max_attempts": self.max_attempts}

    # -- internals ------------------------------------------------------------------------------
    def _call(self, system: str, user: str, schema: dict, schema_name: str) -> Completion:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _sleep(attempt: int):
        time.sleep(min(60.0, 2.0 * 2 ** (attempt - 1)) + random.uniform(0, 1))


def bounded_schema(schema, max_items: int = 8):
    """Copy of a JSON schema with ``maxItems`` on every array. Grammar-constrained local
    decoders enforce it, which stops small models from looping inside a list until the
    token limit. (Hosted structured-output APIs do not all accept ``maxItems``.)"""
    if isinstance(schema, dict):
        out = {k: bounded_schema(v, max_items) for k, v in schema.items()}
        if out.get("type") == "array":
            out.setdefault("maxItems", max_items)
        return out
    if isinstance(schema, list):
        return [bounded_schema(v, max_items) for v in schema]
    return schema


def _strip_fences(text: str) -> str:
    """Some local servers wrap JSON in ``` fences despite a schema; remove them."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


class _HTTPProvider(_Base):
    def __init__(self, profile: dict):
        super().__init__(profile)
        import requests
        self.requests = requests
        self.session = requests.Session()
        key_env = profile.get("api_key_env")
        self.api_key = os.getenv(key_env) if key_env else None
        if key_env and not self.api_key:
            raise ProviderConfigError(f"{self.name}: environment variable {key_env} is not set (API key).")

    def _post(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        try:
            r = self.session.post(url, json=payload, headers=headers or {}, timeout=self.timeout)
        except self.requests.Timeout as e:
            raise TransientError(f"timeout after {self.timeout:.0f}s") from e
        except self.requests.ConnectionError as e:
            raise TransientError(f"connection error: {e.__class__.__name__}") from e
        if r.status_code in (408, 409, 425, 429) or r.status_code >= 500:
            raise TransientError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code in (401, 403):
            raise ProviderConfigError(f"{self.name}: HTTP {r.status_code} (check the API key): {r.text[:300]}")
        if r.status_code == 404:
            raise ProviderConfigError(f"{self.name}: HTTP 404 (unknown model or endpoint {url}): {r.text[:300]}")
        if r.status_code >= 400:
            raise ProviderConfigError(f"{self.name}: HTTP {r.status_code}: {r.text[:500]}")
        try:
            return r.json()
        except ValueError as e:
            raise TransientError(f"non-JSON HTTP body: {r.text[:200]}") from e


class OllamaProvider(_HTTPProvider):
    provider = "ollama"

    def __init__(self, profile: dict):
        super().__init__(profile)
        self.base_url = (profile.get("base_url") or "http://localhost:11434").rstrip("/")
        self.num_ctx = int(profile.get("num_ctx") or 16384)
        self.max_num_ctx = int(profile.get("max_num_ctx") or 65536)
        self.think = profile.get("think")
        try:
            tags = self.session.get(f"{self.base_url}/api/tags", timeout=10).json()
        except (self.requests.RequestException, ValueError) as e:
            raise ProviderConfigError(f"No Ollama server at {self.base_url} ({e.__class__.__name__}). "
                                      "Start it with `ollama serve`.") from e
        names = {m["name"] for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise ProviderConfigError(f"Model {self.model!r} ({self.name}) is not installed. Run: ollama pull {self.model}")

    def describe(self) -> dict:
        return {**super().describe(), "base_url": self.base_url, "num_ctx": self.num_ctx, "think": self.think}

    def _call(self, system, user, schema, schema_name):
        # Ollama silently truncates prompts longer than num_ctx: size the window per request.
        needed = _approx_tokens(system, user) + self.max_tokens + 256
        num_ctx = self.num_ctx
        while num_ctx < needed and num_ctx < self.max_num_ctx:
            num_ctx *= 2
        if needed > num_ctx:
            return Completion(parsed=None, raw_text="", error=f"prompt too long for the context window "
                              f"(~{needed} tokens > max_num_ctx {self.max_num_ctx})")
        data = self._post(f"{self.base_url}/api/chat", {
            "model": self.model, "stream": False, "format": bounded_schema(schema),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **({"think": self.think} if self.think is not None else {}),
            "options": {"temperature": self.temperature, "seed": self.seed, "num_predict": self.max_tokens,
                        "num_ctx": num_ctx}})
        text = (data.get("message") or {}).get("content", "")
        usage = {"input_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count"), "num_ctx": num_ctx}
        stop = data.get("done_reason")
        c = Completion(parsed=None, raw_text=text, served_model=data.get("model") or self.model, stop_reason=stop, usage=usage)
        if usage["input_tokens"] and usage["input_tokens"] >= num_ctx:
            c.error = "prompt was truncated by the context window"
        elif stop == "length":
            c.error = f"output truncated at max_tokens={self.max_tokens}"
        return c


class OpenAICompatibleProvider(_HTTPProvider):
    provider = "openai_compatible"

    def __init__(self, profile: dict):
        super().__init__(profile)
        self.base_url = (profile.get("base_url") or "http://localhost:8080").rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        # Hosted APIs are fast and rate-limited per minute; local servers are usually single-slot.
        if not profile.get("concurrency"):
            self.concurrency = 4 if self.api_key else 1
        # Newer OpenAI reasoning models reject `max_tokens`/`temperature`; both are configurable.
        self.max_tokens_param = profile.get("max_tokens_param") or "max_tokens"
        self.strict = bool(profile.get("strict_schema", True))
        self.extra = profile.get("extra_body") or {}

    def describe(self) -> dict:
        return {**super().describe(), "base_url": self.base_url, "max_tokens_param": self.max_tokens_param,
                "extra_body": self.extra}

    def _call(self, system, user, schema, schema_name):
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   self.max_tokens_param: self.max_tokens,
                   "response_format": {"type": "json_schema",
                                       "json_schema": {"name": schema_name, "schema": schema, "strict": self.strict}},
                   **self.extra}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.seed is not None:
            payload["seed"] = self.seed
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = self._post(f"{self.base_url}/v1/chat/completions", payload, headers)
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as e:
            raise TransientError(f"unexpected response shape: {str(data)[:200]}") from e
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        u = data.get("usage") or {}
        c = Completion(parsed=None, raw_text=text, served_model=data.get("model") or self.model,
                       stop_reason=choice.get("finish_reason"), request_id=data.get("id"),
                       usage={"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")})
        if msg.get("refusal"):
            c.error, c.stop_reason = f"refusal: {msg['refusal'][:200]}", "refusal"
        elif c.stop_reason == "length":
            c.error = f"output truncated at {self.max_tokens_param}={self.max_tokens}"
        return c


class AnthropicProvider(_Base):
    """Claude via the official SDK: structured outputs + prompt caching of the
    (identical for every item) system prompt."""
    provider = "anthropic"
    default_concurrency = 4

    def __init__(self, profile: dict):
        super().__init__(profile)
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - declared dependency
            raise ProviderConfigError("The `anthropic` package is missing: .venv/bin/pip install anthropic") from e
        self.anthropic = anthropic
        key_env = profile.get("api_key_env") or "ANTHROPIC_API_KEY"
        key = os.getenv(key_env)
        if not key:
            raise ProviderConfigError(f"{self.name}: environment variable {key_env} is not set (Anthropic API key).")
        # The SDK already retries 408/409/429/5xx/connection errors with backoff.
        self.client = anthropic.Anthropic(api_key=key, timeout=self.timeout, max_retries=6)
        self.effort = profile.get("effort")
        self.fallbacks = profile.get("fallbacks")
        # Current Claude models reject sampling parameters; only send them when explicitly configured.
        self.temperature = profile.get("temperature")
        self.seed = None
        if not profile.get("max_tokens"):
            self.max_tokens = 16000  # adaptive thinking shares this budget with the JSON answer

    def describe(self) -> dict:
        return {**super().describe(), "effort": self.effort, "fallbacks": self.fallbacks,
                "thinking": "adaptive (model default)"}

    def _call(self, system, user, schema, schema_name):
        a = self.anthropic
        kwargs = {
            "model": self.model, "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": schema},
                              **({"effort": self.effort} if self.effort else {})},
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            if self.fallbacks:
                resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                        fallbacks=self.fallbacks, **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
        except a.AuthenticationError as e:
            raise ProviderConfigError(f"{self.name}: authentication failed (check ANTHROPIC_API_KEY): {e}") from e
        except a.PermissionDeniedError as e:
            raise ProviderConfigError(f"{self.name}: permission denied: {e}") from e
        except a.NotFoundError as e:
            raise ProviderConfigError(f"{self.name}: unknown model {self.model!r}: {e}") from e
        except a.BadRequestError as e:
            raise ProviderConfigError(f"{self.name}: request rejected: {e}") from e
        except (a.RateLimitError, a.InternalServerError, a.APITimeoutError, a.APIConnectionError) as e:
            raise TransientError(f"{e.__class__.__name__}: {e}") from e
        except a.APIStatusError as e:
            if e.status_code >= 500:
                raise TransientError(f"HTTP {e.status_code}: {e}") from e
            raise ProviderConfigError(f"{self.name}: HTTP {e.status_code}: {e}") from e
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        u = resp.usage
        usage = {"input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None),
                 "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
                 "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None)}
        c = Completion(parsed=None, raw_text=text, served_model=resp.model, stop_reason=resp.stop_reason,
                       usage=usage, request_id=getattr(resp, "_request_id", None))
        if any(getattr(b, "type", None) == "fallback" for b in resp.content):
            c.notes.append(f"served by fallback model {resp.model}")
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            c.error = f"refusal ({getattr(details, 'category', None)})"
        elif resp.stop_reason == "max_tokens":
            c.error = f"output truncated at max_tokens={self.max_tokens}"
        return c


PROVIDERS = {"ollama": OllamaProvider, "openai_compatible": OpenAICompatibleProvider, "anthropic": AnthropicProvider}


def make_provider(profile: dict) -> _Base:
    kind = profile.get("provider", "ollama")
    if kind not in PROVIDERS:
        raise ProviderConfigError(f"{profile.get('profile')}: unknown provider {kind!r} (use one of {sorted(PROVIDERS)})")
    return PROVIDERS[kind](profile)


def requirements(profile: dict) -> list[tuple[str, str]]:
    """What a profile needs before it can run: [("env", VAR) | ("ollama", tag)]."""
    kind = profile.get("provider", "ollama")
    out = []
    base = (profile.get("base_url") or "").rstrip("/")
    local = not base or any(h in base for h in ("://localhost", "://127.0.0.1"))
    if kind == "ollama" and local:
        out.append(("ollama", profile["model"]))
    elif kind == "ollama":
        out.append(("server", f"{base}/api/tags {profile['model']}"))
    elif kind == "openai_compatible" and not profile.get("api_key_env"):
        out.append(("server", f"{base or 'http://localhost:8080'}/v1/models {profile['model']}"))
    if kind == "anthropic":
        out.append(("env", profile.get("api_key_env") or "ANTHROPIC_API_KEY"))
    elif profile.get("api_key_env"):
        out.append(("env", profile["api_key_env"]))
    return out
