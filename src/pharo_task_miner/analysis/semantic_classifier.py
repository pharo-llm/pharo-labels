"""Module 9 — optional semantic analysis of methods and repositories with a
local open-weight LLM (default: Qwen2.5-Coder-7B-Instruct via Ollama).

Disabled by default (``classification.operations.semantic_analysis: false``).
Only a *selected* subset of methods is sent (ambiguous/unknown/representative),
never the whole corpus. Output is constrained to a JSON schema and validated:

* operation ids must exist in the taxonomy (otherwise they are kept only as
  *proposed* new categories, never as labels);
* every label must cite evidence that literally occurs in the method (a sent
  selector, a referenced global or a source substring) — unsupported labels are
  rejected and the rejection is stored;
* model id, prompt version and timestamp are recorded with every result.
"""
from __future__ import annotations

import json
import os

from ..storage.provenance import now_iso

METHOD_PROMPT_VERSION = "method-semantics-v1"
DOMAIN_PROMPT_VERSION = "repository-domain-v1"

METHOD_SYSTEM = """You analyse methods from real Pharo (Smalltalk) repositories for an empirical study.
Describe what the method does and which programming operations it performs, choosing operation ids ONLY
from the provided taxonomy. For each operation cite concrete evidence copied verbatim from the method
(selectors, class names or short source fragments). If an operation is not in the taxonomy, put it in
proposed_new_operations instead. If the purpose cannot be determined, say so in uncertainty and return no
operations. Receiver types are generally not statically known; do not assume them."""

METHOD_SCHEMA = {
    "type": "object",
    "properties": {
        "method_summary": {"type": "string"},
        "operations": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}},
            "required": ["id", "evidence"], "additionalProperties": False}},
        "proposed_new_operations": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                           "evidence": {"type": "array", "items": {"type": "string"}}},
            "required": ["name", "description", "evidence"], "additionalProperties": False}},
        "uncertainty": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["method_summary", "operations", "proposed_new_operations", "uncertainty"],
    "additionalProperties": False,
}

DOMAIN_SYSTEM = """You classify public Pharo repositories by application domain for an empirical study.
Use only the provided documentation and metadata. Choose zero or more domain ids from the list; if none fits,
propose a new domain name. Every label must quote a short excerpt from the provided text as evidence."""

DOMAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "domains": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "excerpt": {"type": "string"}, "source": {"type": "string"}},
            "required": ["id", "excerpt", "source"], "additionalProperties": False}},
        "proposed_domains": {"type": "array", "items": {"type": "string"}},
        "uncertainty": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["domains", "proposed_domains", "uncertainty"],
    "additionalProperties": False,
}


class SemanticUnavailable(RuntimeError):
    pass


def load_registry(cfg) -> dict:
    from ..config import load_yaml
    path = cfg.resolve(cfg["semantic"].get("models_file") or "models.yaml")
    return load_yaml(path) if path.exists() else {"default": None, "models": {}}


def resolve_model(cfg, name: str | None = None) -> dict:
    """Model profile: a name from models.yaml, or any raw Ollama tag (e.g. ``mistral:7b``)."""
    reg = load_registry(cfg)
    name = name or cfg["semantic"].get("use") or reg.get("default") or "qwen2.5-coder-7b"
    base = {k: v for k, v in cfg["semantic"].items() if k not in ("use", "models_file", "max_methods")}
    prof = (reg.get("models") or {}).get(name)
    if prof is None:
        prof = {"provider": "ollama", "model": name}
    out = {**base, **prof, "profile": name}
    return out


def default_ollama_url() -> str:
    """The local Ollama server: OLLAMA_HOST when set (as the ollama command reads it), else localhost:11434."""
    host = os.environ.get("OLLAMA_HOST", "").strip().rstrip("/")
    if not host:
        return "http://localhost:11434"
    return host if "://" in host else f"http://{host}"


def installed_ollama_models(base_url: str | None = None) -> set[str] | None:
    import requests
    base_url = base_url or default_ollama_url()
    try:
        tags = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5).json()
    except (requests.RequestException, ValueError):
        return None
    names = set()
    for m in tags.get("models", []):
        names.add(m["name"])
        if m["name"].endswith(":latest"):
            names.add(m["name"][: -len(":latest")])
    return names


class LLMClient:
    """Structured-output calls to a LOCAL open-weight model.

    * ``provider: ollama`` (default) — Ollama's /api/chat with a JSON-schema ``format``.
    * ``provider: openai_compatible`` — any local server exposing /v1/chat/completions
      with ``response_format`` json_schema (llama.cpp ``llama-server``, LM Studio, vLLM).

    Decoding is deterministic (temperature 0, fixed seed) so runs are reproducible.
    """

    def __init__(self, cfg: dict):
        import requests
        self.requests = requests
        self.provider = cfg.get("provider", "ollama")
        self.model = cfg.get("model", "qwen2.5-coder:7b")
        self.base_url = (cfg.get("base_url") or (default_ollama_url() if self.provider == "ollama"
                                                  else "http://localhost:8080")).rstrip("/")
        self.timeout = int(cfg.get("timeout_seconds", 300))
        self.max_tokens = int(cfg.get("max_tokens", 1500))
        self.num_ctx = int(cfg.get("num_ctx", 8192))
        self.seed = int(cfg.get("seed", 42))
        self.think = cfg.get("think")
        self.profile = cfg.get("profile", self.model)
        try:
            if self.provider == "ollama":
                tags = requests.get(f"{self.base_url}/api/tags", timeout=10).json()
                names = {m["name"] for m in tags.get("models", [])}
                if self.model not in names and f"{self.model}:latest" not in names:
                    raise SemanticUnavailable(f"Model {self.model!r} ({self.profile}) is not installed. "
                                              f"Run: ollama pull {self.model}   (see: pharo-miner models)")
            else:
                requests.get(f"{self.base_url}/v1/models", timeout=10)
        except requests.RequestException as e:
            raise SemanticUnavailable(f"No local LLM server at {self.base_url} ({e}). Start it with `ollama serve` "
                                      "or point semantic.base_url at your llama.cpp / LM Studio server.") from e

    def structured(self, system: str, user: str, schema: dict) -> tuple[dict | None, str, str | None]:
        """Return (parsed_json, raw_text, error)."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            if self.provider == "ollama":
                r = self.requests.post(f"{self.base_url}/api/chat", timeout=self.timeout, json={
                    "model": self.model, "messages": messages, "stream": False, "format": schema,
                    **({"think": self.think} if self.think is not None else {}),
                    "options": {"temperature": 0, "seed": self.seed, "num_predict": self.max_tokens, "num_ctx": self.num_ctx}})
                r.raise_for_status()
                text = r.json().get("message", {}).get("content", "")
            else:
                r = self.requests.post(f"{self.base_url}/v1/chat/completions", timeout=self.timeout, json={
                    "model": self.model, "messages": messages, "temperature": 0, "seed": self.seed,
                    "max_tokens": self.max_tokens,
                    "response_format": {"type": "json_schema",
                                        "json_schema": {"name": "result", "schema": schema, "strict": True}}})
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
        except self.requests.Timeout:
            return None, "", f"timeout after {self.timeout}s"
        except self.requests.RequestException as e:
            return None, "", f"request failed: {e}"
        except (KeyError, IndexError, ValueError) as e:
            return None, "", f"unexpected response: {e}"
        try:
            return json.loads(text), text, None
        except json.JSONDecodeError as e:
            return None, text, f"invalid JSON: {e}"


def _method_user_prompt(method: dict, features: dict, domain: str | None, taxonomy_ids: list[dict]) -> str:
    payload = {
        "repository_domain": domain or "unknown",
        "class_name": method["class_name"] + (" class" if method["side"] == "class" else ""),
        "method_selector": method["selector"],
        "method_source": method["source"],
        "structural_features": {
            "message_sends": sorted({s["selector"] for s in method.get("message_sends") or []}),
            "referenced_globals": sorted({g["name"] for g in method.get("referenced_globals") or []}),
            "flags": sorted(k for k, v in features.items() if v is True),
        },
    }
    return ("Taxonomy (id: definition):\n" + "\n".join(f"- {t['id']}: {t['definition']}" for t in taxonomy_ids)
            + "\n\nMethod:\n" + json.dumps(payload, indent=1))


def validate_method_output(out: dict, method: dict, taxonomy_ids: set[str]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (accepted, rejected, proposals)."""
    sends = {s["selector"] for s in method.get("message_sends") or []}
    globs = {g["name"] for g in method.get("referenced_globals") or []}
    source = method.get("source") or ""
    accepted, rejected = [], []
    for op in out.get("operations") or []:
        oid = op.get("id", "")
        ev = [e for e in op.get("evidence") or [] if e and (e in sends or e in globs or e in source)]
        if oid not in taxonomy_ids:
            rejected.append({"id": oid, "reason": "not a taxonomy category"})
        elif not ev:
            rejected.append({"id": oid, "reason": "no cited evidence occurs in the method", "cited": op.get("evidence")})
        else:
            accepted.append({"id": oid, "evidence": ev})
    proposals = [p for p in out.get("proposed_new_operations") or []
                 if any(e and e in source for e in p.get("evidence") or [])]
    return accepted, rejected, proposals


def describe_method(client: LLMClient, method: dict, features: dict, domain: str | None,
                    taxonomy_cats: list[dict]) -> dict:
    ids = [{"id": c["id"], "definition": c.get("definition", "")} for c in taxonomy_cats if c.get("level") == 3]
    out, raw, err = client.structured(METHOD_SYSTEM, _method_user_prompt(method, features, domain, ids), METHOD_SCHEMA)
    rec = {"model": client.model, "prompt_version": METHOD_PROMPT_VERSION, "created_at": now_iso(), "raw_response": raw,
           "error": err}
    if out is None:
        rec.update({"summary": None, "accepted": [], "rejected": [], "proposals": [], "uncertainty": [err]})
        return rec
    acc, rej, prop = validate_method_output(out, method, {i["id"] for i in ids})
    rec.update({"summary": out.get("method_summary"), "accepted": acc, "rejected": rej, "proposals": prop,
                "uncertainty": out.get("uncertainty") or [], "evidence": sorted({e for a in acc for e in a["evidence"]})})
    return rec


def classify_repository_domain(client: LLMClient, repo: dict, domains: dict) -> dict:
    text = {"full_name": repo.get("full_name"), "description": repo.get("description"), "topics": repo.get("topics"),
            "readme": (repo.get("readme") or "")[:12000], "packages": (repo.get("packages") or [])[:60],
            "dependencies": repo.get("dependencies")}
    user = ("Domains (id: name):\n" + "\n".join(f"- {k}: {v['name']}" for k, v in domains.items())
            + "\n\nRepository:\n" + json.dumps(text, indent=1))
    out, raw, err = client.structured(DOMAIN_SYSTEM, user, DOMAIN_SCHEMA)
    rec = {"model": client.model, "prompt_version": DOMAIN_PROMPT_VERSION, "created_at": now_iso(), "error": err,
           "raw_response": raw, "labels": [], "rejected": [], "proposed": []}
    if out is None:
        return rec
    corpus = json.dumps(text)
    for d in out.get("domains") or []:
        ex = (d.get("excerpt") or "").strip()
        if d.get("id") not in domains:
            rec["rejected"].append({"id": d.get("id"), "reason": "unknown domain id"})
        elif not ex or ex[:60] not in corpus:
            rec["rejected"].append({"id": d.get("id"), "reason": "excerpt not found in provided documentation"})
        else:
            rec["labels"].append({"label": d["id"], "evidence": [{"source": d.get("source"), "excerpt": ex}]})
    rec["proposed"] = out.get("proposed_domains") or []
    return rec
