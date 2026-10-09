"""LLM annotation panel: label the blind annotation sample with one or more
language models acting as independent annotators, then aggregate.

Design (what a methods section can state):

* **Same task as human annotators.** Each model sees exactly what a human sees
  in ``annotation_sample.csv`` (repository, its domains, class, selector,
  protocol, source) and the *same* written guidelines
  (``taxonomy/annotation_guidelines.md``: "For every method" and "Decision
  rules") plus the level-3 category definitions with include/exclude notes.
  Rule-based labels are never shown (blind annotation).
* **Constrained, verifiable output.** Answers follow a JSON schema whose
  category field is an enum of taxonomy ids. Every label must cite evidence
  copied from the method; with ``evidence_policy: enforce`` labels whose
  evidence does not occur in the method are rejected and the rejection is kept.
* **Independent annotators.** Every model labels every item, with no access to
  the others' answers. Inter-model agreement is computed with the same metrics
  as for humans (Cohen's kappa, Krippendorff's alpha, Jaccard).
* **Aggregation.** ``majority`` (a category is kept when more than half of the
  models that answered assign it), ``unanimous``, or an **adjudicator model**
  that sees the anonymised candidate label sets of disagreeing items.
* **Reproducibility and audit.** Raw responses, served model ids, token usage,
  prompt hashes, settings and timestamps are written to
  ``validation/llm/``. Responses are cached per (model, prompt hash, item), so
  an interrupted run resumes and a finished run is not paid for twice.
* **Human calibration.** A random subset (``human_spotcheck.csv``) can be
  labelled blind by people; :func:`spotcheck_report` measures how far the LLM
  labels agree with them (and humans with each other).

LLM labels are stored as ``label_source = llm_adjudicated`` and are never
reported as human-validated.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import threading
import time

from .agreement import agreement_report, cohen_kappa, jaccard
from .llm_providers import ProviderConfigError, make_provider
from ..storage.database import Database, dumps
from ..storage.provenance import now_iso

PROMPT_VERSION = "llm-annotation-v1"
ADJUDICATION_PROMPT_VERSION = "llm-adjudication-v1"
SPECIAL = ("unknown", "insufficient_context")
LABELLED_STATUSES = ("agreed", "majority", "adjudicated", "single_annotation")
REVIEWER_COLUMNS = ["sample_id", "annotator", "operations", "new_category", "too_broad", "too_narrow",
                    "multiple_operations", "evidence_sufficient", "missing_operations", "notes"]


# ============================================================================ prompt
def guideline_sections(text: str, wanted=("For every method", "Decision rules")) -> str:
    """The parts of the human annotation guidelines that define the task."""
    parts = re.split(r"(?m)^## ", text)
    keep = [p for p in parts if any(p.startswith(w) for w in wanted)]
    return "\n".join("## " + p.strip() + "\n" for p in keep) if keep else text


def level3_categories(tax: dict) -> list[dict]:
    names = {c["id"]: c.get("name", c["id"]) for c in tax["categories"]}
    return [{"id": c["id"], "parent": c.get("parent"), "parent_name": names.get(c.get("parent"), c.get("parent")),
             "name": c.get("name", c["id"]), "definition": c.get("definition", ""),
             "include": c.get("include") or [], "exclude": c.get("exclude") or []}
            for c in tax["categories"] if c.get("level") == 3]


def category_reference(cats: list[dict]) -> str:
    by_parent = defaultdict(list)
    for c in cats:
        by_parent[(c["parent"], c["parent_name"])].append(c)
    lines = []
    for (pid, pname), items in by_parent.items():
        lines.append(f"\n### {pname} ({pid})")
        for c in items:
            s = f"- `{c['id']}` — {c['definition']}"
            if c["include"]:
                s += f" Includes: {'; '.join(c['include'])}."
            if c["exclude"]:
                s += f" Excludes: {'; '.join(c['exclude'])}."
            lines.append(s)
    return "\n".join(lines)


SYSTEM_TEMPLATE = """You are an expert Pharo (Smalltalk) developer acting as an annotator in an empirical
software-engineering study. You label methods sampled from real public Pharo repositories with
the programming operations they perform, using a fixed taxonomy. You work independently and
follow the same written guidelines as the human annotators of the study.

# Annotation guidelines (shared with human annotators)
{guidelines}
# How to give your answer
You answer with one JSON object (the schema is enforced). It corresponds to the human annotation
form as follows:
- `operations`: every level-3 category the method performs (multi-label). For each one, `evidence`
  lists short fragments copied verbatim from the method (selectors, class names or code) that show
  the operation. Labels without verbatim evidence are discarded.
- `special_status`: `unknown` if no category fits a non-trivial method (then describe the operation
  in `new_category` and `new_category_description`); `insufficient_context` if the method is too
  small or context-dependent to judge; otherwise `none`. Use a special status only with an empty
  `operations` list.
- Keep evidence short: at most three fragments per category, each a few words long.
- `too_broad`, `too_narrow`, `multiple_operations`, `evidence_sufficient`, `missing_operations`:
  the review questions of the guidelines.
- `rationale`: one to three sentences explaining your decision. `notes`: optional short remark.
Smalltalk is dynamically typed: do not assume receiver types that the code does not show.

# Level-3 categories (the only allowed ids)
{categories}
"""

USER_TEMPLATE = """Annotate this method.

Repository: {repository}
Repository domains (automatic, may be empty): {repository_domains}
Class: {class}
Selector: {selector}
Protocol: {protocol}

Source:
```smalltalk
{source}
```"""

ADJUDICATION_SUFFIX = """

# Adjudication
Several annotators labelled this method independently and disagreed. Their anonymised answers are
below. Decide the final labels yourself by applying the guidelines to the source; do not count
votes and do not favour any annotator. You may keep, drop or add categories. Answer with the same
JSON object as an annotator.

{candidates}"""


def output_schema(cat_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "operations": {"type": "array", "items": {
                "type": "object",
                "properties": {"id": {"type": "string", "enum": sorted(cat_ids)},
                               "evidence": {"type": "array", "items": {"type": "string"}}},
                "required": ["id", "evidence"], "additionalProperties": False}},
            "special_status": {"type": "string", "enum": ["none", "unknown", "insufficient_context"]},
            "new_category": {"type": "string"},
            "new_category_description": {"type": "string"},
            "too_broad": {"type": "boolean"},
            "too_narrow": {"type": "boolean"},
            "multiple_operations": {"type": "boolean"},
            "evidence_sufficient": {"type": "string", "enum": ["yes", "partly", "no"]},
            "missing_operations": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["rationale", "operations", "special_status", "new_category", "new_category_description",
                     "too_broad", "too_narrow", "multiple_operations", "evidence_sufficient", "missing_operations",
                     "notes"],
        "additionalProperties": False,
    }


class Prompt:
    def __init__(self, guidelines_text: str, tax: dict):
        self.cats = level3_categories(tax)
        self.ids = [c["id"] for c in self.cats]
        self.system = SYSTEM_TEMPLATE.format(guidelines=guideline_sections(guidelines_text).strip() + "\n",
                                             categories=category_reference(self.cats).strip())
        self.schema = output_schema(self.ids)
        self.sha256 = hashlib.sha256((PROMPT_VERSION + self.system + USER_TEMPLATE
                                      + json.dumps(self.schema, sort_keys=True)).encode()).hexdigest()

    @staticmethod
    def user(item: dict) -> str:
        return USER_TEMPLATE.format(**{k: (item.get(k) or "") for k in
                                       ("repository", "repository_domains", "class", "selector", "protocol", "source")})

    def adjudication_user(self, item: dict, candidates: list[dict]) -> str:
        blocks = []
        for i, c in enumerate(candidates):
            blocks.append(f"Annotator {chr(65 + i)}: labels = {', '.join(sorted(c['labels'])) or '(none)'}\n"
                          f"  rationale: {c.get('rationale') or '-'}")
        return self.user(item) + ADJUDICATION_SUFFIX.format(candidates="\n".join(blocks))


# ============================================================================ validation of one answer
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


def interpret(parsed: dict, item: dict, valid_ids: set[str], evidence_policy: str = "enforce") -> dict:
    """Turn a schema-valid answer into an annotation decision (same fields as a human row)."""
    context = _norm(" ".join(str(item.get(k) or "") for k in ("class", "selector", "protocol", "source")))
    accepted, rejected, notes = [], [], []
    seen = set()
    for op in parsed.get("operations") or []:
        oid = (op.get("id") or "").strip()
        if oid in seen:
            continue
        seen.add(oid)
        ev = [e for e in (op.get("evidence") or []) if _norm(e) and _norm(e) in context]
        if oid not in valid_ids:
            rejected.append({"id": oid, "reason": "not a taxonomy category"})
        elif evidence_policy == "enforce" and not ev:
            rejected.append({"id": oid, "reason": "no cited evidence occurs in the method", "cited": op.get("evidence")})
        else:
            accepted.append({"id": oid, "evidence": ev})
    special = parsed.get("special_status") or "none"
    labels = {a["id"] for a in accepted}
    if labels and special in SPECIAL:
        notes.append(f"special_status {special} ignored because operations were given")
    if not labels:
        if special in SPECIAL:
            labels = {special}
        elif rejected:
            labels = {"unknown"}
            notes.append("all proposed operations were rejected (no verifiable evidence)")
        else:
            labels = {"unknown"}
            notes.append("no operations and no special status given")
    return {
        "labels": sorted(labels), "accepted": accepted, "rejected": rejected,
        "new_category": (parsed.get("new_category") or "").strip(),
        "new_category_description": (parsed.get("new_category_description") or "").strip(),
        "too_broad": bool(parsed.get("too_broad")), "too_narrow": bool(parsed.get("too_narrow")),
        "multiple_operations": bool(parsed.get("multiple_operations")),
        "evidence_sufficient": parsed.get("evidence_sufficient") or "",
        "missing_operations": (parsed.get("missing_operations") or "").strip(),
        "rationale": (parsed.get("rationale") or "").strip(), "notes": (parsed.get("notes") or "").strip(),
        "interpretation_notes": notes,
    }


def decision_row(sample_id: str, annotator: str, d: dict) -> dict:
    """A row in the human ``reviewer_decisions.csv`` format."""
    note = " | ".join(x for x in (d.get("rationale"), d.get("notes")) if x)
    return {"sample_id": sample_id, "annotator": annotator, "operations": ";".join(d["labels"]),
            "new_category": d.get("new_category") or "", "too_broad": "1" if d.get("too_broad") else "",
            "too_narrow": "1" if d.get("too_narrow") else "",
            "multiple_operations": "1" if d.get("multiple_operations") else "",
            "evidence_sufficient": d.get("evidence_sufficient") or "",
            "missing_operations": d.get("missing_operations") or "", "notes": note}


# ============================================================================ aggregation
def aggregate(labels_by_annotator: dict[str, set[str]], rule: str = "majority") -> tuple[set[str], str, dict]:
    """Combine independent label sets. Returns (labels, status, votes)."""
    n = len(labels_by_annotator)
    votes = Counter(l for s in labels_by_annotator.values() for l in s)
    if n == 0:
        return set(), "incomplete", {}
    if n == 1:
        return set(next(iter(labels_by_annotator.values()))), "single_annotation", dict(votes)
    sets = {frozenset(s) for s in labels_by_annotator.values()}
    if len(sets) == 1:
        return set(next(iter(sets))), "agreed", dict(votes)
    need = n if rule == "unanimous" else n // 2 + 1
    chosen = {l for l, v in votes.items() if v >= need}
    if chosen - set(SPECIAL):
        chosen -= set(SPECIAL)
    return chosen, ("majority" if chosen else "no_consensus"), dict(votes)


# ============================================================================ I/O helpers
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], cols: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


class ResponseCache:
    """Append-only JSONL of model responses; one file per annotator. Only
    successful answers are reused, so failed items are retried on the next run."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.done: dict[tuple, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut by an interrupted run
                if r.get("status") == "ok":
                    self.done[(r["sample_id"], r["cache_key"])] = r

    def get(self, sample_id: str, key: str) -> dict | None:
        return self.done.get((sample_id, key))

    def add(self, rec: dict):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec.get("status") == "ok":
                self.done[(rec["sample_id"], rec["cache_key"])] = rec


# ============================================================================ running a model
def _record(kind, annotator, prov, item, key, prompt_sha, prompt_version, comp, decision) -> dict:
    return {"kind": kind, "sample_id": item["sample_id"], "method_id": item.get("method_id"), "annotator": annotator,
            "profile": prov.name, "provider": prov.provider, "model": prov.model, "served_model": comp.served_model,
            "prompt_version": prompt_version, "prompt_sha256": prompt_sha, "cache_key": key, "created_at": now_iso(),
            "status": "ok" if decision is not None else "failed", "error": comp.error, "stop_reason": comp.stop_reason,
            "attempts": comp.attempts, "latency_seconds": comp.latency_seconds, "usage": comp.usage,
            "request_id": comp.request_id, "notes": comp.notes, "raw_response": comp.raw_text, "parsed": comp.parsed,
            "decision": decision}


def _progress(log, label, done, total, t0):
    el = time.monotonic() - t0
    rem = el / done * (total - done) if done else 0
    log(f"  {label}: {done}/{total} — elapsed {el / 60:.0f} min, remaining ~{rem / 60:.0f} min")


def annotate_with(prov, prompt: Prompt, items: list[dict], cache: ResponseCache, evidence_policy: str, log,
                  concurrency: int | None = None) -> dict[str, dict]:
    """Label every item with one model. Returns sample_id -> record (ok or failed)."""
    annotator = f"llm:{prov.name}"
    key = hashlib.sha256(f"{prompt.sha256}|{json.dumps(prov.describe(), sort_keys=True, default=str)}".encode()).hexdigest()[:16]
    out, todo = {}, []
    for it in items:
        hit = cache.get(it["sample_id"], key)
        if hit:
            out[it["sample_id"]] = hit
        else:
            todo.append(it)
    if out:
        log(f"  {annotator}: {len(out)} cached, {len(todo)} to label")
    if not todo:
        return out
    valid = set(prompt.ids)

    def one(it):
        comp = prov.complete(prompt.system, prompt.user(it), prompt.schema)
        dec = interpret(comp.parsed, it, valid, evidence_policy) if comp.parsed is not None and comp.error is None else None
        rec = _record("annotation", annotator, prov, it, key, prompt.sha256, PROMPT_VERSION, comp, dec)
        cache.add(rec)
        return rec

    workers = max(1, int(concurrency or prov.concurrency))
    t0, done, step = time.monotonic(), 0, max(1, min(25, len(todo) // 10 or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, it): it for it in todo}
        try:
            for f in as_completed(futs):
                rec = f.result()  # ProviderConfigError propagates and stops the run
                out[rec["sample_id"]] = rec
                done += 1
                if rec["status"] != "ok":
                    log(f"  {annotator}: {rec['sample_id']} failed: {rec['error']}")
                if done % step == 0 or done == len(todo):
                    _progress(log, annotator, done, len(todo), t0)
        except BaseException:
            for f in futs:
                f.cancel()
            raise
    return out


def adjudicate_with(prov, prompt: Prompt, items: dict[str, dict], disagreements: dict[str, list[dict]],
                    cache: ResponseCache, evidence_policy: str, log, seed: int) -> dict[str, dict]:
    annotator = f"llm-adjudicator:{prov.name}"
    base = hashlib.sha256(f"{ADJUDICATION_PROMPT_VERSION}|{prompt.sha256}|"
                          f"{json.dumps(prov.describe(), sort_keys=True, default=str)}".encode()).hexdigest()[:12]
    out, valid = {}, set(prompt.ids)
    t0, n = time.monotonic(), len(disagreements)
    for i, (sid, cands) in enumerate(sorted(disagreements.items()), 1):
        cands = list(cands)
        random.Random(f"{seed}:{sid}").shuffle(cands)  # anonymise: order carries no model identity
        ckey = base + hashlib.sha256(json.dumps([sorted(c["labels"]) for c in cands]).encode()).hexdigest()[:8]
        hit = cache.get(sid, ckey)
        if hit:
            out[sid] = hit
            continue
        item = items[sid]
        user = prompt.adjudication_user(item, cands)
        comp = prov.complete(prompt.system, user, prompt.schema, "adjudication")
        dec = interpret(comp.parsed, item, valid, evidence_policy) if comp.parsed is not None and comp.error is None else None
        rec = _record("adjudication", annotator, prov, item, ckey, prompt.sha256, ADJUDICATION_PROMPT_VERSION, comp, dec)
        cache.add(rec)
        out[sid] = rec
        if rec["status"] != "ok":
            log(f"  {annotator}: {sid} failed: {rec['error']}")
        if i % 10 == 0 or i == n:
            _progress(log, annotator, i, n, t0)
    return out


# ============================================================================ the whole run
def run_panel(cfg, db: Database, profiles: list[dict], adjudicator: dict | None, tax: dict, guidelines_text: str,
              log, limit: int = 0, consensus: str = "majority", evidence_policy: str = "enforce",
              concurrency: int | None = None, spotcheck_size: int = 30, seed: int = 42) -> dict:
    vdir = cfg.validation_dir
    odir = vdir / "llm"
    items = read_csv(vdir / "annotation_sample.csv")
    if not items:
        raise FileNotFoundError(f"{vdir / 'annotation_sample.csv'} not found: run the mining step (export-annotations) first")
    items.sort(key=lambda r: r["sample_id"])
    if limit:
        items = items[:limit]
    by_id = {r["sample_id"]: r for r in items}
    prompt = Prompt(guidelines_text, tax)
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "prompt_system.md").write_text(prompt.system, encoding="utf-8")
    (odir / "prompt_user_example.md").write_text(prompt.user(items[0]), encoding="utf-8")
    (odir / "output_schema.json").write_text(json.dumps(prompt.schema, indent=1), encoding="utf-8")

    # Connect every provider before any request: a wrong key or model fails in seconds, not after an hour.
    providers = [make_provider(p) for p in profiles]
    names = [p.name for p in providers]
    if len(set(names)) != len(names):
        raise ProviderConfigError(f"the same model appears twice in the panel: {names}")
    adj = make_provider(adjudicator) if adjudicator else None
    started = now_iso()
    log(f"LLM annotation of {len(items)} items by {len(providers)} model(s): {', '.join(names)}"
        + (f"; adjudicator: {adj.name}" if adj else f"; aggregation: {consensus}"))
    if len(providers) == 1:
        log("  note: a single model gives no inter-annotator agreement; use 2–3 diverse models for a paper.")
    if adj and adj.name in names:
        log(f"  note: the adjudicator {adj.name} is also a panel member; an independent model is preferable.")

    results = {}
    for prov in providers:
        cache = ResponseCache(odir / "raw" / f"{_safe(prov.name)}.jsonl")
        results[f"llm:{prov.name}"] = annotate_with(prov, prompt, items, cache, evidence_policy, log, concurrency)

    # ---- per-item decisions and inter-model agreement
    decisions = defaultdict(dict)  # sid -> annotator -> set
    rows = []
    for ann, recs in results.items():
        for sid, rec in recs.items():
            if rec["status"] == "ok":
                decisions[sid][ann] = set(rec["decision"]["labels"])
                rows.append(decision_row(sid, ann, rec["decision"]))
    rows.sort(key=lambda r: (r["sample_id"], r["annotator"]))
    write_csv(odir / "llm_decisions.csv", rows, REVIEWER_COLUMNS)
    agreement = agreement_report(decisions, sorted(set(prompt.ids) | set(SPECIAL)))

    # ---- aggregation
    panel = [f"llm:{n}" for n in names]
    min_answers = len(panel) if len(panel) <= 2 else len(panel) // 2 + 1
    final, disagreements = {}, {}
    for sid in by_id:
        got = decisions.get(sid, {})
        if len(got) < min_answers:
            final[sid] = (set(), "incomplete", dict(Counter(l for s in got.values() for l in s)), "")
            continue
        labels, status, votes = aggregate(got, consensus)
        final[sid] = (labels, status, votes, consensus if status == "majority" else "")
        if adj and status in ("majority", "no_consensus"):
            disagreements[sid] = [{"labels": got[a], "rationale": results[a][sid]["decision"]["rationale"]}
                                  for a in panel if a in got]
    adj_recs = {}
    if adj and disagreements:
        log(f"  adjudicating {len(disagreements)} disagreement(s) with {adj.name}")
        cache = ResponseCache(odir / "raw" / f"adjudicator__{_safe(adj.name)}.jsonl")
        adj_recs = adjudicate_with(adj, prompt, by_id, disagreements, cache, evidence_policy, log, seed)
        for sid, rec in adj_recs.items():
            if rec["status"] == "ok":
                final[sid] = (set(rec["decision"]["labels"]), "adjudicated", final[sid][2], f"adjudicator:{adj.name}")
            else:
                final[sid] = (final[sid][0], final[sid][1], final[sid][2], final[sid][3] + " (adjudication failed)")

    adj_rows = []
    for sid in sorted(by_id):
        labels, status, votes, how = final[sid]
        adj_rows.append({"sample_id": sid, "method_id": by_id[sid]["method_id"], "operations": ";".join(sorted(labels)),
                         "status": status, "decided_by": how or status, "n_annotators": len(decisions.get(sid, {})),
                         "votes": json.dumps(dict(sorted(votes.items()))),
                         "annotator_labels": json.dumps({a: sorted(s) for a, s in sorted(decisions.get(sid, {}).items())})})
    write_csv(odir / "llm_adjudicated_labels.csv", adj_rows,
              ["sample_id", "method_id", "operations", "status", "decided_by", "n_annotators", "votes", "annotator_labels"])
    write_csv(odir / "llm_disagreements.csv",
              [r for r in adj_rows if r["status"] in ("majority", "no_consensus", "adjudicated")],
              ["sample_id", "method_id", "operations", "status", "decided_by", "votes", "annotator_labels"])

    # ---- failures and per-model statistics
    fails = [{"annotator": a, "sample_id": sid, "error": r["error"], "attempts": r["attempts"]}
             for a, recs in list(results.items()) + [(f"llm-adjudicator:{adj.name}" if adj else "-", adj_recs)]
             for sid, r in recs.items() if r["status"] != "ok"]
    write_csv(odir / "failures.csv", fails, ["annotator", "sample_id", "error", "attempts"])
    per_model = {a: _model_stats(recs) for a, recs in results.items()}
    if adj:
        per_model[f"llm-adjudicator:{adj.name}"] = _model_stats(adj_recs)

    # ---- spot-check template for humans (never overwritten once it exists)
    spot = _export_spotcheck(odir / "human_spotcheck.csv", by_id, final, spotcheck_size, seed)

    status_counts = dict(Counter(v[1] for v in final.values()))
    manifest = {
        "started_at": started, "finished_at": now_iso(), "prompt_version": PROMPT_VERSION,
        "adjudication_prompt_version": ADJUDICATION_PROMPT_VERSION if adj else None,
        "prompt_sha256": prompt.sha256, "taxonomy_version": str(tax.get("version")),
        "taxonomy_level3_categories": len(prompt.ids),
        "guidelines_sha256": hashlib.sha256(guidelines_text.encode()).hexdigest(),
        "items": len(items), "limit": limit or None, "panel": [p.describe() | {"name": p.name} for p in providers],
        "adjudicator": (adj.describe() | {"name": adj.name}) if adj else None,
        "aggregation": "adjudicator" if adj else consensus, "min_answers_per_item": min_answers,
        "evidence_policy": evidence_policy, "status_counts": status_counts, "per_model": per_model,
        "served_models": {a: sorted({r.get("served_model") or "?" for r in recs.values() if r["status"] == "ok"})
                          for a, recs in results.items()},
        "agreement": agreement, "spotcheck": spot,
        "note": "Labels produced by language models. Report them as LLM-annotated, not human-validated.",
    }
    (odir / "run_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    (odir / "agreement_report.json").write_text(json.dumps(agreement, indent=1), encoding="utf-8")
    n_imported = import_llm_labels(db, adj_rows, rows, by_id, panel, PROMPT_VERSION)
    return {"items": len(items), "models": names, "adjudicator": adj.name if adj else None,
            "status_counts": status_counts, "labelled_items": n_imported, "failures": len(fails),
            "mean_jaccard_between_models": agreement.get("mean_jaccard"),
            "exact_agreement_between_models": agreement.get("exact_set_agreement"),
            "output_dir": str(odir), "spotcheck": spot}


def _model_stats(recs: dict[str, dict]) -> dict:
    ok = [r for r in recs.values() if r["status"] == "ok"]
    lat = [r["latency_seconds"] for r in ok if r.get("latency_seconds") is not None]
    usage = Counter()
    for r in recs.values():
        for k, v in (r.get("usage") or {}).items():
            if isinstance(v, (int, float)) and k != "num_ctx":
                usage[k] += v
    acc = sum(len(r["decision"]["accepted"]) for r in ok)
    rej = sum(len(r["decision"]["rejected"]) for r in ok)
    return {"answered": len(ok), "failed": len(recs) - len(ok), "labels_accepted": acc, "labels_rejected": rej,
            "label_rejection_rate": round(rej / (acc + rej), 3) if acc + rej else None,
            "mean_labels_per_item": round(sum(len(r["decision"]["labels"]) for r in ok) / len(ok), 2) if ok else None,
            "mean_latency_seconds": round(sum(lat) / len(lat), 2) if lat else None, "tokens": dict(usage)}


SPOTCHECK_COLUMNS = ["sample_id", "annotator", "operations", "new_category", "notes",
                     "repository", "class", "selector", "source"]


def _export_spotcheck(path: Path, items: dict[str, dict], final: dict, size: int, seed: int) -> dict:
    if path.exists():
        filled = sum(1 for r in read_csv(path) if (r.get("operations") or "").strip())
        return {"path": str(path), "status": "kept_existing", "filled_rows": filled}
    pool = sorted(s for s, v in final.items() if v[1] in LABELLED_STATUSES)
    if not size or not pool:
        return {"path": None, "status": "not_requested"}
    chosen = sorted(random.Random(seed).sample(pool, min(size, len(pool))))
    rows = [{"sample_id": s, "annotator": "human_1", **{k: items[s].get(k, "") for k in
                                                         ("repository", "class", "selector", "source")}} for s in chosen]
    write_csv(path, rows, SPOTCHECK_COLUMNS)
    return {"path": str(path), "status": "exported", "items": len(rows)}


# ============================================================================ database
def import_llm_labels(db: Database, adj_rows: list[dict], decision_rows: list[dict], items: dict[str, dict],
                      panel: list[str], prompt_version: str) -> int:
    db.execute("DELETE FROM annotation_decisions WHERE kind IN ('llm_annotation','llm_adjudicated')")
    db.execute("DELETE FROM method_labels WHERE label_source='llm_adjudicated'")
    ts = now_iso()
    for r in decision_rows:
        flags = {k: r[k] == "1" for k in ("too_broad", "too_narrow", "multiple_operations")}
        flags["evidence_sufficient"] = r["evidence_sufficient"] or None
        flags["missing_operations"] = r["missing_operations"] or None
        db.execute("INSERT INTO annotation_decisions (sample_id, method_id, annotator, kind, operations_json, flags_json,"
                   " new_category, notes, decided_at, source_file, imported_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (r["sample_id"], int(items[r["sample_id"]]["method_id"]), r["annotator"], "llm_annotation",
                    dumps(sorted(x for x in r["operations"].split(";") if x)), dumps(flags), r["new_category"] or None,
                    r["notes"] or None, None, "llm/llm_decisions.csv", ts))
    n = 0
    for r in adj_rows:
        if r["status"] not in LABELLED_STATUSES:
            continue
        mid = int(r["method_id"])
        ops = sorted(x for x in r["operations"].split(";") if x)
        db.execute("INSERT INTO annotation_decisions (sample_id, method_id, annotator, kind, operations_json, flags_json,"
                   " new_category, notes, decided_at, source_file, imported_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (r["sample_id"], mid, r["decided_by"], "llm_adjudicated", dumps(ops),
                    dumps({"status": r["status"], "votes": json.loads(r["votes"])}), None, None, None,
                    "llm/llm_adjudicated_labels.csv", ts))
        for o in ops:
            db.upsert("method_labels", {"method_id": mid, "operation_id": o, "label_source": "llm_adjudicated",
                                        "label_status": "llm_consensus", "score": None, "score_kind": None,
                                        "taxonomy_version": None, "review_status": r["status"], "model": ",".join(panel),
                                        "prompt_version": prompt_version, "created_at": ts, "run_id": None},
                      ["method_id", "operation_id", "label_source"])
        n += 1
    db.commit()
    return n


# ============================================================================ human spot-check
def _bootstrap_ci(values: list[float], seed: int, iters: int = 2000, level: float = 0.95):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(iters))
    lo, hi = means[int((1 - level) / 2 * iters)], means[int((1 + level) / 2 * iters) - 1]
    return [round(lo, 3), round(hi, 3)]


def compare(reference: dict[str, set], other: dict[str, set], categories: list[str], seed: int = 42) -> dict:
    """Agreement of `other` with `reference` (a human) on their common items."""
    common = sorted(set(reference) & set(other))
    if not common:
        return {"items": 0}
    jac = [jaccard(reference[s], other[s]) for s in common]
    tp = sum(len(reference[s] & other[s]) for s in common)
    fp = sum(len(other[s] - reference[s]) for s in common)
    fn = sum(len(reference[s] - other[s]) for s in common)
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    kappas = {}
    for c in categories:
        a = [int(c in reference[s]) for s in common]
        b = [int(c in other[s]) for s in common]
        if sum(a) + sum(b) == 0:
            continue
        k = cohen_kappa(a, b)
        kappas[c] = {"reference_positive": sum(a), "other_positive": sum(b), "kappa": None if k is None else round(k, 3)}
    valid_k = [v["kappa"] for v in kappas.values() if v["kappa"] is not None and v["reference_positive"] >= 3]
    return {"items": len(common), "exact_set_agreement": round(sum(1 for s in common if reference[s] == other[s]) / len(common), 3),
            "mean_jaccard": round(sum(jac) / len(jac), 3), "mean_jaccard_95ci": _bootstrap_ci(jac, seed),
            "micro_precision": None if p is None else round(p, 3), "micro_recall": None if r is None else round(r, 3),
            "micro_f1": None if f1 is None else round(f1, 3),
            "mean_kappa_categories_with_3plus_reference_positives": round(sum(valid_k) / len(valid_k), 3) if valid_k else None,
            "per_category": kappas}


def spotcheck_report(vdir: Path, categories: list[str], seed: int = 42) -> dict:
    odir = vdir / "llm"
    rows = [r for r in read_csv(odir / "human_spotcheck.csv") if (r.get("operations") or "").strip()]
    if not rows:
        return {"status": "no_human_answers", "path": str(odir / "human_spotcheck.csv")}
    split = lambda v: {x.strip() for x in (v or "").replace(",", ";").split(";") if x.strip()}  # noqa: E731
    humans = defaultdict(dict)
    unknown_ids = Counter()
    for r in rows:
        ops = split(r["operations"])
        unknown_ids.update(ops - set(categories) - set(SPECIAL) - {"ambiguous"})
        humans[(r.get("annotator") or "human_1").strip()][r["sample_id"]] = ops
    llm = defaultdict(dict)
    for r in read_csv(odir / "llm_decisions.csv"):
        llm[r["annotator"]][r["sample_id"]] = split(r["operations"])
    for r in read_csv(odir / "llm_adjudicated_labels.csv"):
        if r["status"] in LABELLED_STATUSES:
            llm["llm:final_labels"][r["sample_id"]] = split(r["operations"])
    cats = sorted(set(categories) | set(SPECIAL))
    out = {"status": "computed", "human_annotators": sorted(humans), "items_answered": {h: len(v) for h, v in humans.items()},
           "unknown_operation_ids": dict(unknown_ids), "comparisons": {}}
    for h, ref in sorted(humans.items()):
        for name, other in sorted(llm.items()):
            out["comparisons"][f"{h} vs {name}"] = compare(ref, other, cats, seed)
        for h2, other in sorted(humans.items()):
            if h2 != h:
                out["comparisons"][f"{h} vs {h2}"] = compare(ref, other, cats, seed)
    (odir / "spotcheck_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    write_csv(odir / "spotcheck_summary.csv",
              [{"comparison": k, **{m: v.get(m) for m in ("items", "exact_set_agreement", "mean_jaccard", "micro_precision",
                                                         "micro_recall", "micro_f1",
                                                         "mean_kappa_categories_with_3plus_reference_positives")},
                "mean_jaccard_95ci": json.dumps(v.get("mean_jaccard_95ci"))} for k, v in out["comparisons"].items()],
              ["comparison", "items", "exact_set_agreement", "mean_jaccard", "mean_jaccard_95ci", "micro_precision",
               "micro_recall", "micro_f1", "mean_kappa_categories_with_3plus_reference_positives"])
    return out
