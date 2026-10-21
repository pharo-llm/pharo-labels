"""Phase 7 — Independent structured annotation.

Each of the three annotators, alone, classifies every item of the ``annotation`` split (literature tasks
and Pharo methods the taxonomy was NOT built from) with the approved taxonomy: 1 to 3 behaviour IDs (the
answer is constrained to the taxonomy's IDs plus UNKNOWN and INSUFFICIENT), each with its evidence and a
confidence, plus an explanation and an uncertainty level. One item per request, the same prompt for every
model; the taxonomy prefix is identical across items (the model's prompt cache is reused).

Saved for every (item, model): item ID and source, model name and version (digest), taxonomy version,
predicted category / subcategory / behaviour IDs, evidence (and whether it occurs in the item), explanation,
uncertainty, prompt version and request hash, tokens and latency, validity and attempts.

Outputs
  annotations_raw.jsonl            every answer, valid or not, exactly as parsed
  annotations_validated.parquet    valid answers, one row per (item, model), derived columns
  figures/: categories_by_model.png, unknown_insufficient.png, behaviors_per_item.png, validity.png
"""
from __future__ import annotations

from collections import Counter
import re

import pandas as pd

from .. import figures as F
from .. import taxonomy as T
from ..common import append_jsonl, now_iso, read_jsonl, stable_hash, write_parquet
from ..items import load_items
from ..llm import annotators, ledger_summary

PROMPT_VERSION = "annotation-v1"

SYSTEM = """You classify code items with a fixed taxonomy of PROGRAMMING BEHAVIOURS. An item is either a Pharo (Smalltalk) method or a task of a code benchmark (then classify what the solution code must do).

Rules:
- Choose 1 to MAXB behaviour IDs from the taxonomy below, the most important first. Several only when the item clearly combines them.
- Classify what the code DOES (or must do), not the kind of benchmark task, not the application domain alone.
- Follow the USE WHEN / NOT WHEN rules to choose between neighbouring behaviours.
- evidence: a short quote (at most 15 words) copied exactly from the item.
- confidence: 0 to 1, how sure you are of this behaviour.
- UNKNOWN: no behaviour of the taxonomy fits; then describe the missing behaviour in missing_behavior. INSUFFICIENT: the item does not show enough to decide. UNKNOWN and INSUFFICIENT are used alone.
- uncertainty: low, medium or high for the whole answer.

TAXONOMY
"""


def schema(ids: list[str], maxb: int) -> dict:
    return {"type": "object", "properties": {
        "behaviors": {"type": "array", "minItems": 1, "maxItems": maxb, "items": {"type": "object", "properties": {
            "behavior_id": {"type": "string", "enum": ids + list(T.RESERVED)},
            "evidence": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
            "required": ["behavior_id", "evidence", "confidence"]}},
        "explanation": {"type": "string"}, "uncertainty": {"type": "string", "enum": ["low", "medium", "high"]},
        "missing_behavior": {"type": "string"}},
        "required": ["behaviors", "explanation", "uncertainty"]}


def _validator(data: dict) -> list[str]:
    ids = [b.get("behavior_id") for b in data.get("behaviors", [])]
    errs = []
    if len(set(ids)) != len(ids):
        errs.append("the same behaviour_id is given twice")
    if any(i in T.RESERVED for i in ids) and len(ids) > 1:
        errs.append("UNKNOWN or INSUFFICIENT must be the only answer")
    return errs


def _norm(s: str) -> str:
    return " ".join(re.sub(r"\s+", " ", str(s).lower()).split())


def evidence_found(evidence: str, text: str) -> bool:
    """The quote occurs in the item (ignoring case and layout), or most of its words do (quotes get trimmed)."""
    e, t = _norm(evidence).strip(" .\"'`"), _norm(text)
    if not e:
        return False
    if e in t:
        return True
    words = [w for w in re.findall(r"[a-z0-9_:#]+", e) if len(w) > 2]
    return bool(words) and sum(w in t for w in words) / len(words) >= 0.8


def taxonomy_digest(tax: dict) -> str:
    """Content hash of a taxonomy: two different taxonomies can carry the same version label (a draft rebuilt
    by phase 6), so answers are matched to the taxonomy they were made with by this hash, never by the label."""
    return stable_hash({k: tax[k] for k in ("categories", "subcategories", "behaviors")})


def classify(ph, client, tax: dict, items: pd.DataFrame, raw_path, phase_label: str) -> list[dict]:
    """Annotate items with one client; resumes from raw_path (items already answered are not asked again,
    and the LLM cache makes a full rerun free)."""
    maxb = int(ph.cfg["annotation"]["max_behaviors_per_item"])
    ids = T.behavior_ids(tax)
    system = SYSTEM.replace("MAXB", str(maxb)) + T.prompt_text(tax)
    sch = schema(ids, maxb)
    digest = taxonomy_digest(tax)
    done = {(r["item_id"], r["model"], r.get("taxonomy_digest")) for r in read_jsonl(raw_path)}
    info = client.info()
    out = []
    todo = [r for r in items.itertuples() if (r.item_id, client.name, digest) not in done]
    if len(todo) < len(items):
        ph.log.info(f"{client.name}: {len(items) - len(todo)} items already annotated (resumed)")
    for r in ph.progress(todo, f"{client.name}: items annotated", every_s=60):
        ans = client.ask(system=system, user=f"ITEM\n{r.text}", schema=sch, max_tokens=700,
                         prompt_version=PROMPT_VERSION, phase=phase_label, item_id=r.item_id, validator=_validator)
        behs = (ans.data or {}).get("behaviors", []) if ans.valid else []
        for b in behs:
            b["evidence_found"] = evidence_found(b.get("evidence", ""), r.text)
            b["category_id"], b["subcategory_id"], _ = T.behavior_path(tax, b["behavior_id"])
        rec = {"item_id": r.item_id, "source": r.source, "group": r.group, "purpose": r.purpose,
               "model": client.name, "model_id": info.get("model"), "model_digest": info.get("digest"),
               "provider": info.get("provider"), "taxonomy_version": tax["version"], "taxonomy_digest": digest,
               "valid": ans.valid,
               "errors": ans.errors, "behaviors": behs,
               "behavior_ids": [b["behavior_id"] for b in behs],
               "category_ids": list(dict.fromkeys(b["category_id"] for b in behs)),
               "subcategory_ids": list(dict.fromkeys(b["subcategory_id"] for b in behs)),
               "explanation": (ans.data or {}).get("explanation") if ans.valid else None,
               "uncertainty": (ans.data or {}).get("uncertainty") if ans.valid else None,
               "missing_behavior": (ans.data or {}).get("missing_behavior") if ans.valid else None,
               "prompt_version": PROMPT_VERSION, "request_hash": ans.request_hash, "attempts": ans.attempts,
               "input_tokens": ans.usage.get("input_tokens"), "output_tokens": ans.usage.get("output_tokens"),
               "latency_s": ans.usage.get("latency_s"), "cached": ans.cached, "annotated_at": now_iso(),
               "raw_answer": ans.text}
        append_jsonl(raw_path, rec)
        out.append(rec)
    return out


def flatten(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for r in raw:
        if not r["valid"]:
            continue
        behs = r["behaviors"]
        ids = r["behavior_ids"]
        rows.append({"item_id": r["item_id"], "source": r["source"], "group": r["group"], "purpose": r["purpose"],
                     "model": r["model"], "model_digest": r["model_digest"], "taxonomy_version": r["taxonomy_version"],
                     "behavior_ids": ids, "category_ids": r["category_ids"], "subcategory_ids": r["subcategory_ids"],
                     "primary_behavior": ids[0] if ids else None, "primary_category": r["category_ids"][0] if ids else None,
                     "n_behaviors": len(ids), "is_unknown": T.UNKNOWN in ids, "is_insufficient": T.INSUFFICIENT in ids,
                     "mean_confidence": round(sum(float(b.get("confidence", 0)) for b in behs) / max(len(behs), 1), 3),
                     "evidence_found_share": round(sum(bool(b.get("evidence_found")) for b in behs) / max(len(behs), 1), 3),
                     "uncertainty": r["uncertainty"], "missing_behavior": r.get("missing_behavior"),
                     "request_hash": r["request_hash"], "prompt_version": r["prompt_version"]})
    return pd.DataFrame(rows)


def latest_raw(raw: list[dict], tax: dict) -> list[dict]:
    """The last answer per (item, model) made with this exact taxonomy (an append-only file may hold reruns and
    answers made with earlier taxonomies; those are kept in the file, never used)."""
    digest = taxonomy_digest(tax)
    last = {}
    for r in raw:
        if r.get("taxonomy_digest") == digest:
            last[(r["item_id"], r["model"])] = r
    return list(last.values())


def run(ph):
    tax = T.load(ph.upstream("taxonomy", f"taxonomy_{ph.cfg['taxonomy'].get('version', 'v1')}.json"))
    if ph.upstream_state("taxonomy").get("status") == "provisional":
        ph.mark_provisional(f"annotations use taxonomy {tax['version']} that experts have not reviewed yet")
    items = load_items(ph, "annotation")
    ph.log.info(f"taxonomy {tax['version']} ({tax['status']}): {len(T.behavior_ids(tax))} behaviours; "
                f"{len(items)} items ({(items['source'] == 'literature').sum()} literature, "
                f"{(items['source'] == 'pharo').sum()} Pharo)")
    raw_path = ph.path("annotations_raw.jsonl")
    for client in annotators(ph.cfg):
        info = client.info()
        ph.log.info(f"annotator {client.name}: {info.get('model')} digest {info.get('digest')}")
        classify(ph, client, tax, items, raw_path, ph.key)
    raw = latest_raw(read_jsonl(raw_path), tax)
    item_ids = set(items["item_id"])
    raw = [r for r in raw if r["item_id"] in item_ids]
    df = flatten(raw)
    write_parquet(ph.path("annotations_validated.parquet"), df)

    models = [a["name"] for a in ph.cfg["llm"]["annotators"]]
    invalid = Counter(r["model"] for r in raw if not r["valid"])
    for m in models:
        if invalid[m]:
            ph.warn(f"{m}: {invalid[m]} items without a valid answer after retries (kept in annotations_raw.jsonl)")
    ph.metric("items", len(items))
    ph.metric("answers_valid", int(len(df)))
    ph.metric("answers_invalid", int(sum(invalid.values())))
    ph.metric("unknown_share", round(float(df["is_unknown"].mean()), 3) if len(df) else None)
    ph.metric("insufficient_share", round(float(df["is_insufficient"].mean()), 3) if len(df) else None)
    ph.metric("evidence_found_share", round(float(df["evidence_found_share"].mean()), 3) if len(df) else None)
    ph.metric("llm", ledger_summary(ph.cfg, ph.key))
    figures(ph, tax, df, raw, models)


def figures(ph, tax, df, raw, models):
    if df.empty:
        return
    cats = {c["id"]: c["name"] for c in T.active(tax, "category")}
    ex = df.explode("category_ids")
    ex = ex[ex["category_ids"].isin(cats)]
    order = ex["category_ids"].value_counts().index.tolist()
    for src in ("pharo", "literature"):
        sub = ex[ex["source"] == src]
        if sub.empty:
            continue
        tab = sub.groupby(["category_ids", "model"])["item_id"].nunique().unstack(fill_value=0).reindex(order, fill_value=0)
        n_items = df[df["source"] == src].groupby("model")["item_id"].nunique()
        mat = (tab / n_items.reindex(tab.columns).values * 100).fillna(0)
        F.heatmap(mat.values, [f"{c} {cats[c]}" for c in mat.index], list(mat.columns),
                  f"{src.capitalize()} items: % of items each model puts in a category", ph.figure(f"categories_by_model_{src}.png"),
                  fmt="{:.0f}", cbar_label="% of items", note="an item can have several categories (multi-label)")
    rates = df.groupby("model").agg(unknown=("is_unknown", "mean"), insufficient=("is_insufficient", "mean"))
    F.stacked_barh(list(rates.index), {"UNKNOWN": (rates["unknown"] * 100).tolist(),
                                        "INSUFFICIENT": (rates["insufficient"] * 100).tolist()},
                   "Answers outside the taxonomy", "% of items", ph.figure("unknown_insufficient.png"))
    nb = df.groupby(["model", "n_behaviors"]).size().unstack(fill_value=0)
    F.stacked_barh(list(nb.index), {f"{c} behaviour(s)": nb[c].tolist() for c in nb.columns},
                   "Behaviours per item (multi-label)", "items", ph.figure("behaviors_per_item.png"))
    val = pd.DataFrame([{"model": r["model"], "status": ("valid, 1st try" if r["valid"] and r["attempts"] == 1 else
                                                         "valid after retry" if r["valid"] else "invalid")} for r in raw])
    tab = val.groupby(["model", "status"]).size().unstack(fill_value=0)
    F.stacked_barh(list(tab.index), {c: tab[c].tolist() for c in ["valid, 1st try", "valid after retry", "invalid"] if c in tab},
                   "Structured output validity", "answers", ph.figure("validity.png"),
                   colors={"valid, 1st try": F.STATUS["good"], "valid after retry": F.STATUS["warning"],
                           "invalid": F.STATUS["critical"]})
