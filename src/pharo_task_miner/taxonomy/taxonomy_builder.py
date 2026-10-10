"""Module 11 — build a versioned, evidence-backed taxonomy document.

The output combines (a) the seed categories, (b) corpus evidence (prevalence,
diverse examples, related categories), (c) human validation results when
available (validated/rejected categories, false-positive negative examples,
annotator-proposed categories) and (d) bottom-up discovery groups as
*proposed* categories. Category status is one of: ``validated`` (rule labels
reach the precision threshold against adjudicated human labels),
``llm_supported`` (the same, against the LLM annotation panel — not a human
validation), ``provisional`` (automatic evidence only), ``needs_revision``
(precision below threshold), ``proposed`` (from annotators or open discovery).
"""
from __future__ import annotations

from collections import Counter

from .evidence_linker import cooccurrence, evidence_examples
from ..storage.database import Database, loads
from ..storage.provenance import now_iso


# reference label source -> (adjudicated decision kind, per-annotator decision kind)
REFERENCE_KINDS = {"human_adjudicated": ("adjudicated", "annotation"),
                   "llm_adjudicated": ("llm_adjudicated", "llm_annotation")}


def human_label_stats(db: Database, basis: str = "human_adjudicated") -> tuple[dict, dict, Counter]:
    """Per-category precision/recall of rule labels vs the reference labels
    (adjudicated human labels, or the LLM panel's aggregated labels), plus
    negative examples and annotator-proposed categories."""
    adjudicated_kind, annotation_kind = REFERENCE_KINDS[basis]
    adjud = {}
    for r in db.query("SELECT method_id, operations_json FROM annotation_decisions WHERE kind=?", (adjudicated_kind,)):
        adjud[r["method_id"]] = set(loads(r["operations_json"], []))
    proposed = Counter()
    for r in db.query("SELECT new_category FROM annotation_decisions WHERE kind=? AND new_category IS NOT NULL"
                      " AND new_category != ''", (annotation_kind,)):
        for c in str(r["new_category"]).split(";"):
            if c.strip():
                proposed[c.strip()] += 1
    if not adjud:
        return {}, {}, proposed
    auto = {}
    ids = sorted(adjud)
    for start in range(0, len(ids), 5000):  # whole repositories may be labelled: too many ids for one statement
        for r in db.query("SELECT method_id, operation_id FROM method_labels WHERE label_source='rule_based' AND method_id IN (%s)"
                          % ",".join(map(str, ids[start:start + 5000]))):
            auto.setdefault(r["method_id"], set()).add(r["operation_id"])
    cats = set().union(*adjud.values(), *auto.values()) if auto else set().union(*adjud.values())
    stats, negatives = {}, {}
    for c in cats:
        tp = sum(1 for m, h in adjud.items() if c in h and c in auto.get(m, set()))
        fp_methods = [m for m, h in adjud.items() if c not in h and c in auto.get(m, set())]
        fn = sum(1 for m, h in adjud.items() if c in h and c not in auto.get(m, set()))
        fp = len(fp_methods)
        stats[c] = {"human_positive": tp + fn, "auto_positive": tp + fp, "true_positive": tp,
                    "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                    "recall": round(tp / (tp + fn), 3) if tp + fn else None}
        negatives[c] = fp_methods[:5]
    return stats, negatives, proposed


def build(db: Database, tax: dict, prevalence: dict[str, dict], discovery: dict, version: str,
          precision_threshold: float = 0.7, k_examples: int = 5, basis: str | None = "human_adjudicated") -> dict:
    stats, negatives, proposed = human_label_stats(db, basis) if basis else ({}, {}, Counter())
    related = cooccurrence(db)
    validated_any = bool(stats)
    method_ref = {}
    if negatives:
        ids = sorted({m for v in negatives.values() for m in v})
        for start in range(0, len(ids), 5000):
            for r in db.query("SELECT m.method_id, r.full_name, m.class_name, m.selector FROM methods m JOIN repositories r"
                              " ON r.repo_id=m.repo_id WHERE m.method_id IN (%s)" % ",".join(map(str, ids[start:start + 5000]))):
                method_ref[r["method_id"]] = {"repository": r["full_name"], "method": f"{r['class_name']}>>{r['selector']}"}
    cats = []
    for c in tax["categories"]:
        cid = c["id"]
        s = stats.get(cid)
        if s and s["precision"] is not None and s["human_positive"] >= 1:
            ok = "validated" if basis == "human_adjudicated" else "llm_supported"
            status = ok if s["precision"] >= precision_threshold else "needs_revision"
        else:
            status = "provisional"
        cats.append({
            "id": cid, "parent": c.get("parent"), "level": c["level"], "name": c.get("name", cid),
            "concept_type": c.get("concept_type", "general"), "definition": c.get("definition"),
            "include": c.get("include") or [], "exclude": c.get("exclude") or [],
            "status": status, "reference_validation": s,
            "prevalence": prevalence.get(cid),
            "evidence_examples": evidence_examples(db, cid, k=k_examples) if c["level"] == 3 else [],
            "negative_examples": [method_ref.get(m, {"method_id": m}) for m in negatives.get(cid, [])],
            "related_categories": related.get(cid, []),
        })
    source = "annotators" if basis != "llm_adjudicated" else "llm_annotators"
    proposals = [{"id": f"proposed:{name}", "source": source, "votes": n, "status": "proposed"}
                 for name, n in proposed.most_common()]
    proposals += [{"id": f"proposed:signature:{g['signature']}", "source": "open_discovery_signature_group",
                   "methods": g["methods"], "repositories": g["repositories"], "status": "proposed",
                   "example_method_ids": g["example_method_ids"]} for g in discovery.get("signature_groups", [])[:30]]
    proposals += [{"id": "proposed:description:" + "-".join(g["keywords"][:3]), "source": "llm_description_group",
                   "methods": g["size"], "repositories": g["repositories"], "status": "proposed",
                   "example_summaries": g["example_summaries"]} for g in discovery.get("description_groups", [])[:30]]
    return {
        "taxonomy_version": version, "generated_at": now_iso(), "seed_version": str(tax.get("version")),
        "validation_basis": ("automatic_only_unvalidated" if not validated_any else
                             "human_adjudicated_sample" if basis == "human_adjudicated" else
                             "llm_panel_annotated_sample_NOT_human_validated"),
        "special_statuses": tax.get("special_statuses", {}),
        "categories": cats, "proposed_categories": proposals,
    }
