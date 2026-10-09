"""Import annotator decisions, list disagreements, import adjudicated labels and
developer feedback. Automatic labels are never treated as validated: only
adjudicated human labels (label_source ``human_adjudicated``) are."""
from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path

from .agreement import agreement_report
from ..storage.database import Database, dumps
from ..storage.provenance import now_iso

TRUE = {"1", "y", "yes", "true", "x"}


def _ops(v: str | None) -> set[str]:
    return {x.strip() for x in (v or "").replace(",", ";").split(";") if x.strip()}


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def import_annotations(db: Database, vdir: Path, known_categories: set[str]) -> dict:
    sample = {r["sample_id"]: int(r["method_id"]) for r in _read(vdir / "annotation_sample.csv")}
    if not sample:
        return {"status": "no_sample", "note": "run export-annotations first"}
    decisions = defaultdict(dict)
    unknown_ops = defaultdict(int)
    db.execute("DELETE FROM annotation_decisions WHERE kind IN ('annotation','adjudicated')")
    for r in _read(vdir / "reviewer_decisions.csv"):
        sid, ann = r.get("sample_id"), (r.get("annotator") or "").strip()
        filled = any((r.get(k) or "").strip() for k in ("operations", "new_category", "notes", "missing_operations"))
        if sid not in sample or not ann or not filled:
            continue
        ops = _ops(r.get("operations"))
        for o in ops - known_categories - {"unknown", "ambiguous", "insufficient_context"}:
            unknown_ops[o] += 1
        decisions[sid][ann] = ops
        flags = {k: (r.get(k) or "").strip().lower() in TRUE for k in ("too_broad", "too_narrow", "multiple_operations")}
        flags["evidence_sufficient"] = (r.get("evidence_sufficient") or "").strip().lower() or None
        flags["missing_operations"] = r.get("missing_operations") or None
        db.execute("INSERT INTO annotation_decisions (sample_id, method_id, annotator, kind, operations_json, flags_json,"
                   " new_category, notes, decided_at, source_file, imported_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (sid, sample[sid], ann, "annotation", dumps(sorted(ops)), dumps(flags), r.get("new_category") or None,
                    r.get("notes") or None, None, "reviewer_decisions.csv", now_iso()))
    report = agreement_report(decisions, sorted(known_categories))
    # disagreements and consensus
    disagreements, consensus = [], {}
    for sid, d in decisions.items():
        sets = {frozenset(v) for v in d.values()}
        if len(d) >= 2 and len(sets) > 1:
            disagreements.append({"sample_id": sid, "method_id": sample[sid],
                                  **{f"labels_{a}": ";".join(sorted(v)) for a, v in d.items()}})
        elif len(d) >= 2:
            consensus[sid] = next(iter(d.values()))
    ann_names = sorted({a for d in decisions.values() for a in d})
    _write(vdir / "disagreements.csv", disagreements, ["sample_id", "method_id"] + [f"labels_{a}" for a in ann_names])
    # adjudicated labels: keep human-entered rows; add consensus and pending rows for new items
    adj_path = vdir / "adjudicated_labels.csv"
    existing = {r["sample_id"]: r for r in _read(adj_path)}
    for sid, ops in consensus.items():
        existing.setdefault(sid, {"sample_id": sid, "operations": ";".join(sorted(ops)), "adjudicator": "consensus",
                                  "status": "agreed", "notes": ""})
    for row in disagreements:
        existing.setdefault(row["sample_id"], {"sample_id": row["sample_id"], "operations": "", "adjudicator": "",
                                               "status": "needs_adjudication", "notes": ""})
    single = [sid for sid, d in decisions.items() if len(d) == 1]
    for sid in single:
        existing.setdefault(sid, {"sample_id": sid, "operations": ";".join(sorted(next(iter(decisions[sid].values())))),
                                  "adjudicator": "single_annotator", "status": "single_annotation", "notes": ""})
    _write(adj_path, sorted(existing.values(), key=lambda r: r["sample_id"]),
           ["sample_id", "operations", "adjudicator", "status", "notes"])
    # store adjudicated labels (not pending rows)
    db.execute("DELETE FROM method_labels WHERE label_source='human_adjudicated'")
    n_adj = 0
    for r in existing.values():
        if r.get("status") == "needs_adjudication" and not (r.get("operations") or "").strip():
            continue
        if r["sample_id"] not in sample:
            continue
        mid = sample[r["sample_id"]]
        ops = _ops(r.get("operations"))
        db.execute("INSERT INTO annotation_decisions (sample_id, method_id, annotator, kind, operations_json, flags_json,"
                   " new_category, notes, decided_at, source_file, imported_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (r["sample_id"], mid, r.get("adjudicator") or "unknown", "adjudicated", dumps(sorted(ops)),
                    dumps({"status": r.get("status")}), None, r.get("notes") or None, None, "adjudicated_labels.csv", now_iso()))
        for o in ops:
            db.upsert("method_labels", {"method_id": mid, "operation_id": o, "label_source": "human_adjudicated",
                                        "label_status": "validated", "score": None, "score_kind": None,
                                        "taxonomy_version": None, "review_status": "adjudicated", "model": None,
                                        "prompt_version": None, "created_at": now_iso(), "run_id": None},
                      ["method_id", "operation_id", "label_source"])
        n_adj += 1
    db.commit()
    return {"status": "imported", "annotated_items": len(decisions), "disagreements": len(disagreements),
            "adjudicated_items": n_adj, "unknown_operation_ids": dict(unknown_ops), "agreement": report}


def import_developer_feedback(db: Database, path: Path) -> dict:
    rows = [r for r in _read(path) if any((r.get(k) or "").strip() for k in
                                          ("recognizable_1to5", "important_in_work_1to5", "deserves_benchmark_yes_no",
                                           "missing_capabilities", "notes"))]
    db.execute("DELETE FROM developer_feedback WHERE source_file=?", (path.name,))

    def num(v):
        try:
            return int(str(v).strip())
        except (TypeError, ValueError):
            return None
    for r in rows:
        db.execute("INSERT INTO developer_feedback (respondent, respondent_role, category_id, recognizable, important_in_work,"
                   " deserves_benchmark, examples_representative, missing_capabilities, notes, source_file, imported_at)"
                   " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (r.get("respondent"), r.get("respondent_role"), r.get("category_id"), num(r.get("recognizable_1to5")),
                    num(r.get("important_in_work_1to5")),
                    1 if (r.get("deserves_benchmark_yes_no") or "").strip().lower() in TRUE else (
                        0 if (r.get("deserves_benchmark_yes_no") or "").strip() else None),
                    num(r.get("examples_representative_1to5")), r.get("missing_capabilities"), r.get("notes"),
                    path.name, now_iso()))
    db.commit()
    return {"feedback_rows": len(rows)}


def _write(path: Path, rows: list[dict], cols: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
