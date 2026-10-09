"""Import benchmark task metadata (JSON, JSONL, YAML or CSV).

Each task must declare its operations explicitly (taxonomy ids), normally after
manual mapping. The importer NEVER infers operations from description keywords.
Expected fields: benchmark, task_id, description, operations, structure,
required_capabilities, transfer_status, difficulty, preserves_capability, origin.

``transfer_status`` vocabulary: not_evaluated | portable | needs_adaptation |
transferred_validated | native_pharo | not_transferable.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..config import load_yaml
from ..storage.database import Database, dumps
from ..storage.provenance import now_iso

TRANSFER_STATUSES = {"not_evaluated", "portable", "needs_adaptation", "transferred_validated", "native_pharo",
                     "not_transferable"}


def _listify(v) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in str(v).replace(",", ";").split(";") if x.strip()]


def read_tasks(path: Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("tasks", [])
    if path.suffix in (".yaml", ".yml"):
        data = load_yaml(path)
        return data if isinstance(data, list) else data.get("tasks", [])
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def import_benchmarks(db: Database, paths: list[Path], taxonomy_ids: set[str]) -> dict:
    summary = {"tasks": 0, "unmapped_tasks": 0, "unknown_operation_ids": {}, "invalid_transfer_status": 0}
    for p in paths:
        for t in read_tasks(p):
            if not str(t.get("task_id") or "").strip():
                summary["skipped_without_task_id"] = summary.get("skipped_without_task_id", 0) + 1
                continue
            ops = _listify(t.get("operations"))
            unknown = [o for o in ops if o not in taxonomy_ids]
            for o in unknown:
                summary["unknown_operation_ids"][o] = summary["unknown_operation_ids"].get(o, 0) + 1
            status = (t.get("transfer_status") or "not_evaluated").strip()
            if status not in TRANSFER_STATUSES:
                summary["invalid_transfer_status"] += 1
                status = "not_evaluated"
            if not [o for o in ops if o in taxonomy_ids]:
                summary["unmapped_tasks"] += 1
            db.upsert("benchmark_tasks", {
                "benchmark": str(t.get("benchmark") or Path(p).stem), "task_id": str(t.get("task_id")),
                "description": t.get("description"), "operations_json": dumps([o for o in ops if o in taxonomy_ids]),
                "structure_json": dumps(_listify(t.get("structure"))),
                "required_capabilities_json": dumps(_listify(t.get("required_capabilities"))),
                "transfer_status": status, "difficulty": t.get("difficulty"),
                "preserves_capability": (str(t.get("preserves_capability")).lower() if t.get("preserves_capability") is not None else None),
                "origin": t.get("origin"), "source_file": str(p), "unknown_operations_json": dumps(unknown),
                "imported_at": now_iso()}, ["benchmark", "task_id"])
            summary["tasks"] += 1
    db.commit()
    return summary
