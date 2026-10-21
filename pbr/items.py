"""Items to annotate — literature tasks and Pharo methods — and how they are shown to models and experts.

An item has a global ``item_id``: ``L:<task_uid>`` for a literature task, ``P:<method_id>`` for a Pharo
method. The text shown is the same for every annotator (models and experts), so their answers are
comparable: for a literature task, what the evaluated model receives, the reference answer and how it is
checked; for a Pharo method, its source with its class context (superclass, instance variables, comment).
"""
from __future__ import annotations

import pandas as pd

MAX_SOURCE = 2500
MAX_CARD = 2600


def _cut(s, n):
    s = str(s or "")
    return s if len(s) <= n else s[:n].rstrip() + "\n[... cut]"


def load_items(ph, split: str, include_targeted: bool = True) -> pd.DataFrame:
    """The items of a split (phase 4), with the columns every phase needs: item_id, source ('literature' or
    'pharo'), group, purpose, weight, text, title, provenance."""
    sample = pd.read_csv(ph.upstream("sampling", "pilot_sample.csv"))
    sample = sample[sample["split"] == split]
    if not include_targeted:
        sample = sample[sample["purpose"] == "random"]
    lit_s = pd.read_csv(ph.upstream("sampling", "literature_sample.csv"))
    lit_s = lit_s[lit_s["split"] == split]
    rows = []
    if len(lit_s):
        lit = pd.read_parquet(ph.upstream("literature", "literature_tasks.parquet"),
                              columns=["task_uid", "benchmark", "benchmark_name", "item_id", "card", "task_type",
                                       "language", "source_dataset", "source_revision"])
        lit = lit_s.merge(lit, on=["task_uid", "benchmark", "task_type"], how="left")
        for r in lit.itertuples():
            rows.append({"item_id": f"L:{r.task_uid}", "source": "literature", "group": r.benchmark, "purpose": "random",
                         "subset": "literature",
                         "weight": r.weight, "repository": None, "title": f"{r.benchmark_name} {r.item_id}",
                         "task_type": r.task_type, "language": r.language,
                         "provenance": f"{r.source_dataset}@{str(r.source_revision)[:12]} item {r.item_id}",
                         "text": literature_text(r)})
    if len(sample):
        cols = ["method_id", "package", "class_name", "side", "selector", "protocol", "superclass",
                "class_instance_variables", "class_comment", "source", "is_test", "url", "repository", "commit_sha",
                "source_path", "line"]
        m = pd.read_parquet(ph.upstream("corpus", "methods.parquet"), columns=cols,
                            filters=[("method_id", "in", list(sample["method_id"]))])
        keep = [c for c in ("method_id", "group", "purpose", "signal", "weight", "subset", "filter_decision", "filter_reasons")
                if c in sample.columns]
        m = sample[keep].merge(m, on="method_id", how="left")
        context = ph.cfg.get("annotation", {}).get("context", "class")
        for r in m.itertuples():
            rows.append({"item_id": f"P:{r.method_id}", "source": "pharo", "group": r.group, "purpose": r.purpose,
                         "subset": getattr(r, "subset", "production"), "filter_decision": getattr(r, "filter_decision", None),
                         "filter_reasons": getattr(r, "filter_reasons", None),
                         "weight": r.weight, "repository": r.repository, "title": f"{r.class_name}{' class' if r.side == 'class' else ''}>>{r.selector}",
                         "task_type": None, "language": "Pharo", "signal": r.signal,
                         "provenance": r.url, "text": pharo_text(r, context == "class")})
    return pd.DataFrame(rows)


def literature_text(r) -> str:
    return (f"Literature benchmark task ({r.benchmark_name}, language {r.language}).\n"
            f"{_cut(r.card, MAX_CARD)}")


def pharo_text(r, with_class: bool = True) -> str:
    head = f"Pharo method {r.class_name}{' class' if r.side == 'class' else ''}>>{r.selector}"
    ctx = [f"package {r.package}", f"protocol {r.protocol}"]
    if r.is_test:
        ctx.append("test method")
    lines = [head + " (" + ", ".join(str(c) for c in ctx if c) + ")"]
    if with_class:
        ivars = list(r.class_instance_variables) if r.class_instance_variables is not None else []
        lines.append(f"Class {r.class_name}: subclass of {r.superclass or '?'}"
                     + (f", instance variables: {' '.join(ivars[:20])}" if len(ivars) else ""))
        if isinstance(r.class_comment, str) and r.class_comment.strip():
            lines.append("Class comment: " + _cut(" ".join(r.class_comment.split()), 300))
    lines.append("```smalltalk\n" + _cut(r.source, MAX_SOURCE) + "\n```")
    return "\n".join(lines)
