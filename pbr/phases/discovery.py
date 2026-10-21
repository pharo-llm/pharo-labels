"""Phase 5 — Taxonomy discovery by three independent LLM annotators, separately for each source.

Each annotator (three model families, config ``llm.annotators``), on its own and never shown another model's
answers, builds TWO taxonomies: one from the sampled Pharo methods (phase 4: production and test methods that
passed the method filters of phase 3), one from the sampled tasks of the existing benchmarks — six in all.

1. **Open proposals.** For each item (in batches of one source), up to ``max_behaviors_per_item`` programming
   behaviours at three levels — category › subcategory › concrete behaviour — with the evidence (a quote of the
   item). Behaviours are named in language-neutral terms so Python tasks and Pharo methods can be compared; the
   evaluation task type is NOT a behaviour (§2).
2. **Consolidation.** The model organises its own proposals of that source into a hierarchy with a definition per
   concept, inclusion and exclusion rules, and the items each behaviour covers.

Outputs, per model and source, ``<model>/<pharo|literature>/``:
  taxonomy.png       the taxonomy as a tree; a small bar and a number give the items classified under each behaviour
  taxonomy.json      the full hierarchy: definitions, rules, items, evidence, proposals not placed by the model
  taxonomy.csv       one row per behaviour (category, subcategory, behaviour, definitions, rules, n_items, examples)
  items.csv          each item with the category / subcategory / behaviour this model gave it (one row per behaviour)
  categories.png     the categories by number of item-behaviour assignments
  proposals.jsonl    every item's proposals;  raw_responses.jsonl   every raw answer (all attempts: results/cache/llm/)
and discovery_summary.csv (one row per model and source); figures/: the six taxonomies compared.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import re

import pandas as pd

from .. import figures as F
from ..common import read_jsonl, write_csv, write_json, write_jsonl
from ..items import load_items
from ..llm import annotators, ledger_summary

PROMPT_VERSION = "discovery-v1"

SYSTEM = """You are an expert software engineer building a taxonomy of PROGRAMMING BEHAVIOURS, to compare what existing code benchmarks evaluate with what real Pharo (Smalltalk) code does.

A programming behaviour is what the code does (for a code benchmark task: what the solution code must do), described at three levels:
- category: a broad, language-neutral area of programming (e.g. "Collections", "String processing", "Error handling", "Object construction and state", "Testing").
- subcategory: a narrower family inside the category (e.g. "Indexed access").
- behavior: a concrete behaviour, an imperative phrase of 3 to 8 words (e.g. "Retrieve an element by index").

Rules:
- Name behaviours in language-neutral terms: no Python, Smalltalk or library names in category, subcategory or behavior.
- Describe what the code DOES, not the kind of benchmark task (generation, repair, completion are not behaviours), not the application domain alone, and not code quality.
- Give 1 to MAXB behaviours per item, the most important first. Several behaviours when the item clearly combines them.
- evidence: a short quote (at most 15 words) copied from the item that shows the behaviour.
- insufficient_context: true only when the item does not show enough code or description to decide."""

CONSOLIDATE = """You proposed the programming behaviours listed below (id, category › subcategory › behavior, number of items).
Organise them into ONE clean hierarchical taxonomy:
- at most 16 categories, each with 1 to 8 subcategories, each with 1 to 10 behaviours;
- merge proposals that mean the same thing, even if worded differently; keep different behaviours apart;
- every concept gets a one-sentence definition; every behaviour gets "include" rules (when to use it) and "exclude" rules (when not to, naming the neighbouring behaviour to use instead);
- "proposals": the ids of the proposals that each behaviour covers; assign every proposal id to exactly one behaviour;
- names stay language-neutral (no Python, Smalltalk or library names).

Proposals:
"""


def item_schema(local_ids: list[str], maxb: int) -> dict:
    beh = {"type": "object", "properties": {
        "category": {"type": "string", "minLength": 2}, "subcategory": {"type": "string", "minLength": 2},
        "behavior": {"type": "string", "minLength": 3}, "evidence": {"type": "string"}},
        "required": ["category", "subcategory", "behavior", "evidence"]}
    return {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string", "enum": local_ids},
        "behaviors": {"type": "array", "items": beh, "minItems": 1, "maxItems": maxb},
        "insufficient_context": {"type": "boolean"}}, "required": ["id", "behaviors", "insufficient_context"]},
        "minItems": len(local_ids), "maxItems": len(local_ids)}}, "required": ["items"]}


TAX_SCHEMA = {"type": "object", "properties": {"categories": {"type": "array", "minItems": 1, "items": {
    "type": "object", "properties": {
        "name": {"type": "string"}, "definition": {"type": "string"},
        "subcategories": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "definition": {"type": "string"},
            "behaviors": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "definition": {"type": "string"},
                "include": {"type": "array", "items": {"type": "string"}},
                "exclude": {"type": "array", "items": {"type": "string"}},
                "proposals": {"type": "array", "items": {"type": "string"}}},
                "required": ["name", "definition", "include", "exclude", "proposals"]}}},
            "required": ["name", "definition", "behaviors"]}}},
    "required": ["name", "definition", "subcategories"]}}}, "required": ["categories"]}


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).split())


def batches(items: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    out = []
    for _, part in items.groupby("source", sort=True):      # literature and Pharo items in separate batches
        part = part.reset_index(drop=True)
        out += [part.iloc[i:i + size] for i in range(0, len(part), size)]
    return out


def propose(ph, client, batch_list, maxb, out_dir) -> list[dict]:
    rows, raws = [], []
    for b in ph.progress(batch_list, f"{client.name}: proposal batches", every_s=60):
        local = [f"I{i + 1}" for i in range(len(b))]
        user = "Items:\n\n" + "\n\n".join(f"### {lid}\n{t}" for lid, t in zip(local, b["text"]))
        ans = client.ask(system=SYSTEM.replace("MAXB", str(maxb)), user=user, schema=item_schema(local, maxb),
                         max_tokens=300 * len(b) + 200, prompt_version=PROMPT_VERSION, phase=ph.key,
                         item_id=",".join(b["item_id"]))
        raws.append({"items": list(b["item_id"]), "request_hash": ans.request_hash, "valid": ans.valid,
                     "errors": ans.errors, "attempts": ans.attempts, "raw": ans.text})
        got = {x.get("id"): x for x in (ans.data or {}).get("items", []) if isinstance(x, dict)} if ans.data else {}
        for lid, (_, it) in zip(local, b.iterrows()):
            x = got.get(lid)
            rows.append({"item_id": it["item_id"], "source": it["source"], "group": it["group"], "model": client.name,
                         "valid": bool(ans.valid and x), "behaviors": (x or {}).get("behaviors", []),
                         "insufficient_context": bool((x or {}).get("insufficient_context", False)),
                         "request_hash": ans.request_hash})
    write_jsonl(out_dir / "proposals.jsonl", rows)
    write_jsonl(out_dir / "raw_responses.jsonl", raws)
    return rows


def consolidate(ph, client, rows, source: str, out_dir) -> dict:
    """The model organises its own proposals of one source into a hierarchy; proposals keep their items."""
    props = defaultdict(lambda: {"items": [], "evidence": []})
    for r in rows:
        for b in r["behaviors"]:
            k = (b["category"].strip(), b["subcategory"].strip(), b["behavior"].strip())
            key = (norm(k[0]), norm(k[1]), norm(k[2]))
            p = props[key]
            p.setdefault("label", k)
            p["items"].append(r["item_id"])
            p["evidence"].append(b.get("evidence", ""))
    plist = sorted(props.values(), key=lambda p: (-len(p["items"]), p["label"]))
    for i, p in enumerate(plist):
        p["id"] = f"p{i + 1}"
    lines = [f"{p['id']}: {p['label'][0]} › {p['label'][1]} › {p['label'][2]} ({len(p['items'])})" for p in plist]
    pid = {p["id"]: p for p in plist}
    ans = client.ask(system="You are an expert software engineer and a careful taxonomist. Answer with JSON only.",
                     user=CONSOLIDATE + "\n".join(lines), schema=TAX_SCHEMA, max_tokens=12000,
                     prompt_version=PROMPT_VERSION, phase=ph.key, item_id=f"consolidation:{source}")
    tax = ans.data if ans.valid else None
    assigned = Counter()
    candidate = {"model": client.name, "source": source, "n_items_shown": len(rows), "model_info": client.info(),
                 "prompt_version": PROMPT_VERSION,
                 "valid": ans.valid, "errors": ans.errors, "request_hash": ans.request_hash, "categories": [],
                 "n_proposals": len(plist)}
    if tax:
        for c in tax["categories"]:
            cc = {"name": c["name"].strip(), "definition": c["definition"].strip(), "subcategories": []}
            for s in c["subcategories"]:
                ss = {"name": s["name"].strip(), "definition": s["definition"].strip(), "behaviors": []}
                for b in s["behaviors"]:
                    covered = [x for x in b.get("proposals", []) if x in pid]
                    assigned.update(covered)
                    items = list(dict.fromkeys(i for x in covered for i in pid[x]["items"]))
                    ev = [e for x in covered for e in pid[x]["evidence"] if e][:3]
                    ss["behaviors"].append({"name": b["name"].strip(), "definition": b["definition"].strip(),
                                            "include": b.get("include", []), "exclude": b.get("exclude", []),
                                            "proposals": [pid[x]["label"] for x in covered], "examples": items[:8],
                                            "items": items, "n_items": len(items), "evidence": ev})
                cc["subcategories"].append(ss)
            candidate["categories"].append(cc)
    # proposals the consolidation forgot are kept, never dropped
    candidate["unassigned_proposals"] = [{"label": p["label"], "items": p["items"]} for p in plist if p["id"] not in assigned]
    candidate["assigned_share"] = round(len(assigned) / max(len(plist), 1), 3)
    write_json(out_dir / "taxonomy.json", candidate)
    return candidate


def export_tables(out_dir, cand: dict, titles: dict):
    """Usable outputs next to the figure: taxonomy.csv (one row per behaviour) and items.csv (each item and the
    category / subcategory / behaviour this model gives it; an item with several behaviours has several rows)."""
    trows, irows = [], []
    for c in cand["categories"]:
        for s_ in c["subcategories"]:
            for b in s_["behaviors"]:
                trows.append({"category": c["name"], "category_definition": c["definition"], "subcategory": s_["name"],
                              "subcategory_definition": s_["definition"], "behavior": b["name"],
                              "behavior_definition": b["definition"], "include": "; ".join(b.get("include", [])),
                              "exclude": "; ".join(b.get("exclude", [])), "n_items": b["n_items"],
                              "example_items": "; ".join(b["examples"]), "evidence": " | ".join(b.get("evidence", []))})
                irows += [{"item_id": i, "title": titles.get(i, ""), "category": c["name"], "subcategory": s_["name"],
                           "behavior": b["name"], "placed_by_model": True} for i in b.get("items", [])]
    for p_ in cand.get("unassigned_proposals", []):
        c, s_, b = p_["label"]
        trows.append({"category": c, "subcategory": s_, "behavior": b, "n_items": len(p_["items"]),
                      "example_items": "; ".join(p_["items"][:8]), "behavior_definition": "(not placed by the model's consolidation)"})
        irows += [{"item_id": i, "title": titles.get(i, ""), "category": c, "subcategory": s_, "behavior": b,
                   "placed_by_model": False} for i in p_["items"]]
    write_csv(out_dir / "taxonomy.csv", pd.DataFrame(trows))
    write_csv(out_dir / "items.csv", pd.DataFrame(irows).sort_values(["item_id", "category"]) if irows else pd.DataFrame(irows))


def counts(candidate: dict) -> tuple[int, int, int]:
    cats = candidate["categories"]
    subs = [s for c in cats for s in c["subcategories"]]
    return len(cats), len(subs), sum(len(s["behaviors"]) for s in subs)


SOURCES = {"pharo": "Pharo methods", "literature": "existing benchmark tasks"}


def run(ph):
    items = load_items(ph, "discovery")
    maxb = int(ph.cfg["discovery"]["max_behaviors_per_item"])
    titles = dict(zip(items["item_id"], items["title"]))
    by_source = {src: items[items["source"] == src] for src in SOURCES}
    ph.log.info(f"{len(items)} discovery items: " + ", ".join(f"{len(v)} {SOURCES[k]}" for k, v in by_source.items())
                + "; each model builds one taxonomy per source, alone")
    summary, all_rows, cands = [], [], {}
    for client in annotators(ph.cfg):
        info = client.info()
        ph.log.info(f"annotator {client.name}: {info.get('model')} ({info.get('family') or ''} "
                    f"{info.get('parameter_size') or ''} {info.get('quantization') or ''}, digest {info.get('digest', '')})")
        for src, its in by_source.items():
            if its.empty:
                continue
            out = ph.path(client.name, src, "x").parent
            rows = propose(ph, client, batches(its, int(ph.cfg["discovery"]["batch_size"])), maxb, out)
            bad = sum(not r["valid"] for r in rows)
            if bad:
                ph.warn(f"{client.name}/{src}: {bad} items without a valid proposal (kept in proposals.jsonl, valid=false)")
            ph.log.info(f"{client.name}/{src}: consolidating {sum(len(r['behaviors']) for r in rows)} proposals into a hierarchy")
            cand = consolidate(ph, client, rows, src, out)
            if not cand["valid"]:
                ph.warn(f"{client.name}/{src}: consolidation invalid after retries ({cand['errors'][:2]}) — candidate kept empty")
            export_tables(out, cand, titles)
            nc, ns, nb = counts(cand)
            ph.log.info(f"{client.name}/{src}: {nc} categories, {ns} subcategories, {nb} behaviours; "
                        f"{cand['assigned_share']:.0%} of proposals placed")
            summary.append({"model": client.name, "source": src, "model_id": info.get("model"), "digest": info.get("digest"),
                            "items": len(rows), "invalid_items": bad,
                            "insufficient_context": sum(r["insufficient_context"] for r in rows),
                            "proposals": sum(len(r["behaviors"]) for r in rows), "distinct_proposals": cand["n_proposals"],
                            "categories": nc, "subcategories": ns, "behaviors": nb,
                            "assigned_share": cand["assigned_share"], "consolidation_valid": cand["valid"]})
            all_rows += rows
            cands[(client.name, src)] = cand
            model_figures(ph, client.name, src, cand, out)
    sdf = pd.DataFrame(summary)
    write_csv(ph.path("discovery_summary.csv"), sdf)
    for r in summary:
        ph.metric(f"{r['model']}_{r['source']}_behaviors", r["behaviors"])
    ph.metric("items", len(items))
    ph.metric("llm", ledger_summary(ph.cfg, ph.key))

    # ---------------------------------------------------------------- figures (the taxonomies compared)
    import numpy as np
    fig, axes = F.plt.subplots(1, len(SOURCES), figsize=(12, 3.8), sharey=True)
    levels = ["categories", "subcategories", "behaviors"]
    w = 0.8 / len(levels)
    for ax, src in zip(np.atleast_1d(axes), SOURCES):
        d = sdf[sdf["source"] == src] if len(sdf) else sdf
        x = np.arange(len(d))
        for i, lv in enumerate(levels):
            ax.bar(x + (i - 1) * w, d[lv], width=w - 0.02, color=F.color(i), label=lv)
            for xi, v in zip(x, d[lv]):
                ax.text(xi + (i - 1) * w, v, str(v), ha="center", va="bottom", fontsize=8, color=F.INK_2)
        ax.set_xticks(x, d["model"])
        ax.grid(axis="x", visible=False)
        ax.set_title(f"from {SOURCES[src]}", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("concepts")
    np.atleast_1d(axes)[0].legend()
    fig.suptitle("Candidate taxonomies: size per model and source (each model alone)", x=0.01, ha="left",
                 fontsize=12, weight="bold")
    F.save(fig, ph.figure("concepts_per_model.png"))

    rp = pd.DataFrame([{"who": f"{r['model']} · {r['source']}", "n": len(r["behaviors"])} for r in all_rows])
    if len(rp):
        tab = rp.groupby(["who", "n"]).size().unstack(fill_value=0)
        F.stacked_barh(list(tab.index), {f"{c} behaviour(s)": tab[c].tolist() for c in tab.columns},
                       "Behaviours proposed per item", "items", ph.figure("proposals_per_item.png"))


def model_figures(ph, name: str, src: str, cand: dict, out):
    """In <model>/<source>/: the taxonomy as a tree (taxonomy.png, with the number of items of each behaviour as a
    small bar) and its categories by size (categories.png)."""
    if not cand["categories"]:
        return
    blocks = []
    for c in cand["categories"]:
        subs = [(s["name"], [{"label": b["name"], "count": b["n_items"]} for b in s["behaviors"]])
                for s in c["subcategories"]]
        blocks.append((c["name"], subs))
    left = cand.get("unassigned_proposals") or []
    if left:      # what the model proposed but did not place in its own hierarchy: shown, never hidden
        groups = {}
        for p in left:
            c, s_, b = p["label"]
            groups.setdefault(f"{c} › {s_}", []).append({"label": b, "count": len(p["items"])})
        blocks.append((f"Not placed by the model's own consolidation ({len(left)} proposals)",
                       [(k, v) for k, v in sorted(groups.items())]))
    nb = sum(len(b) for _, subs in blocks[:len(cand["categories"])] for _, b in subs)
    info = cand.get("model_info") or {}
    F.taxonomy_tree(blocks, f"{name} ({info.get('model', '')}) — taxonomy of the {SOURCES[src]} ({cand.get('n_items_shown', '?')} items): "
                    f"{len(cand['categories'])} categories, "
                    f"{sum(len(c['subcategories']) for c in cand['categories'])} subcategories, {nb} behaviours; "
                    f"{cand.get('assigned_share', 0):.0%} of its proposals placed",
                    out / "taxonomy.png",
                    note="bar and number = items this model put under the behaviour; built by this model alone, before any merge (phase 6)",
                    last_apart=bool(left))
    cs = sorted(((c["name"], sum(b["n_items"] for s_ in c["subcategories"] for b in s_["behaviors"]))
                 for c in cand["categories"]), key=lambda t: -t[1])
    F.barh([c for c, _ in cs], [n for _, n in cs], f"{name}: categories of its taxonomy of the {SOURCES[src]}",
           "item-behaviour assignments", out / "categories.png")
