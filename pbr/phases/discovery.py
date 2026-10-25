"""Phase 5 — Taxonomy discovery by three independent LLM annotators, separately for each source.

Each annotator (three model families, config ``llm.annotators``), on its own and never shown another model's
answers, builds TWO taxonomies: one from the sampled Pharo methods (phase 4: production and test methods that
passed the method filters of phase 3), one from the sampled tasks of the existing benchmarks — six in all.

0. **Sample and saturation.** Each annotator starts with a sample of 95% confidence (±5%, Cochran with finite
   population correction: ~384 Pharo methods, ~354 benchmark problems; phase 4 draws the ordered pool) and goes on with
   rounds of ``round_size`` new items until ``patience`` rounds in a row bring no new behaviour (derived from
   ``min_share``: a behaviour of 0.5% prevalence would have appeared with 95% probability — the rule of the earlier
   labelling study; inductive thematic saturation). A behaviour is new when a group of similar proposals (embedding
   cosine ≥ ``novelty_threshold``) reaches ``min_support`` items. Curve: ``saturation.csv`` / ``saturation.png``.
1. **Open proposals.** For each item (in batches of one source), up to ``max_behaviors_per_item`` programming
   behaviours at three levels — category › subcategory › concrete behaviour — with the evidence (a quote of the
   item). Behaviours are named in language-neutral terms so Python tasks and Pharo methods can be compared; the
   evaluation task type is NOT a behaviour (§2).
2. **Consolidation.** The model organises its own proposals of that source into a hierarchy with a definition per
   concept, inclusion and exclusion rules, and the items each behaviour covers. Every call stays small whatever the
   number of proposals: similar proposals are grouped (embeddings); the model designs the categories and
   subcategories from its most frequent groups, places every group in one subcategory, then writes the behaviours of
   each subcategory in its own call (caps: ``discovery.consolidation``). Nothing is dropped: groups placed nowhere are
   kept as unassigned proposals.

Outputs, per model and source, ``<model>/<pharo|literature>/``:
  taxonomy.png       the taxonomy as a tree; a small bar and a number give the items classified under each behaviour
  taxonomy.json      the full hierarchy: definitions, rules, items, evidence, proposals not placed by the model
  taxonomy.csv       one row per behaviour (category, subcategory, behaviour, definitions, rules, n_items, examples)
  items.csv          each item with the category / subcategory / behaviour this model gave it (one row per behaviour)
  categories.png     the categories by number of item-behaviour assignments
  saturation.csv/png the saturation curve: items seen, new behaviours per round, rounds without a new one, status
  proposals.jsonl    every item's proposals;  raw_responses.jsonl   every raw answer (all attempts: results/cache/llm/)
and discovery_summary.csv (one row per model and source), saturation_curves.csv; figures/: the six taxonomies
compared, saturation_pharo.png and saturation_literature.png (the three annotators' curves).
"""
from __future__ import annotations

from collections import defaultdict
import re

import pandas as pd

from .. import figures as F
from ..common import parallel, pmap, read_json, read_jsonl, write_csv, write_json, write_jsonl
from ..items import load_items
from ..llm import Client, annotators, ledger_summary

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

SKELETON = """You proposed the programming behaviours listed below (id, category › subcategory › behavior, number of items).
Each line is a group of proposals that mean nearly the same thing; the less frequent groups are not listed.
Design the TOP of ONE clean hierarchical taxonomy: its categories and their subcategories only (the behaviours come later).
- at most MAXC categories, each with 1 to MAXS subcategories;
- together they cover every proposal listed; merge what means the same thing, keep different things apart;
- every category and subcategory gets a definition of ONE short sentence;
- names stay language-neutral (no Python, Smalltalk or library names).

Proposals:
"""

ROUTE = """Here are the categories and subcategories of your taxonomy (id: category › subcategory — definition):

SUBCATEGORIES

Place each proposal below (id: category › subcategory › behavior) in the ONE subcategory that covers it, by id;
answer "none" only when no subcategory fits.

Proposals:
"""

BEHAVIOURS = """Subcategory "SUBNAME" of the category "CATNAME": SUBDEF
The other subcategories of this category: SIBLINGS

You placed these proposals in this subcategory (id: behavior, number of items):

PROPOSALS

Write the BEHAVIOURS of this subcategory (1 to MAXB):
- a behaviour is a concrete thing code does: an imperative phrase of 3 to 8 words, language-neutral;
- merge proposals that mean the same thing, even if worded differently; keep different behaviours apart;
- definition: one sentence; include: 1 to 3 rules (when to use it); exclude: 1 to 3 rules (when not to, naming the
  neighbouring behaviour to use instead);
- proposals: the ids of the proposals the behaviour covers; every proposal id in exactly one behaviour."""

ASSIGN = """Here are the behaviours of one subcategory of your taxonomy (id: behavior — definition):

BEHAVIOURS

Place each proposal below (id: behavior) under the ONE behaviour that covers it, by id; answer "none" only when no
behaviour fits.

Proposals:
"""

CONSOLIDATE_VERSION = "discovery-consolidate-v3"    # the proposals keep PROMPT_VERSION: their cached answers stay valid
ASSIGN_BATCH = 80            # proposal groups per placement call
SUB_BATCH = 120              # proposal groups shown when a subcategory's behaviours are written (the rest: placed after)
TAXONOMIST = "You are an expert software engineer and a careful taxonomist. Answer with JSON only."


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


def skeleton_schema(maxc: int, maxs: int) -> dict:
    named = {"name": {"type": "string", "minLength": 2}, "definition": {"type": "string", "minLength": 10}}
    return {"type": "object", "properties": {"categories": {"type": "array", "minItems": 1, "maxItems": maxc, "items": {
        "type": "object", "properties": {**named, "subcategories": {"type": "array", "minItems": 1, "maxItems": maxs, "items": {
            "type": "object", "properties": named, "required": ["name", "definition"]}}},
        "required": ["name", "definition", "subcategories"]}}}, "required": ["categories"]}


def behaviours_schema(group_ids: list[str], maxb: int) -> dict:
    rule = {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string"}}
    return {"type": "object", "properties": {"behaviors": {"type": "array", "minItems": 1, "maxItems": maxb, "items": {
        "type": "object", "properties": {
            "name": {"type": "string", "minLength": 3}, "definition": {"type": "string", "minLength": 10},
            "include": rule, "exclude": rule,
            "proposals": {"type": "array", "items": {"type": "string", "enum": group_ids}}},
        "required": ["name", "definition", "include", "exclude", "proposals"]}}}, "required": ["behaviors"]}


def assign_schema(group_ids: list[str], target_ids: list[str]) -> dict:
    return {"type": "object", "properties": {"assignments": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string", "enum": group_ids}, "target": {"type": "string", "enum": target_ids + ["none"]}},
        "required": ["id", "target"]}, "minItems": len(group_ids), "maxItems": len(group_ids)}},
        "required": ["assignments"]}


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).split())


def batches(items: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    out = []
    for _, part in items.groupby("source", sort=True):      # literature and Pharo items in separate batches
        part = part.reset_index(drop=True)
        out += [part.iloc[i:i + size] for i in range(0, len(part), size)]
    return out


def propose(ph, client, batch_list, maxb, label: str = "") -> tuple[list[dict], list[dict]]:
    rows, raws = [], []

    def ask(b):
        local = [f"I{i + 1}" for i in range(len(b))]
        user = "Items:\n\n" + "\n\n".join(f"### {lid}\n{t}" for lid, t in zip(local, b["text"]))
        return client.ask(system=SYSTEM.replace("MAXB", str(maxb)), user=user, schema=item_schema(local, maxb),
                          max_tokens=300 * len(b) + 200, prompt_version=PROMPT_VERSION, phase=ph.key,
                          item_id=",".join(b["item_id"]))
    # the batches of a round are independent: asked in parallel (compute.llm_parallel), kept in their order
    for b, ans in zip(batch_list, ph.pmap(ask, batch_list, f"{client.name}{label}: proposal batches", every_s=120)):
        local = [f"I{i + 1}" for i in range(len(b))]
        raws.append({"items": list(b["item_id"]), "request_hash": ans.request_hash, "valid": ans.valid,
                     "errors": ans.errors, "attempts": ans.attempts, "raw": ans.text})
        got = {x.get("id"): x for x in (ans.data or {}).get("items", []) if isinstance(x, dict)} if ans.data else {}
        for lid, (_, it) in zip(local, b.iterrows()):
            x = got.get(lid)
            rows.append({"item_id": it["item_id"], "source": it["source"], "group": it["group"], "model": client.name,
                         "valid": bool(ans.valid and x), "behaviors": (x or {}).get("behaviors", []),
                         "insufficient_context": bool((x or {}).get("insufficient_context", False)),
                         "request_hash": ans.request_hash})
    return rows, raws


class Novelty:
    """Behaviours an annotator has proposed so far for one source, grouped by embedding similarity. A proposal joins
    the most similar group (cosine >= threshold) or starts a new one; a group counts as a behaviour once it reaches
    ``min_support`` items. ``add`` returns how many behaviours became established in this round (the new ones)."""

    def __init__(self, embedder, threshold: float, min_support: int):
        self.embedder, self.threshold, self.min_support = embedder, threshold, min_support
        self.vecs, self.counts, self.labels = [], [], []

    def add(self, rows: list[dict]) -> int:
        import numpy as np
        texts = [f"clustering: {b['category']} › {b['subcategory']} › {b['behavior']}" for r in rows for b in r["behaviors"]]
        if not texts:
            return 0
        new_est = 0
        for v, t in zip(self.embedder.embed(texts), texts):
            v = np.asarray(v)
            if self.vecs:
                sims = np.asarray(self.vecs) @ v
                j = int(np.argmax(sims))
                if sims[j] >= self.threshold:
                    self.counts[j] += 1
                    new_est += self.counts[j] == self.min_support
                    continue
            self.vecs.append(v)
            self.counts.append(1)
            self.labels.append(t.removeprefix("clustering: "))
            new_est += self.min_support == 1
        return new_est

    @property
    def established(self) -> int:
        return sum(c >= self.min_support for c in self.counts)


def saturation_loop(ph, client, src: str, pool, extras, plan: dict, maxb: int, embedder) -> tuple[list, list, list]:
    """95% start sample, then rounds of new items until ``patience`` rounds in a row add no new behaviour."""
    d = ph.cfg["discovery"]
    nov = Novelty(embedder, float(d["novelty_threshold"]), int(d.get("min_support", 2)))
    rows, raws, curve = [], [], []
    start, R, patience, max_rounds = plan["initial"], plan["round_size"], plan["patience"], plan["max_rounds"]
    streak, rnd, seen = 0, 0, 0
    status = "pool exhausted before saturation"
    while True:
        if rnd == 0:
            batch_items = pd.concat([pool.iloc[:start], extras])
        else:
            lo = start + (rnd - 1) * R
            batch_items = pool.iloc[lo:lo + R]
        if batch_items.empty:
            break
        r_rows, r_raws = propose(ph, client, batches(batch_items, int(d["batch_size"])), maxb, f"/{src} round {rnd}")
        new = nov.add(r_rows)
        rows += r_rows
        raws += r_raws
        seen += len(batch_items)
        if rnd > 0:
            streak = streak + 1 if new == 0 else 0
        curve.append({"round": rnd, "items_added": len(batch_items), "items_seen": seen,
                      "proposals": sum(len(r["behaviors"]) for r in r_rows), "new_behaviors": new,
                      "behaviors": nov.established, "proposal_groups": len(nov.counts), "rounds_without_new": streak})
        ph.log.info(f"{client.name}/{src} round {rnd}: +{len(batch_items)} items ({seen} seen), {new} new behaviour(s), "
                    f"{nov.established} so far; {streak}/{patience} rounds without a new one")
        if rnd > 0 and streak >= patience:
            status = "saturated"
            break
        if rnd >= max_rounds:
            status = f"stopped at max_rounds ({max_rounds}) before saturation"
            ph.warn(f"{client.name}/{src}: not saturated after {max_rounds} rounds ({seen} items)")
            break
        rnd += 1
    ph.log.info(f"{client.name}/{src}: {status} after {seen} items ({rnd} round(s) after the start sample)")
    for c in curve:
        c["status"] = status
    return rows, raws, curve


def group_proposals(plist: list[dict], embedder, threshold: float) -> list[list[dict]]:
    """Proposals of one model grouped by embedding similarity (as in the saturation loop): a proposal joins the most
    similar group (cosine >= threshold) or starts one. Proposals come most frequent first: a group's first is its label."""
    import numpy as np
    texts = [f"clustering: {' › '.join(p['label'])}" for p in plist]
    vecs, groups = [], []
    for p, v in zip(plist, embedder.embed(texts)):
        v = np.asarray(v)
        if vecs:
            sims = np.asarray(vecs) @ v
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                groups[j].append(p)
                continue
        vecs.append(v)
        groups.append([p])
    return groups


def place(ph, client, groups: list[str], label: dict, targets: dict, prompt: str, item_id: str) -> tuple[dict, int]:
    """{group id: target id} for the groups the model places under one of ``targets`` ({id: line}), in batches;
    the number of invalid calls."""
    chunks = [groups[k:k + ASSIGN_BATCH] for k in range(0, len(groups), ASSIGN_BATCH)]

    def ask(chunk):
        return client.ask(system=TAXONOMIST, user=prompt.replace("TARGETS", "\n".join(targets.values()))
                          + "\n".join(f"{g}: {label[g]}" for g in chunk),
                          schema=assign_schema(chunk, list(targets)), max_tokens=40 * len(chunk) + 300,
                          prompt_version=CONSOLIDATE_VERSION, phase=ph.key, item_id=f"{item_id}:{chunk[0]}-{chunk[-1]}")
    out, bad = {}, 0
    for chunk, a in zip(chunks, ph.pmap(ask, chunks, f"{item_id}: placing proposal groups")):
        bad += not a.valid
        for x in (a.data or {}).get("assignments", []) if a.valid else []:
            if x.get("id") in chunk and x["id"] not in out and x.get("target") in targets:
                out[x["id"]] = x["target"]
    return out, bad


def consolidate(ph, client, rows, source: str, out_dir, embedder) -> dict:
    """The model organises its own proposals of one source into a hierarchy; proposals keep their items.
    Every call and answer stays small, whatever the number of proposals:
      1. proposals that mean nearly the same thing are grouped (embedding cosine >= novelty_threshold);
      2. the model designs the categories and subcategories from its most frequent groups (``overview_chars``);
      3. it places every group in one subcategory (batches of ``ASSIGN_BATCH``);
      4. for each subcategory, it writes the behaviours (definition, inclusion and exclusion rules) and says which
         groups each covers; groups it leaves out are placed under one of them afterwards.
    Groups placed nowhere are kept as unassigned proposals (phase 6 still uses them), never dropped."""
    cc = ph.cfg["discovery"]["consolidation"]
    maxc, maxs, maxb = int(cc["max_categories"]), int(cc["max_subcategories"]), int(cc["max_behaviors"])
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
    groups = group_proposals(plist, embedder, float(ph.cfg["discovery"]["novelty_threshold"]))
    gitems = [list(dict.fromkeys(i for p in g for i in p["items"])) for g in groups]
    order = sorted(range(len(groups)), key=lambda j: (-len(gitems[j]), groups[j][0]["label"]))
    gid = {j: f"g{n + 1}" for n, j in enumerate(order)}
    G = {gid[j]: {"members": groups[j], "items": gitems[j], "label": groups[j][0]["label"]} for j in order}
    gids = [gid[j] for j in order]
    line = {g: f"{g}: {' › '.join(G[g]['label'])} ({len(G[g]['items'])})" for g in gids}
    shown, size = [], 0
    for g in gids:
        if size + len(line[g]) > int(cc["overview_chars"]):
            break
        shown.append(line[g])
        size += len(line[g]) + 1
    ph.log.info(f"{client.name}/{source}: {len(plist)} distinct proposals in {len(groups)} groups; the categories and "
                f"subcategories are designed from the {len(shown)} most frequent")
    ans = client.ask(system=TAXONOMIST, user=SKELETON.replace("MAXC", str(maxc)).replace("MAXS", str(maxs)) + "\n".join(shown),
                     schema=skeleton_schema(maxc, maxs), max_tokens=60 * maxc * (maxs + 1) + 300,
                     prompt_version=CONSOLIDATE_VERSION, phase=ph.key, item_id=f"consolidation:{source}")
    candidate = {"model": client.name, "source": source, "n_items_shown": len(rows), "model_info": client.info(),
                 "prompt_version": CONSOLIDATE_VERSION, "valid": ans.valid, "errors": ans.errors,
                 "request_hash": ans.request_hash, "categories": [], "n_proposals": len(plist),
                 "n_proposal_groups": len(groups), "groups_in_overview": len(shown)}
    placed, problems = {}, []
    if ans.valid:
        subs = {}                                    # subcategory id -> (category, subcategory) of the skeleton
        for c in ans.data["categories"]:
            for s_ in c["subcategories"]:
                subs[f"s{len(subs) + 1}"] = (c, s_)
        lbl = {g: " › ".join(G[g]["label"]) for g in gids}
        route, bad = place(ph, client, gids, lbl, {i: f"{i}: {c['name'].strip()} › {s_['name'].strip()} — {s_['definition'].strip()}"
                                                   for i, (c, s_) in subs.items()},
                           ROUTE.replace("SUBCATEGORIES", "TARGETS"), f"consolidation:{source}:route")
        if bad:
            problems.append(f"{bad} routing call(s) invalid")
        members = defaultdict(list)
        for g in gids:                               # most frequent first
            if g in route:
                members[route[g]].append(g)

        def write_behaviours(sid):
            c, s_ = subs[sid]
            mine = members[sid]
            head = mine[:SUB_BATCH]                  # the rest is placed afterwards
            sib = [x["name"].strip() for x in c["subcategories"] if x is not s_]
            r = client.ask(system=TAXONOMIST, user=BEHAVIOURS.replace("SUBNAME", s_["name"].strip())
                           .replace("CATNAME", c["name"].strip()).replace("SUBDEF", s_["definition"].strip())
                           .replace("SIBLINGS", "; ".join(sib) or "none").replace("MAXB", str(maxb))
                           .replace("PROPOSALS", "\n".join(f"{g}: {G[g]['label'][2]} ({len(G[g]['items'])})" for g in head)),
                           schema=behaviours_schema(head, maxb), max_tokens=maxb * 250 + 16 * len(head) + 400,
                           prompt_version=CONSOLIDATE_VERSION, phase=ph.key, item_id=f"consolidation:{source}:{sid}")
            if not r.valid:
                return sid, [], {}, 1
            behs, got = [], {}
            for k, b in enumerate(r.data["behaviors"]):
                bid = f"{sid}b{k + 1}"
                behs.append((bid, b))
                for g in b["proposals"]:
                    got.setdefault(g, bid)
            left = [g for g in mine if g not in got]          # left out of the answer, or not shown (rest)
            bad = 0
            if left:
                more, bad = place(ph, client, left, {g: G[g]["label"][2] for g in left},
                                  {bid: f"{bid}: {b['name'].strip()} — {b['definition'].strip()}" for bid, b in behs},
                                  ASSIGN.replace("BEHAVIOURS", "TARGETS"), f"consolidation:{source}:{sid}:place")
                got.update(more)
            return sid, behs, got, bad

        built = {}
        for sid, behs, got, bad in ph.pmap(write_behaviours, [x for x in subs if members[x]],
                                           f"{client.name}/{source}: behaviours per subcategory"):
            if bad:
                problems.append(f"subcategory {sid}: {bad} invalid call(s)")
            built[sid] = (behs, got)
        for c in ans.data["categories"]:
            cc_ = {"name": c["name"].strip(), "definition": c["definition"].strip(), "subcategories": []}
            for s_ in c["subcategories"]:
                sid = next(i for i, (_, x) in subs.items() if x is s_)
                behs, got = built.get(sid, ([], {}))
                ss = {"name": s_["name"].strip(), "definition": s_["definition"].strip(), "behaviors": []}
                for bid, b in behs:
                    mem = [g for g in members[sid] if got.get(g) == bid]
                    if not mem:
                        continue                    # a behaviour no proposal supports is not kept
                    placed.update({g: bid for g in mem})
                    ps = [p for g in mem for p in G[g]["members"]]
                    items = list(dict.fromkeys(i for p in ps for i in p["items"]))
                    ss["behaviors"].append({"name": b["name"].strip(), "definition": b["definition"].strip(),
                                            "include": b["include"], "exclude": b["exclude"],
                                            "proposals": [p["label"] for p in ps], "examples": items[:8], "items": items,
                                            "n_items": len(items), "evidence": [e for p in ps for e in p["evidence"] if e][:3]})
                if ss["behaviors"]:
                    cc_["subcategories"].append(ss)
            if cc_["subcategories"]:
                candidate["categories"].append(cc_)
    for pr in problems:
        ph.warn(f"{client.name}/{source} consolidation: {pr} — those proposals stay unassigned")
    # proposals the consolidation did not place are kept, never dropped
    rest = [p for g in gids if g not in placed for p in G[g]["members"]]
    candidate["unassigned_proposals"] = [{"label": p["label"], "items": p["items"]} for p in rest]
    candidate["assigned_share"] = round(1 - len(rest) / max(len(plist), 1), 3)
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
    plans = read_json(ph.upstream("sampling", "sampling_design.json"))["discovery"]["plans"]
    embedder = Client({"name": "embedder", **ph.cfg["llm"]["embedder"]}, ph.cfg)
    summary, all_rows, cands, curves = [], [], {}, []
    clients = annotators(ph.cfg)
    for client in clients:
        info = client.info()
        ph.log.info(f"annotator {client.name}: {info.get('model')} ({info.get('family') or ''} "
                    f"{info.get('parameter_size') or ''} {info.get('quantization') or ''}, digest {info.get('digest', '')})")

    def one(task):
        """One annotator on one source, alone: saturation loop, then consolidation (no figure: not thread-safe)."""
        client, src = task
        its = by_source[src]
        out = ph.path(client.name, src, "x").parent
        pool = its[its["order"].notna()].sort_values("order")
        extras = its[its["order"].isna()]            # targeted and test methods: in the first round
        rows, raws, curve = saturation_loop(ph, client, src, pool, extras, plans[src], maxb, embedder)
        write_jsonl(out / "proposals.jsonl", rows)
        write_jsonl(out / "raw_responses.jsonl", raws)
        write_csv(out / "saturation.csv", pd.DataFrame(curve))
        ph.log.info(f"{client.name}/{src}: consolidating {sum(len(r['behaviors']) for r in rows)} proposals into a hierarchy")
        cand = consolidate(ph, client, rows, src, out, embedder)
        export_tables(out, cand, titles)
        return client, src, out, rows, curve, cand

    # (annotator, source) tasks are independent: compute.models_parallel at once (sources interleaved, so the
    # tasks running together use different models)
    tasks = [(c, src) for src, its in by_source.items() if not its.empty for c in clients]
    for client, src, out, rows, curve, cand in pmap(one, tasks, parallel(ph.cfg, "models_parallel")):
        info = client.info()
        curves += [{**c, "model": client.name, "source": src} for c in curve]
        saturation_figure(curve, f"{client.name}: saturation on the {SOURCES[src]}", out / "saturation.png", plans[src])
        bad = sum(not r["valid"] for r in rows)
        if bad:
            ph.warn(f"{client.name}/{src}: {bad} items without a valid proposal (kept in proposals.jsonl, valid=false)")
        if not cand["valid"]:
            ph.warn(f"{client.name}/{src}: consolidation invalid after retries ({cand['errors'][:2]}) — candidate kept empty")
        nc, ns, nb = counts(cand)
        ph.log.info(f"{client.name}/{src}: {nc} categories, {ns} subcategories, {nb} behaviours; "
                    f"{cand['assigned_share']:.0%} of proposals placed")
        summary.append({"model": client.name, "source": src, "model_id": info.get("model"), "digest": info.get("digest"),
                        "items": len(rows), "invalid_items": bad, "saturation": curve[-1]["status"] if curve else None,
                        "rounds": curve[-1]["round"] if curve else 0,
                        "insufficient_context": sum(r["insufficient_context"] for r in rows),
                        "proposals": sum(len(r["behaviors"]) for r in rows), "distinct_proposals": cand["n_proposals"],
                        "categories": nc, "subcategories": ns, "behaviors": nb,
                        "assigned_share": cand["assigned_share"], "consolidation_valid": cand["valid"]})
        all_rows += rows
        cands[(client.name, src)] = cand
        model_figures(ph, client.name, src, cand, out)
    sdf = pd.DataFrame(summary)
    write_csv(ph.path("discovery_summary.csv"), sdf)
    cdf = pd.DataFrame(curves)
    write_csv(ph.path("saturation_curves.csv"), cdf)
    for src in SOURCES:
        sub = cdf[cdf["source"] == src] if len(cdf) else cdf
        if len(sub):
            fig, ax = F.plt.subplots(figsize=(8.5, 4.2))
            for i, (mname, g) in enumerate(sub.groupby("model", sort=False)):
                ax.plot(g["items_seen"], g["behaviors"], color=F.color(i), lw=2, marker="o", ms=3.5, label=mname)
                ax.annotate(f"{g['status'].iloc[-1].split(' ')[0]}", (g["items_seen"].iloc[-1], g["behaviors"].iloc[-1]),
                            xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=F.INK_2, va="center")
            ax.axvline(plans[src]["initial"], color=F.GRID, lw=1.2, ls="--")
            ax.text(plans[src]["initial"], ax.get_ylim()[1] * 0.97, f"  95% start sample ({plans[src]['initial']})",
                    fontsize=7.5, color=F.INK_2, va="top")
            ax.set_xlabel("items seen")
            ax.set_ylabel(f"behaviours (≥ {ph.cfg['discovery'].get('min_support', 2)} items)")
            ax.set_title(f"Saturation on the {SOURCES[src]}: rounds of {plans[src]['round_size']} until "
                         f"{plans[src]['patience']} rounds add nothing")
            ax.legend()
            F.save(fig, ph.figure(f"saturation_{src}.png"))
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


def saturation_figure(curve: list, title: str, path, plan: dict):
    if not curve:
        return
    c = pd.DataFrame(curve)
    fig, ax = F.plt.subplots(figsize=(7.5, 3.8))
    ax.plot(c["items_seen"], c["behaviors"], color=F.SERIES[0], lw=2, marker="o", ms=4)
    for r in c.itertuples():
        if r.round > 0 and r.new_behaviors == 0:
            ax.plot(r.items_seen, r.behaviors, "o", ms=7, mfc="none", mec=F.STATUS["good"])
    ax.axvline(plan["initial"], color=F.GRID, lw=1.2, ls="--")
    ax.set_xlabel("items seen")
    ax.set_ylabel("behaviours established")
    ax.set_title(title)
    F.save(fig, path, note=f"{c['status'].iloc[-1]}; green ring = a round without a new behaviour "
                           f"({plan['patience']} in a row = saturation); dashed = end of the 95% start sample")


def model_figures(ph, name: str, src: str, cand: dict, out):
    """In <model>/<source>/: the taxonomy as a tree (taxonomy.png, with the number of items of each behaviour as a
    small bar) and its categories by size (categories.png)."""
    if not cand["categories"]:
        return
    blocks = []
    for c in cand["categories"]:
        subs, c_items = [], set()
        c_sum = 0
        for s in c["subcategories"]:
            s_items = {i for b in s["behaviors"] for i in (b.get("items") or [])}
            s_sum = sum(int(b.get("n_items") or 0) for b in s["behaviors"])   # when there are no item lists
            c_items |= s_items
            c_sum += s_sum
            subs.append(({"label": s["name"], "count": len(s_items) or s_sum},
                         [{"label": b["name"], "count": b["n_items"]} for b in s["behaviors"]]))
        blocks.append(({"label": c["name"], "count": len(c_items) or c_sum}, subs))
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
    F.taxonomy_tree(blocks, f"{name} ({info.get('model', '')}) — taxonomy of the {SOURCES.get(src, src)} ({cand.get('n_items_shown', '?')} items): "
                    f"{len(cand['categories'])} categories, "
                    f"{sum(len(c['subcategories']) for c in cand['categories'])} subcategories, {nb} behaviours; "
                    f"{cand.get('assigned_share', 0):.0%} of its proposals placed",
                    out / "taxonomy.png",
                    note="bar and number = items this model put under the behaviour; built by this model alone, before any merge (phase 6)",
                    last_apart=bool(left))
    cs = sorted(((c["name"], sum(b["n_items"] for s_ in c["subcategories"] for b in s_["behaviors"]))
                 for c in cand["categories"]), key=lambda t: -t[1])
    F.barh([c for c, _ in cs], [n for _, n in cs], f"{name}: categories of its taxonomy of the {SOURCES.get(src, src)}",
           "item-behaviour assignments", out / "categories.png")
