"""Phase 7 — AI agreement and expert validation.

A. **AI agreement** on the annotation split, with standardised IDs: per item, full / partial / no
   agreement, or insufficient context / unknown (at least two models say so); Krippendorff's α with the
   MASI distance on behaviour sets (multi-label) and nominal on the primary behaviour and category; Fleiss'
   κ and Gwet's AC1; pairwise Cohen's κ; α per behaviour (present / absent).
B. **Blinded expert workbook** (``expert/expert_workbook.xlsx``): items stratified by AI agreement level
   (more where models disagree; strata sizes kept for stratified estimators), with the code / task and
   the taxonomy definitions — AI predictions are NOT shown. Sheets: Method Annotation, Taxonomy (reference),
   Taxonomy Review, Behavior Relevance.
C. **Import** of completed workbooks (``expert/returned/*.xlsx``): validated (known rows, existing IDs,
   reviewer identity); every decision kept with its reviewer, file and sha256. Experts' agreement among
   themselves (the human baseline), reference labels, then human vs AI: precision / recall / F1, exact
   match, accuracy of the primary behaviour and category (overall, stratified, and per AI agreement level —
   AI majority agreement is not treated as correctness).
D. **Adjudication**: items where experts disagree among themselves or with the AI majority go to
   ``adjudication/adjudication.xlsx`` (now with the AI answers revealed); returned decisions override.
E. **Corrected taxonomy**: Taxonomy Review decisions → ``taxonomy_v2.json`` (v1 annotations stay as they are;
   the map v1 → v2 is in the file).

Without returned workbooks the phase exports the workbook, computes AI agreement only and is provisional.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import re

import numpy as np
import pandas as pd

from .. import agreement as A
from .. import figures as F
from .. import taxonomy as T
from .. import workbook as W
from ..common import now_iso, read_json, write_csv, write_json, write_jsonl, write_text
from ..items import load_items

ID_RE = re.compile(r"^\s*(B\d{3,}|UNKNOWN|INSUFFICIENT)\b", re.I)
LEVELS = ["full", "partial", "none", "insufficient", "unknown", "missing"]


# ---------------------------------------------------------------------------------------------- A. AI agreement
def item_table(ann: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for item_id, g in ann.groupby("item_id"):
        by = {r.model: r for r in g.itertuples()}
        sets = [frozenset(by[m].behavior_ids) if m in by else None for m in models]
        present = [s for s in sets if s is not None]
        n_ins = sum(T.INSUFFICIENT in s for s in present)
        n_unk = sum(T.UNKNOWN in s for s in present)
        if len(present) < 2:
            level = "missing"
        elif n_ins >= 2:
            level = "insufficient"
        elif n_unk >= 2:
            level = "unknown"
        else:
            level = A.agreement_level(present)
        maj = A.majority_set(present) if present else frozenset()
        # the AI-majority primary behaviour: the primary most models gave (ties: annotator order), when it is in
        # the majority set; else the majority set's first member
        prims = [by[m].primary_behavior for m in models if m in by]
        top = Counter(prims).most_common(1)[0] if prims else (None, 0)
        maj_primary = top[0] if top[0] in maj and top[1] >= 2 else (sorted(maj)[0] if maj else None)
        first = g.iloc[0]
        rows.append({"item_id": item_id, "source": first.source, "group": first.group, "purpose": first.purpose,
                     "level": level, "n_models": len(present), "majority": sorted(maj), "majority_primary": maj_primary,
                     **{f"{m}_behaviors": sorted(by[m].behavior_ids) if m in by else None for m in models},
                     **{f"{m}_primary": by[m].primary_behavior if m in by else None for m in models},
                     **{f"{m}_primary_category": by[m].primary_category if m in by else None for m in models}})
    return pd.DataFrame(rows)


def ai_metrics(items: pd.DataFrame, models: list[str], tax: dict) -> dict:
    def block(df):
        sets = [[frozenset(r[f"{m}_behaviors"]) if r[f"{m}_behaviors"] is not None else None for m in models]
                for _, r in df.iterrows()]
        prim = [[r[f"{m}_primary"] for m in models] for _, r in df.iterrows()]
        pcat = [[r[f"{m}_primary_category"] for m in models] for _, r in df.iterrows()]
        csets = [[frozenset(T.behavior_path(tax, b)[0] for b in s) if s is not None else None for s in u] for u in sets]
        by_model = {m: [r[f"{m}_primary"] for _, r in df.iterrows()] for m in models}
        return {"items": int(len(df)),
                "alpha_masi_behaviors": A.krippendorff_alpha(sets, A.masi),
                "alpha_nominal_primary_behavior": A.krippendorff_alpha(prim),
                "alpha_masi_categories": A.krippendorff_alpha(csets, A.masi),
                "alpha_nominal_primary_category": A.krippendorff_alpha(pcat),
                "fleiss_kappa_primary_behavior": A.fleiss_kappa(prim),
                "gwet_ac1_primary_behavior": A.gwet_ac1(prim),
                "gwet_ac1_primary_category": A.gwet_ac1(pcat),
                "pairwise_cohen_primary_behavior": A.pairwise(by_model, A.cohen_kappa),
                "levels": df["level"].value_counts().reindex(LEVELS, fill_value=0).to_dict()}
    out = {"overall": block(items)}
    for src in ("literature", "pharo"):
        sub = items[items["source"] == src]
        if len(sub):
            out[src] = block(sub)
    # per behaviour: presence / absence across models
    per = {}
    for b in T.behavior_ids(tax):
        units = [[(b in r[f"{m}_behaviors"]) if r[f"{m}_behaviors"] is not None else None for m in models]
                 for _, r in items.iterrows()]
        support = sum(any(u) for u in units if None not in u)
        if support >= 3:
            per[b] = {"alpha": A.krippendorff_alpha(units), "items_with_any_model": int(support)}
    out["per_behavior"] = per
    return out


# ---------------------------------------------------------------------------------------------- B. expert workbook
def select_expert_items(ph, items: pd.DataFrame) -> pd.DataFrame:
    n = int(ph.sizes["expert_items"])
    shares = ph.cfg["validation"]["expert_strata"]
    stratum = items["level"].map(lambda l: l if l in ("full", "partial") else "none")
    items = items.assign(stratum=stratum)
    rng = np.random.default_rng(ph.seed + 7)
    want = {s: int(round(n * float(shares.get(s, 0)))) for s in ("full", "partial", "none")}
    picked, short = [], 0
    for s in ("none", "partial", "full"):
        pool = items[items["stratum"] == s]
        k = min(len(pool), want[s] + short)
        short = want[s] + short - k
        if k:
            picked.append(pool.iloc[np.sort(rng.choice(len(pool), size=k, replace=False))])
    sel = pd.concat(picked) if picked else items.head(0)
    sizes = items["stratum"].value_counts()
    sel = sel.assign(stratum_N=sel["stratum"].map(sizes), stratum_m=sel["stratum"].map(sel["stratum"].value_counts()))
    sel["stratum_weight"] = sel["stratum_N"] / sel["stratum_m"]
    return sel.sample(frac=1, random_state=ph.seed)       # shuffled: the strata are not visible in the order


def behavior_choices(tax: dict) -> list[str]:
    return [f"{b['id']} {b['name']}" for b in T.active(tax, "behavior")] + [T.UNKNOWN, T.INSUFFICIENT]


def write_expert_workbook(ph, tax: dict, sel: pd.DataFrame, texts: dict):
    idx = T.index(tax)
    rows = [{"row_key": r.item_id, "source": r.source, "title": texts[r.item_id]["title"],
             "provenance": texts[r.item_id]["provenance"], "item": texts[r.item_id]["text"]} for r in sel.itertuples()]
    wb = W.new_workbook("Expert validation (phase 7) — blinded", [
        "Sheet 'Method Annotation': for each item (a Pharo method or a code benchmark task), choose 1 to 3 behaviours "
        "from the drop-down (most important first), quote the evidence, give your confidence. "
        "UNKNOWN = no behaviour fits (describe it in missing_behavior); INSUFFICIENT = not enough context.",
        "The AI annotators' answers are deliberately NOT shown: annotate independently. "
        "Definitions and rules: sheet 'Taxonomy' (and annotation_guidelines in results/05_taxonomy/).",
        "Sheet 'Taxonomy Review' (optional): accept / rename / redefine / merge / split / reject / add / move concepts; "
        "your decisions produce the next taxonomy version.",
        "Sheet 'Behavior Relevance': how important is each behaviour in real Pharo code (1 = marginal ... 5 = essential), "
        "and for a Pharo benchmark. Do not edit row_key or id columns.",
        "Save the file as expert/returned/<your name>.xlsx and put your name in B3 above.",
    ], {"taxonomy version": tax["version"], "items": len(rows), "generated": now_iso()})
    W.add_sheet(wb, "Method Annotation", rows,
                [("row_key", "row_key", 16, False), ("source", "source", 10, False), ("title", "title", 26, False),
                 ("provenance", "provenance", 26, False), ("item", "item", 80, False),
                 ("behavior_1", "behavior_1", 30, True), ("behavior_2", "behavior_2", 30, True),
                 ("behavior_3", "behavior_3", 30, True), ("evidence", "evidence", 30, True),
                 ("confidence", "confidence", 11, True), ("missing_behavior", "missing_behavior", 24, True),
                 ("notes", "notes", 24, True)],
                choices={"behavior_1": behavior_choices(tax), "behavior_2": behavior_choices(tax),
                         "behavior_3": behavior_choices(tax), "confidence": ["low", "medium", "high"]}, row_height=160)
    trows = []
    for lv in ("category", "subcategory", "behavior"):
        for c in T.active(tax, lv):
            p = c.get("subcategory_id") or c.get("category_id") or ""
            trows.append({"id": c["id"], "level": lv, "parent": f"{p} {idx[p]['name']}" if p in idx else "",
                          "name": c["name"], "definition": c["definition"], "use when": c.get("include"),
                          "not when": c.get("exclude")})
    W.add_sheet(wb, "Taxonomy", trows, [(k, k, w, False) for k, w in (("id", 8), ("level", 11), ("parent", 28),
                                                                      ("name", 30), ("definition", 60),
                                                                      ("use when", 40), ("not when", 40))])
    W.add_sheet(wb, "Taxonomy Review", [{k: t[k] for k in ("id", "level", "parent", "name", "definition")} for t in trows],
                [("id", "id", 8, False), ("level", "level", 11, False), ("parent", "current_parent", 26, False),
                 ("name", "name", 30, False), ("definition", "definition", 50, False), ("action", "action", 11, True),
                 ("new_name", "new_name", 24, True), ("new_definition", "new_definition", 34, True),
                 ("merge_into", "merge_into", 11, True), ("split_into", "split_into", 26, True),
                 ("parent_id", "parent_id", 10, True), ("comment", "comment", 30, True)],
                choices={"action": list(T.REVIEW_ACTIONS)})
    W.add_sheet(wb, "Behavior Relevance", [{"id": b["id"], "name": b["name"], "definition": b["definition"]}
                                           for b in T.active(tax, "behavior")],
                [("id", "id", 8, False), ("name", "name", 32, False), ("definition", "definition", 60, False),
                 ("relevance_pharo", "relevance_pharo", 14, True), ("relevance_benchmark", "relevance_benchmark", 16, True),
                 ("comment", "comment", 30, True)],
                choices={"relevance_pharo": ["1", "2", "3", "4", "5"], "relevance_benchmark": ["1", "2", "3", "4", "5"]})
    W.save(wb, ph.path("expert", "expert_workbook.xlsx"))


# ---------------------------------------------------------------------------------------------- C. import
def parse_ids(vals, valid: set) -> tuple[list[str], list[str]]:
    ids, bad = [], []
    for v in vals:
        if v is None or not str(v).strip():
            continue
        m = ID_RE.match(str(v))
        x = m.group(1).upper() if m else None
        if x and (x in valid or x in T.RESERVED):
            ids.append(x)
        else:
            bad.append(str(v)[:40])
    return list(dict.fromkeys(ids)), bad


def import_experts(ph, tax: dict, keys: set) -> tuple[pd.DataFrame, list, list, pd.DataFrame, list]:
    """(human labels, taxonomy decisions, relevance rows, raw rows, problems)."""
    valid = set(T.behavior_ids(tax))
    labels, decisions, relevance, raw, problems = [], [], [], [], []
    for f in W.returned_files(ph.path("expert", "returned")):
        reviewer, rows, prov = W.read_sheet(f, "Method Annotation")
        if prov.get("error"):
            problems.append(f"{f.name}: {prov['error']}")
            continue
        if prov["reviewer_from"] == "file name":
            problems.append(f"{f.name}: README!B3 (reviewer_id) is empty — the file name '{reviewer}' is used")
        seen = set()
        for r in rows:
            key = r.get("row_key")
            raw.append({**{k: (str(v) if v is not None else None) for k, v in r.items()}, "reviewer": reviewer, **prov})
            if key not in keys:
                problems.append(f"{f.name}: unknown row_key {key!r} (edited or from another workbook) — ignored")
                continue
            if key in seen:
                problems.append(f"{f.name}: row_key {key} appears twice — first kept")
                continue
            seen.add(key)
            ids, bad = parse_ids([r.get("behavior_1"), r.get("behavior_2"), r.get("behavior_3")], valid)
            for b in bad:
                problems.append(f"{f.name} {key}: '{b}' is not an ID of taxonomy {tax['version']} — ignored")
            if any(i in T.RESERVED for i in ids) and len(ids) > 1:
                problems.append(f"{f.name} {key}: UNKNOWN/INSUFFICIENT with other behaviours — reserved answer dropped")
                ids = [i for i in ids if i not in T.RESERVED]
            if not ids:
                continue          # not answered (yet)
            labels.append({"item_id": key, "reviewer": reviewer, "behavior_ids": ids,
                           "evidence": r.get("evidence"), "confidence": r.get("confidence"),
                           "missing_behavior": r.get("missing_behavior"), "notes": r.get("notes"),
                           "file": f.name, "sha256": prov["sha256"], "imported_at": prov["imported_at"]})
        _, trows, _ = W.read_sheet(f, "Taxonomy Review")
        decisions += [{**{k: r.get(k) for k in ("id", "action", "new_name", "new_definition", "merge_into",
                                                 "split_into", "parent_id", "comment")}, "reviewer": reviewer, "file": f.name}
                      for r in trows if r.get("action") and str(r.get("action")).strip().lower() != "accept"]
        _, rrows, _ = W.read_sheet(f, "Behavior Relevance")
        for r in rrows:
            if r.get("relevance_pharo") or r.get("relevance_benchmark"):
                relevance.append({"behavior_id": r.get("id"), "reviewer": reviewer,
                                  "relevance_pharo": _num(r.get("relevance_pharo")),
                                  "relevance_benchmark": _num(r.get("relevance_benchmark")), "comment": r.get("comment"),
                                  "file": f.name})
    return pd.DataFrame(labels), decisions, relevance, pd.DataFrame(raw), problems


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def reference_labels(human: pd.DataFrame, adjud: dict) -> pd.DataFrame:
    """One reference set per item: adjudicated if decided, else the experts' majority (one expert: theirs)."""
    rows = []
    for item_id, g in human.groupby("item_id"):
        sets = [frozenset(x) for x in g["behavior_ids"]]
        if item_id in adjud:
            ref, how = frozenset(adjud[item_id]["behavior_ids"]), "adjudicated"
        elif len(sets) == 1:
            ref, how = sets[0], "single expert"
        else:
            maj = A.majority_set(sets, k=(len(sets) + 1) // 2)
            ref, how = (maj, "expert majority") if maj else (frozenset(), "experts disagree: needs adjudication")
        rows.append({"item_id": item_id, "reference": sorted(ref), "how": how, "n_experts": len(sets),
                     "reviewers": sorted(g["reviewer"])})
    return pd.DataFrame(rows)


def human_vs_ai(ref: pd.DataFrame, items: pd.DataFrame, sel: pd.DataFrame, models: list[str], tax: dict) -> dict:
    df = ref[ref["reference"].map(len) > 0].merge(items, on="item_id").merge(
        sel[["item_id", "stratum", "stratum_N", "stratum_m"]], on="item_id", how="left")
    if df.empty:
        return {}
    out = {"items": int(len(df))}
    answers = {m: f"{m}_behaviors" for m in models}
    answers["ai_majority"] = "majority"
    strata = {s: {"N": int(g["stratum_N"].iloc[0]), "m": int(len(g))} for s, g in df.groupby("stratum")}
    for name, col in answers.items():
        pred = [set(x) if x is not None else None for x in df[col]]
        gold = [set(x) for x in df["reference"]]
        prf = A.multilabel_prf(gold, pred)
        primary_pred = list(df[f"{name}_primary"] if name != "ai_majority" else df["majority_primary"])
        prim_ok = [p is not None and p in set(g) for p, g in zip(primary_pred, df["reference"])]
        cat_ok = [p is not None and T.behavior_path(tax, p)[0] in {T.behavior_path(tax, b)[0] for b in g}
                  for p, g in zip(primary_pred, df["reference"])]
        vals = defaultdict(list)
        for s, ok in zip(df["stratum"], cat_ok):
            vals[s].append(1.0 if ok else 0.0)
        by_level = {lv: round(float(np.mean([o for o, l in zip(prim_ok, df["level"]) if l == lv])), 3)
                    for lv in df["level"].unique()}
        out[name] = {"micro": prf["micro"], "macro": prf["macro"], "exact_match": prf["exact_match"],
                     "primary_behavior_in_reference": round(float(np.mean(prim_ok)), 4),
                     "primary_category_in_reference": round(float(np.mean(cat_ok)), 4),
                     "primary_category_accuracy_stratified": A_stratified(strata, vals),
                     "primary_behavior_correct_by_ai_agreement_level": by_level,
                     "cohen_kappa_primary_vs_expert": A.cohen_kappa(primary_pred, [g[0] for g in df["reference"]]),
                     "per_label": {k: v for k, v in prf["per_label"].items() if v["support"] or v["predicted"]}}
    return out


def A_stratified(strata: dict, values: dict) -> dict:
    """Stratified mean of 0/1 values (Cochran 1977, ch. 5): strata weights from the AI agreement levels."""
    n = sum(s["N"] for s in strata.values())
    est = var = 0.0
    for h, s in strata.items():
        v = values.get(h) or []
        if not v or not n:
            continue
        w, m = s["N"] / n, len(v)
        mean = sum(v) / m
        s2 = sum((x - mean) ** 2 for x in v) / (m - 1) if m > 1 else 0.25   # one value: most conservative
        pt = (sum(v) + 0.5) / (m + 1)       # floor: a stratum that is all right (or all wrong) is not certain
        s2 = max(s2, pt * (1 - pt))
        est += w * mean
        # no finite-population correction: the annotation split is itself a sample of the corpus, and the
        # question is about the corpus (a stratum checked entirely would otherwise get a zero-width interval)
        var += w * w * s2 / m
    half = A.z_value(0.95) * var ** 0.5
    return {"estimate": round(est, 4), "ci_low": round(max(0, est - half), 4), "ci_high": round(min(1, est + half), 4)}


def write_adjudication(ph, tax, ref, human, items, models):
    names = {b["id"]: b["name"] for b in T.active(tax, "behavior")}

    def show(ids):
        return "; ".join(f"{i} {names.get(i, '')}".strip() for i in (ids or []))
    df = ref.merge(items, on="item_id")
    need = []
    for r in df.itertuples():
        expert_sets = [frozenset(x) for x in human.loc[human["item_id"] == r.item_id, "behavior_ids"]]
        experts_disagree = len(set(expert_sets)) > 1
        ai_differs = set(r.reference) != set(r.majority)
        if experts_disagree or ai_differs or not r.reference:
            need.append({"row_key": r.item_id, "why": ("experts disagree; " if experts_disagree else "")
                         + ("AI majority differs from experts" if ai_differs else ""),
                         "experts": " || ".join(f"{rv}: {show(ids)}" for rv, ids in
                                                zip(human.loc[human["item_id"] == r.item_id, "reviewer"],
                                                    human.loc[human["item_id"] == r.item_id, "behavior_ids"])),
                         "ai_majority": show(r.majority),
                         **{f"ai_{m}": show(getattr(r, f"{m}_behaviors")) for m in models}})
    if not need:
        return 0
    wb = W.new_workbook("Adjudication (phase 7) — AI answers revealed", [
        "Items where the experts disagree among themselves or with the AI majority. Now that independent labels exist, "
        "the AI answers are shown. Decide the final behaviours (final_1..3) and why (decision).",
        "decision: expert_correct · ai_correct · both_partially · taxonomy_problem (explain in comment). "
        "Save as adjudication/returned/<your name>.xlsx with your name in B3."], {"generated": now_iso()})
    cols = [("row_key", "row_key", 16, False), ("why", "why", 22, False), ("experts", "experts", 40, False),
            ("ai_majority", "ai_majority", 30, False)] + [(f"ai_{m}", f"ai_{m}", 28, False) for m in models] + [
        ("final_1", "final_1", 28, True), ("final_2", "final_2", 28, True), ("final_3", "final_3", 28, True),
        ("decision", "decision", 16, True), ("comment", "comment", 30, True)]
    ch = behavior_choices(tax)
    W.add_sheet(wb, "Adjudication", need, cols, choices={"final_1": ch, "final_2": ch, "final_3": ch,
                                                         "decision": ["expert_correct", "ai_correct", "both_partially",
                                                                      "taxonomy_problem"]})
    W.save(wb, ph.path("adjudication", "adjudication.xlsx"))
    return len(need)


def import_adjudication(ph, tax) -> tuple[dict, list, list]:
    valid = set(T.behavior_ids(tax))
    out, rows_all, problems = {}, [], []
    for f in W.returned_files(ph.path("adjudication", "returned")):
        reviewer, rows, prov = W.read_sheet(f, "Adjudication")
        for r in rows:
            ids, bad = parse_ids([r.get("final_1"), r.get("final_2"), r.get("final_3")], valid)
            problems += [f"{f.name} {r.get('row_key')}: '{b}' unknown" for b in bad]
            rec = {"item_id": r.get("row_key"), "behavior_ids": ids, "decision": r.get("decision"),
                   "comment": r.get("comment"), "reviewer": reviewer, "file": f.name, "sha256": prov["sha256"],
                   "imported_at": prov["imported_at"]}
            rows_all.append(rec)
            if ids:
                out[rec["item_id"]] = rec
    return out, rows_all, problems


# ---------------------------------------------------------------------------------------------- phase
def run(ph):
    version = ph.cfg["taxonomy"].get("version", "v1")
    tax = T.load(ph.upstream(5, f"taxonomy_{version}.json"))
    ann = pd.read_parquet(ph.upstream(6, "annotations_validated.parquet"))
    models = [a["name"] for a in ph.cfg["llm"]["annotators"]]
    items = item_table(ann, models)
    write_csv(ph.path("item_agreement.csv"), items)
    metrics = ai_metrics(items, models, tax)
    write_json(ph.path("agreement.json"), metrics)
    o = metrics["overall"]
    ph.log.info(f"AI agreement on {o['items']} items: α(MASI, behaviours)={o['alpha_masi_behaviors']}, "
                f"α(primary behaviour)={o['alpha_nominal_primary_behavior']}, α(primary category)={o['alpha_nominal_primary_category']}, "
                f"AC1(primary category)={o['gwet_ac1_primary_category']}; levels {o['levels']}")
    dis = items[items["level"].isin(["partial", "none", "unknown", "insufficient"])]
    write_csv(ph.path("disagreement_report.csv"), dis)

    # expert workbook (stable selection: same seed and items -> same workbook)
    texts = load_items(ph, "annotation").set_index("item_id")[["title", "provenance", "text"]].to_dict("index")
    sel = select_expert_items(ph, items)
    write_csv(ph.path("gold_selection.csv"), sel[["item_id", "source", "level", "stratum", "stratum_N", "stratum_m", "stratum_weight"]])
    write_expert_workbook(ph, tax, sel, texts)
    ph.log.info(f"expert workbook: {len(sel)} items ({sel['stratum'].value_counts().to_dict()}), AI answers hidden "
                f"→ results/07_validation/expert/expert_workbook.xlsx")

    human, decisions, relevance, raw, problems = import_experts(ph, tax, set(sel["item_id"]))
    write_jsonl(ph.path("expert", "imported_rows.jsonl"), raw.to_dict("records") if len(raw) else [])
    adjud, adj_rows, adj_problems = import_adjudication(ph, tax)
    problems += adj_problems
    for p in problems[:20]:
        ph.warn(p)
    report = {"generated": now_iso(), "taxonomy_version": version, "ai_agreement": {k: v for k, v in metrics.items() if k != "per_behavior"}}
    ph.metric("items", int(len(items)))
    ph.metric("alpha_masi_behaviors", o["alpha_masi_behaviors"])
    ph.metric("alpha_primary_category", o["alpha_nominal_primary_category"])
    ph.metric("full_agreement_share", round(o["levels"].get("full", 0) / max(o["items"], 1), 3))
    ph.metric("expert_items", int(len(sel)))

    reviewers = sorted(human["reviewer"].unique()) if len(human) else []
    simulated = [r for r in reviewers if str(r).upper().startswith("SIMULATED")]
    if simulated:
        ph.warn(f"returned workbooks include SIMULATED reviewer(s) {simulated}: the human metrics below are a test of the "
                "import path, not a validation (delete expert/returned/SIMULATED*.xlsx before real use)")
        ph.mark_provisional("expert answers are simulated")
    if len(human) == 0:
        ph.mark_provisional("no completed expert workbook in expert/returned/ yet: AI-only agreement; human vs AI "
                            "metrics, adjudication and taxonomy v2 wait for it")
        write_csv(ph.path("human_labels.csv"), pd.DataFrame(columns=["item_id", "reviewer", "behavior_ids"]))
    else:
        write_csv(ph.path("human_labels.csv"), human.assign(behavior_ids=human["behavior_ids"].map(";".join)))
        ph.log.info(f"imported {len(human)} expert labels from {len(reviewers)} reviewer(s): {reviewers}")
        if len(reviewers) >= 2:
            piv = human.pivot_table(index="item_id", columns="reviewer", values="behavior_ids", aggfunc="first")
            units = [[frozenset(v) if isinstance(v, list) else None for v in row] for row in piv.itertuples(index=False)]
            report["expert_agreement"] = {"alpha_masi": A.krippendorff_alpha(units, A.masi),
                                          "alpha_primary": A.krippendorff_alpha([[v[0] if isinstance(v, list) else None for v in row]
                                                                                 for row in piv.itertuples(index=False)]),
                                          "reviewers": reviewers, "items_with_2plus": int((piv.notna().sum(axis=1) >= 2).sum())}
            ph.metric("expert_alpha_masi", report["expert_agreement"]["alpha_masi"])
        ref = reference_labels(human, adjud)
        write_csv(ph.path("human_reference_labels.csv"), ref.assign(reference=ref["reference"].map(";".join)))
        hv = human_vs_ai(ref, items, sel, models, tax)
        report["human_vs_ai"] = hv
        write_json(ph.path("human_vs_ai.json"), hv)
        for m in models + ["ai_majority"]:
            if m in hv:
                ph.metric(f"{m}_f1_micro", hv[m]["micro"]["f1"])
        n_adj = write_adjudication(ph, tax, ref, human, items, models)
        ph.metric("adjudication_items", n_adj)
        if adj_rows:
            write_csv(ph.path("adjudication_decisions.csv"), pd.DataFrame(adj_rows).assign(
                behavior_ids=lambda d: d["behavior_ids"].map(";".join)))
        elif n_adj:
            ph.mark_provisional(f"{n_adj} items wait for adjudication (adjudication/adjudication.xlsx)")
        if relevance:
            rel = pd.DataFrame(relevance)
            agg = rel.groupby("behavior_id").agg(relevance_pharo=("relevance_pharo", "mean"),
                                                 relevance_benchmark=("relevance_benchmark", "mean"),
                                                 reviewers=("reviewer", "nunique")).reset_index()
            write_csv(ph.path("behavior_relevance.csv"), agg)
        if decisions:
            nxt = f"v{int(re.sub(r'[^0-9]', '', version) or 1) + 1}"
            v2, probs = T.apply_decisions(tax, decisions, nxt, source="07_validation/expert/returned")
            for p in probs:
                ph.warn(f"taxonomy decision not applied: {p}")
            v2["status"] = "corrected by experts"
            T.save(v2, ph.path(f"taxonomy_{nxt}.json"))
            write_csv(ph.path("taxonomy_decisions.csv"), pd.DataFrame(decisions))
            ph.metric("taxonomy_next_version", nxt)
            ph.log.info(f"taxonomy {nxt}: {len(v2['changes'])} changes from the experts")
    write_json(ph.path("validation_report.json"), report)
    write_text(ph.path("validation_report.md"), md_report(report, problems, models))
    figures(ph, items, metrics, report, models, tax)


def md_report(rep: dict, problems: list, models: list) -> str:
    o = rep["ai_agreement"]["overall"]
    lines = ["# Agreement and expert validation", "", f"Taxonomy {rep['taxonomy_version']} · {rep['generated']}", "",
             "## AI–AI agreement (three models, standardised IDs)", "",
             "| | items | α MASI behaviours | α primary behaviour | α primary category | AC1 primary category |",
             "|---|---|---|---|---|---|"]
    for k in ("overall", "literature", "pharo"):
        if k in rep["ai_agreement"]:
            b = rep["ai_agreement"][k]
            lines.append(f"| {k} | {b['items']} | {b['alpha_masi_behaviors']} | {b['alpha_nominal_primary_behavior']} | "
                         f"{b['alpha_nominal_primary_category']} | {b['gwet_ac1_primary_category']} |")
    lines += ["", f"Agreement levels: {o['levels']}", "",
              "α ≥ 0.8 reliable, ≥ 0.667 tentative (Krippendorff 2004). Model agreement is NOT correctness: see below.", ""]
    if rep.get("expert_agreement"):
        e = rep["expert_agreement"]
        lines += ["## Experts among themselves (human baseline)", "", f"α MASI {e['alpha_masi']}, α primary {e['alpha_primary']} "
                  f"({e['items_with_2plus']} items with ≥ 2 experts; reviewers {', '.join(e['reviewers'])})", ""]
    hv = rep.get("human_vs_ai") or {}
    if hv:
        lines += ["## Human reference vs AI", "", f"{hv['items']} items with a reference label.", "",
                  "| annotator | micro F1 | macro F1 | exact match | primary behaviour ∈ reference | primary category ∈ reference "
                  "| stratified category accuracy (95% CI) |", "|---|---|---|---|---|---|---|"]
        for m in models + ["ai_majority"]:
            if m in hv:
                x = hv[m]
                s = x["primary_category_accuracy_stratified"]
                lines.append(f"| {m} | {x['micro']['f1']} | {x['macro']['f1']} | {x['exact_match']} | "
                             f"{x['primary_behavior_in_reference']} | {x['primary_category_in_reference']} | "
                             f"{s['estimate']} ({s['ci_low']}–{s['ci_high']}) |")
        lines += ["", "Primary behaviour correct by AI agreement level (ai_majority): "
                  + str(hv.get("ai_majority", {}).get("primary_behavior_correct_by_ai_agreement_level")), ""]
    else:
        lines += ["## Human reference vs AI", "", "No completed expert workbook yet (expert/returned/).", ""]
    if problems:
        lines += ["## Import problems", ""] + [f"- {p}" for p in problems] + [""]
    return "\n".join(lines) + "\n"


def figures(ph, items, metrics, report, models, tax):
    lv = items.groupby(["source", "level"]).size().unstack(fill_value=0).reindex(columns=LEVELS, fill_value=0)
    lv = lv.loc[:, lv.sum() > 0]
    F.stacked_barh(list(lv.index), {c: lv[c].tolist() for c in lv.columns}, "AI agreement level per item (3 models)",
                   "items", ph.figure("agreement_levels.png"),
                   colors={"full": F.STATUS["good"], "partial": F.STATUS["warning"], "none": F.STATUS["critical"],
                           "insufficient": F.STATUS["neutral"], "unknown": "#6250d6", "missing": "#cfcdc7"})
    keys = [("alpha_masi_behaviors", "α MASI, behaviour sets"), ("alpha_nominal_primary_behavior", "α, primary behaviour"),
            ("alpha_masi_categories", "α MASI, category sets"), ("alpha_nominal_primary_category", "α, primary category"),
            ("gwet_ac1_primary_category", "AC1, primary category")]
    srcs = [s for s in ("overall", "literature", "pharo") if s in metrics]
    mat = [[metrics[s][k] if metrics[s][k] is not None else np.nan for s in srcs] for k, _ in keys]
    F.heatmap(mat, [lab for _, lab in keys], srcs, "Agreement between the three AI annotators", ph.figure("alpha_summary.png"),
              fmt="{:.2f}", vmin=0, vmax=1, cbar_label="coefficient",
              note="≥ 0.80 reliable · ≥ 0.667 tentative (Krippendorff); agreement between models is not correctness")
    pw = metrics["overall"]["pairwise_cohen_primary_behavior"]
    mat = np.eye(len(models))
    for k, v in pw.items():
        a, b = k.split("~")
        mat[models.index(a), models.index(b)] = mat[models.index(b), models.index(a)] = v if v is not None else np.nan
    F.heatmap(mat, models, models, "Pairwise Cohen's κ, primary behaviour", ph.figure("pairwise_kappa.png"), fmt="{:.2f}",
              vmin=0, vmax=1)
    per = metrics.get("per_behavior") or {}
    if per:
        names = {b["id"]: b["name"] for b in T.active(tax, "behavior")}
        s = sorted(per.items(), key=lambda kv: (kv[1]["alpha"] if kv[1]["alpha"] is not None else -1))
        vals = [v["alpha"] if v["alpha"] is not None else 0 for _, v in s]
        F.barh([f"{k} {names.get(k, '')} (n={v['items_with_any_model']})" for k, v in s][::-1], vals[::-1],
               "Agreement per behaviour (present / absent, α)", "Krippendorff's α", ph.figure("per_behavior_alpha.png"),
               fmt="{:.2f}", colors=[F.STATUS["good"] if v >= 0.8 else F.STATUS["warning"] if v >= 0.667 else F.STATUS["critical"]
                                     for v in vals[::-1]], xlim=(min(0, min(vals)) - 0.05, 1.1))
    hv = report.get("human_vs_ai") or {}
    if hv:
        names_ = [m for m in models + ["ai_majority"] if m in hv]
        mets = [("micro F1", lambda x: x["micro"]["f1"]), ("exact match", lambda x: x["exact_match"]),
                ("primary category ∈ ref.", lambda x: x["primary_category_in_reference"])]
        fig, ax = F.plt.subplots(figsize=(8.5, 3.8))
        w = 0.8 / len(mets)
        x = np.arange(len(names_))
        for i, (lab, fn) in enumerate(mets):
            vals = [fn(hv[n]) or 0 for n in names_]
            ax.bar(x + (i - 1) * w, vals, width=w - 0.02, color=F.color(i), label=lab)
            for xi, v in zip(x, vals):
                ax.text(xi + (i - 1) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5, color=F.INK_2)
        ax.set_xticks(x, names_)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"AI annotators against the human reference ({hv['items']} items)")
        ax.legend(ncol=3, loc="upper left")
        sim = any(str(r).upper().startswith("SIMULATED") for r in (report.get("expert_agreement") or {}).get("reviewers", []))
        F.save(fig, ph.figure("human_vs_ai.png"), note="SIMULATED expert: import-path test only" if sim else None)
        bl = hv.get("ai_majority", {}).get("primary_behavior_correct_by_ai_agreement_level") or {}
        if bl:
            order = [l for l in LEVELS if l in bl]
            F.barh(order, [bl[l] * 100 for l in order], "Is the AI majority right when the models agree?",
                   "% items whose AI-majority primary behaviour is in the human reference", ph.figure("accuracy_by_agreement.png"),
                   fmt="{:.0f}%", xlim=(0, 110))
