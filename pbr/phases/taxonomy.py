"""Phase 6 — The two final taxonomies: one for the Pharo methods, one for the existing benchmark tasks.

For each source separately (``pharo``, ``literature``), the taxonomies the three annotators built on their own in
phase 5 become one shared taxonomy, written to ``06_taxonomy/<source>/``:

1. **Flatten** the three candidate taxonomies of phase 5 into concepts (model, level, name, definition,
   rules, example items, parent).
2. **Candidate groups** per level, on a COMBINED similarity: 0.6 × cosine of the text embeddings ("name:
   definition", ``llm.embedder``) + 0.4 × the overlap (Jaccard) of the items the concepts classify, on the items both
   annotators saw (``taxonomy.similarity_weights``); a category or subcategory holds every item under it. The
   annotators classified the same items: the same items are evidence of the same meaning, different items of a
   different one. Average-linkage clustering at ``taxonomy.candidate_threshold``; groups larger than
   ``max_group_size`` re-split. Embeddings alone rate "iterate" and "filter" closer than two wordings of "filter".
3. **Judged merges**: the editor model reads each candidate group (members with their number of items, largest
   first) and partitions it into concepts whose members mean the same thing, writing one name, definition, inclusion and exclusion rules per concept
   (a single proposal keeps its model's text). Groups the editor split are listed for review
   (``merge_candidates.csv``, sheet "Merge candidates"): matching names alone are never enough (§5).
4. **Hierarchy, weighted by items**: a behaviour goes under the subcategory where most of its members' items were
   placed; a subcategory under the category where most of its items were (so the result is a tree); concept
   centres are weighted by items too. Each concept records the
   models that proposed it.
5. **Stable IDs**: ``results/taxonomy_registry.json`` remembers which members got which ID, so a rerun
   (or a new version) gives the same concept the same ID.
6. **Review**: ``review/taxonomy_review.xlsx`` (Taxonomy Review sheet: accept / rename / redefine / merge
   / split / reject / add / move; Merge candidates sheet: near-misses to check). Completed files in
   ``review/returned/`` are applied to the draft to produce taxonomy_v1. Without them, v1 is the
   unreviewed draft and the phase is *provisional* (``taxonomy.require_review: true`` stops instead).

Outputs per source (06_taxonomy/<source>/): taxonomy_v0_draft.json, taxonomy_v1.json, taxonomy_aliases.csv,
merge_candidates.csv, annotation_guidelines_v1.md, review/taxonomy_review.xlsx (completed files: review/returned/),
figures: taxonomy_v1.png (●●○ = annotators that proposed a behaviour; bar and number = items classified in each
category, subcategory and behaviour), model_overlap.png, support_per_behavior.png.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from .. import figures as F
from .. import taxonomy as T
from .. import workbook as W
from ..common import PhaseWaiting, now_iso, read_json, results_dir, write_csv, write_json, write_text
from ..llm import Client, annotators, ledger_summary

PROMPT_VERSION = "taxonomy-editor-v3"   # judged merges; harmonisation prompts share it

_CONCEPT = {"type": "object", "properties": {
    "members": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
    "name": {"type": "string", "minLength": 2}, "definition": {"type": "string", "minLength": 10},
    "include": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    "exclude": {"type": "array", "items": {"type": "string"}, "maxItems": 4}},
    "required": ["members", "name", "definition", "include", "exclude"]}
EDIT_SCHEMA = {"type": "object", "properties": {"concepts": {"type": "array", "items": _CONCEPT, "minItems": 1}},
               "required": ["concepts"]}

EDIT_SYSTEM = """You edit a taxonomy of programming behaviours built by three independent annotators. You receive numbered concepts that an embedding model found SIMILAR; similar is not the same.
Partition them: put members in the same concept ONLY if they mean the same thing (synonyms, or one is a rewording of the other). Related but different behaviours (e.g. reading vs updating an element, iterating vs filtering) stay in separate concepts. Every member number appears in exactly one concept.
Each member shows how many items the annotator classified under it: when you write a merged concept, follow the
members that hold the most items (they are the best-supported meaning).
For each concept write:
- name: short and language-neutral (category or subcategory: a noun phrase; behaviour: an imperative phrase of 3 to 8 words);
- definition: one precise sentence;
- include: when to use it; exclude: when not to (name the neighbouring concept to use instead).
Answer with JSON only."""


def partition_validator(n: int):
    def check(data):
        seen = [m for c in data.get("concepts", []) for m in c.get("members", [])]
        errs = []
        if sorted(seen) != list(range(1, n + 1)):
            missing = sorted(set(range(1, n + 1)) - set(seen))
            dup = sorted({m for m in seen if seen.count(m) > 1})
            bad = sorted({m for m in seen if not 1 <= m <= n})
            errs.append(f"every member 1..{n} must appear exactly once (missing {missing}, repeated {dup}, unknown {bad})")
        return errs
    return check


# ---------------------------------------------------------------------------------------------- flatten
def flatten(cands: dict) -> pd.DataFrame:
    rows = []
    for (model, source), cand in cands.items():
        tag = f"{model}/{source}"
        for c in cand.get("categories", []):
            ck = f"{tag}|C|{c['name']}"
            rows.append({"key": ck, "model": model, "source": source, "level": "category", "name": c["name"], "definition": c["definition"],
                         "parent": None, "include": [], "exclude": [], "examples": [], "evidence": [], "n_items": 0})
            for s in c.get("subcategories", []):
                sk = f"{tag}|S|{c['name']}|{s['name']}"
                rows.append({"key": sk, "model": model, "source": source, "level": "subcategory", "name": s["name"],
                             "definition": s["definition"], "parent": ck, "include": [], "exclude": [], "examples": [],
                             "evidence": [], "n_items": 0})
                for b in s.get("behaviors", []):
                    rows.append({"key": f"{tag}|B|{c['name']}|{s['name']}|{b['name']}", "model": model, "source": source,
                                 "level": "behavior", "name": b["name"], "definition": b["definition"], "parent": sk,
                                 "include": b.get("include", []), "exclude": b.get("exclude", []),
                                 "examples": b.get("items") or b.get("examples", []), "evidence": b.get("evidence", []),
                                 "n_items": b.get("n_items", 0)})
        # proposals the model's own consolidation left out still count: under the category / subcategory the
        # model proposed for them (a weak consolidation must not erase a model's evidence)
        known = {r["key"] for r in rows}
        for p in cand.get("unassigned_proposals", []):
            c, s, b = (str(x).strip() for x in p["label"])
            ck, sk = f"{tag}|C|{c}", f"{tag}|S|{c}|{s}"
            if ck not in known:
                rows.append({"key": ck, "model": model, "source": source, "level": "category", "name": c, "definition": c, "parent": None,
                             "include": [], "exclude": [], "examples": [], "evidence": [], "n_items": 0})
                known.add(ck)
            if sk not in known:
                rows.append({"key": sk, "model": model, "source": source, "level": "subcategory", "name": s, "definition": f"{s} ({c})",
                             "parent": ck, "include": [], "exclude": [], "examples": [], "evidence": [], "n_items": 0})
                known.add(sk)
            rows.append({"key": f"{tag}|B|{c}|{s}|{b}", "model": model, "source": source, "level": "behavior", "name": b,
                         "definition": f"{b} ({c} › {s}); proposal not organised by the model's consolidation",
                         "parent": sk, "include": [], "exclude": [], "examples": list(p.get("items", [])),
                         "evidence": [], "n_items": len(p.get("items", []))})
    df = pd.DataFrame(rows)
    return df.drop_duplicates("key")


def cluster(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Agglomerative clustering (average linkage) on the distance 1 - similarity; cut at 1 - threshold."""
    if len(sim) == 1:
        return np.array([0])
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    d = np.clip(1 - sim, 0, 2)
    np.fill_diagonal(d, 0)
    z = linkage(squareform((d + d.T) / 2, checks=False), method="average")
    return fcluster(z, t=1 - threshold, criterion="distance") - 1


def candidate_groups(sim: np.ndarray, threshold: float, max_size: int) -> np.ndarray:
    """Loose clusters; a cluster larger than max_size is clustered again with a stricter threshold."""
    lab = cluster(sim, threshold)
    out = np.empty(len(sim), dtype=object)
    for l in np.unique(lab):
        idx = np.where(lab == l)[0]
        if len(idx) <= max_size or threshold >= 0.97:
            for i in idx:
                out[i] = f"{l}"
            continue
        sub = candidate_groups(sim[np.ix_(idx, idx)], threshold + 0.03, max_size)
        for i, s_ in zip(idx, sub):
            out[i] = f"{l}.{s_}"
    return out


def item_sets(df: pd.DataFrame) -> list[set]:
    """The items each concept classifies: a behaviour its own items; a subcategory or a category every item under
    it (the same numbers the taxonomy figures show)."""
    items = {r.key: set(r.examples or []) for r in df.itertuples() if r.level == "behavior"}
    for lv in ("subcategory", "category"):
        for r in df[df["level"] == lv].itertuples():
            items[r.key] = set().union(*[items.get(c.key, set()) for c in df[df["parent"] == r.key].itertuples()])
    return [items.get(k, set()) for k in df["key"]]


def similarity(emb: np.ndarray, items: list[set], w_emb: float, w_items: float, seen: list[set] | None = None) -> np.ndarray:
    """Combined similarity of concepts: w_emb × cosine of their text embeddings + w_items × Jaccard of the items
    they classify. The annotators classified the same items, so two concepts holding the same items are evidence of
    the same meaning even when they are worded differently; two concepts with similar words but disjoint items are
    evidence of different meanings."""
    cos = emb @ emb.T
    n = len(items)
    jac = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            a, b = items[i], items[j]
            if seen is not None:          # compare on the items both annotators saw (they may stop at different rounds)
                common = seen[i] & seen[j]
                a, b = a & common, b & common
            u = len(a | b)
            jac[i, j] = jac[j, i] = (len(a & b) / u) if u else 0.0
    tot = (w_emb + w_items) or 1
    return (w_emb * cos + w_items * jac) / tot


def majority(xs):
    xs = [x for x in xs if x is not None]
    return Counter(xs).most_common(1)[0][0] if xs else None


# ---------------------------------------------------------------------------------------------- registry
def assign_ids(clusters: dict, level: str, registry: dict) -> dict:
    """cluster label -> stable ID: reuse the ID whose recorded members overlap most (Jaccard >= 0.5)."""
    prefix = T.LEVELS[level][1]
    known = registry.setdefault(level, {})        # id -> member keys
    out, taken = {}, set()
    for lab, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        ms = set(members)
        best, best_j = None, 0.0
        for cid, prev in known.items():
            if cid in taken:
                continue
            j = len(ms & set(prev)) / len(ms | set(prev))
            if j > best_j:
                best, best_j = cid, j
        if best and best_j >= 0.5:
            cid = best
        else:
            n = max([int(i[1:]) for i in known] + [0]) + 1
            cid = f"{prefix}{n:03d}"
        known[cid] = sorted(ms)
        taken.add(cid)
        out[lab] = cid
    return out


# ---------------------------------------------------------------------------------------------- phase
SOURCES = {"pharo": "Pharo methods", "literature": "existing benchmark tasks"}


class SourcePhase:
    """The phase seen from one source: files in <phase>/<source>/, metrics prefixed by the source."""

    def __init__(self, ph, src: str):
        self._ph, self.src = ph, src

    def __getattr__(self, name):
        return getattr(self._ph, name)

    def path(self, *parts):
        return self._ph.path(self.src, *parts)

    def figure(self, name: str):
        return self._ph.path(self.src, "figures", name)

    def metric(self, key: str, value):
        return self._ph.metric(f"{self.src}_{key}", value)

    def mark_provisional(self, reason: str):
        return self._ph.mark_provisional(f"{self.src}: {reason}")

    def warn(self, msg: str):
        return self._ph.warn(f"{self.src}: {msg}")


def run(ph):
    """Two final taxonomies: one from the three annotators' Pharo taxonomies, one from their benchmark taxonomies."""
    for src, label in SOURCES.items():
        ph.log.info(f"— shared taxonomy of the {label} (from the three annotators' {label} taxonomies) —")
        build(SourcePhase(ph, src), src)


def build(ph, src: str):
    names = [a["name"] for a in ph.cfg["llm"]["annotators"]]
    cands = {}
    for n in names:                       # the candidate taxonomy of each model for this source (phase 5)
        c = read_json(ph.upstream("discovery", n, src, "taxonomy.json"))
        if not c or not c.get("categories"):
            ph.warn(f"{n}/{src}: no usable candidate taxonomy (phase 5) — left out of the merge")
            continue
        cands[(n, src)] = c
    if len(cands) < 2:
        raise RuntimeError("fewer than two candidate taxonomies: cannot build a shared one")
    df = flatten(cands)
    ph.log.info("concepts: " + ", ".join(f"{lv}={int((df['level'] == lv).sum())}" for lv in T.LEVELS))
    embedder = Client({"name": "embedder", **ph.cfg["llm"]["embedder"]}, ph.cfg)
    texts = [f"clustering: {r.name}. {r.definition}" for r in df.itertuples()]
    emb = np.array(embedder.embed(texts))
    df = df.reset_index(drop=True)
    df["emb_idx"] = range(len(df))
    df["item_set"] = item_sets(df)
    df["size"] = df["item_set"].map(len)           # items a concept classifies: its weight in the merge
    seen_by = {m: set().union(*g.loc[g["level"] == "behavior", "item_set"]) for m, g in df.groupby("model")}
    thr = ph.cfg["taxonomy"]["candidate_threshold"]
    wts = ph.cfg["taxonomy"].get("similarity_weights", {"embedding": 0.6, "items": 0.4})
    max_size = int(ph.cfg["taxonomy"].get("max_group_size", 9))
    df["group"] = None
    sims = {}
    for lv in T.LEVELS:
        sub = df[df["level"] == lv]
        sm = similarity(emb[sub["emb_idx"].to_numpy()], list(sub["item_set"]), float(wts["embedding"]), float(wts["items"]),
                        seen=[seen_by[m] for m in sub["model"]])
        sims[lv] = (list(sub.index), sm)
        lab = candidate_groups(sm, float(thr[lv]), max_size)
        df.loc[sub.index, "group"] = [f"{lv}:{l}" for l in lab]
    pos = {lv: {ix: k for k, ix in enumerate(ixs)} for lv, (ixs, _) in sims.items()}

    def pair_sim(a, b) -> float:            # combined similarity of two concepts of one level
        lv = df.at[a, "level"]
        return float(sims[lv][1][pos[lv][a], pos[lv][b]])

    # judged merges: the editor partitions every candidate group into concepts of the same meaning
    editor = Client(ph.cfg["llm"]["editor"], ph.cfg)
    labels, texts_by_cluster, split_rows = {}, {}, []
    groups = list(df.groupby("group", sort=True))
    n_multi = sum(len(g) > 1 for _, g in groups)
    ph.log.info(f"{len(groups)} candidate groups ({n_multi} with several members) → the editor judges each one")

    def judge(group):
        gname, g = group
        rows_ = sorted(g.itertuples(), key=lambda r: -r.size)     # members that classify the most items first
        if len(rows_) < 2:
            return None
        members = "\n".join(f"{i}. [{r.model}, {r.size} item{'s' if r.size != 1 else ''}] {r.name}: {r.definition}"
                             + (f" | use when: {'; '.join(r.include)}" if len(r.include) else "")
                             for i, r in enumerate(rows_, 1))
        return editor.ask(system=EDIT_SYSTEM, user=f"Level: {g['level'].iloc[0]}\nMembers:\n{members}", schema=EDIT_SCHEMA,
                          max_tokens=400 + 260 * len(rows_), prompt_version=PROMPT_VERSION, phase=ph.key,
                          item_id=gname, validator=partition_validator(len(rows_)))
    # the groups are independent: judged in parallel (compute.llm_parallel), then applied in order
    for (gname, g), ans in zip(groups, ph.pmap(judge, groups, "candidate groups judged")):
        lv = g["level"].iloc[0]
        rows_ = sorted(g.itertuples(), key=lambda r: -r.size)
        parts = None
        if ans is not None:
            if ans.valid:
                parts = [([rows_[m - 1] for m in c["members"]], c, True) for c in ans.data["concepts"]]
            else:
                ph.warn(f"{gname}: editor answer invalid after retries — members kept as separate concepts")
        if parts is None:
            parts = [([r], {"name": r.name, "definition": r.definition, "include": list(r.include),
                            "exclude": list(r.exclude)}, False) for r in rows_]
        for k, (mem, text, edited) in enumerate(parts):
            cl = f"{gname}#{k}"
            for r in mem:
                labels[r.key] = cl
            texts_by_cluster[cl] = {**text, "edited_by_model": edited and len(mem) > 1, "candidate_group": gname,
                                    "group_split": len(parts) > 1}
        if len(parts) > 1 and len(rows_) > 1:     # similar but judged different: the expert should see them
            for (ma, ta, _), (mb, tb, _) in combinations(parts, 2):
                for ra in ma:
                    for rb in mb:
                        if ra.model != rb.model:
                            split_rows.append({"level": lv, "candidate_group": gname, "cluster_a": labels[ra.key],
                                               "name_a": ra.name, "model_a": ra.model, "cluster_b": labels[rb.key],
                                               "name_b": rb.name, "model_b": rb.model,
                                               "similarity": round(pair_sim(ra.Index, rb.Index), 3)})
    df["cluster"] = df["key"].map(labels)

    # hierarchy by majority of members' parents
    parent_cluster = {}
    for lv in ("subcategory", "behavior"):
        for cl, g in df[df["level"] == lv].groupby("cluster"):
            votes = Counter()                    # the parent where most of the members' items were placed
            for pkey, size in zip(g["parent"], g["size"]):
                votes[labels.get(pkey)] += max(int(size), 1)
            parent_cluster[cl] = votes.most_common(1)[0][0] if votes else None
    behaviours = df[df["level"] == "behavior"]
    used_sub = set(parent_cluster[c] for c in behaviours["cluster"].unique())
    used_cat = set(parent_cluster[c] for c in used_sub)

    registry_path = results_dir(ph.cfg) / f"taxonomy_registry_{src}.json"     # stable IDs across reruns
    registry = read_json(registry_path, {})
    ids = {}
    for lv in T.LEVELS:
        cl_members = {cl: list(g["key"]) for cl, g in df[df["level"] == lv].groupby("cluster")
                      if (lv != "subcategory" or cl in used_sub) and (lv != "category" or cl in used_cat)}
        ids.update(assign_ids(cl_members, lv, registry))

    sim_rows, alias_rows, concepts = [], [], {}
    for cl, g in df.groupby("cluster"):
        if cl not in ids:
            continue
        lv = g["level"].iloc[0]
        cid = ids[cl]
        text = texts_by_cluster[cl]
        idx = g["emb_idx"].to_numpy()
        wts_ = np.maximum(g["size"].to_numpy(dtype=float), 1.0)       # members weighted by the items they classify
        centroid = (emb[idx] * wts_[:, None]).sum(axis=0)
        centroid /= np.linalg.norm(centroid) or 1
        to_centroid = emb[idx] @ centroid
        pair = [pair_sim(a, b) for a, b in combinations(list(g.index), 2)]
        jac = [len(df.at[a, "item_set"] & df.at[b, "item_set"]) / max(len(df.at[a, "item_set"] | df.at[b, "item_set"]), 1)
               for a, b in combinations(list(g.index), 2)]
        models = sorted(set(g["model"]))
        sources = sorted(set(g["source"]))
        # every item the merged concept classifies (for a category or subcategory: every item under it)
        examples = list(dict.fromkeys(i for st in g["item_set"] for i in sorted(st)))
        evidence = [e for ev in g["evidence"] for e in ev][:4]
        concepts[cid] = {"id": cid, "level": lv, "cluster": cl, "name": str(text["name"]).strip(),
                         "definition": str(text["definition"]).strip(), "include": text.get("include", []),
                         "exclude": text.get("exclude", []), "proposed_by": models, "n_models": len(models), "sources": sources,
                         "aliases": sorted(set(g["name"])), "examples": [{"item_id": e} for e in examples[:10]],
                         "items": examples, "evidence": evidence, "n_items": len(examples),
                         "edited_by_model": text["edited_by_model"], "same_meaning": True,
                         "candidate_group": text["candidate_group"], "group_split": text["group_split"],
                         "min_pair_similarity": round(min(pair), 3) if pair else None}
        for r, s_ in zip(g.itertuples(), to_centroid):
            alias_rows.append({"alias": r.name, "model": r.model, "level": lv, "id": cid, "items": int(r.size),
                               "canonical_name": concepts[cid]["name"], "similarity_to_centroid": round(float(s_), 3)})
        if pair:
            sim_rows.append({"id": cid, "level": lv, "members": len(g), "models": ", ".join(models),
                             "min_similarity": round(min(pair), 3), "mean_similarity": round(float(np.mean(pair)), 3),
                             "mean_item_overlap": round(float(np.mean(jac)), 3), "items": len(examples),
                             "canonical_name": concepts[cid]["name"],
                             "member_names": " | ".join(f"{n} ({k})" for n, k in zip(g["name"], g["size"]))})

    near = [{"level": r["level"], "id_a": ids.get(r["cluster_a"]), "name_a": r["name_a"], "model_a": r["model_a"],
             "id_b": ids.get(r["cluster_b"]), "name_b": r["name_b"], "model_b": r["model_b"], "similarity": r["similarity"]}
            for r in split_rows if ids.get(r["cluster_a"]) and ids.get(r["cluster_b"])]
    near_df = pd.DataFrame(near).sort_values("similarity", ascending=False) if near else pd.DataFrame(
        columns=["level", "id_a", "name_a", "model_a", "id_b", "name_b", "model_b", "similarity"])
    near_df = near_df.drop_duplicates(["id_a", "id_b"]).head(300)

    # ---------------------------------------------------------------- draft taxonomy (v0)
    draft = T.empty("v0-draft")
    draft["status"] = "draft (automatic merge, not reviewed)"
    draft["describes"] = src                       # pharo | literature: one final taxonomy per source
    draft["source"] = {"phase": 4, "models": list(cands), "embedder": ph.cfg["llm"]["embedder"],
                       "thresholds": thr, "editor": ph.cfg["llm"]["editor"], "prompt_version": PROMPT_VERSION}
    cat_of_sub = {}
    for cl, cid in ids.items():
        c = concepts[cid]
        base = {k: c[k] for k in ("id", "name", "definition", "include", "exclude", "proposed_by", "sources", "aliases",
                                  "examples", "items", "evidence", "n_items", "edited_by_model", "same_meaning")}
        base["status"] = "active"
        if c["level"] == "category":
            draft["categories"].append(base)
        elif c["level"] == "subcategory":
            base["category_id"] = ids[parent_cluster[cl]]
            cat_of_sub[cid] = base["category_id"]
            draft["subcategories"].append(base)
    for cl, cid in ids.items():
        c = concepts[cid]
        if c["level"] == "behavior":
            sid = ids[parent_cluster[cl]]
            draft["behaviors"].append({**{k: c[k] for k in ("id", "name", "definition", "include", "exclude",
                                                            "proposed_by", "sources", "aliases", "examples", "items", "evidence",
                                                            "n_items", "edited_by_model", "same_meaning")},
                                       "subcategory_id": sid, "category_id": cat_of_sub[sid], "status": "active",
                                       "min_models_met": c["n_models"] >= int(ph.cfg["taxonomy"]["min_models_for_behavior"])})
    for lv in T.LEVELS:      # stable, readable order
        draft[T.LEVELS[lv][0]].sort(key=lambda x: x["id"])
    _dedupe_names(draft)
    n_before = len(T.active(draft, "behavior"))
    draft = harmonize(ph, editor, draft)
    ph.log.info(f"harmonised draft: {len(T.active(draft, 'category'))} categories, {len(T.active(draft, 'subcategory'))} "
                f"subcategories, {len(T.active(draft, 'behavior'))} behaviours (from {n_before}); {len(draft['changes'])} recorded changes")
    errs = T.validate(draft)
    if errs:
        raise RuntimeError("draft taxonomy invalid: " + "; ".join(errs[:5]))
    write_json(ph.path("taxonomy_v0_draft.json"), draft)
    write_json(registry_path, registry)

    # ---------------------------------------------------------------- review workbook + decisions
    write_review_workbook(ph, draft, near_df, pd.DataFrame(sim_rows))
    returned = W.returned_files(ph.path("review", "returned"))
    decisions, import_notes = [], []
    for f in returned:
        reviewer, rows, prov = W.read_sheet(f, "Taxonomy Review")
        import_notes.append(f"- {f.name}: reviewer {reviewer} ({prov['reviewer_from']}), {len(rows)} rows, sha256 {prov['sha256'][:12]}")
        for r in rows:
            if r.get("action"):
                decisions.append({**{k: r.get(k) for k in ("id", "action", "new_name", "new_definition", "merge_into",
                                                            "split_into", "parent_id", "include", "exclude", "comment")},
                                  "reviewer": reviewer, "file": f.name})
    version = ph.cfg["taxonomy"].get("version", "v1")
    if decisions:
        v1, problems = T.apply_decisions(draft, decisions, version, source="review/returned")
        for p in problems:
            ph.warn(f"review decision not applied: {p}")
        v1["status"] = "reviewed"
        v1["reviewers"] = sorted({d["reviewer"] for d in decisions})
        write_csv(ph.path("review", "decisions_imported.csv"), pd.DataFrame(decisions))
    else:
        if ph.cfg["taxonomy"].get("require_review"):
            raise PhaseWaiting("fill results/06_taxonomy/review/taxonomy_review.xlsx (sheet 'Taxonomy Review', "
                               "column 'action'; your name in README!B3) and save it in results/06_taxonomy/review/returned/")
        v1 = T.clone(draft)
        v1.update(version=version, parent_version="v0-draft", status="draft-unreviewed", created=now_iso())
        ph.mark_provisional("taxonomy_v1 is the automatic merge: no expert decision in review/returned/ yet "
                            "(the file to fill is review/taxonomy_review.xlsx)")
    T.save(v1, ph.path(f"taxonomy_{version}.json"))
    write_text(ph.path("review", "import_report.md"), "# Taxonomy review import\n\n"
               + ("\n".join(import_notes) if import_notes else "No returned workbook yet.") + "\n")
    al = pd.DataFrame(alias_rows)
    al["final_id"] = al["id"].map(lambda i: ";".join(T.translate([i], draft, draft)) if i.startswith("B") else
                                  (T.index(draft)[i].get("replaced_by") or [i])[0] if i in T.index(draft) else i)
    write_csv(ph.path("taxonomy_aliases.csv"), al.sort_values(["level", "id", "model"]))
    write_csv(ph.path("merge_candidates.csv"), pd.concat([pd.DataFrame(sim_rows).assign(kind="merged by the editor"),
                                                         near_df.assign(kind="similar, judged different by the editor")],
                                                        ignore_index=True))
    write_text(ph.path(f"annotation_guidelines_{version}.md"), guidelines(v1))

    nb = len(T.active(v1, "behavior"))
    ph.metric("categories", len(T.active(v1, "category")))
    ph.metric("subcategories", len(T.active(v1, "subcategory")))
    ph.metric("behaviors", nb)
    ph.metric("behaviors_from_3_models", sum(len(b.get("proposed_by", [])) >= 3 for b in T.active(v1, "behavior")))
    ph.metric("behaviors_single_model", sum(len(b.get("proposed_by", [])) == 1 for b in T.active(v1, "behavior")))
    ph.metric("candidate_groups_split_by_editor", int(sum(c["group_split"] for c in concepts.values())))
    ph.metric("near_miss_pairs", int(len(near_df)))
    ph.metric("review_decisions", len(decisions))
    ph.metric("llm", ledger_summary(ph.cfg, ph.key))

    figures(ph, v1, df, ids, names)


HARM_SYSTEM = """You restructure a draft taxonomy of programming behaviours merged from three annotators. Keep it faithful: never invent concepts, only merge existing ones (by their IDs) and rename survivors. Language-neutral names. Answer with JSON only."""

CAT_SCHEMA = {"type": "object", "properties": {
    "merges": {"type": "array", "items": {"type": "object", "properties": {"from": {"type": "string"}, "into": {"type": "string"}},
                                          "required": ["from", "into"]}},
    "renames": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "name": {"type": "string"}, "definition": {"type": "string"}}, "required": ["id", "name", "definition"]}}},
    "required": ["merges", "renames"]}
WITHIN_SCHEMA = {"type": "object", "properties": {
    "subcategory_merges": {"type": "array", "items": {"type": "object", "properties": {"from": {"type": "string"}, "into": {"type": "string"}},
                                                      "required": ["from", "into"]}},
    "behavior_merges": {"type": "array", "items": {"type": "object", "properties": {"from": {"type": "string"}, "into": {"type": "string"}},
                                                   "required": ["from", "into"]}}},
    "required": ["subcategory_merges", "behavior_merges"]}


def _merge_validator(allowed: dict):
    """allowed: level prefix -> set of ids; a merge must join two different existing ids of one level."""
    def check(data):
        errs = []
        for key, val in data.items():
            if not isinstance(val, list) or key == "renames":
                continue
            for m in val:
                a, b = m.get("from"), m.get("into")
                if not a or not b or a == b or a[0] != b[0] or a not in allowed.get(a[0], ()) or b not in allowed.get(b[0], ()):
                    errs.append(f"{key}: {a} -> {b} is not a merge of two existing ids of the same level")
        return errs
    return check


def harmonize(ph, editor, draft: dict) -> dict:
    """Global pass: (A) merge categories down to taxonomy.max_categories, (B) per category merge duplicate
    subcategories and behaviours. Applied with taxonomy.apply_decisions: every operation is a recorded change."""
    maxc = int(ph.cfg["taxonomy"].get("max_categories", 16))
    changes = []
    cats = T.active(draft, "category")
    lines = []
    for c in cats:
        subs = [s for s in T.active(draft, "subcategory") if s["category_id"] == c["id"]]
        nb = sum(1 for b in T.active(draft, "behavior") if b["category_id"] == c["id"])
        n_it = len({i for b in T.active(draft, "behavior") if b["category_id"] == c["id"] for i in (b.get("items") or [])})
        lines.append(f"{c['id']} {c['name']} ({nb} behaviours, {n_it} items): {c['definition']} | subcategories: "
                     + "; ".join(s["name"] for s in subs[:12]) + (f"; … ({len(subs) - 12} more)" if len(subs) > 12 else ""))
    ans = editor.ask(system=HARM_SYSTEM, schema=CAT_SCHEMA, max_tokens=max(3000, 60 * len(cats)), prompt_version=PROMPT_VERSION,
                     phase=ph.key, item_id="harmonize-categories",
                     validator=_merge_validator({"C": {c["id"] for c in cats}}),
                     user=f"There are {len(cats)} categories; merge them into at most {maxc} broad categories of programming "
                          "(merge a small or overlapping category — few behaviours, few items — into the broader one that contains it; "
                          "the survivor is the one holding more items). A category that is "
                          "a merge target is never itself merged. Rename a survivor when its name no longer covers what it "
                          "absorbed.\n\n" + "\n".join(lines))
    if ans.valid:
        targets = {m["into"] for m in ans.data["merges"]}
        dec = [{"id": m["from"], "action": "merge", "merge_into": m["into"], "reviewer": "editor-model (harmonisation)"}
               for m in ans.data["merges"] if m["from"] not in targets]
        dec += [{"id": r["id"], "action": "rename", "new_name": r["name"], "reviewer": "editor-model (harmonisation)"}
                for r in ans.data["renames"] if r["id"] in {c["id"] for c in cats}]
        dec += [{"id": r["id"], "action": "redefine", "new_definition": r["definition"], "reviewer": "editor-model (harmonisation)"}
                for r in ans.data["renames"] if r["id"] in {c["id"] for c in cats} and r.get("definition")]
        draft, probs = T.apply_decisions(draft, dec, draft["version"], "harmonisation A")
        changes += draft["changes"]
        for pr in probs:
            ph.warn(f"harmonisation A: {pr}")
    else:
        ph.warn("harmonisation of categories: editor answer invalid — categories kept")
    ph.log.info(f"harmonisation A: {len(cats)} → {len(T.active(draft, 'category'))} categories")
    _dedupe_names(draft)

    for c in ph.progress(T.active(draft, "category"), "harmonisation B (per category)", every_s=60):
        subs = [x for x in T.active(draft, "subcategory") if x["category_id"] == c["id"]]
        behs = [b for b in T.active(draft, "behavior") if b["category_id"] == c["id"]]
        if len(behs) < 2 and len(subs) < 2:
            continue
        text = []
        for sc in subs:
            text.append(f"{sc['id']} {sc['name']}: {sc['definition']}")
            text += [f"    {b['id']} {b['name']} ({len(b.get('items') or [])} items): {b['definition']}"
                     for b in behs if b["subcategory_id"] == sc["id"]]
        ans = editor.ask(system=HARM_SYSTEM, schema=WITHIN_SCHEMA, max_tokens=max(2500, 30 * (len(behs) + len(subs))),
                         prompt_version=PROMPT_VERSION,
                         phase=ph.key, item_id=f"harmonize-{c['id']}",
                         validator=_merge_validator({"S": {x["id"] for x in subs}, "B": {b["id"] for b in behs}}),
                         user=f"Category {c['id']} {c['name']}: {c['definition']}\nIts subcategories (S...) and behaviours (B...):\n"
                              + "\n".join(text) + "\n\nMerge subcategories that mean the same thing or are too small to stand "
                              "alone, and behaviours that mean the same thing (synonyms, rewordings). Do NOT merge behaviours that "
                              "differ (reading vs writing, iterating vs filtering). A merge target is never itself merged.")
        if not ans.valid:
            ph.warn(f"harmonisation B {c['id']}: editor answer invalid — kept as is")
            continue
        merges = ans.data["subcategory_merges"] + ans.data["behavior_merges"]
        targets = {m["into"] for m in merges}
        dec = [{"id": m["from"], "action": "merge", "merge_into": m["into"], "reviewer": "editor-model (harmonisation)"}
               for m in merges if m["from"] not in targets]
        if dec:
            draft, probs = T.apply_decisions(draft, dec, draft["version"], f"harmonisation B {c['id']}")
            changes += draft["changes"]
            for pr in probs:
                ph.warn(f"harmonisation B {c['id']}: {pr}")
        _dedupe_names(draft)
    # a subcategory (or category) left without behaviours by the merges is retired, as a recorded change
    for lv, key in (("subcategory", "subcategory_id"), ("category", "category_id")):
        holders = {b[key] for b in T.active(draft, "behavior")}
        for x in T.active(draft, lv):
            if x["id"] not in holders:
                x.update(status="retired", replaced_by=[T.UNKNOWN], retired_reason="no behaviour left after harmonisation")
                changes.append({"action": "retire_empty", "id": x["id"], "reviewer": "editor-model (harmonisation)"})
    draft["changes"] = changes
    draft["parent_version"] = None
    return draft


def _dedupe_names(tax):
    """Two concepts with the same name under one parent (possible after editing) get a suffix, reported."""
    for lv in T.LEVELS:
        seen = Counter()
        for c in tax[T.LEVELS[lv][0]]:
            k = (c["name"].lower(), c.get("category_id"), c.get("subcategory_id"))
            seen[k] += 1
            if seen[k] > 1:
                c["name"] = f"{c['name']} ({c['id']})"


def write_review_workbook(ph, draft, near_df, sim_df):
    idx = T.index(draft)
    rows = []
    for lv in ("category", "subcategory", "behavior"):
        for c in T.active(draft, lv):
            parent = c.get("subcategory_id") or c.get("category_id") or ""
            rows.append({"id": c["id"], "level": lv, "parent_id": parent,
                         "parent_name": idx[parent]["name"] if parent in idx else "", "name": c["name"],
                         "definition": c["definition"], "include": c.get("include"), "exclude": c.get("exclude"),
                         "proposed_by": c.get("proposed_by"), "aliases": c.get("aliases"), "n_items": c.get("n_items"),
                         "examples": [e["item_id"] for e in c.get("examples", [])[:4]],
                         "merge_flag": "merged from " + str(len(c.get("aliases", []))) + " proposals" if len(c.get("aliases", [])) > 1 else ""})
    cols = [("id", "id", 8, False), ("level", "level", 11, False), ("parent_id", "parent_id", 9, False),
            ("parent_name", "parent_name", 22, False), ("name", "name", 30, False), ("definition", "definition", 50, False),
            ("include", "include", 34, False), ("exclude", "exclude", 34, False), ("proposed_by", "proposed_by", 16, False),
            ("aliases", "aliases", 34, False), ("n_items", "n_items", 8, False), ("examples", "examples", 28, False),
            ("merge_flag", "merge_flag", 18, False),
            ("action", "action", 11, True), ("new_name", "new_name", 24, True), ("new_definition", "new_definition", 34, True),
            ("merge_into", "merge_into", 11, True), ("split_into", "split_into", 26, True), ("parent_id_new", "parent_id", 10, True),
            ("comment", "comment", 30, True)]
    # the expert's parent (move / add) goes in a column whose header is "parent_id"; the read-only one is renamed
    cols[2] = ("parent_id", "current_parent", 9, False)
    wb = W.new_workbook("Taxonomy review (phase 6)", [
        "You review the shared taxonomy built from three AI models' independent proposals. Each row is one concept "
        "(category C…, subcategory S…, behaviour B…). Grey columns are read-only; fill the yellow ones.",
        "action: accept (or leave empty) · rename (new_name) · redefine (new_definition, include, exclude) · "
        "merge (merge_into = the id that absorbs this one) · split (split_into = 'name A | name B', behaviours only) · "
        "reject · move (parent_id = new parent) · add (a new row: new_name, new_definition, parent_id; leave id empty).",
        "Merges were proposed by embedding similarity and judged by an editor model; check that merged members (aliases) "
        "really mean the same. Sheet 'Merge candidates' lists similar proposals the editor kept apart: merge them here if they are synonyms.",
        "IDs are stable: never edit the id column. Save the file under review/returned/<your name>.xlsx.",
    ], {"taxonomy version": draft["version"], "generated": now_iso()})
    W.add_sheet(wb, "Taxonomy Review", rows, cols, choices={"action": list(T.REVIEW_ACTIONS)})
    W.add_sheet(wb, "Merge candidates", near_df.to_dict("records"),
                [(c, c, 16 if "name" not in c else 32, False) for c in near_df.columns])
    if len(sim_df):
        W.add_sheet(wb, "Merged clusters", sim_df.to_dict("records"),
                    [(c, c, 14 if c != "member_names" else 70, False) for c in sim_df.columns])
    W.save(wb, ph.path("review", "taxonomy_review.xlsx"))


def guidelines(tax: dict) -> str:
    out = [f"# Annotation guidelines — taxonomy {tax['version']}", "",
           f"Status: **{tax['status']}** · created {tax['created']}", "",
           "## What to annotate", "",
           "- The **programming behaviour**: what the code does (Pharo method), or what the solution code must do "
           "(literature task). Not the kind of evaluation task (generation, repair...), not the application domain, not quality.",
           "- Give **1 to 3 behaviour IDs**, the most important first. Several only when the item clearly combines them.",
           "- For each behaviour, quote the **evidence** (a few words of the item).",
           f"- `{T.UNKNOWN}`: no behaviour fits (describe the missing one). `{T.INSUFFICIENT}`: the item does not show "
           "enough to decide. Do not force a behaviour.",
           "- Use the *use when* / *not when* rules of each behaviour to decide between neighbours.",
           "- Category and subcategory follow from the behaviour (they are not chosen separately).", "",
           "## Evaluation task types (separate classification)", ""]
    out += [f"- `{k}`: {v}" for k, v in T.TASK_TYPES.items()]
    out += ["", "## Taxonomy", ""]
    subs = T.active(tax, "subcategory")
    for c in T.active(tax, "category"):
        out += [f"### {c['id']} {c['name']}", "", c["definition"], ""]
        for s in [s for s in subs if s["category_id"] == c["id"]]:
            out += [f"#### {s['id']} {s['name']}", "", s["definition"], ""]
            for b in [b for b in T.active(tax, "behavior") if b["subcategory_id"] == s["id"]]:
                out.append(f"- **{b['id']} {b['name']}** — {b['definition']}")
                if b.get("include"):
                    out.append(f"  - use when: {'; '.join(b['include'])}")
                if b.get("exclude"):
                    out.append(f"  - not when: {'; '.join(b['exclude'])}")
                if b.get("evidence"):
                    out.append(f"  - e.g. “{b['evidence'][0]}”")
                out.append(f"  - proposed by: {', '.join(b.get('proposed_by', []))}")
            out.append("")
    return "\n".join(out) + "\n"


def draw_taxonomy(tax: dict, models: list, path, title: str | None = None):
    """The taxonomy figure: boxes per category, ●●○ per behaviour (one ● per annotator that proposed it) and, as a
    thin bar with the number, the items classified there — under the ●●○ of a behaviour, on the right of a category
    or a subcategory (distinct items; when a taxonomy has no item lists, the behaviours' counts are summed)."""
    cats, subs, behs = T.active(tax, "category"), T.active(tax, "subcategory"), T.active(tax, "behavior")
    blocks = []

    def n_of(b):
        return len(set(b.get("items") or [])) or int(b.get("n_items") or 0)
    for c in cats:
        lines, c_items, c_sum = [], set(), 0
        for s_ in [x for x in subs if x["category_id"] == c["id"]]:
            bl, s_items, s_sum = [], set(), 0
            for b in [b for b in behs if b["subcategory_id"] == s_["id"]]:
                k = len(b.get("proposed_by", []))
                s_items |= set(b.get("items") or [])
                s_sum += n_of(b)
                bl.append({"marker": "●" * k + "○" * max(0, len(models) - k), "count": n_of(b),
                           "label": f"{b['id']} {b['name']}"})
            c_items |= s_items
            c_sum += s_sum
            lines.append(({"label": f"{s_['id']} {s_['name']}", "count": len(s_items) or s_sum}, bl))
        blocks.append(({"label": f"{c['id']} {c['name']}", "count": len(c_items) or c_sum}, lines))
    n_all = len({i for b in behs for i in (b.get("items") or [])})
    src = SOURCES.get(tax.get("describes"))
    F.taxonomy_tree(blocks, title or (f"Taxonomy {tax['version']}{' of the ' + src if src else ''} ({tax['status']}): "
                                      f"{len(cats)} categories, {len(subs)} subcategories, {len(behs)} behaviours   "
                                      f"● = proposed by one model ({len(models)} models)"),
                    path, note="bar and number = items classified there by the annotators"
                               + (f" ({n_all} distinct items)" if n_all else "")
                               + " — under ●●○ for a behaviour, on the right for a category or a subcategory")


def figures(ph, tax, df, ids, models):
    draw_taxonomy(tax, models, ph.figure(f"taxonomy_{tax['version']}.png"))

    # model overlap: share of model A's behaviours that were merged with one of model B's
    b = df[(df["level"] == "behavior") & df["cluster"].isin(ids)]
    clusters_of = {m: set(b.loc[b["model"] == m, "cluster"]) for m in models}
    mat = [[len(clusters_of[a] & clusters_of[c]) / max(len(clusters_of[a]), 1) * 100 for c in models] for a in models]
    F.heatmap(mat, models, models, "Behaviours of the row model matched by the column model (%)",
              ph.figure("model_overlap.png"), fmt="{:.0f}", vmin=0, vmax=100, cbar_label="%")

    support = Counter(len(x.get("proposed_by", [])) for x in T.active(tax, "behavior"))
    F.barh([f"proposed by {k} model(s)" for k in sorted(support, reverse=True)],
           [support[k] for k in sorted(support, reverse=True)], "How many models proposed each behaviour",
           "behaviours", ph.figure("support_per_behavior.png"),
           note="a behaviour proposed by one model only is kept but needs the expert's attention")
