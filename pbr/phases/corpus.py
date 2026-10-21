"""Phase 3 — Pharo corpus preparation.

Input: the methods of the repositories selected in phase 1 (repositories: hard filters + weighted quality score),
read from the parsed files of the corpus snapshot (``data/parsed/<owner>__<repo>.jsonl.gz``; Hugging Face
``pharo-llm/pharo-labels``, downloaded by setup), with the GitHub metadata of every repository and its domain.

What this phase adds:
* **Provenance**: every method keeps repository, commit, file, line, class, side, selector, and a
  permalink ``https://github.com/<repo>/blob/<commit>/<file>#L<line>``.
* **Stable IDs**: ``method_id`` = hash of (repository, file, class, side, selector) — independent of the
  line, of the parse order and of this pipeline's runs.
* **Method-level context**: package, protocol, class superclass, instance variables, class comment.
* **Structure**: size, message sends, sent selectors, referenced globals, pragmas, and the parser's
  boolean features (loops, conditionals, exception handling, reflection, accessors, ...).
* **Deduplication**: exact (same source) and normalised (same code once comments and layout are
  removed) duplicates; the canonical copy is the one in the oldest repository. A repository whose methods
  are mostly copies of another one is flagged as a mirror. Trivial methods (one statement) are flagged
  separately: their repetition is an idiom, not copying.

Outputs
  repositories.parquet   one row per repository: metadata, domain, size, test share, duplication, status
  method_filters.parquet one row per method: decision (exclude / review / keep / keep_test) and every reason
                         (rules: pbr/method_filters.py); method_filters_summary.csv: methods per decision and rule
  methods.parquet        one row per method (source included)
  duplicates.parquet     one row per non-canonical copy: method_id, canonical_method_id, kind, scope, group size
  figures/: methods_per_repository.png, repository_groups.png, duplication.png, method_size.png, features.png
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import figures as F
from .. import method_filters as MF
from ..common import data_dir, resolve, write_csv, write_parquet

FEATURES = ["has_conditional", "has_loop", "has_collection_iteration", "has_exception_handler", "has_exception_signal",
            "has_recursion", "has_reflection", "has_metaprogramming", "has_dynamic_send", "has_super_send",
            "has_instance_creation", "has_nonlocal_return", "has_cascade", "has_block_evaluation", "is_getter",
            "is_setter", "is_lazy_initializer", "is_empty", "is_abstract"]


def strip_comments(src: str) -> str:
    """Smalltalk source without comments ("...") and with whitespace collapsed; string literals ('...',
    with '' escapes) and character literals ($") are kept verbatim."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c == "$" and i + 1 < n:              # character literal, e.g. $" or $'
            out.append(src[i:i + 2])
            i += 2
        elif c == "'":
            j = i + 1
            while j < n:
                if src[j] == "'":
                    if j + 1 < n and src[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            out.append(src[i:j + 1])
            i = j + 1
        elif c == '"':
            j = src.find('"', i + 1)
            i = n if j < 0 else j + 1
            out.append(" ")
        else:
            out.append(c)
            i += 1
    return " ".join("".join(out).split())


def method_id(repo: str, path: str, cls: str, side: str, selector: str) -> str:
    return "M" + hashlib.sha1(f"{repo}|{path}|{cls}|{side}|{selector}".encode()).hexdigest()[:15]


def find_sources(ph) -> tuple[Path, Path | None]:
    """(folder of parsed *.jsonl.gz, folder of the snapshot with metadata or None)."""
    c = ph.cfg["corpus"]
    snap = data_dir(ph.cfg) / "pharo-corpus" / c["snapshot_revision"]
    for d in c.get("local_parsed_dirs") or []:
        p = resolve(d)
        if p.is_dir() and any(p.glob("*.jsonl.gz")):
            ph.log.info(f"parsed methods: local folder {p}")
            return p, (snap / "snapshot" if (snap / "snapshot").exists() else None)
    parsed = snap / "snapshot" / "data" / "parsed"
    if not parsed.is_dir() or len(list(parsed.glob("*.jsonl.gz"))) < 100:
        ph.log.info(f"downloading the corpus snapshot {c['snapshot_repo']}@{c['snapshot_revision']} "
                    f"(parsed methods + metadata, about 1.3 GB) into {snap}")
        from huggingface_hub import snapshot_download
        snapshot_download(c["snapshot_repo"], repo_type="dataset", revision=c["snapshot_revision"], local_dir=str(snap),
                          max_workers=8, allow_patterns=[
                              "snapshot/data/parsed/*", "snapshot/data/raw/repositories.jsonl",
                              "snapshot/data/processed/sample.json", "snapshot/data/processed/domain_labels.jsonl",
                              "snapshot/output/tables/repository_selection.csv", "snapshot/manifest.json"])
    return parsed, snap / "snapshot"


def load_metadata(ph, snap: Path | None) -> tuple[dict, set | None, dict]:
    """(full_name -> GitHub metadata, repositories that passed filter 1 or None, full_name -> domain)."""
    meta, selected, domains = {}, None, {}
    if snap is None:
        ph.warn("no snapshot metadata folder: repositories get only what the parsed files contain")
        return meta, selected, domains
    rj = snap / "data" / "raw" / "repositories.jsonl"
    if rj.exists():
        with rj.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                name = d.get("full_name") or d.get("repository")
                if name:   # the GitHub API record is under "raw"
                    meta[name] = {**(d.get("raw") if isinstance(d.get("raw"), dict) else {}), "full_name": name}
    # filter-1 table of ../labeling-pharo-repos: primary domain, contributors, commits, project kind
    st = snap / "output" / "tables" / "repository_selection.csv"
    if st.exists():
        tab = pd.read_csv(st, low_memory=False)
        for r in tab.to_dict("records"):
            m = meta.setdefault(r["full_name"], {"full_name": r["full_name"]})
            m["selection"] = {k: (None if pd.isna(v) else v) for k, v in r.items()}
            if isinstance(r.get("primary_domain"), str) and r["primary_domain"] != "unknown":
                domains[r["full_name"]] = r["primary_domain"]
    sj = snap / "data" / "processed" / "sample.json"
    if sj.exists():
        s = json.loads(sj.read_text(encoding="utf-8"))
        sel = s.get("selected") or s.get("repositories") or s.get("sample")
        if isinstance(sel, list) and sel:
            selected = {x if isinstance(x, str) else (x.get("full_name") or x.get("repository")) for x in sel}
    dj = snap / "data" / "processed" / "domain_labels.jsonl"
    if dj.exists():          # fallback when the filter-1 table has no primary domain: highest-scoring label
        with dj.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                name = d.get("repository") or d.get("full_name")
                labs = sorted(d.get("domain_labels") or [], key=lambda x: -float(x.get("score") or 0))
                if name and labs and name not in domains:
                    domains[name] = labs[0]["label"]
    return meta, selected, domains


def _g(d: dict, *keys, default=None):
    for k in keys:
        cur = d
        for part in k.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if cur not in (None, ""):
            return cur
    return default


SCHEMA = pa.schema([
    ("method_id", pa.string()), ("parser_method_key", pa.string()), ("repository", pa.string()),
    ("commit_sha", pa.string()), ("package", pa.string()), ("class_name", pa.string()), ("side", pa.string()),
    ("selector", pa.string()), ("protocol", pa.string()), ("source_path", pa.string()), ("line", pa.int32()),
    ("container_format", pa.string()), ("url", pa.string()), ("is_test", pa.bool_()), ("is_extension", pa.bool_()),
    ("in_trait", pa.bool_()), ("parse_status", pa.string()), ("superclass", pa.string()),
    ("class_instance_variables", pa.list_(pa.string())), ("class_comment", pa.string()),
    ("source", pa.string()), ("source_chars", pa.int32()), ("n_lines", pa.int32()), ("n_statements", pa.int32()),
    ("n_ast_nodes", pa.int32()), ("n_message_sends", pa.int32()), ("n_blocks", pa.int32()),
    ("max_block_depth", pa.int32()), ("n_arguments", pa.int32()), ("n_temporaries", pa.int32()),
    ("sent_selectors", pa.list_(pa.string())), ("referenced_globals", pa.list_(pa.string())),
    ("pragmas", pa.list_(pa.string())),
    *[(f, pa.bool_()) for f in FEATURES],
    ("source_hash", pa.string()), ("norm_hash", pa.string()), ("is_trivial", pa.bool_()), ("over_long", pa.bool_()),
    ("duplicate_of", pa.string()), ("duplicate_kind", pa.string()),
    # AST facts for the method filters (pbr/method_filters.analyse): reasons, statements at every level, globals
    ("ast_exclude", pa.string()), ("ast_review", pa.string()), ("n_statements_all", pa.int32()),
    ("ast_globals", pa.list_(pa.string())),
])


def apply_method_filters(ph, methods_path, mirrors: set):
    """The decision of every method (pbr/method_filters.py, decided on its AST and its class definition):
    exclude / review / keep / keep_test, with every reason."""
    cols = ["method_id", "repository", "class_name", "superclass", "is_test", "parse_status", "duplicate_of",
            "duplicate_kind", "n_lines", "ast_exclude", "ast_review", "n_statements_all", "ast_globals"]
    df = pd.read_parquet(methods_path, columns=cols)
    gpath = resolve(ph.cfg["pharo"]["dir"]) / "globals.txt"
    image_globals = set(gpath.read_text().split()) if gpath.exists() else set()
    if not image_globals:
        ph.warn("no Pharo image globals (data/pharo-image/globals.txt, written by setup): the rule 'requires "
                "dependencies' is skipped")
    repo_classes = df.groupby("repository")["class_name"].agg(lambda x: set(x)).to_dict()
    ph.log.info(f"method filters on {len(df):,} methods ({len(image_globals):,} Pharo image globals known)")
    parts = []
    for i in ph.progress(range(0, len(df), 100_000), "method filters (batches of 100,000)", every_s=30):
        parts.append(MF.classify(df.iloc[i:i + 100_000], ph.cfg, mirrors, image_globals, repo_classes))
    dec = pd.concat(parts, ignore_index=True)
    dec["rule"] = dec["reasons"].map(MF.first_rule)
    write_parquet(ph.path("method_filters.parquet"), dec)
    summ = (dec.assign(rule=np.where(dec["decision"].isin(["exclude", "review"]), dec["rule"], dec["decision"]))
            .groupby(["decision", "rule"]).size().rename("methods").reset_index().sort_values(["decision", "methods"],
                                                                                              ascending=[True, False]))
    write_csv(ph.path("method_filters_summary.csv"), summ)
    counts = dec["decision"].value_counts()
    for d in ("keep", "review", "keep_test", "exclude"):
        ph.metric(f"methods_{d}", int(counts.get(d, 0)))
    ph.log.info("method decisions: " + ", ".join(f"{d} {int(counts.get(d, 0)):,}" for d in ("keep", "review", "keep_test", "exclude")))
    ex = summ[summ["decision"] == "exclude"]
    # review: a method can have several reasons; each reason counts every method that has it
    rv_reasons = ["simple message forwarding", "non-trivial initialisation", "very long method", "requires dependencies"]
    rv_df = dec[dec["decision"] == "review"]
    rv = pd.DataFrame({"rule": rv_reasons, "methods": [int(rv_df["reasons"].str.contains(r, regex=False).sum()) for r in rv_reasons]})
    rv = rv[rv["methods"] > 0].sort_values("methods", ascending=False)
    labels = [f"excluded: {r}" for r in ex["rule"]] + [f"review: {r}" for r in rv["rule"]] + ["kept", "kept separately: tests"]
    vals = list(ex["methods"]) + list(rv["methods"]) + [int(counts.get("keep", 0)), int(counts.get("keep_test", 0))]
    cols_ = [F.STATUS["critical"]] * len(ex) + [F.STATUS["warning"]] * len(rv) + [F.STATUS["good"], F.SERIES[0]]
    F.barh(labels, vals, f"Method filters on {len(dec):,} methods of the selected repositories", "methods",
           ph.figure("method_filters.png"), colors=cols_, fmt="{:,.0f}", max_items=60,
           note="excluded: counted under its first (strongest) rule; review: every reason counted (a method can have several) — kept in the samples, marked")


def run(ph):
    c = ph.cfg["corpus"]
    parsed, snap = find_sources(ph)
    meta, selected, domains = load_metadata(ph, snap)
    files = sorted(parsed.glob("*.jsonl.gz"))
    name_of = {f: f.name[:-len(".jsonl.gz")].replace("__", "/", 1) for f in files}
    # the repositories phase 1 selected (hard filters + weighted quality score)
    chosen = set(ph.upstream("repositories", "selected_repos.txt").read_text().split())
    scores = pd.read_csv(ph.upstream("repositories", "repository_scores.csv")).set_index("repository")
    ph.log.info(f"{len(files)} parsed repositories in the snapshot; {len(chosen)} selected by phase 1; "
                f"GitHub metadata for {len(meta)}; domains for {len(domains)}")
    kept = [f for f in files if name_of[f] in chosen]
    # canonical copy = the oldest repository (then name), so process the repositories in that order
    created = {f: str(_g(meta.get(name_of[f], {}), "created_at", "github.created_at", default="9999")) for f in kept}
    kept.sort(key=lambda f: (created[f], name_of[f]))

    max_chars, triv = int(c["max_source_chars"]), int(c["trivial_max_statements"])
    out_path = ph.path("methods.parquet")
    tmp_path = out_path.with_name(".methods.parquet.tmp")
    writer = pq.ParquetWriter(tmp_path, SCHEMA, compression="zstd")
    first_raw, first_norm = {}, {}       # hash -> (method_id, repository)
    dup_rows, repo_rows = [], []
    seen_ids = set()
    try:
        for f in ph.progress(kept, "repositories parsed", every_s=20):
            repo = name_of[f]
            rows, classes, snapshot_rec = [], {}, {}
            try:
                with gzip.open(f, "rt", encoding="utf-8") as fh:
                    recs = [json.loads(line) for line in fh if line.strip()]
            except (OSError, ValueError) as e:
                ph.warn(f"{repo}: unreadable parsed file ({e}) — skipped")
                continue
            for r in recs:
                if r.get("record") == "snapshot":
                    snapshot_rec = r
                    for cl in r.get("classes") or []:
                        classes[cl.get("name")] = cl
            commit = snapshot_rec.get("commit_sha") or ""
            n_fail = 0
            for r in recs:
                if r.get("record") != "method":
                    continue
                path, cls, side, sel = r.get("source_path") or "", r.get("class_name") or "", r.get("side") or "instance", r.get("selector") or ""
                mid = method_id(repo, path, cls, side, sel)
                if mid in seen_ids:          # same method twice in one file (rare parser artefact): disambiguate by line
                    mid = method_id(repo, f"{path}:{r.get('line')}", cls, side, sel)
                    if mid in seen_ids:
                        continue
                seen_ids.add(mid)
                src = r.get("source") or ""
                feats = r.get("features") or {}
                ok = r.get("parse_status") == "ok"
                n_fail += not ok
                cl = classes.get(cls) or {}
                norm = strip_comments(src)
                raw_h = r.get("source_hash") or hashlib.sha1(src.encode()).hexdigest()
                norm_h = hashlib.sha1(norm.encode()).hexdigest()
                nstat = int(feats.get("n_statements") or 0)
                dup_of = kind = None
                if raw_h in first_raw:
                    dup_of, kind = first_raw[raw_h][0], "exact"
                elif norm_h in first_norm:
                    dup_of, kind = first_norm[norm_h][0], "normalized"
                else:
                    first_raw[raw_h] = (mid, repo)
                first_norm.setdefault(norm_h, (mid, repo))
                if dup_of:
                    dup_rows.append((mid, repo, dup_of, kind, norm_h))
                line = int(r.get("line") or 0)
                facts = MF.analyse(r.get("ast") if ok else None, MF.class_variables(cl, side), sel, src,
                                   [p.get("selector") for p in r.get("pragmas") or [] if p.get("selector")])
                rows.append({
                    "method_id": mid, "parser_method_key": r.get("method_key"), "repository": repo,
                    "commit_sha": commit or r.get("commit_sha"), "package": r.get("package"), "class_name": cls,
                    "side": side, "selector": sel, "protocol": r.get("protocol"), "source_path": path, "line": line,
                    "container_format": r.get("container_format"),
                    "url": f"https://github.com/{repo}/blob/{commit or r.get('commit_sha')}/{path}#L{line}",
                    "is_test": bool(r.get("is_test")), "is_extension": bool(r.get("is_extension")),
                    "in_trait": bool(r.get("in_trait")), "parse_status": r.get("parse_status"),
                    "superclass": cl.get("superclass"), "class_instance_variables": [str(x) for x in cl.get("instance_variables") or []],
                    "class_comment": (cl.get("comment") or "")[:800] or None,
                    "source": src, "source_chars": len(src), "n_lines": int(feats.get("n_lines") or src.count("\n") + 1),
                    "n_statements": nstat, "n_ast_nodes": int(feats.get("n_ast_nodes") or r.get("n_ast_nodes") or 0),
                    "n_message_sends": int(feats.get("n_message_sends") or 0), "n_blocks": int(feats.get("n_blocks") or 0),
                    "max_block_depth": int(r.get("max_block_depth") or feats.get("max_nesting_depth") or 0),
                    "n_arguments": int(feats.get("n_arguments") or 0), "n_temporaries": int(feats.get("n_temporaries") or 0),
                    "sent_selectors": sorted({m.get("selector") for m in r.get("message_sends") or [] if m.get("selector")}),
                    # the parser records each reference as {"name": ..., "line": ...}: keep the names
                    "referenced_globals": sorted({str(x.get("name")) if isinstance(x, dict) else str(x)
                                                  for x in r.get("referenced_globals") or []} - {"None"}),
                    "pragmas": [p.get("selector") for p in r.get("pragmas") or [] if p.get("selector")],
                    **{k: bool(feats.get(k)) for k in FEATURES},
                    "source_hash": raw_h, "norm_hash": norm_h, "is_trivial": nstat <= triv,
                    "over_long": len(src) > max_chars, "duplicate_of": dup_of, "duplicate_kind": kind,
                    "ast_exclude": "; ".join(facts["exclude"]), "ast_review": "; ".join(facts["review"]),
                    "n_statements_all": facts["n_statements_all"], "ast_globals": facts["globals"],
                })
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
            m = meta.get(repo, {})
            n = len(rows)
            n_test = sum(x["is_test"] for x in rows)
            fmts = Counter(x["container_format"] for x in rows)
            repo_rows.append({
                "repository": repo, "commit_sha": commit, "url": f"https://github.com/{repo}",
                "description": _g(m, "description", "github.description"), "stars": _g(m, "stargazers_count", "stars", default=0),
                "forks": _g(m, "forks_count", "forks", default=0),
                "contributors": _g(m, "selection.contributors", "contributors"),
                "commit_count": _g(m, "selection.commit_count"), "project_kind": _g(m, "selection.project_kind"),
                "default_branch": _g(m, "default_branch"),
                "created_at": _g(m, "created_at"), "pushed_at": _g(m, "pushed_at"),
                "license": _g(m, "license.spdx_id", "license.key", "license"), "topics": json.dumps(_g(m, "topics", default=[])),
                "archived": bool(_g(m, "archived", default=False)), "domain": domains.get(repo, "unknown"),
                "n_methods": n, "n_test_methods": n_test, "test_share": round(n_test / n, 4) if n else 0.0,
                "n_classes": len(classes), "n_packages": len({x["package"] for x in rows}),
                "parse_failures": n_fail, "formats": json.dumps(dict(fmts)),
                "status": "included" if n else "excluded: no methods",
            })
    finally:
        writer.close()
    tmp_path.replace(out_path)

    # ---------------------------------------------------------------- duplicates and mirrors
    excluded = [{"repository": name_of[f], "n_methods": 0,
                 "status": "excluded: " + (str(scores.loc[name_of[f], "reason"]) if name_of[f] in scores.index
                                           else "not in the snapshot's filter-1 selection")}
                for f in files if f not in set(kept)]
    group_size = Counter(h for *_, h in dup_rows)
    repo_of_canon = {}
    dups = pd.DataFrame(dup_rows, columns=["method_id", "repository", "canonical_method_id", "kind", "norm_hash"])
    if len(dups):
        canon_ids = set(dups["canonical_method_id"])
        canon_repo = pd.read_parquet(out_path, columns=["method_id", "repository"], filters=[("method_id", "in", list(canon_ids))])
        repo_of_canon = dict(zip(canon_repo["method_id"], canon_repo["repository"]))
        dups["canonical_repository"] = dups["canonical_method_id"].map(repo_of_canon)
        dups["scope"] = np.where(dups["canonical_repository"] == dups["repository"], "same_repository", "cross_repository")
        dups["group_size"] = dups["norm_hash"].map(lambda h: group_size[h] + 1)
    write_parquet(ph.path("duplicates.parquet"), dups)

    repos = pd.DataFrame(repo_rows)
    if len(dups):
        cross = dups[dups["scope"] == "cross_repository"]
        per = cross.groupby("repository").size()
        top_src = cross.groupby("repository")["canonical_repository"].agg(lambda s: s.value_counts().index[0])
        repos["duplicated_methods"] = repos["repository"].map(per).fillna(0).astype(int)
        repos["duplicated_share"] = (repos["duplicated_methods"] / repos["n_methods"].clip(lower=1)).round(4)
        repos["main_copy_source"] = repos["repository"].map(top_src)
        repos["is_mirror"] = repos["duplicated_share"] >= 0.5
    else:
        repos["duplicated_methods"], repos["duplicated_share"], repos["main_copy_source"], repos["is_mirror"] = 0, 0.0, None, False
    q = repos.loc[repos["n_methods"] > 0, "n_methods"].quantile([1 / 3, 2 / 3]).tolist() if len(repos) else [0, 0]
    repos["size_class"] = pd.cut(repos["n_methods"], [-1, q[0], q[1], np.inf], labels=["small", "medium", "large"]).astype(str)
    repos = pd.concat([repos, pd.DataFrame(excluded)], ignore_index=True)
    repos["quality_score"] = repos["repository"].map(scores["score"])      # phase 1 (selection only, never a weight)
    write_parquet(ph.path("repositories.parquet"), repos)

    inc = repos[repos["status"] == "included"]
    n_methods = int(inc["n_methods"].sum())
    apply_method_filters(ph, out_path, set(inc.loc[inc["is_mirror"].astype(bool), "repository"]))
    ph.metric("repositories", int(len(inc)))
    ph.metric("repositories_excluded", int(len(repos) - len(inc)))
    ph.metric("methods", n_methods)
    ph.metric("test_methods", int(inc["n_test_methods"].sum()))
    ph.metric("duplicates_exact", int((dups["kind"] == "exact").sum()) if len(dups) else 0)
    ph.metric("duplicates_normalized", int((dups["kind"] == "normalized").sum()) if len(dups) else 0)
    ph.metric("duplicate_share", round(len(dups) / max(n_methods, 1), 4))
    ph.metric("mirror_repositories", int(inc["is_mirror"].sum()))
    ph.metric("parse_failures", int(inc["parse_failures"].sum()))
    ph.metric("domains", inc["domain"].value_counts().to_dict())
    ph.log.info(f"methods.parquet: {n_methods:,} methods from {len(inc)} repositories; {len(dups):,} non-canonical copies")

    # ---------------------------------------------------------------- figures
    s = np.sort(inc["n_methods"].to_numpy())[::-1]
    fig, ax = F.plt.subplots(figsize=(8.5, 4.2))
    ax.plot(np.arange(1, len(s) + 1), np.cumsum(s) / max(s.sum(), 1), color=F.SERIES[0], lw=2)
    for frac in (0.01, 0.05, 0.10):
        k = max(1, int(len(s) * frac))
        share = s[:k].sum() / max(s.sum(), 1)
        ax.annotate(f"top {frac:.0%} of repositories\n= {share:.0%} of methods", (k, share), xytext=(12, -30),
                    textcoords="offset points", fontsize=8, color=F.INK_2, arrowprops={"arrowstyle": "-", "color": F.INK_2})
    ax.set_xscale("log")
    ax.set_xlabel("repositories, largest first (log scale)")
    ax.set_ylabel("cumulative share of methods")
    ax.set_title("A few large repositories hold most methods (why sampling is stratified and capped)")
    F.save(fig, ph.figure("methods_per_repository.png"))

    dv = inc.groupby("domain").agg(repos=("repository", "size"), methods=("n_methods", "sum")).sort_values("repos", ascending=False)
    F.barh(dv.index, dv["repos"], "Repositories per application domain (repository groups)", "repositories",
           ph.figure("repository_groups.png"), note="domain = label of ../labeling-pharo-repos domain classifier (highest score)")

    if len(dups):
        parts = {"exact copy": [], "normalised copy": []}
        cats = ["cross_repository", "same_repository"]
        for sc in cats:
            parts["exact copy"].append(int(((dups["scope"] == sc) & (dups["kind"] == "exact")).sum()))
            parts["normalised copy"].append(int(((dups["scope"] == sc) & (dups["kind"] == "normalized")).sum()))
        F.stacked_barh(["copied from another repository", "repeated in the same repository"], parts,
                       f"Duplicated methods ({len(dups):,} of {n_methods:,} = {len(dups) / max(n_methods, 1):.1%})",
                       "methods", ph.figure("duplication.png"))

    sizes = pd.read_parquet(out_path, columns=["n_statements", "is_test", "duplicate_of"])
    sizes = sizes[sizes["duplicate_of"].isna()]
    fig, ax = F.plt.subplots(figsize=(8.5, 4))
    bins = np.arange(0, 41)
    for i, (lab, sub) in enumerate([("production methods", sizes[~sizes["is_test"]]), ("test methods", sizes[sizes["is_test"]])]):
        ax.hist(sub["n_statements"].clip(upper=40), bins=bins, histtype="step", lw=2, color=F.color(i), label=lab,
                density=True)
    ax.set_xlabel("statements per method (40 = 40 or more)")
    ax.set_ylabel("share of methods")
    ax.set_title("Method size (canonical copies)")
    ax.legend()
    F.save(fig, ph.figure("method_size.png"))

    feats = pd.read_parquet(out_path, columns=FEATURES + ["duplicate_of"])
    feats = feats[feats["duplicate_of"].isna()]
    share = feats[FEATURES].mean().sort_values(ascending=False)
    F.barh([x.replace("_", " ") for x in share.index], share.values * 100, "Structural features of canonical methods",
           "% of methods", ph.figure("features.png"), fmt="{:.1f}%",
           note="parser features of ../labeling-pharo-repos (python-st-parser); a method can have several")
