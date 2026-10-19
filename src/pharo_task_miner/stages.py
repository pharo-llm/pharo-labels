"""Pipeline stages. Each stage reads its inputs from the evidence database or
from earlier stages' files, writes its own outputs, and records a run with
provenance, so any stage can be re-run independently (e.g. re-classify without
re-downloading)."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import json
import math
import os
from pathlib import Path

from .config import load_yaml
from .discovery.deduplication import content_duplicates, metadata_duplicates
from .discovery.github_client import GitHubClient
from .discovery.metadata import clone_dir_name, collect_metadata, find_local_clone, load_discovered, verify_repositories
from .discovery.repository_search import DiscoveryStore, import_search_dump, run_discovery
from .extraction.method_extractor import extract_repository
from .extraction.pharo_parser_bridge import make_backend
from .extraction.repository_cloner import clone, reuse
from .analysis.api_usage import FrameworkMatcher, RepositoryIndex, api_references, resolve_send
from .analysis.domain_classifier import DomainClassifier, clean_readme, emergent_terms
from .analysis.structural_features import compute_features
from .sampling.corpus_filter import study_repo_ids
from .sampling.repository_filters import eligibility, repository_profiles, selection_digest
from .sampling.diversity_sampling import sample as draw_sample, stratum_of
from .storage.database import Database, dumps, loads
from .storage.provenance import now_iso, tracked_run, write_json, write_jsonl, read_jsonl


class StageBlocked(RuntimeError):
    """Raised when a stage needs human-reviewed data that is not available."""


def log(msg):
    print(msg, flush=True)


def open_db(cfg) -> Database:
    return Database(cfg.database_path)


def github_client(cfg, args, required: bool) -> GitHubClient | None:
    token = os.getenv(cfg["github"]["token_env"])
    offline = getattr(args, "offline", False)
    if not token and not offline:
        if required:
            raise StageBlocked(f"Set {cfg['github']['token_env']} (GitHub token) or use --offline with cached responses")
        return None
    return GitHubClient(cfg["github"], cfg.resolve(cfg["github"]["cache_dir"]), token=token, offline=offline,
                        refresh=getattr(args, "refresh", False), log=log)


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or (list(rows[0].keys()) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


# --------------------------------------------------------------------------- discovery / verification / metadata
def discover(cfg, args):
    raw = cfg.data_dir / "raw"
    with open_db(cfg) as db, tracked_run(db, "discover", cfg, vars(args)) as run:
        store = DiscoveryStore(raw)
        if args.import_dump:
            for p in args.import_dump:
                n = import_search_dump(store, Path(p))
                log(f"Imported {n} repositories from {p}")
        else:
            client = github_client(cfg, args, required=True)
            run.counts.update(run_discovery(cfg, client, store, args.strategy or None, log))
        store.save()
        run.counts["unique_repositories"] = len(store.repos)
        run.counts["loaded_into_db"] = load_discovered(db, raw)
        run.counts["incomplete_tasks"] = store.manifest.get("incomplete_tasks")
        log(f"Discovery: {len(store.repos)} unique repositories → {store.repos_path}")


def verify(cfg, args):
    with open_db(cfg) as db, tracked_run(db, "verify", cfg, vars(args)) as run:
        load_discovered(db, cfg.data_dir / "raw")
        client = None if args.local else github_client(cfg, args, required=False)
        if client is None:
            log("No GitHub client: verifying from local clones where available (others become 'uncertain').")
        counts = verify_repositories(cfg, db, client, args.only, prefer_local=args.local,
                                     refresh=getattr(args, "refresh", False), log=log)
        run.counts.update(counts)
        run.counts["duplicates"] = metadata_duplicates(db, run.run_id)
        rows = [{"full_name": r["full_name"], "pharo_status": r["pharo_status"],
                 "evidence": loads(r["pharo_evidence_json"], []), "counter_evidence": loads(r["pharo_counter_evidence_json"], []),
                 "verification_source": r["verification_source"]} for r in db.query("SELECT * FROM repositories ORDER BY full_name")]
        write_jsonl(cfg.data_dir / "processed" / "pharo_verification.jsonl", rows)
        write_csv(cfg.output_dir / "tables" / "manual_review_uncertain.csv",
                  [{"full_name": r["full_name"], "evidence": r["evidence"], "counter_evidence": r["counter_evidence"]}
                   for r in rows if r["pharo_status"] == "uncertain"], ["full_name", "evidence", "counter_evidence"])
        log(f"Verification: {counts}")


def metadata(cfg, args):
    with open_db(cfg) as db, tracked_run(db, "metadata", cfg, vars(args)) as run:
        load_discovered(db, cfg.data_dir / "raw")
        client = github_client(cfg, args, required=True)
        run.counts.update(collect_metadata(cfg, db, client, cfg.data_dir / "raw", args.only,
                                           refresh=getattr(args, "refresh", False), log=log))
        run.counts["duplicates"] = metadata_duplicates(db, run.run_id)
        log(f"Metadata: {run.counts}")


# --------------------------------------------------------------------------- domains
def _repo_domain_inputs(db: Database) -> dict[int, dict]:
    deps, pkgs, fw = defaultdict(set), defaultdict(set), defaultdict(Counter)
    latest = {r[0]: r[1] for r in db.query("SELECT repo_id, MAX(snapshot_id) FROM repository_snapshots GROUP BY repo_id")}
    snap_repo = {v: k for k, v in latest.items()}
    for r in db.query("SELECT snapshot_id, to_name FROM package_dependencies WHERE kind='external_project'"):
        if r["snapshot_id"] in snap_repo:
            deps[snap_repo[r["snapshot_id"]]].add(r["to_name"])
    for r in db.query("SELECT snapshot_id, name FROM packages"):
        if r["snapshot_id"] in snap_repo:
            pkgs[snap_repo[r["snapshot_id"]]].add(r["name"])
    for r in db.query("SELECT m.repo_id, m.snapshot_id, a.framework, SUM(a.occurrences) n FROM api_references a"
                      " JOIN methods m ON m.method_id=a.method_id WHERE a.defined_in_repository=0 GROUP BY m.repo_id, m.snapshot_id, a.framework"):
        if latest.get(r["repo_id"]) == r["snapshot_id"]:
            fw[r["repo_id"]][r["framework"] or "(unattributed)"] += r["n"]
    out = {}
    for r in db.query("SELECT * FROM repositories"):
        total = sum(fw[r["repo_id"]].values())
        out[r["repo_id"]] = {
            "full_name": r["full_name"], "name": r["name"], "description": r["description"], "readme": r["readme_text"],
            "readme_path": r["readme_path"], "topics": loads(r["topics_json"], []), "dependencies": sorted(deps[r["repo_id"]]),
            "packages": sorted(pkgs[r["repo_id"]]),
            "framework_shares": {k: v / total for k, v in fw[r["repo_id"]].items()} if total else {}}
    return out


def classify_domains(cfg, args):
    dcfg = cfg["classification"]["domains"]
    spec = load_yaml(cfg.resolve(dcfg["definitions"]))
    clf = DomainClassifier(spec, float(dcfg.get("min_score", 2.0)), float(dcfg.get("framework_reference_min_share", 0.02)))
    with open_db(cfg) as db, tracked_run(db, "classify-domains", cfg, vars(args)) as run:
        inputs = _repo_domain_inputs(db)
        db.execute("DELETE FROM domain_labels WHERE method='deterministic_rules'")
        results, counts = [], Counter()
        for rid, repo in inputs.items():
            res = clf.classify(repo)
            results.append(res)
            for lab in res["domain_labels"]:
                counts[lab["label"]] += 1
                db.upsert("domain_labels", {"repo_id": rid, "label": lab["label"], "method": "deterministic_rules",
                                            "score": lab["score"], "score_kind": lab["score_kind"],
                                            "evidence_json": dumps(lab["evidence"]), "review_status": "pending",
                                            "model": None, "prompt_version": f"domains.yaml v{clf.version}",
                                            "run_id": run.run_id, "created_at": now_iso()}, ["repo_id", "label", "method"])
        db.commit()
        write_jsonl(cfg.data_dir / "processed" / "domain_labels.jsonl", results)
        unknown_docs = {r["repository"]: " ".join(filter(None, [inputs[k]["description"], clean_readme(inputs[k]["readme"])]))
                        for k, r in zip(inputs, results) if r["domain_labels"][0]["label"] == "unknown"}
        all_docs = {v["full_name"]: " ".join(filter(None, [v["description"], clean_readme(v["readme"])])) for v in inputs.values()}
        write_csv(cfg.output_dir / "tables" / "emergent_domain_terms.csv", emergent_terms(unknown_docs, all_docs),
                  ["term", "tfidf", "unknown_repositories", "corpus_repositories"])
        run.counts.update(dict(counts))
        log(f"Domain labels: {dict(counts.most_common())}")


# --------------------------------------------------------------------------- sampling
def sample(cfg, args):
    scfg = dict(cfg["sampling"])
    strategy = args.strategy or scfg["strategy"]
    if strategy in ("all", "all_repositories"):
        size = args.size or int(cfg["repositories"]["max_repositories"])
    else:
        size = args.size or scfg["sample_size"]
    seed = args.seed if args.seed is not None else scfg["random_seed"]
    with open_db(cfg) as db, tracked_run(db, "sample", cfg, vars(args)) as run:
        try:
            previous = set(_selected_repo_ids(db, cfg, None))
        except StageBlocked:
            previous = None  # first sample
        profiles = repository_profiles(db, cfg)
        eligible, decisions = [], []
        for p in profiles:
            ok, reason = eligibility(p, cfg)
            if ok:
                eligible.append(p)
            else:
                decisions.append((p, "excluded", reason))
        manual = cfg.resolve(args.manual_file or scfg.get("manual_selection_file"))
        chosen, skipped, info = draw_sample(eligible, strategy, size, seed, scfg, manual)
        decisions += [(p, "included", "selected by " + info["strategy"]) for p in chosen]
        decisions += [(p, "not_selected", reason) for p, reason in skipped]
        for p, dec, reason in decisions:
            db.upsert("selection_decisions", {"run_id": run.run_id, "repo_id": p["repo_id"], "strategy": info["strategy"],
                                              "decision": dec, "reason": reason, "stratum": stratum_of(p, scfg["strata"])},
                      ["run_id", "repo_id"])
        db.commit()
        out = {"run_id": run.run_id, **info, "selection_digest": selection_digest(cfg),
               "selected": [p["full_name"] for p in chosen], "counts": dict(Counter(d for _, d, _ in decisions))}
        if previous is not None and previous != {p["repo_id"] for p in chosen}:
            out["reselection"] = _reset_for_new_selection(cfg, db, {p["repo_id"] for p in chosen})
        write_json(cfg.data_dir / "processed" / "sample.json", out)
        write_csv(cfg.output_dir / "tables" / "repository_selection.csv",
                  [{**{k: p.get(k) for k in ("full_name", "pharo_status", "primary_domain", "size_category", "owner",
                                              "project_kind", "educational", "core_infrastructure", "stars", "forks",
                                              "contributors", "commit_count", "n_pharo_source_files", "has_tests",
                                              "coursework")},
                    "decision": d, "reason": r} for p, d, r in decisions])
        run.counts.update(out["counts"])
        log(f"Sample ({info['strategy']}, seed {seed}): {len(chosen)} selected of {len(eligible)} eligible; {out['counts']}")


def _reset_for_new_selection(cfg, db: Database, selected: set[int]) -> dict:
    """The set of analysed repositories changed. Everything derived from the old set would
    silently mix with the new one, so: parsed data of repositories that are no longer selected
    is removed from the database (their clones stay on disk) and the labelling folder is moved
    aside (never deleted) so that a new sample is drawn from the new corpus."""
    gone = [r[0] for r in db.query("SELECT DISTINCT repo_id FROM repository_snapshots") if r[0] not in selected]
    db.execute("CREATE TEMP TABLE IF NOT EXISTS _gone_repos (repo_id INTEGER PRIMARY KEY)")
    db.execute("DELETE FROM _gone_repos")
    db.executemany("INSERT INTO _gone_repos VALUES (?)", [(r,) for r in gone])
    methods = "SELECT method_id FROM methods WHERE repo_id IN (SELECT repo_id FROM _gone_repos)"
    snaps = "SELECT snapshot_id FROM repository_snapshots WHERE repo_id IN (SELECT repo_id FROM _gone_repos)"
    for t in ("message_sends", "api_references", "operation_evidence", "method_labels", "method_status", "method_descriptions"):
        db.execute(f"DELETE FROM {t} WHERE method_id IN ({methods})")
    for t in ("source_files", "packages", "package_dependencies", "classes", "methods"):
        db.execute(f"DELETE FROM {t} WHERE snapshot_id IN ({snaps})")
    db.execute("DELETE FROM repository_snapshots WHERE repo_id IN (SELECT repo_id FROM _gone_repos)")
    res = {"repositories_removed_from_corpus": len(gone), "labelling_moved_to": _reset_labelling(cfg, "reselection")}
    log(f"The selection of repositories changed: parsed data of {len(gone)} repositories that are no longer selected was "
        f"removed, and the previous labelling results were moved to {res['labelling_moved_to']}.")
    return res


def _reset_labelling(cfg, why: str) -> str | None:
    """Labelling results describe the previous set of repositories: the labelling folder is
    moved aside (never deleted) so that the next ./run.sh draws a new sample."""
    ldir = cfg.labelling_dir
    if not ldir.exists():
        return None
    dest = ldir.with_name(f"{ldir.name}.before-{why}-{now_iso()[:19].replace(':', '').replace('-', '')}")
    ldir.rename(dest)
    return str(dest)


# --------------------------------------------------------------------------- study corpus
def filter_corpus(cfg, args):
    """Filter 2: keep the repositories the labelling is based on (the study corpus)."""
    from .sampling import corpus_filter as cf
    if getattr(args, "max_repositories", None) is not None:
        cfg["corpus_filter"]["max_repositories"] = args.max_repositories
    fcfg = cfg["corpus_filter"]
    with open_db(cfg) as db, tracked_run(db, cf.STAGE, cfg, vars(args)) as run:
        if not db.scalar("SELECT COUNT(*) FROM repository_snapshots WHERE n_methods > 0"):
            raise StageBlocked("No mined repositories yet: run ./run.sh mine first.")
        previous = cf.study_repo_ids(db)
        log("Measuring the mined repositories (about a minute on the full database) …")
        metrics, mined = cf.repository_metrics(db, cfg)
        decisions, funnel, info = cf.select(metrics, fcfg)
        log(f"Mined repositories: {info['mined']}")
        for i, r in enumerate(funnel, 1):
            log(f"  {i:>2}. {r['requirement']:<62} -{r['removed']:<5} -> {r['remaining']}")
        chosen = {d["repo_id"] for d in decisions if d["decision"] == "included"}
        if not chosen:
            raise StageBlocked("No repository passes every criterion: relax corpus_filter in config.yaml "
                               "(results/tables/corpus_filter.csv is not written; nothing was changed).")
        for d in decisions:
            db.upsert("corpus_decisions", {"run_id": run.run_id, "repo_id": d["repo_id"], "decision": d["decision"],
                                           "reason": d["reason"], "failed_json": dumps(d["failed"]),
                                           "metrics_json": dumps({k: v for k, v in d.items()
                                                                  if k not in ("decision", "reason", "failed")})},
                      ["run_id", "repo_id"])
        db.commit()
        out = {"run_id": run.run_id, "filter_digest": cf.filter_digest(cfg), "criteria": dict(fcfg), "mining_date": mined,
               **info, "funnel": funnel,
               "selected": sorted(d["full_name"] for d in decisions if d["decision"] == "included")}
        if previous != chosen and cfg.labelling_dir.exists():
            # the sample and its labels describe another set of repositories
            out["labelling_moved_to"] = _reset_labelling(cfg, "filtering")
            log(f"The study corpus changed: the previous labelling results were moved to {out['labelling_moved_to']}.")
        elif previous == chosen:
            log("The study corpus is the same as before: the labelling results are kept.")
        write_json(cfg.data_dir / "processed" / "corpus.json", out)
        cols = ["full_name", "decision", "reason", "failed", "primary_domain", "license", "stars", "forks", "contributors",
                "commit_count", "created_at", "pushed_at", "development_days", "years_since_last_push", "baselines",
                "tonel_files", "classes", "methods", "non_test_methods", "test_methods", "test_method_share",
                "duplicated_method_share", "parse_failure_share", "size_category", "project_kind"]
        write_csv(cfg.output_dir / "tables" / "corpus_filter.csv",
                  [{**d, "failed": ";".join(d["failed"])} for d in sorted(decisions, key=lambda d: d["full_name"].lower())], cols)
        cf.write_report(cfg.output_dir / "reports" / "corpus_filter.md", cfg, funnel, decisions, info, mined)
        run.counts.update({k: v for k, v in info.items()})
        log(f"Study corpus: {info['selected']} repositories"
            + (f" (chosen for diversity among {info['passing']} that pass)" if info["capped"] else "")
            + f". Report: {cfg.output_dir / 'reports' / 'corpus_filter.md'}")


def _selected_repo_ids(db: Database, cfg, args) -> list[int]:
    if getattr(args, "repos", None):
        names = {n.lower() for n in args.repos}
        return [r["repo_id"] for r in db.query("SELECT repo_id, full_name FROM repositories") if r["full_name"].lower() in names]
    run = db.scalar("SELECT run_id FROM analysis_runs WHERE stage='sample' AND status='completed' ORDER BY started_at DESC LIMIT 1")
    if not run:
        raise StageBlocked("No completed sample run; run `pharo-miner sample` first (or pass --repos)")
    return [r[0] for r in db.query("SELECT repo_id FROM selection_decisions WHERE run_id=? AND decision='included'", (run,))]


# --------------------------------------------------------------------------- clone
def clone_stage(cfg, args):
    rcfg = cfg["repositories"]
    dest_root = cfg.resolve(rcfg["clone_dir"])
    with open_db(cfg) as db, tracked_run(db, "clone", cfg, vars(args)) as run:
        ids = _selected_repo_ids(db, cfg, args)
        counts = Counter()
        for i, rid in enumerate(ids, 1):
            r = db.query("SELECT * FROM repositories WHERE repo_id=?", (rid,))[0]
            local = find_local_clone(cfg, r["full_name"])
            if local is not None:
                res = reuse(local, r["head_commit_sha"])  # never fetches or deletes
                path = local
            elif args.local_only:
                res, path = {"status": "missing_local_clone", "commit_sha": None, "error": "--local-only and no clone found"}, None
            else:
                url = r["url"] or f"https://github.com/{r['full_name']}"
                path = dest_root / clone_dir_name(r["full_name"])
                res = clone(url + ".git", path, r["head_commit_sha"], int(rcfg.get("clone_depth", 1)),
                            int(rcfg.get("clone_timeout_seconds", 900)))
            counts[res["status"]] += 1
            existing = db.query("SELECT snapshot_id FROM repository_snapshots WHERE repo_id=? AND commit_sha IS ?", (rid, res.get("commit_sha")))
            row = {"repo_id": rid, "commit_sha": res.get("commit_sha"), "local_path": str(path) if path else None,
                   "clone_status": res["status"], "clone_error": res.get("error") or res.get("note"),
                   "cloned_at": now_iso(), "run_id": run.run_id}
            if existing:
                db.execute("UPDATE repository_snapshots SET local_path=?, clone_status=?, clone_error=?, cloned_at=?, run_id=? WHERE snapshot_id=?",
                           (row["local_path"], row["clone_status"], row["clone_error"], row["cloned_at"], run.run_id, existing[0][0]))
            else:
                db.upsert("repository_snapshots", row, ["repo_id", "commit_sha"])
            log(f"  [{i}/{len(ids)}] {r['full_name']}: {res['status']} {(res.get('commit_sha') or '')[:12]} {res.get('error') or ''}")
            db.commit()
        run.counts.update(counts)


# --------------------------------------------------------------------------- parse
def _store_extraction(db: Database, snapshot_id: int, repo_id: int, ex, features_by_key: dict, store_ast: bool):
    old = [r[0] for r in db.query("SELECT method_id FROM methods WHERE snapshot_id=?", (snapshot_id,))]
    for i in range(0, len(old), 900):
        chunk = old[i:i + 900]
        q = ",".join("?" * len(chunk))
        for t in ("message_sends", "api_references", "operation_evidence", "method_labels", "method_status", "method_descriptions"):
            db.execute(f"DELETE FROM {t} WHERE method_id IN ({q})", chunk)
    for t in ("methods", "classes", "packages", "source_files", "package_dependencies"):
        db.execute(f"DELETE FROM {t} WHERE snapshot_id=?", (snapshot_id,))
    db.executemany("INSERT INTO source_files VALUES (?,?,?,?,?,?,?,?,?)",
                   [(snapshot_id, f["path"], f["format"], f["status"], f["n_methods"], f["n_classes"], f["package"],
                     dumps(f["warnings"]), dumps(f["dialect_hints"])) for f in ex.files])
    for c in ex.classes:
        db.execute("INSERT INTO classes (snapshot_id, package, name, kind, superclass, external_root_superclass, instance_variables_json,"
                   " class_variables_json, class_instance_variables_json, traits, class_traits, category, source_path, line, is_test, comment)"
                   " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (snapshot_id, c["package"], c["name"], c["kind"], c["superclass"], c["external_root_superclass"],
                    dumps(c["instance_variables"]), dumps(c["class_variables"]), dumps(c["class_instance_variables"]),
                    str(c["traits"]) if c["traits"] else None, str(c["class_traits"]) if c["class_traits"] else None,
                    c["category"], c["source_path"], c["line"], int(c["is_test"]), c["comment"]))
    pk_methods, pk_tests = Counter(), Counter()
    for m in ex.methods:
        pk_methods[m["package"]] += 1
        pk_tests[m["package"]] += int(m["is_test"])
    for name, p in ex.packages.items():
        db.execute("INSERT OR IGNORE INTO packages (snapshot_id, name, paths_json, fingerprint, n_methods, is_test) VALUES (?,?,?,?,?,?)",
                   (snapshot_id, name, dumps(p["paths"]), None, pk_methods.get(name, 0),
                    int(pk_methods.get(name, 0) > 0 and pk_tests.get(name, 0) == pk_methods.get(name, 0))))
    for d in ex.baseline_dependencies:
        kind = {"external_project": "external_project", "package": "baseline_package", "requires": "baseline_requires"}[d["kind"]]
        db.execute("INSERT INTO package_dependencies VALUES (?,?,?,?,?)",
                   (snapshot_id, d.get("package") or d.get("from"), d["name"], kind, dumps(d)))
    for m in ex.methods:
        cur = db.execute(
            "INSERT OR REPLACE INTO methods (method_key, snapshot_id, repo_id, package, class_name, side, selector, protocol, source_path, line,"
            " container_format, source, source_hash, arguments_json, temporaries_json, block_arguments_json, literals_json, returns_json,"
            " assignments_json, blocks_json, referenced_globals_json, variables_read_json, pragmas_json, ast_json, parse_status,"
            " parse_errors_json, parser_backend, is_test, is_extension, in_trait, duplicate_of_method_id, features_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (m["method_key"], snapshot_id, repo_id, m["package"], m["class_name"], m["side"], m["selector"], m["protocol"],
             m["source_path"], m["line"], m["container_format"], m["source"], m["source_hash"], dumps(m["arguments"]),
             dumps(m.get("temporaries")), dumps(m.get("block_arguments")), dumps(m.get("literals")), dumps(m.get("returns")),
             dumps(m.get("assignments")), dumps(m.get("blocks")), dumps(m.get("referenced_globals")),
             dumps(m.get("variables_read")), dumps(m.get("pragmas")), dumps(m.get("ast")) if store_ast else None,
             m["parse_status"], dumps(m["parse_errors"]), m["parser_backend"], int(m["is_test"]), int(m["is_extension"]),
             int(m["in_trait"]), None, dumps(features_by_key.get(m["method_key"]))))
        mid = cur.lastrowid
        db.executemany("INSERT INTO message_sends (method_id, ordinal, selector, kind, line, receiver_kind, receiver_name,"
                       " receiver_resolution, resolved_target, resolution_evidence, in_cascade, block_depth, polymorphic_candidate)"
                       " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       [(mid, i, s["selector"], s["kind"], s["line"], s["receiver_kind"], s["receiver_name"], None, None, None,
                         int(s["in_cascade"]), s["block_depth"], 0) for i, s in enumerate(m.get("message_sends") or [])])


def parse(cfg, args):
    pcfg = cfg["parsing"]
    backend = make_backend(pcfg, cfg.root)
    parsed_dir = cfg.data_dir / "parsed"
    with open_db(cfg) as db, tracked_run(db, "parse", cfg, vars(args)) as run:
        ids = set(_selected_repo_ids(db, cfg, args))
        snaps = [s for s in db.query("SELECT s.*, r.full_name FROM repository_snapshots s JOIN repositories r ON r.repo_id=s.repo_id"
                                     " WHERE s.local_path IS NOT NULL AND s.snapshot_id IN (SELECT MAX(snapshot_id) FROM"
                                     " repository_snapshots GROUP BY repo_id)") if s["repo_id"] in ids]
        totals = Counter()
        for i, s in enumerate(snaps, 1):
            root = Path(s["local_path"])
            if not root.exists():
                log(f"  [{i}/{len(snaps)}] {s['full_name']}: local path missing")
                totals["missing_path"] += 1
                continue
            ex = extract_repository(root, s["full_name"], s["commit_sha"], backend, include_tests=pcfg["include_tests"],
                                    include_extensions=pcfg["include_extension_methods"], export_ast=pcfg["export_ast"],
                                    exclude_globs=pcfg.get("exclude_globs") or [])
            cls_info = {c["name"]: c for c in ex.classes if c["kind"] != "extension"}
            feats = {m["method_key"]: compute_features(m, cls_info.get(m["class_name"])) for m in ex.methods if m["parse_status"] == "ok"}
            _store_extraction(db, s["snapshot_id"], s["repo_id"], ex, feats, bool(pcfg.get("store_ast_in_db", False)))
            n_fail = sum(m["parse_status"] != "ok" for m in ex.methods)
            n_lines = sum(m["source"].count("\n") + 1 for m in ex.methods)
            db.execute("UPDATE repository_snapshots SET parsed_at=?, parser_backend=?, n_files=?, n_packages=?, n_classes=?, n_methods=?,"
                       " n_parse_failures=?, n_lines=? WHERE snapshot_id=?",
                       (now_iso(), ex.parser_backend, sum(1 for f in ex.files if f["status"] in ("ok", "partial", "failed")),
                        len(ex.packages), sum(1 for c in ex.classes if c["kind"] != "extension"), len(ex.methods), n_fail, n_lines,
                        s["snapshot_id"]))
            db.commit()
            out = parsed_dir / f"{clone_dir_name(s['full_name'])}.jsonl.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(out, "wt", encoding="utf-8") as f:
                f.write(json.dumps({"record": "snapshot", "repository": s["full_name"], "commit_sha": s["commit_sha"],
                                    "parser_backend": ex.parser_backend, "files": ex.files, "classes": ex.classes,
                                    "baseline_dependencies": ex.baseline_dependencies}, default=str) + "\n")
                for m in ex.methods:
                    f.write(json.dumps({"record": "method", **m, "features": feats.get(m["method_key"])}, default=str) + "\n")
            unsupported = sum(1 for f in ex.files if f["status"] in ("unsupported", "failed", "partial"))
            totals["methods"] += len(ex.methods)
            totals["parse_failures"] += n_fail
            totals["unsupported_or_failed_files"] += unsupported
            log(f"  [{i}/{len(snaps)}] {s['full_name']}: {len(ex.methods)} methods, {n_fail} parse failures, "
                f"{unsupported} unsupported/failed files")
        run.counts.update(totals)


# --------------------------------------------------------------------------- analyze (API usage, resolution, duplicates)
def analyze(cfg, args):
    matcher = FrameworkMatcher(load_yaml(cfg.resolve(cfg["classification"]["frameworks"])))
    with open_db(cfg) as db, tracked_run(db, "analyze", cfg, vars(args)) as run:
        latest = [r[0] for r in db.query("SELECT MAX(snapshot_id) FROM repository_snapshots WHERE n_methods > 0 GROUP BY repo_id")]
        totals = Counter()
        for sid in latest:
            classes = [dict(r) for r in db.query("SELECT name, kind, superclass FROM classes WHERE snapshot_id=?", (sid,))]
            methods = [dict(r) for r in db.query("SELECT method_id, class_name, side, selector, parse_status, referenced_globals_json"
                                                  " FROM methods WHERE snapshot_id=?", (sid,))]
            index = RepositoryIndex(classes, methods)
            by_id = {m["method_id"]: m for m in methods}
            sends = db.query("SELECT ms.method_id, ms.ordinal, ms.selector, ms.receiver_kind, ms.receiver_name FROM message_sends ms"
                             " JOIN methods m ON m.method_id=ms.method_id WHERE m.snapshot_id=?", (sid,))
            updates = []
            for s in sends:
                res = resolve_send(dict(s), by_id[s["method_id"]], index)
                updates.append((res["receiver_resolution"], res["resolved_target"], res["resolution_evidence"],
                                int(res["polymorphic_candidate"]), s["method_id"], s["ordinal"]))
                totals["resolved" if res["resolved_target"] else "unresolved"] += 1
            db.executemany("UPDATE message_sends SET receiver_resolution=?, resolved_target=?, resolution_evidence=?,"
                           " polymorphic_candidate=? WHERE method_id=? AND ordinal=?", updates)
            mids = [m["method_id"] for m in methods]
            for i in range(0, len(mids), 900):
                chunk = mids[i:i + 900]
                db.execute(f"DELETE FROM api_references WHERE method_id IN ({','.join('?' * len(chunk))})", chunk)
            refs = []
            for m in methods:
                m["referenced_globals"] = loads(m["referenced_globals_json"], []) or []
                for a in api_references(m, index, matcher):
                    refs.append((m["method_id"], a["name"], a["kind"], int(a["defined_in_repository"]), a["framework"], a["occurrences"]))
            db.executemany("INSERT OR REPLACE INTO api_references VALUES (?,?,?,?,?,?)", refs)
            # internal package dependencies inferred from references to classes defined in other packages
            pkg_of = {r["name"]: r["package"] for r in db.query("SELECT name, package FROM classes WHERE snapshot_id=? AND kind!='extension'", (sid,))}
            edges = Counter()
            for r in db.query("SELECT m.package, a.name FROM api_references a JOIN methods m ON m.method_id=a.method_id"
                              " WHERE m.snapshot_id=? AND a.defined_in_repository=1", (sid,)):
                tgt = pkg_of.get(r["name"])
                if tgt and r["package"] and tgt != r["package"]:
                    edges[(r["package"], tgt)] += 1
            db.execute("DELETE FROM package_dependencies WHERE snapshot_id=? AND kind='class_reference'", (sid,))
            db.executemany("INSERT INTO package_dependencies VALUES (?,?,?,?,?)",
                           [(sid, a, b, "class_reference", dumps({"references": n})) for (a, b), n in edges.items()])
            db.commit()
            totals["snapshots"] += 1
        run.counts.update(totals)
        run.counts["content_duplicates"] = content_duplicates(db, run.run_id)
        log(f"Analysis: {dict(totals)}; duplicates: {run.counts['content_duplicates']}")
def snapshot(cfg, args):
    from . import snapshot as snap
    scfg = cfg["snapshot"]
    repo = args.repo or scfg["repo"]
    token = os.getenv(scfg.get("token_env") or "HF_TOKEN")
    if args.action == "state":
        print(snap.local_state(cfg))
        return
    try:
        if args.action == "info":
            res = snap.info(cfg, repo, args.revision or scfg["revision"])
        elif args.action == "pull":
            res = snap.pull(cfg, repo, args.revision or scfg["revision"], force=args.force, token=token)
        else:
            if snap.local_state(cfg) == "none":
                raise StageBlocked("No local results to push (run ./run.sh mine first).")
            res = snap.push(cfg, repo, include_api_cache=scfg["include_api_cache"] and not args.no_api_cache,
                            include_clones=scfg["include_clones"] or args.with_clones, message=args.message, token=token,
                            dry_run=args.dry_run)
    except snap.SnapshotError as e:
        raise StageBlocked(str(e)) from e
    log(json.dumps(res, indent=1, default=str))


def status(cfg, args):
    with open_db(cfg) as db:
        for t in ("repositories", "repository_snapshots", "methods", "message_sends", "domain_labels", "corpus_decisions"):
            log(f"{t:24s} {db.scalar(f'SELECT COUNT(*) FROM {t}')}")
        log("pharo_status: " + ", ".join(f"{r[0]}={r[1]}" for r in db.query(
            "SELECT pharo_status, COUNT(*) FROM repositories GROUP BY pharo_status")))
        log("\nRecent runs:")
        for r in db.query("SELECT stage, status, started_at, counts_json FROM analysis_runs ORDER BY started_at DESC LIMIT 12"):
            log(f"  {r['started_at']} {r['stage']:20s} {r['status']:10s} {(r['counts_json'] or '')[:110]}")



# --------------------------------------------------------------------------- labelling
def label(cfg, args):
    """Step 3, inductive phase: sample size, initial sample, saturation loop, codebook draft."""
    from .labelling import export as lexport, sampling as ls, saturation
    from .labelling.coding import PROMPT_VERSION
    from .labelling.models import ModelError, OllamaEmbedder, make_model
    from .sampling.corpus_filter import STAGE as FILTER_STAGE
    s = dict(cfg["labelling"])
    for k in ("model", "sample_size", "confidence", "margin_of_error", "batch_size", "patience",
              "max_rounds", "min_support", "min_share", "embedding_model"):
        if getattr(args, k, None) is not None:
            s[k] = getattr(args, k)
    for k in ("embedding_model",):
        if str(s.get(k) or "").lower() in ("", "none", "no", "off"):
            s[k] = None
    with open_db(cfg) as db, tracked_run(db, "label", cfg, vars(args)) as run:
        study = study_repo_ids(db)
        if not study:
            raise StageBlocked("The study corpus is not filtered yet: run ./run.sh filter first.")
        pop = ls.population(db, study, bool(s["include_tests"]))
        N = sum(len(v) for v in pop.values())
        tests = db.scalar("SELECT COUNT(*) FROM methods WHERE parse_status='ok' AND duplicate_of_method_id IS NULL AND is_test=1"
                          f" AND snapshot_id IN ({ls.latest_snapshots_sql()}) AND repo_id IN ({','.join(map(str, study))})")
        calc = ls.cochran(N, float(s["confidence"]), float(s["margin_of_error"]), float(s["proportion"]))
        quiet = ls.quiet_run_length(float(s.get("min_share") or 0), float(s["confidence"]))
        s["patience_rule"] = "set" if s.get("patience") else "derived"
        if not s.get("patience"):
            s["patience"] = max(3, math.ceil(quiet / int(s["batch_size"]))) if quiet else 3
        s["quiet_methods_needed"] = quiet
        n = int(s["sample_size"] or calc["n"])
        s.update(sample_size=n, prompt_version=PROMPT_VERSION, corpus_run_id=db.scalar(
            "SELECT run_id FROM analysis_runs WHERE stage=? AND status='completed' ORDER BY started_at DESC LIMIT 1",
            (FILTER_STAGE,)))
        log(f"Population: {N:,} methods in {len(pop)} repositories. Sample size (confidence {calc['confidence']:.0%}, "
            f"margin ±{calc['margin_of_error']:.1%}): {calc['n']}" + (f"; using {n} as set" if n != calc["n"] else "") + ".")
        out = cfg.labelling_dir
        saturation.prepare(out, s, log)
        try:
            model = make_model(s["model"], s)
            embed = OllamaEmbedder(s["embedding_model"]) if s["embedding_model"] else None
        except ModelError as e:
            raise StageBlocked(str(e)) from e
        log(f"Labelling model: {model.name}; names merged by meaning with: {s['embedding_model'] or 'off'}")
        plan = {"created_at": now_iso(), "corpus_run_id": s["corpus_run_id"],
                "population": {"repositories": len(pop), "methods": N, "test_methods": tests,
                               "include_tests": bool(s["include_tests"])},
                "sample_size": calc, "initial_sample_size": n,
                "initial_margin": round(ls.margin_for(n, N, calc["confidence"], calc["proportion"]), 4),
                "alternatives": [ls.cochran(N, c, e) for c, e in ((0.90, 0.05), (0.95, 0.05), (0.95, 0.03), (0.99, 0.05))],
                "model": {"name": model.name, "description": model.describe()},
                "settings": s, "settings_digest": s.get("settings_digest") or saturation._settings_digest(s)}
        write_json(out / "plan.json", plan)

        def records(ids):
            return ls.method_records(db, list(ids))
        lrun = saturation.Run(out, s, pop, records, log, embed=embed)
        result = lrun.run(model)
        log(f"Labelling: {result['methods_labelled']} methods, {result['categories']} categories, "
            f"stop: {result['stop_reason']}.")
        plan["result"] = result
        write_json(out / "plan.json", plan)
        summ = lexport.export(lrun, plan, out, cfg.output_dir / "figures", log)
        run.counts.update({k: result[k] for k in ("methods_labelled", "categories", "stop_reason")})
        log(f"Taxonomy (inductive): {out / 'taxonomy.csv'}")
        from .labelling.classify import draft_codebook
        draft_codebook(summ, list(lrun.labels.values()), model, out, log)


def classify(cfg, args):
    """Step 4, deductive phase: several models classify a fresh sample with the codebook;
    agreement, prompt sensitivity, reliability gate, stratified expert sample."""
    from .labelling import classify as lc, sampling as ls
    from .labelling.models import ModelError, make_model
    s = dict(cfg["labelling"])
    for k in ("classify_models", "min_alpha", "gold_size", "classify_sample_size"):
        if getattr(args, k, None) is not None:
            s[k] = getattr(args, k)
    if isinstance(s["classify_models"], str):
        s["classify_models"] = s["classify_models"].split()
    out = cfg.labelling_dir
    if not (out / "plan.json").exists():
        raise StageBlocked("The inductive phase has not run yet: ./run.sh label first.")
    plan = json.loads((out / "plan.json").read_text())
    s["max_method_lines"] = plan["settings"].get("max_method_lines", s["max_method_lines"])
    s["max_method_chars"] = plan["settings"].get("max_method_chars", s.get("max_method_chars"))
    if not s.get("classify_sample_size"):
        s["classify_sample_size"] = plan["sample_size"]["n"]
    with open_db(cfg) as db, tracked_run(db, "classify", cfg, vars(args)) as run:
        pop = ls.population(db, study_repo_ids(db), bool(plan["settings"]["include_tests"]))
        inductive = {json.loads(line)["method_id"] for line in (out / "labels.jsonl").read_text().splitlines() if line}
        try:
            models = [make_model(m, s) for m in s["classify_models"]]
        except ModelError as e:
            raise StageBlocked(str(e)) from e
        cls = lc.Classification(out, s, pop, inductive, lambda ids: ls.method_records(db, list(ids)), log,
                                [m.name for m in models])
        log(f"Codebook ({cls.codebook_kind}): {len(cls.codebook)} categories; {len(cls.sample)} new methods; "
            f"models: {', '.join(m.name for m in models)}")
        res = cls.run(models)
        log(f"Krippendorff's alpha {res['krippendorff_alpha']}, Fleiss' kappa {res['fleiss_kappa']}, "
            f"Gwet's AC1 {res['gwet_ac1']}; levels {res['levels']}; self-check kappa {res['selfcheck']['cohen_kappa']}")
        gate = cls.out / "NOT_RELIABLE.md"
        ev = out / "expert_validation"
        if not res["reliable"]:
            gate.write_text(lc.NOT_RELIABLE.format(k=len(models), n=res["sample"], alpha=res["krippendorff_alpha"],
                                                   t=res["min_alpha"], ac1=res["gwet_ac1"]), encoding="utf-8")
            log(f"Codebook NOT reliable (alpha < {res['min_alpha']}): no expert files; see {gate}")
        else:
            if gate.exists():
                gate.unlink()
            with (out / "labels.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            inductive_rows = [{"category": r["category"], "signature": r["method"], "summary": r["summary"]} for r in rows]
            res["experts"] = lc.expert_pack(cls, res, inductive_rows, ev, log)
            write_json(cls.out / "agreement.json", res)
        run.counts.update({"alpha": res["krippendorff_alpha"], "reliable": res["reliable"]})


def validate(cfg, args):
    """Step 5: read the experts' answers back: accuracy, experts' agreement, corrected shares."""
    from .labelling import validation
    with open_db(cfg) as db, tracked_run(db, "validate", cfg, vars(args)) as run:
        res = validation.validate(cfg.labelling_dir, log)
        if res:
            run.counts.update({"experts": len(res["experts"]), "final_categories": len(res["final_taxonomy"])})


def report(cfg, args):
    from .labelling.report import write
    with open_db(cfg) as db:
        p = write(cfg, db)
    log(f"Report: {p}")
