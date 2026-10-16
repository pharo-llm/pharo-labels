"""Pipeline stages. Each stage reads its inputs from the evidence database or
from earlier stages' files, writes its own outputs, and records a run with
provenance, so any stage can be re-run independently (e.g. re-classify without
re-downloading)."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import json
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
from .analysis.operation_rules import RuleClassifier
from .analysis.structural_features import compute_features
from .analysis import method_descriptions as md
from .sampling.corpus_filter import study_repo_ids
from .sampling.repository_filters import eligibility, repository_profiles, selection_digest
from .sampling.diversity_sampling import sample as draw_sample, stratum_of
from .storage.database import Database, dumps, loads
from .storage.provenance import now_iso, tracked_run, write_json, write_jsonl, read_jsonl
from .taxonomy.category_manager import load_taxonomy, store_taxonomy


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
        if args.semantic or dcfg.get("semantic_analysis"):
            run.counts["semantic"] = _semantic_domains(cfg, db, inputs, spec, run, args.semantic_limit,
                                                      getattr(args, "model", None))
        log(f"Domain labels: {dict(counts.most_common())}")


def _semantic_domains(cfg, db, inputs, spec, run, limit, model=None):
    from .analysis.semantic_classifier import LLMClient, SemanticUnavailable, resolve_model, classify_repository_domain
    try:
        client = LLMClient(resolve_model(cfg, model))
        log(f"Semantic analysis with {client.profile} ({client.provider}:{client.model})")
    except SemanticUnavailable as e:
        raise StageBlocked(str(e))
    n = 0
    targets = [rid for rid in inputs if db.scalar("SELECT pharo_status FROM repositories WHERE repo_id=?", (rid,))
               in ("confirmed_pharo", "probable_pharo", "uncertain")]
    for rid in targets[: limit or len(targets)]:
        rec = classify_repository_domain(client, inputs[rid], spec["domains"])
        for lab in rec["labels"]:
            db.upsert("domain_labels", {"repo_id": rid, "label": lab["label"], "method": "semantic_classifier", "score": None,
                                        "score_kind": "none", "evidence_json": dumps(lab["evidence"]), "review_status": "pending",
                                        "model": f"{client.profile}={client.provider}:{rec['model']}", "prompt_version": rec["prompt_version"], "run_id": run.run_id,
                                        "created_at": rec["created_at"]}, ["repo_id", "label", "method"])
        n += 1
    db.commit()
    return n


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
    is removed from the database (their clones stay on disk), labels given to the old annotation
    sample are removed, and the validation folder is moved aside (never deleted) so that a new
    annotation sample is drawn from the new corpus."""
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
    res = {"repositories_removed_from_corpus": len(gone), "validation_moved_to": _reset_validation(cfg, db, "reselection")}
    log(f"The selection of repositories changed: parsed data of {len(gone)} repositories that are no longer selected was "
        f"removed, and the previous annotation sample and its labels were moved to {res['validation_moved_to']}.")
    return res


def _reset_validation(cfg, db: Database, why: str) -> str | None:
    """Labels given to the previous annotation sample are removed from the database and the
    validation folder is moved aside (never deleted), so that a new sample is drawn."""
    db.execute("DELETE FROM annotation_decisions")
    db.execute("DELETE FROM method_labels WHERE label_source != 'rule_based'")
    db.execute("DELETE FROM method_status WHERE label_source != 'rule_based'")
    db.commit()
    vdir = cfg.validation_dir
    if not vdir.exists():
        return None
    dest = vdir.with_name(f"{vdir.name}.before-{why}-{now_iso()[:19].replace(':', '').replace('-', '')}")
    vdir.rename(dest)
    return str(dest)


# --------------------------------------------------------------------------- study corpus
def filter_corpus(cfg, args):
    """Keep the repositories that steps 2, 3 and 4 are based on (docs/1-2-filtering.md)."""
    from .sampling import corpus_filter as cf
    if getattr(args, "max_repositories", None) is not None:
        cfg["corpus_filter"]["max_repositories"] = args.max_repositories
    fcfg = cfg["corpus_filter"]
    with open_db(cfg) as db, tracked_run(db, cf.STAGE, cfg, vars(args)) as run:
        if not db.scalar("SELECT COUNT(*) FROM repository_snapshots WHERE n_methods > 0"):
            raise StageBlocked("No mined repositories yet: run ./1-mining.sh first.")
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
        labelled = bool(db.scalar("SELECT COUNT(*) FROM method_labels WHERE label_source != 'rule_based'"))
        if previous != chosen and (previous is not None or labelled or (cfg.validation_dir / "annotation_sample.csv").exists()):
            # the annotation sample and its labels describe another set of repositories
            out["validation_moved_to"] = _reset_validation(cfg, db, "filtering")
            log(f"The study corpus changed: the previous annotation sample and its labels were moved to "
                f"{out['validation_moved_to']}. Steps 2, 3 and 4 must be run again.")
        elif previous == chosen:
            log("The study corpus is the same as before: the annotation sample and its labels are kept.")
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


# --------------------------------------------------------------------------- operations
def _method_inputs(db: Database, snapshot_ids: list[int]):
    for sid in snapshot_ids:
        sends = defaultdict(list)
        for s in db.query("SELECT ms.method_id, ms.selector, ms.line, ms.receiver_kind FROM message_sends ms JOIN methods m"
                          " ON m.method_id=ms.method_id WHERE m.snapshot_id=?", (sid,)):
            sends[s["method_id"]].append({"selector": s["selector"], "line": s["line"], "receiver_kind": s["receiver_kind"]})
        for m in db.query("SELECT method_id, repo_id, class_name, side, selector, line, is_test, referenced_globals_json, features_json"
                          " FROM methods WHERE snapshot_id=? AND parse_status='ok'", (sid,)):
            yield {"method_id": m["method_id"], "repo_id": m["repo_id"], "class_name": m["class_name"], "side": m["side"],
                   "selector": m["selector"], "line": m["line"], "is_test": bool(m["is_test"]),
                   "message_sends": sends.get(m["method_id"], []),
                   "referenced_globals": loads(m["referenced_globals_json"], []) or []}, loads(m["features_json"], {}) or {}


def discover_operations(cfg, args):
    ocfg = cfg["classification"]["operations"]
    tax = load_taxonomy(cfg.resolve(ocfg["taxonomy"]))
    clf = RuleClassifier(tax, int(ocfg.get("multiple_operations_threshold", 4)), int(ocfg.get("insufficient_context_max_nodes", 3)))
    with open_db(cfg) as db, tracked_run(db, "discover-operations", cfg, vars(args)) as run:
        store_taxonomy(db, tax)
        latest = [r[0] for r in db.query("SELECT MAX(snapshot_id) FROM repository_snapshots WHERE n_methods > 0 GROUP BY repo_id")]
        db.execute("DELETE FROM operation_evidence WHERE classification='rule_based'")
        db.execute("DELETE FROM method_labels WHERE label_source='rule_based'")
        db.execute("DELETE FROM method_status WHERE label_source='rule_based'")
        status_counts, label_counts = Counter(), Counter()
        unknown_pool = []
        ev_rows, lab_rows, st_rows = [], [], []

        def flush():
            db.executemany("INSERT INTO operation_evidence (method_id, operation_id, evidence_kind, evidence_json, strength, classification,"
                           " taxonomy_version, review_status, run_id) VALUES (?,?,?,?,?,?,?,?,?)", ev_rows)
            db.executemany("INSERT OR REPLACE INTO method_labels VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", lab_rows)
            db.executemany("INSERT OR REPLACE INTO method_status VALUES (?,?,?,?,?,?)", st_rows)
            db.commit()
            ev_rows.clear(), lab_rows.clear(), st_rows.clear()

        ts = now_iso()
        for m, f in _method_inputs(db, latest):
            res = clf.classify(m, f)
            status_counts[res.status] += 1
            st_rows.append((m["method_id"], "rule_based", res.status, len(res.labels), clf.version, run.run_id))
            for op, lab in res.labels.items():
                label_counts[op] += 1
                lab_rows.append((m["method_id"], op, "rule_based", res.status, None, None, clf.version, "pending", None,
                                 f"rules:{clf.version}", ts, run.run_id))
                for e in lab["evidence"]:
                    ev_rows.append((m["method_id"], op, e["kind"], dumps(e), e.get("rule_strength"), "rule_based", clf.version,
                                    "pending", run.run_id))
            if res.status in ("unknown", "ambiguous"):
                unknown_pool.append({"method_id": m["method_id"], "repo_id": m["repo_id"], "label": res.status,
                                     "selectors": {s["selector"] for s in m["message_sends"]}})
            if len(ev_rows) > 20000:
                flush()
        flush()
        # open discovery
        covered = clf.covered_selectors()
        vocab = md.uncovered_vocabulary(
            ((r["repo_id"], r["method_id"], r["selector"], r["class_name"], r["msel"]) for r in db.query(
                "SELECT m.repo_id, m.method_id, ms.selector, m.class_name, m.selector msel FROM message_sends ms"
                " JOIN methods m ON m.method_id=ms.method_id")), covered)
        groups = md.signature_groups(unknown_pool, len(latest))
        discovery = {"run_id": run.run_id, "taxonomy_version": clf.version, "status_counts": dict(status_counts),
                     "uncovered_vocabulary": vocab, "signature_groups": groups, "description_groups": []}
        write_csv(cfg.output_dir / "tables" / "uncovered_selectors.csv", vocab)
        write_csv(cfg.output_dir / "tables" / "candidate_operation_groups.csv", groups)
        if args.semantic or ocfg.get("semantic_analysis"):
            discovery["description_groups"] = _semantic_methods(cfg, db, tax, run, args.semantic_limit,
                                                                  getattr(args, "model", None))
            write_csv(cfg.output_dir / "tables" / "description_groups.csv", discovery["description_groups"])
        write_json(cfg.data_dir / "processed" / "discovery.json", discovery)
        run.counts.update({"status": dict(status_counts), "labels": len(label_counts)})
        log(f"Method statuses: {dict(status_counts)}")
        log(f"Uncovered selectors (>=2 repos): {len(vocab)}; candidate groups: {len(groups)}")


def _semantic_methods(cfg, db, tax, run, limit, model=None):
    from .analysis.semantic_classifier import LLMClient, SemanticUnavailable, resolve_model, describe_method
    try:
        client = LLMClient(resolve_model(cfg, model))
        log(f"Semantic analysis with {client.profile} ({client.provider}:{client.model})")
    except SemanticUnavailable as e:
        raise StageBlocked(str(e))
    domains = {}
    for r in db.query("SELECT repo_id, label FROM domain_labels WHERE method='deterministic_rules' ORDER BY score DESC"):
        domains.setdefault(r["repo_id"], r["label"])
    cands = [{"method_id": r["method_id"], "repo_id": r["repo_id"], "status": r["status"], "domain": domains.get(r["repo_id"])}
             for r in db.query("SELECT m.method_id, m.repo_id, ms.status FROM methods m JOIN method_status ms ON ms.method_id=m.method_id"
                               " AND ms.label_source='rule_based' WHERE m.duplicate_of_method_id IS NULL AND m.is_test=0")]
    n = limit or int(cfg["semantic"]["max_methods"])
    chosen = md.select_methods_for_semantic_analysis(cands, n, int(cfg["sampling"]["random_seed"]))
    described = []
    for i, c in enumerate(chosen, 1):
        row = db.query("SELECT * FROM methods WHERE method_id=?", (c["method_id"],))[0]
        m = {"class_name": row["class_name"], "side": row["side"], "selector": row["selector"], "source": row["source"],
             "message_sends": [{"selector": s["selector"]} for s in db.query("SELECT selector FROM message_sends WHERE method_id=?", (c["method_id"],))],
             "referenced_globals": loads(row["referenced_globals_json"], []) or []}
        rec = describe_method(client, m, loads(row["features_json"], {}) or {}, c["domain"], tax["categories"])
        db.upsert("method_descriptions", {
            "method_id": c["method_id"], "summary": rec["summary"], "proposed_operations_json": dumps(rec["proposals"]),
            "accepted_operations_json": dumps(rec["accepted"]), "rejected_json": dumps(rec["rejected"]),
            "evidence_json": dumps(rec.get("evidence")), "uncertainty_json": dumps(rec["uncertainty"]),
            "model": f"{client.profile}={client.provider}:{rec['model']}",
            "prompt_version": rec["prompt_version"], "created_at": rec["created_at"], "raw_response": rec["raw_response"],
            "run_id": run.run_id}, ["method_id", "model", "prompt_version"])
        for a in rec["accepted"]:
            db.upsert("method_labels", {"method_id": c["method_id"], "operation_id": a["id"], "label_source": "semantic_classifier",
                                        "label_status": "proposed", "score": None, "score_kind": None, "taxonomy_version": str(tax["version"]),
                                        "review_status": "pending", "model": f"{client.profile}={client.provider}:{rec['model']}", "prompt_version": rec["prompt_version"],
                                        "created_at": rec["created_at"], "run_id": run.run_id}, ["method_id", "operation_id", "label_source"])
        if rec["summary"]:
            described.append({"method_id": c["method_id"], "repo_id": c["repo_id"], "summary": rec["summary"]})
        if i % 10 == 0:
            db.commit()
            log(f"  described {i}/{len(chosen)}")
    db.commit()
    return md.description_groups(described)


# --------------------------------------------------------------------------- validation
def export_annotations(cfg, args):
    from .sampling.corpus_filter import repositories_to_label
    from .validation.annotation_export import export, select_proportional, select_repositories, select_sample
    vcfg = cfg["validation"]
    with open_db(cfg) as db, tracked_run(db, "export-annotations", cfg, vars(args)) as run:
        tax_rows = [dict(r) for r in db.query("SELECT category_id, parent_id, level, definition FROM taxonomy_categories"
                                              " WHERE taxonomy_version=(SELECT MAX(taxonomy_version) FROM taxonomy_categories)")]
        size = args.size or vcfg["annotation_sample_size"]
        n_repos = getattr(args, "label_repositories", None)
        n_repos = int(vcfg.get("label_repositories") or 0) if n_repos is None else n_repos
        strategy = getattr(args, "sample_strategy", None) or vcfg.get("sample_strategy") or "category_balanced"
        explicit = bool(args.size) or getattr(args, "label_repositories", None) is not None \
            or getattr(args, "sample_strategy", None) is not None
        study, chosen, repos = study_repo_ids(db), None, []
        if n_repos and study is None and not explicit:
            n_repos = 0  # mining, before the study corpus exists: a first sample, replaced by ./1-2-filtering.sh
        if n_repos:
            # every method of n repositories of the study corpus, instead of a sample of methods
            if study is None:
                raise StageBlocked("Labelling whole repositories needs the study corpus: run ./1-2-filtering.sh "
                                   "(`pharo-miner filter-corpus`) first.")
            repos = repositories_to_label(db, n_repos, cfg["corpus_filter"])
            chosen = select_repositories(db, {r["repo_id"] for r in repos})
        elif strategy == "proportional":
            # a sample that represents the whole study corpus: every repository, in proportion to its size
            chosen = select_proportional(db, size, vcfg["random_seed"], repo_ids=study)
        existing = cfg.validation_dir / "annotation_sample.csv"
        if explicit and existing.exists():
            with existing.open(encoding="utf-8", newline="") as f:
                have = {int(r["method_id"]) for r in csv.DictReader(f)}
            want = chosen if chosen is not None else select_sample(db, size, vcfg["random_seed"], vcfg["max_per_repository"],
                                                                   repo_ids=study)
            if have != {c["method_id"] for c in want}:  # other methods are asked for: the old ones are moved aside
                log(f"The methods to label changed ({len(have)} before, {len(want)} now): the previous ones and their "
                    f"labels were moved to {_reset_validation(cfg, db, 'resampling')}.")
        res = export(db, cfg.validation_dir, size, vcfg["random_seed"],
                     vcfg["max_per_repository"], vcfg["annotators"], float(vcfg["overlap_fraction"]),
                     cfg.resolve("taxonomy/annotation_guidelines.md"), tax_rows, repo_ids=study, chosen=chosen)
        if repos and res["status"] == "exported":
            per = Counter(c["repo_id"] for c in chosen)
            write_csv(cfg.validation_dir / "labelled_repositories.csv",
                      [{"repository": r["full_name"], "domain": r["primary_domain"], "methods_to_label": per[r["repo_id"]]}
                       for r in repos])
            res["repositories"] = len(repos)
            log(f"Every method of {len(repos)} repositories is to be labelled: {len(chosen)} methods "
                f"(list: {cfg.validation_dir / 'labelled_repositories.csv'}).")
        elif strategy == "proportional" and not n_repos and res["status"] == "exported":
            # the allocation, for the paper: each repository's methods in the corpus and in the sample
            sampled = Counter(c["repo_id"] for c in chosen)
            in_corpus = Counter(r["repo_id"] for r in db.query(
                "SELECT m.repo_id FROM methods m"
                " JOIN method_status ms ON ms.method_id=m.method_id AND ms.label_source='rule_based'"
                " WHERE m.parse_status='ok' AND m.duplicate_of_method_id IS NULL"
                " AND m.snapshot_id IN (SELECT MAX(snapshot_id) FROM repository_snapshots GROUP BY repo_id)")
                if study is None or r["repo_id"] in study)
            names = {r["repo_id"]: r["full_name"] for r in db.query("SELECT repo_id, full_name FROM repositories")}
            write_csv(cfg.validation_dir / "sample_allocation.csv",
                      [{"repository": names[k], "methods_in_corpus": n, "methods_sampled": sampled[k]}
                       for k, n in sorted(in_corpus.items(), key=lambda kv: -kv[1])])
            res["repositories"] = len(sampled)
            log(f"Proportional sample: {len(chosen)} methods from {len(sampled)} repositories "
                f"(per repository: {cfg.validation_dir / 'sample_allocation.csv'}).")
        run.counts.update({k: v for k, v in res.items() if k != "strata"})
        log(json.dumps(res, indent=1))


def import_annotations(cfg, args):
    from .validation.adjudication import import_annotations as imp, import_developer_feedback
    with open_db(cfg) as db, tracked_run(db, "import-annotations", cfg, vars(args)) as run:
        cats = {r[0] for r in db.query("SELECT category_id FROM taxonomy_categories")}
        res = imp(db, cfg.validation_dir, cats)
        fb = cfg.validation_dir / "developer_feedback.csv"
        if fb.exists():
            res["developer_feedback"] = import_developer_feedback(db, fb)
        write_json(cfg.validation_dir / "agreement_report.json", res)
        run.counts.update({k: v for k, v in res.items() if k != "agreement"})
        log(json.dumps({k: v for k, v in res.items() if k != "agreement"}, indent=1))
        if res.get("agreement"):
            a = res["agreement"]
            log(f"Agreement: items={a['items']}, multiply annotated={a['items_multiply_annotated']}, mean Jaccard={a['mean_jaccard']}")


def has_validated_labels(db: Database) -> bool:
    return bool(db.scalar("SELECT COUNT(*) FROM method_labels WHERE label_source='human_adjudicated'"))


def label_basis(cfg, db: Database) -> str | None:
    """Which reviewed labels the validation-dependent stages use.

    ``validation.label_source``: ``human`` (adjudicated human annotations only),
    ``llm`` (the LLM annotation panel only) or ``auto`` (human when available,
    otherwise LLM). Returns the ``method_labels.label_source`` value, or None."""
    pref = cfg["validation"].get("label_source", "auto")
    human = has_validated_labels(db)
    llm = bool(db.scalar("SELECT COUNT(*) FROM method_labels WHERE label_source='llm_adjudicated'"))
    if pref == "human":
        return "human_adjudicated" if human else None
    if pref == "llm":
        return "llm_adjudicated" if llm else None
    return "human_adjudicated" if human else ("llm_adjudicated" if llm else None)


def _require_validation(cfg, db, args, stage):
    if cfg["validation"]["require_human_review"] and not label_basis(cfg, db) and not getattr(args, "allow_unvalidated", False):
        raise StageBlocked(
            f"`{stage}` depends on reviewed labels, but none were imported (validation.label_source = "
            f"{cfg['validation'].get('label_source', 'auto')}). Either have annotators fill validation/reviewer_decisions.csv "
            "and run import-annotations, or label the sample with language models (./2-ai-labelling.sh, `pharo-miner "
            "llm-annotate`). Use --allow-unvalidated to produce a clearly-marked PRELIMINARY result.")


# --------------------------------------------------------------------------- LLM annotation
def _llm_profiles(cfg, names: list[str]) -> list[dict]:
    from .analysis.semantic_classifier import load_registry
    reg = load_registry(cfg)
    lcfg = cfg["llm_annotation"]
    out = []
    for n in names:
        prof = (reg.get("models") or {}).get(n)
        if prof is None:
            if ":" not in n:
                raise StageBlocked(f"Model {n!r} is not defined in {cfg['semantic']['models_file']} "
                                   "(and is not an Ollama tag like qwen2.5-coder:7b).")
            prof = {"provider": "ollama", "model": n}
        prof = {**prof, "profile": n}
        for k in ("max_attempts", "timeout_seconds", "seed"):
            prof.setdefault(k, lcfg.get(k))
        out.append(prof)
    return out


def _split_names(v) -> list[str]:
    if not v:
        return []
    if isinstance(v, str):
        v = v.replace(",", " ").split()
    return [x.strip() for x in v if x and x.strip()]


def llm_requirements(cfg, args):
    """Print what the chosen models need (used by 2-ai-labelling.sh): `env VAR` or `ollama TAG` lines."""
    from .validation.llm_providers import requirements
    names = _split_names(args.models) or _split_names(cfg["llm_annotation"]["models"])
    names += _split_names(args.adjudicator) or _split_names(cfg["llm_annotation"].get("adjudicator"))
    names = [n for n in names if n.lower() not in ("none", "no", "off")]
    seen = set()
    for prof in _llm_profiles(cfg, names):
        for kind, val in requirements(prof):
            if (kind, val) not in seen:
                seen.add((kind, val))
                print(f"{kind} {val}")


def llm_annotate(cfg, args):
    from .taxonomy.category_manager import load_taxonomy as _load_tax
    from .validation.llm_annotation import run_panel
    from .validation.llm_providers import ProviderConfigError
    lcfg = cfg["llm_annotation"]
    names = _split_names(args.models) or _split_names(lcfg["models"])
    if not names:
        raise StageBlocked("No models chosen: set MODELS in 2-ai-labelling.sh, llm_annotation.models in config.yaml, "
                           "or pass --models.")
    adj_name = (_split_names(args.adjudicator) or _split_names(lcfg.get("adjudicator")) or [None])[0]
    if adj_name and adj_name.lower() in ("none", "no", "off"):
        adj_name = None
    consensus = args.consensus or lcfg["consensus"]
    if consensus not in ("majority", "unanimous"):
        raise StageBlocked(f"consensus must be majority or unanimous, not {consensus!r}")
    policy = args.evidence_policy or lcfg["evidence_policy"]
    tax = _load_tax(cfg.resolve(cfg["classification"]["operations"]["taxonomy"]))
    gpath = cfg.resolve(lcfg.get("guidelines") or "taxonomy/annotation_guidelines.md")
    with open_db(cfg) as db, tracked_run(db, "llm-annotate", cfg, vars(args)) as run:
        try:
            res = run_panel(cfg, db, _llm_profiles(cfg, names), (_llm_profiles(cfg, [adj_name])[0] if adj_name else None),
                            tax, gpath.read_text(encoding="utf-8"), log,
                            limit=args.limit if args.limit is not None else int(lcfg.get("limit") or 0),
                            consensus=consensus, evidence_policy=policy, concurrency=args.concurrency or lcfg.get("concurrency"),
                            spotcheck_size=int(args.spotcheck_size if args.spotcheck_size is not None else lcfg["spotcheck_size"]),
                            seed=int(lcfg["random_seed"]))
        except ProviderConfigError as e:
            raise StageBlocked(str(e)) from e
        except FileNotFoundError as e:
            raise StageBlocked(str(e)) from e
        run.counts.update({k: v for k, v in res.items() if k not in ("spotcheck",)})
        log(json.dumps(res, indent=1))


def _benchmark_profiles(cfg, args) -> tuple[list[dict], int]:
    """Profiles of the author (first) and the solvers, and how many solvers there are."""
    bcfg = cfg["benchmark_generation"]
    author = (_split_names(getattr(args, "author", None)) or _split_names(bcfg.get("author")) or [None])[0]
    solvers = _split_names(getattr(args, "solvers", None)) or _split_names(bcfg.get("solvers"))
    if solvers and solvers[0].lower() in ("none", "no", "off"):
        solvers, bcfg["require_independent_solution"] = [], False
    if not author:
        raise StageBlocked("No author model: set AUTHOR in 4-benchmark.sh (or benchmark_generation.author in config.yaml).")
    if bcfg.get("require_independent_solution", True) and not solvers:
        raise StageBlocked("No solver model: set SOLVERS in 4-benchmark.sh. The solver checks that a task can be solved "
                           "from its statement alone; use a model other than the author. SOLVERS=none skips that check.")
    return [{**p, "max_tokens": max(int(p.get("max_tokens") or 0), int(bcfg["max_tokens"]))}
            for p in _llm_profiles(cfg, [author] + solvers)], len(solvers)


def benchmark_generate(cfg, args):
    """Synthesise benchmark tasks from mined operations and validate each one in a Pharo image."""
    from .benchgen.image import ImageError
    from .benchgen.pipeline import run_generation
    from .validation.llm_providers import ProviderConfigError, requirements
    for k in ("tasks_per_operation", "max_tasks", "operations", "concurrency"):
        if getattr(args, k, None) is not None:
            cfg["benchmark_generation"][k] = getattr(args, k)
    from .validation.llm_providers import make_provider
    profiles, _ = _benchmark_profiles(cfg, args)
    if getattr(args, "print_requirements", False):
        for kind, what in sorted({r for p in profiles for r in requirements(p)}):
            print(f"{kind}\t{what}")
        return
    try:
        author, *solvers = [make_provider(p) for p in profiles]
    except ProviderConfigError as e:
        raise StageBlocked(str(e)) from e
    if author.name in [s.name for s in solvers]:
        log(f"WARNING: {author.name} is both author and solver: the solvability check is then not independent.")
    tax = load_taxonomy(cfg.resolve(cfg["classification"]["operations"]["taxonomy"]))
    with open_db(cfg) as db, tracked_run(db, "benchmark-generate", cfg, vars(args)) as run:
        try:
            m = run_generation(cfg, db, tax, author, solvers, log, limit=int(getattr(args, "limit", 0) or 0))
        except (ImageError, ProviderConfigError, ValueError) as e:
            raise StageBlocked(str(e)) from e
        run.counts.update({"candidates": m["candidates"], "accepted": m["accepted"], "rejected": m["rejected"]})
    log(f"Benchmark: {cfg.resolve(cfg['benchmark_generation']['dir'])}  report: results/reports/benchmark_generation.md")


def benchmark_validate(cfg, args):
    """Re-validate stored benchmark tasks in the Pharo image; no model is called."""
    from .benchgen.image import ImageError
    from .benchgen.pipeline import run_validation
    bcfg = cfg["benchmark_generation"]
    if getattr(args, "concurrency", None) is not None:
        bcfg["concurrency"] = args.concurrency
    path = Path(args.input) if getattr(args, "input", None) else cfg.resolve(bcfg["dir"]) / "tasks.jsonl"
    if not path.exists():
        raise StageBlocked(f"No tasks to validate: {path} not found (run ./4-benchmark.sh first, or give --input).")
    tasks = read_jsonl(path) if path.suffix == ".jsonl" else (json.loads(path.read_text(encoding="utf-8")))
    tasks = tasks.get("tasks", tasks) if isinstance(tasks, dict) else tasks
    tax = load_taxonomy(cfg.resolve(cfg["classification"]["operations"]["taxonomy"]))
    try:
        m = run_validation(cfg, tax, tasks, log)
    except ImageError as e:
        raise StageBlocked(str(e)) from e
    log(f"Verification: {m['accepted']} of {m['candidates']} tasks pass every gate "
        f"(report: results/reports/benchmark_verification.md)")
    if m["accepted"] != m["candidates"] and getattr(args, "strict", False):
        raise SystemExit(1)


def llm_spotcheck(cfg, args):
    from .validation.llm_annotation import spotcheck_report
    with open_db(cfg) as db, tracked_run(db, "llm-spotcheck", cfg, vars(args)) as run:
        cats = sorted({r[0] for r in db.query("SELECT category_id FROM taxonomy_categories WHERE level=3")})
        res = spotcheck_report(cfg.validation_dir, cats, int(cfg["llm_annotation"]["random_seed"]))
        run.counts["status"] = res["status"]
        if res["status"] != "computed":
            log(f"Spot-check: no human answers yet in {res['path']}")
            return
        log("Spot-check (human = reference):")
        for k, v in res["comparisons"].items():
            if v.get("items"):
                log(f"  {k:55s} items={v['items']:3d}  exact={v['exact_set_agreement']}  "
                    f"Jaccard={v['mean_jaccard']} {v['mean_jaccard_95ci']}  P={v['micro_precision']} "
                    f"R={v['micro_recall']} F1={v['micro_f1']}")
        if res["unknown_operation_ids"]:
            log(f"  unknown ids in human answers (typos?): {res['unknown_operation_ids']}")
        log(f"Written to {cfg.validation_dir / 'llm' / 'spotcheck_report.json'}")


# --------------------------------------------------------------------------- taxonomy
def build_taxonomy(cfg, args):
    from .statistics.prevalence import Corpus, operation_prevalence
    from .taxonomy.taxonomy_builder import build
    tax = load_taxonomy(cfg.resolve(cfg["classification"]["operations"]["taxonomy"]))
    with open_db(cfg) as db, tracked_run(db, "build-taxonomy", cfg, vars(args)) as run:
        _require_validation(cfg, db, args, "build-taxonomy")
        corpus = Corpus(db, "deduplicated")
        prev = {r["operation"]: {k: r[k] for k in ("repositories", "repo_prevalence", "methods", "method_prevalence",
                                                     "repo_balanced_frequency")} for r in operation_prevalence(corpus, cfg["statistics"])}
        discovery = json.loads((cfg.data_dir / "processed" / "discovery.json").read_text()) if (cfg.data_dir / "processed" / "discovery.json").exists() else {}
        basis = label_basis(cfg, db)
        version = args.version or ("1" if basis else "draft")
        doc = build(db, tax, prev, discovery, version, basis=basis)
        name = ("taxonomy_draft_UNVALIDATED.json" if not basis else
                f"taxonomy_v{version}.json" if basis == "human_adjudicated" else f"taxonomy_v{version}-llm.json")
        write_json(cfg.validation_dir / name, doc)
        write_csv(cfg.output_dir / "tables" / "taxonomy.csv",
                  [{"id": c["id"], "parent": c["parent"], "level": c["level"], "name": c["name"], "concept_type": c["concept_type"],
                    "status": c["status"], "definition": c["definition"],
                    "repositories": (c["prevalence"] or {}).get("repositories"),
                    "repo_prevalence": (c["prevalence"] or {}).get("repo_prevalence"),
                    "reference_basis": doc["validation_basis"],
                    "reference_precision": (c["reference_validation"] or {}).get("precision"),
                    "reference_recall": (c["reference_validation"] or {}).get("recall"),
                    "examples": "; ".join(f"{e['repository']}:{e['class']}>>{e['method']}" for e in c["evidence_examples"])}
                   for c in doc["categories"]])
        run.counts.update({"categories": len(doc["categories"]), "proposed": len(doc["proposed_categories"]),
                           "basis": doc["validation_basis"]})
        log(f"Taxonomy written to {cfg.validation_dir / name} ({doc['validation_basis']})")


# --------------------------------------------------------------------------- statistics
def statistics(cfg, args):
    from .statistics.prevalence import (Corpus, api_prevalence, language_feature_prevalence, operation_prevalence,
                                        selector_cooccurrence, selector_prevalence, status_prevalence)
    from .statistics.domain_analysis import domain_operation_matrix
    scfg = cfg["statistics"]
    with open_db(cfg) as db, tracked_run(db, "statistics", cfg, vars(args)) as run:
        basis = label_basis(cfg, db)
        if cfg["validation"]["require_human_review"] and not basis and not args.allow_unvalidated:
            _require_validation(cfg, db, args, "statistics")
        cats = {r["category_id"]: dict(r) for r in db.query("SELECT * FROM taxonomy_categories WHERE taxonomy_version="
                                                              "(SELECT MAX(taxonomy_version) FROM taxonomy_categories)")}
        variants = {}
        for v in ("all", "deduplicated", "non_test"):
            corpus = Corpus(db, v)
            ops = operation_prevalence(corpus, scfg)
            for r in ops:
                c = cats.get(r["operation"], {})
                r.update({"level": c.get("level"), "parent": c.get("parent_id"), "concept_type": c.get("concept_type"),
                          "name": c.get("name")})
            variants[v] = {"corpus": corpus, "operations": ops}
        main_variant = "non_test" if scfg.get("exclude_tests") else "deduplicated"
        corpus = variants[main_variant]["corpus"]
        ops = variants[main_variant]["operations"]
        cells, cross = domain_operation_matrix(db, corpus)
        level_of = {k: c["level"] for k, c in cats.items()}
        for r in cross:
            r["level"] = level_of.get(r["operation"])
        cross_by = {r["operation"]: r["domains_with_operation"] for r in cross}
        for r in ops:
            r["domain_count"] = cross_by.get(r["operation"])
        # sensitivity: repo prevalence across variants
        sens = []
        base = {r["operation"]: r for r in variants["all"]["operations"]}
        for v in ("deduplicated", "non_test"):
            other = {r["operation"]: r for r in variants[v]["operations"]}
            for op, r in base.items():
                o = other.get(op)
                sens.append({"operation": op, "variant": v, "repo_prevalence_all": r["repo_prevalence"],
                             "repo_prevalence_variant": o["repo_prevalence"] if o else 0.0,
                             "method_prevalence_all": r["method_prevalence"],
                             "method_prevalence_variant": o["method_prevalence"] if o else 0.0})
        features = language_feature_prevalence(corpus, scfg)
        statuses = status_prevalence(corpus, scfg)
        selectors = selector_prevalence(db, corpus)
        api_names, frameworks = api_prevalence(db, corpus)
        cooc = selector_cooccurrence(db, corpus, int(cfg["analysis"]["cooccurrence_min_repos"]), int(cfg["analysis"]["cooccurrence_top"]))
        snaps = db.query("SELECT SUM(n_methods) m, SUM(n_parse_failures) f, COUNT(*) n FROM repository_snapshots WHERE n_methods > 0")[0]
        unsupported = db.scalar("SELECT COUNT(*) FROM source_files WHERE status IN ('unsupported','failed','partial')")
        domain_rows = [dict(r) for r in db.query(
            "SELECT label domain, COUNT(DISTINCT repo_id) repositories FROM domain_labels WHERE method='deterministic_rules'"
            " AND repo_id IN (%s) GROUP BY label ORDER BY repositories DESC" % (",".join(map(str, corpus.repos)) or "NULL"))]
        uncertainty = {
            "analysed_repositories": len(corpus.repos), "analysed_methods": len(corpus.method_repo),
            "study_corpus": "filtered: see reports/corpus_filter.md" if study_repo_ids(db) is not None
                            else "every mined repository (filter-corpus was not run)",
            "main_variant": main_variant, "parsed_methods_all_snapshots": snaps["m"], "parse_failures": snaps["f"],
            "unsupported_or_failed_files": unsupported,
            "method_status_distribution": {r["status"]: r["methods"] for r in statuses},
            "label_basis": {"human_adjudicated": "human_validated", "llm_adjudicated": "llm_panel_annotated"}.get(
                basis, "automatic_rule_based_UNVALIDATED"),
            "confidence_interval_note": "Wilson/bootstrap intervals reflect sampling variability under random sampling only; "
                                        "the corpus is not a random sample of all Pharo code.",
            "interpretation_note": "Prevalence measures observed source-code patterns, not developer demand or task difficulty.",
            "repositories_without_domain": sum(1 for r in corpus.repos if not db.scalar(
                "SELECT COUNT(*) FROM domain_labels WHERE repo_id=? AND method='deterministic_rules' AND label!='unknown'", (r,))),
        }

        def strip(rows):
            return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        stats = {"run_id": run.run_id, "generated_at": now_iso(), "uncertainty": uncertainty,
                 "operations": strip(ops), "operations_all_variants": {v: strip(d["operations"]) for v, d in variants.items()},
                 "statuses": strip(statuses), "language_features": strip(features), "selectors": selectors,
                 "api_names": api_names, "frameworks": frameworks, "selector_cooccurrence": cooc,
                 "domain_operation": cells, "cross_domain": cross, "domains": domain_rows, "sensitivity": sens}
        write_json(cfg.data_dir / "processed" / "statistics.json", stats)
        t = cfg.output_dir / "tables"
        write_csv(t / "operation_statistics.csv", strip(ops))
        write_csv(t / "language_features.csv", strip(features))
        write_csv(t / "selector_prevalence.csv", selectors)
        write_csv(t / "api_usage.csv", api_names)
        write_csv(t / "framework_usage.csv", frameworks)
        write_csv(t / "selector_cooccurrence.csv", cooc)
        write_csv(t / "domain_operation_matrix.csv", cells)
        write_csv(t / "cross_domain_operations.csv", cross)
        write_csv(t / "sensitivity_analysis.csv", sens)
        run.counts.update({k: uncertainty[k] for k in ("analysed_repositories", "analysed_methods")})
        log(f"Statistics over {len(corpus.repos)} repositories / {len(corpus.method_repo)} methods ({main_variant}); "
            f"basis: {uncertainty['label_basis']}")


# --------------------------------------------------------------------------- benchmarks
def benchmark_coverage(cfg, args):
    from .benchmarks.benchmark_importer import import_benchmarks
    from .benchmarks.capability_mapping import task_coverage
    from .benchmarks.coverage_analysis import RULES_DOC, coverage_rows, recommend
    bcfg = cfg["benchmarks"]
    with open_db(cfg) as db, tracked_run(db, "benchmark-coverage", cfg, vars(args)) as run:
        _require_validation(cfg, db, args, "benchmark-coverage")
        cats = {r["category_id"]: dict(r) for r in db.query("SELECT * FROM taxonomy_categories WHERE taxonomy_version="
                                                              "(SELECT MAX(taxonomy_version) FROM taxonomy_categories)")}
        inputs = [cfg.resolve(p) for p in (args.input or bcfg.get("inputs") or [])]
        if inputs:
            run.counts["import"] = import_benchmarks(db, inputs, set(cats))
            log(f"Imported benchmark tasks: {run.counts['import']}")
        if not db.scalar("SELECT COUNT(*) FROM benchmark_tasks"):
            raise StageBlocked("No benchmark tasks imported. Provide --input benchmarks/<file>.json (see benchmarks/README.md).")
        stats_path = cfg.data_dir / "processed" / "statistics.json"
        if not stats_path.exists():
            raise StageBlocked("Run `pharo-miner statistics` first.")
        stats = json.loads(stats_path.read_text())
        prev = {r["operation"]: r for r in stats["operations"]}
        validated_cats, cat_status = set(), {}
        basis = label_basis(cfg, db)
        suffix = "-llm.json" if basis == "llm_adjudicated" else ".json"
        tax_files = sorted(p for p in cfg.validation_dir.glob("taxonomy_v*.json")
                           if (basis == "llm_adjudicated") == p.name.endswith("-llm.json"))
        if tax_files:
            doc = json.loads(tax_files[-1].read_text())
            cat_status = {c["id"]: c["status"] for c in doc["categories"]}
            # categories supported by the reference labels (human, or the LLM panel when that is the basis)
            validated_cats = {k for k, v in cat_status.items() if v in ("validated", "llm_supported")}
        elif basis:
            log(f"note: no taxonomy_v*{suffix} found; run build-taxonomy first for category statuses.")
        parent = {k: c["parent_id"] for k, c in cats.items()}
        cov = task_coverage(db, parent)
        feedback = defaultdict(list)
        for r in db.query("SELECT category_id, important_in_work, deserves_benchmark FROM developer_feedback"):
            feedback[r["category_id"]].append(r)
        support = defaultdict(list)
        study = study_repo_ids(db)
        for r in db.query("SELECT DISTINCT ml.operation_id, r.full_name, r.repo_id FROM method_labels ml JOIN methods m ON m.method_id=ml.method_id"
                          " JOIN repositories r ON r.repo_id=m.repo_id WHERE ml.label_source='rule_based'"):
            if len(support[r["operation_id"]]) < 10 and (study is None or r["repo_id"] in study):
                support[r["operation_id"]].append(r["full_name"])
        ops = [{"id": k, "name": c["name"], "level": c["level"], "concept_type": c["concept_type"],
                "status": cat_status.get(k, "unvalidated"),
                "repo_prevalence": (prev.get(k) or {}).get("repo_prevalence"),
                "repositories": (prev.get(k) or {}).get("repositories"), "domains": (prev.get(k) or {}).get("domain_count")}
               for k, c in cats.items()]
        rows = coverage_rows(ops, cov, int(bcfg.get("min_covering_tasks_full", 2)))
        recs = []
        for r in rows:
            r["supporting_repositories"] = support.get(r["operation"], [])
            fb = feedback.get(r["operation"], [])
            expert = None
            if fb:
                imp = [x["important_in_work"] for x in fb if x["important_in_work"] is not None]
                ben = [x["deserves_benchmark"] for x in fb if x["deserves_benchmark"] is not None]
                expert = {"respondents": len(fb), "mean_importance": round(sum(imp) / len(imp), 2) if imp else None,
                          "share_deserves_benchmark": round(sum(ben) / len(ben), 2) if ben else None}
            if r["level"] == 3:
                recs.append(recommend(r, float(bcfg.get("min_repo_prevalence", 0.1)), r["operation"] in validated_cats, expert))
        out_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        write_csv(cfg.output_dir / "tables" / "benchmark_coverage.csv", out_rows)
        write_json(cfg.data_dir / "processed" / "benchmark_coverage.json",
                   {"run_id": run.run_id, "rules": RULES_DOC, "coverage": out_rows, "recommendations": recs})
        write_csv(cfg.output_dir / "tables" / "benchmark_recommendations.csv",
                  [{**r, "reasons": "; ".join(r["reasons"]), "supporting_evidence": ";".join(r["supporting_evidence"])} for r in recs])
        run.counts.update(dict(Counter(r["recommendation"] for r in recs)))
        log(f"Benchmark coverage: {dict(Counter(r['existing_benchmark_coverage'] for r in out_rows if r['level'] == 3))}")
        log(f"Recommendations: {dict(Counter(r['recommendation'] for r in recs))}")


# --------------------------------------------------------------------------- report
def report(cfg, args):
    from .reporting.report_generator import generate
    with open_db(cfg) as db, tracked_run(db, "report", cfg, vars(args)) as run:
        res = generate(cfg, db, allow_unvalidated=args.allow_unvalidated)
        run.counts.update(res)
        log(f"Reports written to {cfg.output_dir / 'reports'}; figures to {cfg.output_dir / 'figures'}")


def models(cfg, args):
    from .analysis.semantic_classifier import installed_ollama_models, load_registry, resolve_model
    reg = load_registry(cfg)
    current = resolve_model(cfg)["profile"]
    have = installed_ollama_models()
    if have is None:
        log("(Ollama is not running: install status unknown. Start it with `ollama serve`.)")
    log(f"{'':2}{'name':27}{'provider':19}{'model':38}{'size':9}status")
    for name, m in (reg.get("models") or {}).items():
        key_env = m.get("api_key_env") or ("ANTHROPIC_API_KEY" if m.get("provider") == "anthropic" else None)
        if m.get("provider", "ollama") == "ollama":
            st = "?" if have is None else ("installed" if m["model"] in have else f"ollama pull {m['model']}")
        elif key_env:
            st = f"API ({key_env} {'set' if os.getenv(key_env) else 'not set'})"
        else:
            st = f"server at {m.get('base_url')}"
        mark = "* " if name == current else "  "
        log(f"{mark}{name:27}{m.get('provider', 'ollama'):19}{m['model']:38}{str(m.get('size', '')):9}{st}")
        if m.get("notes"):
            log(f"{'':29}{m['notes']}")
    log(f"\n* = used by default (change `default:` in {cfg['semantic']['models_file']} or `semantic.use` in config.yaml).")
    log("Use another: --model <name>, or any Ollama tag (e.g. --model mistral:7b).")


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
                raise StageBlocked("No local results to push (run ./1-mining.sh first).")
            res = snap.push(cfg, repo, include_api_cache=scfg["include_api_cache"] and not args.no_api_cache,
                            include_clones=scfg["include_clones"] or args.with_clones, message=args.message, token=token,
                            dry_run=args.dry_run)
    except snap.SnapshotError as e:
        raise StageBlocked(str(e)) from e
    log(json.dumps(res, indent=1, default=str))


def status(cfg, args):
    with open_db(cfg) as db:
        for t in ("repositories", "repository_snapshots", "methods", "message_sends", "method_labels", "domain_labels",
                  "annotation_decisions", "benchmark_tasks"):
            log(f"{t:24s} {db.scalar(f'SELECT COUNT(*) FROM {t}')}")
        log("pharo_status: " + ", ".join(f"{r[0]}={r[1]}" for r in db.query(
            "SELECT pharo_status, COUNT(*) FROM repositories GROUP BY pharo_status")))
        log("\nRecent runs:")
        for r in db.query("SELECT stage, status, started_at, counts_json FROM analysis_runs ORDER BY started_at DESC LIMIT 12"):
            log(f"  {r['started_at']} {r['stage']:20s} {r['status']:10s} {(r['counts_json'] or '')[:110]}")
