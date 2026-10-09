"""``pharo-miner`` command-line interface."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from . import stages
from .config import load_config


def _common(p: argparse.ArgumentParser, top: bool = False):
    p.add_argument("--config", default=None if top else argparse.SUPPRESS,
                   help="YAML/JSON configuration (default: ./config.yaml if present)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pharo-miner", description="PharoTaskMiner — empirical discovery of programming "
                                 "operations in Pharo repositories")
    _common(ap, top=True)
    sub = ap.add_subparsers(dest="command", required=True)

    def cmd(name, help_):
        p = sub.add_parser(name, help=help_)
        _common(p)
        return p

    p = cmd("discover", "Discover repositories on GitHub (topics, language, organisations, indicators, curated lists)")
    p.add_argument("--strategy", action="append", choices=["topics", "languages", "organizations", "indicators", "curated"])
    p.add_argument("--import-dump", action="append", help="Import a saved JSON list of GitHub repository objects instead")
    p.add_argument("--offline", action="store_true", help="Use cached API responses only")
    p.add_argument("--refresh", action="store_true", help="Ignore the response cache")

    p = cmd("verify", "Classify repositories as confirmed/probable Pharo, uncertain or non-Pharo, with evidence")
    p.add_argument("--local", action="store_true", help="Use local clones only (no API)")
    p.add_argument("--only", nargs="*")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--refresh", action="store_true")

    p = cmd("metadata", "Collect full repository metadata (languages, contributors, commits, license, revision)")
    p.add_argument("--only", nargs="*")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--refresh", action="store_true")

    p = cmd("classify-domains", "Multi-label application-domain classification with evidence")
    p.add_argument("--semantic", action="store_true", help="Also run the local LLM classifier (see `pharo-miner models`)")
    p.add_argument("--semantic-limit", type=int, default=0)
    p.add_argument("--model", help="Model name from models.yaml, or any Ollama tag")

    p = cmd("sample", "Select repositories (all|random|stratified|diversity|manual)")
    p.add_argument("--strategy")
    p.add_argument("--size", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--manual-file")

    p = cmd("clone", "Shallow-clone selected repositories (or reuse existing clones)")
    p.add_argument("--repos", nargs="*", help="owner/name list instead of the latest sample")
    p.add_argument("--local-only", action="store_true", help="Never clone; only register existing clones")

    p = cmd("parse", "Parse Pharo sources (Tonel, FileTree, chunk) and extract method records")
    p.add_argument("--repos", nargs="*")

    cmd("analyze", "Receiver resolution, API/framework usage, package dependencies, duplicated code")

    p = cmd("discover-operations", "Rule-based candidate operations + open discovery (+ optional LLM descriptions)")
    p.add_argument("--semantic", action="store_true")
    p.add_argument("--semantic-limit", type=int, default=0)
    p.add_argument("--model", help="Model name from models.yaml, or any Ollama tag")

    p = cmd("export-annotations", "Export a stratified blind annotation sample and templates")
    p.add_argument("--size", type=int)

    cmd("import-annotations", "Import annotator decisions, compute agreement, import adjudicated labels and feedback")

    p = cmd("llm-annotate", "Label the annotation sample with a panel of LLMs (local or API) and aggregate their labels")
    p.add_argument("--models", help="Space/comma-separated model names from models.yaml (or Ollama tags)")
    p.add_argument("--adjudicator", help="Model that resolves disagreements (default: majority vote; 'none' to disable)")
    p.add_argument("--consensus", choices=["majority", "unanimous"], help="Aggregation without an adjudicator")
    p.add_argument("--evidence-policy", choices=["enforce", "record"],
                   help="enforce: drop labels whose cited evidence is not in the method (default)")
    p.add_argument("--limit", type=int, help="Label only the first N items (a cheap trial run); 0 = all")
    p.add_argument("--concurrency", type=int, help="Parallel requests per model (default: per provider)")
    p.add_argument("--spotcheck-size", type=int, help="Items exported for a blind human spot-check (0 = none)")
    p.add_argument("--print-requirements", action="store_true", help=argparse.SUPPRESS)

    cmd("llm-spotcheck", "Compare LLM labels with human answers in validation/llm/human_spotcheck.csv")

    p = cmd("build-taxonomy", "Build the versioned evidence-backed taxonomy")
    p.add_argument("--version")
    p.add_argument("--allow-unvalidated", action="store_true")

    p = cmd("statistics", "Prevalence, domain association, language features, uncertainty, sensitivity")
    p.add_argument("--allow-unvalidated", action="store_true")

    p = cmd("benchmark-coverage", "Compare the taxonomy with imported benchmark tasks and recommend candidates")
    p.add_argument("--input", action="append")
    p.add_argument("--allow-unvalidated", action="store_true")

    p = cmd("report", "Generate figures, tables and Markdown reports")
    p.add_argument("--allow-unvalidated", action="store_true")

    p = cmd("snapshot", "Save results to / restore results from a Hugging Face dataset")
    p.add_argument("action", choices=["push", "pull", "info", "state"],
                   help="push: upload local results; pull: download a snapshot; info: what is local and remote; "
                        "state: print the local state only (complete/partial/none)")
    p.add_argument("--repo", help="Hugging Face dataset id (default: snapshot.repo in config.yaml)")
    p.add_argument("--revision", help="pull/info: branch, tag (snapshot-YYYYMMDD-HHMMSS) or commit (default: main)")
    p.add_argument("--force", action="store_true", help="pull: replace existing local results (kept as a backup folder)")
    p.add_argument("--message", help="push: commit message")
    p.add_argument("--dry-run", action="store_true", help="push: prepare and list the snapshot without uploading")
    p.add_argument("--no-api-cache", action="store_true", help="push: leave out the raw GitHub API responses")
    p.add_argument("--with-clones", action="store_true", help="push: include the cloned repositories (third-party code)")

    cmd("status", "Show database contents and recent runs")
    cmd("models", "List the local LLMs in models.yaml and whether they are installed")

    p = cmd("pipeline", "Run all stages; stops before validation-dependent stages without human review")
    p.add_argument("--import-dump", action="append", help="Use a saved repository dump instead of live discovery")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--local", action="store_true", help="Verify with local clones instead of the API")
    p.add_argument("--local-only", action="store_true", help="Do not clone; only use existing clones")
    p.add_argument("--strategy")
    p.add_argument("--size", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--semantic", action="store_true", help="Run the optional local LLM step")
    p.add_argument("--semantic-limit", type=int, default=0)
    p.add_argument("--model", help="Model name from models.yaml, or any Ollama tag")
    p.add_argument("--allow-unvalidated", action="store_true",
                   help="Continue past the validation gate, producing clearly marked PRELIMINARY outputs")
    return ap


def pipeline(cfg, args):
    has_token = bool(os.getenv(cfg["github"]["token_env"]))
    ns = argparse.Namespace
    common = dict(offline=args.offline, refresh=args.refresh)
    stages.discover(cfg, ns(strategy=None, import_dump=args.import_dump, **common))
    stages.verify(cfg, ns(local=args.local or not has_token, only=None, **common))
    if has_token or args.offline:
        try:
            stages.metadata(cfg, ns(only=None, **common))
        except stages.StageBlocked as e:
            print(f"metadata skipped: {e}")
    else:
        print(f"metadata skipped: {cfg['github']['token_env']} not set (search-level metadata only)")
    stages.classify_domains(cfg, ns(semantic=False, semantic_limit=0))
    stages.sample(cfg, ns(strategy=args.strategy, size=args.size, seed=args.seed, manual_file=None))
    stages.clone_stage(cfg, ns(repos=None, local_only=args.local_only))
    stages.parse(cfg, ns(repos=None))
    stages.analyze(cfg, ns())
    stages.classify_domains(cfg, ns(semantic=args.semantic, semantic_limit=args.semantic_limit, model=args.model))
    stages.discover_operations(cfg, ns(semantic=args.semantic, semantic_limit=args.semantic_limit, model=args.model))
    stages.export_annotations(cfg, ns(size=None))
    stages.import_annotations(cfg, ns())
    with stages.open_db(cfg) as db:
        validated = stages.label_basis(cfg, db)
    if not validated and cfg["validation"]["require_human_review"] and not args.allow_unvalidated:
        print("\nPipeline stopped at the validation gate: no adjudicated human annotations are available.\n"
              f"  1. Annotators fill {cfg.validation_dir / 'reviewer_decisions.csv'} (see annotation_guidelines.md)\n"
              "  2. pharo-miner import-annotations  (then resolve validation/adjudicated_labels.csv)\n"
              "  3. pharo-miner pipeline … again, or run build-taxonomy / statistics / benchmark-coverage / report\n"
              "Or label the sample with language models instead: ./2-ai-labelling.sh, then ./3-results.sh\n"
              "Use --allow-unvalidated for clearly marked PRELIMINARY statistics and reports.")
        return 3
    flags = ns(allow_unvalidated=args.allow_unvalidated, version=None, input=None)
    stages.build_taxonomy(cfg, flags)
    stages.statistics(cfg, flags)
    try:
        stages.benchmark_coverage(cfg, flags)
    except stages.StageBlocked as e:
        print(f"benchmark-coverage skipped: {e}")
    stages.report(cfg, flags)
    return 0


COMMANDS = {
    "discover": stages.discover, "verify": stages.verify, "metadata": stages.metadata,
    "classify-domains": stages.classify_domains, "sample": stages.sample, "clone": stages.clone_stage,
    "parse": stages.parse, "analyze": stages.analyze, "discover-operations": stages.discover_operations,
    "export-annotations": stages.export_annotations, "import-annotations": stages.import_annotations,
    "llm-spotcheck": stages.llm_spotcheck, "snapshot": stages.snapshot,
    "build-taxonomy": stages.build_taxonomy, "statistics": stages.statistics,
    "benchmark-coverage": stages.benchmark_coverage, "report": stages.report, "status": stages.status,
    "models": stages.models,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = args.config or ("config.yaml" if Path("config.yaml").exists() else None)
    cfg = load_config(cfg_path)
    try:
        if args.command == "pipeline":
            return pipeline(cfg, args)
        if args.command == "llm-annotate":
            (stages.llm_requirements if args.print_requirements else stages.llm_annotate)(cfg, args)
            return 0
        COMMANDS[args.command](cfg, args)
        return 0
    except stages.StageBlocked as e:
        print(f"\nBLOCKED: {e}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nInterrupted; completed work is saved and the stage can be resumed.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
