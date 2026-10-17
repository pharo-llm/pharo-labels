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
    ap = argparse.ArgumentParser(prog="pharo-miner", description="PharoTaskMiner — mine Pharo repositories and label "
                                 "the programming tasks their methods perform")
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

    cmd("classify-domains", "Multi-label application-domain classification with evidence")

    p = cmd("sample", "Select repositories (all|random|stratified|diversity|manual)")
    p.add_argument("--strategy")
    p.add_argument("--size", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--manual-file")

    p = cmd("filter-corpus", "Filter 2: keep the repositories the labelling is based on (the study corpus)")
    p.add_argument("--max-repositories", type=int, dest="max_repositories",
                   help="Size limit of the study corpus (default: corpus_filter.max_repositories)")

    p = cmd("clone", "Shallow-clone selected repositories (or reuse existing clones)")
    p.add_argument("--repos", nargs="*", help="owner/name list instead of the latest sample")
    p.add_argument("--local-only", action="store_true", help="Never clone; only register existing clones")

    p = cmd("parse", "Parse Pharo sources (Tonel, FileTree, chunk) and extract method records")
    p.add_argument("--repos", nargs="*")

    cmd("analyze", "Receiver resolution, API/framework usage, package dependencies, duplicated code")

    p = cmd("label", "Sample size, open coding of task categories with a small model, saturation loop, "
                     "expert-validation export")
    p.add_argument("--model", help="Labelling model: an Ollama tag, or hf:<repo>[@revision] (default: labelling.model)")
    p.add_argument("--check-model", dest="check_model",
                   help="Second model that re-labels the review subset to measure agreement ('none' to skip)")
    p.add_argument("--sample-size", type=int, dest="sample_size", help="Initial sample (default: computed)")
    p.add_argument("--confidence", type=float, help="Confidence level of the sample-size calculator, e.g. 0.95")
    p.add_argument("--margin", type=float, dest="margin_of_error", help="Margin of error, e.g. 0.05")
    p.add_argument("--batch-size", type=int, dest="batch_size", help="Methods per saturation round")
    p.add_argument("--patience", type=int, help="Rounds without a new category that mean saturation")
    p.add_argument("--max-rounds", type=int, dest="max_rounds", help="Cap on saturation rounds")
    p.add_argument("--min-support", type=int, dest="min_support", help="Methods a category needs to be admitted")
    p.add_argument("--min-share", type=float, dest="min_share",
                   help="Share of the labelled methods a category needs to be admitted, e.g. 0.005")
    p.add_argument("--embedding-model", dest="embedding_model",
                   help="Ollama embedding model that merges names with the same meaning ('none' = lexical only)")

    cmd("report", "Write results/REPORT.md: from GitHub to the expert-validation export")

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

    p = cmd("pipeline", "Mine: discover, verify, metadata, filter 1, clone, parse, analyze")
    p.add_argument("--import-dump", action="append", help="Use a saved repository dump instead of live discovery")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--local", action="store_true", help="Verify with local clones instead of the API")
    p.add_argument("--local-only", action="store_true", help="Do not clone; only use existing clones")
    p.add_argument("--strategy")
    p.add_argument("--size", type=int)
    p.add_argument("--seed", type=int)
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
    stages.classify_domains(cfg, ns())
    stages.sample(cfg, ns(strategy=args.strategy, size=args.size, seed=args.seed, manual_file=None))
    stages.clone_stage(cfg, ns(repos=None, local_only=args.local_only))
    stages.parse(cfg, ns(repos=None))
    stages.analyze(cfg, ns())
    stages.classify_domains(cfg, ns())
    return 0


COMMANDS = {
    "discover": stages.discover, "verify": stages.verify, "metadata": stages.metadata,
    "classify-domains": stages.classify_domains, "sample": stages.sample, "filter-corpus": stages.filter_corpus,
    "clone": stages.clone_stage, "parse": stages.parse, "analyze": stages.analyze,
    "label": stages.label, "report": stages.report, "snapshot": stages.snapshot, "status": stages.status,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = args.config or ("config.yaml" if Path("config.yaml").exists() else None)
    cfg = load_config(cfg_path)
    try:
        if args.command == "pipeline":
            return pipeline(cfg, args)
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
