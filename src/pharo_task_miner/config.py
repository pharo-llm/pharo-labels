"""Configuration loading: YAML file deep-merged over documented defaults."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a declared dependency
    yaml = None

DEFAULTS: dict = {
    "project": {"name": "PharoTaskMiner", "data_dir": "results/data", "output_dir": "results",
                 "validation_dir": "results/validation", "database": "results/data/processed/pharo_task_miner.sqlite"},
    "github": {
        "token_env": "GITHUB_TOKEN",
        "api_url": "https://api.github.com",
        "cache_dir": "results/data/raw/api_cache",
        "cache_ttl_hours": None,  # None = cached responses never expire (reproducible re-runs)
        "max_retries": 6,
        "backoff_seconds": 2.0,
        "search_min_interval_seconds": 2.2,
        "timeout_seconds": 45,
        "discovery": {
            "start_date": "2008-01-01",
            "end_date": None,
            "initial_window_days": 365,
            "topics": ["pharo", "smalltalk"],
            "languages": ["Smalltalk"],
            "organizations": ["pharo-project", "pharo-contributions"],
            "indicator_repository_queries": ['pharo in:readme', 'BaselineOf in:readme', 'Metacello in:readme', 'tonel in:readme'],
            "indicator_code_queries": ["filename:.smalltalk.ston", "BaselineOf extension:st", "filename:.project srcDirectory"],
            "curated_lists": [],
            "strategies": ["topics", "languages", "organizations", "indicators", "curated"],
        },
    },
    "verification": {"max_ci_files": 5},
    "metadata": {"statuses": ["confirmed_pharo", "probable_pharo", "uncertain"]},
    "repositories": {
        "include_archived": True,
        "include_forks": False,
        "max_repositories": 500,
        "clone_depth": 1,
        "clone_dir": "results/data/repositories",
        "existing_clone_dirs": [],
        "accepted_pharo_status": ["confirmed_pharo", "probable_pharo"],
        "core_owners": ["pharo-project", "pharo-contributions", "pharo-spec", "pharo-graphics", "pharo-vcs",
                        "pharo-ide", "pharo-nosql", "pharo-rdbms", "pharo-containers", "pharo-ai"],
        "clone_timeout_seconds": 900,
        # inclusion criteria, all off by default (see sampling.repository_filters.quality)
        "quality": {"require_pharo_source": False, "exclude_coursework": False, "min_commits": 0,
                    "min_pharo_source_files": 0, "require_tests": False, "require_external_interest": False},
    },
    "sampling": {"strategy": "stratified", "sample_size": 50, "random_seed": 42, "max_per_owner": 5,
                  "manual_selection_file": None, "strata": ["primary_domain", "size_category"],
                  "diversity_attributes": ["primary_domain", "size_category", "owner", "project_kind",
                                           "educational", "core_infrastructure"],
                  "size_categories": {"tiny": 10, "small": 50, "medium": 250, "large": 1000}},
    # Study corpus (`pharo-miner filter-corpus`, ./1-2-filtering.sh); see docs/1-2-filtering.md
    "corpus_filter": {
        "max_repositories": 150, "require_license": True, "min_contributors": 2, "min_commits": 50,
        "min_development_days": 180, "max_years_since_last_push": 5, "require_baseline": True, "require_tonel": True,
        "min_non_test_methods": 200, "max_methods": 20000, "min_test_methods": 20, "min_test_method_share": 0.10,
        "max_duplicated_method_share": 0.10, "max_parse_failure_share": 0.01,
        "diversity_attributes": ["primary_domain", "size_category", "project_kind"], "max_per_owner": 5,
        "random_seed": 42},
    "parsing": {"backend": "python", "pharo_vm": None, "pharo_image": None, "pharo_timeout": 900,
                 "primary_format": "tonel", "export_ast": True, "include_tests": True,
                 "include_extension_methods": True, "exclude_globs": []},
    "analysis": {"extract_message_sends": True, "extract_blocks": True, "extract_control_flow": True,
                  "extract_api_usage": True, "cooccurrence_min_repos": 3, "cooccurrence_top": 300},
    "classification": {
        "domains": {"multi_label": True, "allow_unknown": True, "definitions": "taxonomy/domains.yaml",
                     "min_score": 2.0, "framework_reference_min_share": 0.02, "semantic_analysis": False},
        "operations": {"multi_label": True, "allow_unknown": True, "taxonomy": "taxonomy/initial_categories.yaml",
                        "semantic_analysis": False, "multiple_operations_threshold": 4,
                        "insufficient_context_max_nodes": 3},
        "frameworks": "taxonomy/frameworks.yaml",
    },
    "semantic": {"models_file": "models.yaml", "use": None, "max_methods": 200,
                  "max_tokens": 1500, "num_ctx": 8192, "timeout_seconds": 300, "seed": 42},
    "validation": {"require_human_review": True, "label_source": "auto", "annotation_sample_size": 300, "label_repositories": 0,
                    "overlap_fraction": 0.3, "max_per_repository": 15, "random_seed": 42,
                    "annotators": ["annotator_1", "annotator_2"]},
    # LLM annotation panel (`pharo-miner llm-annotate`, ./2-ai-labelling.sh); models are defined in models.yaml
    "llm_annotation": {"models": [], "adjudicator": None, "consensus": "majority", "evidence_policy": "enforce",
                        "limit": 0, "concurrency": None, "spotcheck_size": 30, "random_seed": 42,
                        "max_attempts": 4, "timeout_seconds": 600, "seed": 42,
                        "guidelines": "taxonomy/annotation_guidelines.md"},
    "statistics": {"bootstrap_iterations": 1000, "random_seed": 42, "confidence_level": 0.95,
                    "exclude_tests": False},
    "benchmarks": {"inputs": [], "min_covering_tasks_full": 2, "min_repo_prevalence": 0.1},
    # Benchmark generation (`pharo-miner benchmark-generate`, ./4-benchmark.sh); see docs/4-benchmark.md
    "benchmark_generation": {
        "dir": "results/benchmark", "baseline": "PharoTaskBench", "class_prefix": "Ptb",
        "pharo_version": "130", "mutalk_version": "v3.0.8", "pharo_vm": None, "pharo_image": None,
        "author": None, "solvers": [], "operations": "auto", "tasks_per_operation": 3, "max_tasks": 0,
        "seed_min_lines": 5, "seed_max_lines": 40, "random_seed": 42, "concurrency": 2,
        "max_repair_rounds": 3, "max_amplification_rounds": 2, "solver_attempts": 2, "max_tokens": 6000,
        "min_tests": 4, "max_target_methods": 2, "repeat": 3, "test_time_limit_seconds": 10, "job_timeout_seconds": 600,
        "min_mutation_score": 0.8, "min_mutants": 4, "max_seed_similarity": 0.6,
        "require_operation": True, "require_independent_solution": True},
    # Results snapshots on the Hugging Face Hub (`pharo-miner snapshot …`)
    "snapshot": {"repo": "pharo-llm/pharo-labels", "revision": "main", "include_api_cache": True,
                 "include_clones": False, "token_env": "HF_TOKEN"},
    "output": {"formats": ["jsonl", "csv", "markdown"]},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config(dict):
    """Config dictionary that also knows its project root and source file."""

    def __init__(self, data: dict, root: Path, path: Path | None):
        super().__init__(data)
        self.root = root
        self.path = path

    def resolve(self, rel: str | os.PathLike | None) -> Path | None:
        if rel is None:
            return None
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    @property
    def data_dir(self) -> Path:
        return self.resolve(self["project"]["data_dir"])

    @property
    def output_dir(self) -> Path:
        return self.resolve(self["project"]["output_dir"])

    @property
    def validation_dir(self) -> Path:
        return self.resolve(self["project"]["validation_dir"])

    @property
    def database_path(self) -> Path:
        return self.resolve(self["project"]["database"])

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self, sort_keys=True, default=str).encode()).hexdigest()


def load_config(path: str | os.PathLike | None = None, root: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else None
    data = {}
    if cfg_path is not None:
        if not cfg_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
        text = cfg_path.read_text(encoding="utf-8")
        if cfg_path.suffix == ".json":
            data = json.loads(text)
        else:
            if yaml is None:
                raise RuntimeError("PyYAML is required to read YAML configuration (pip install pyyaml)")
            data = yaml.safe_load(text) or {}
    project_root = Path(root) if root else (cfg_path.resolve().parent if cfg_path else Path.cwd())
    return Config(deep_merge(DEFAULTS, data), project_root, cfg_path)


def load_yaml(path: Path):
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required (pip install pyyaml)")
    return yaml.safe_load(text)
