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
                 "labelling_dir": "results/labelling", "database": "results/data/processed/pharo_task_miner.sqlite"},
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
    # Filter 2, the study corpus (`pharo-miner filter-corpus`, ./run.sh filter)
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
        "domains": {"definitions": "taxonomy/domains.yaml", "min_score": 2.0, "framework_reference_min_share": 0.02},
        "frameworks": "taxonomy/frameworks.yaml",
    },
    # Labelling of the study corpus (`pharo-miner label`, ./run.sh label): sample size, open
    # coding of task categories with a small model, saturation loop, expert-validation export
    "labelling": {
        "model": "qwen3.8:27b",          # inductive phase and codebook definitions (Ollama tag)
        "confidence": 0.95, "margin_of_error": 0.05, "proportion": 0.5,  # sample-size calculator (Cochran + FPC)
        "sample_size": None,                    # None = computed from the three values above
        "batch_size": 50,                       # methods drawn per saturation round
        "patience": None,                       # rounds in a row without a new category = saturation; None = derived
                                                # from min_share: a 0.5% task would have shown with 95% probability
        "max_rounds": 40,                       # safety cap on saturation rounds
        "min_support": 2,                       # methods a proposed category needs before it enters the taxonomy
        "min_share": 0.005,                     # ... and the share of the labelled methods it needs (0.5%)
        "merge_similarity": 0.88,               # name similarity (typos, spelling) above which two names are one category
        "embedding_model": "all-minilm",        # Ollama embedding model that merges names with the same meaning; None = off
        "semantic_threshold": 0.78,             # cosine similarity above which two names mean the same task
        "include_tests": True,
        "max_method_lines": 40,                 # longer methods are cut (the model sees the start)
        "max_method_chars": 4000,               # ... and so are methods with very long lines (generated data),
                                                # which would not fit in the model's context
        "examples_per_category": 5,             # examples per category in the expert-validation file
        # deductive phase (step 4): models of three different families classify a new sample with the codebook
        "classify_models": ["qwen3.8:27b", "gemma4:12b", "mistral-small3.2:24b"],
        "classify_sample_size": None,           # None = the sample-size calculator's n (384 at 95% / ±5%)
        "selfcheck_size": 100,                  # methods the first model classifies again, prompt shuffled
        "expert_prefill": True,                 # the experts see the models' majority label and correct it (False: blind)
        "min_alpha": 0.667,                     # Krippendorff's alpha needed before the experts (tentative level)
        "gold_size": 200,                       # methods the experts classify (±5% on accuracy)
        "gold_min_per_stratum": 20,             # at least this many per agreement level (3/3, 2/3, none)
        "gold_error_guess": {"3/3": 0.1, "2/3": 0.35, "none": 0.6},  # guessed error rates: Neyman allocation
        "random_seed": 42, "temperature": 0, "max_tokens": 80, "timeout_seconds": 300,
    },
    # What the labelling would cost on a commercial API (REPORT.md, section 6). USD per million tokens,
    # standard tier, short context; prices of 2026-10-18 from the providers' pages.
    "api_cost": {
        "prices_date": "2026-10-18",
        "sources": ["https://developers.openai.com/api/docs/pricing", "https://claude.com/pricing",
                    "https://ai.google.dev/gemini-api/docs/pricing"],
        "reference_model": "GPT-6 Astra",       # the model the savings are computed on
        "batch_discount": 0.5,                  # Batch API: half price, answers within 24 h
        "cache_min_prefix_tokens": 1024,        # a shared prefix shorter than this is never cached
        "chat_overhead_tokens": 20,             # system/user message wrapping of a chat API, per call
        "reasoning_tokens_per_call": 250,       # hidden reasoning at low effort (an assumption: measure it)
        "budget_factor": [1.5, 2.0],            # pilot, retries, check model, a rerun
        "models": [   # token_factor: tokens of this provider per GPT token (Claude counts more on code)
            {"name": "GPT-6 Astra", "provider": "OpenAI", "input": 10.0, "cached_input": 1.0, "output": 50.0},
            {"name": "GPT-6.1 Sol", "provider": "OpenAI", "input": 2.0, "cached_input": 0.10, "output": 10.0},
            {"name": "GPT-6 Luna", "provider": "OpenAI", "input": 0.10, "cached_input": 0.01, "output": 0.50},
            {"name": "GPT-5.4 mini", "provider": "OpenAI", "input": 0.75, "cached_input": 0.075, "output": 4.50},
            {"name": "Claude Fable 5.1", "provider": "Anthropic", "input": 10.0, "cached_input": 0.25, "output": 50.0,
             "token_factor": 1.25},
            {"name": "Claude Opus 5.5", "provider": "Anthropic", "input": 4.0, "cached_input": 0.20, "output": 20.0,
             "token_factor": 1.25},
            {"name": "Claude Sonnet 5.5", "provider": "Anthropic", "input": 2.0, "cached_input": 0.10, "output": 10.0,
             "token_factor": 1.25},
            {"name": "Claude Haiku 5.5", "provider": "Anthropic", "input": 0.10, "cached_input": 0.01, "output": 0.50,
             "token_factor": 1.25},
            {"name": "Gemini 4 Argon", "provider": "Google", "input": 2.0, "cached_input": 0.10, "output": 10.0},
            {"name": "Gemini 3.8 Flash", "provider": "Google", "input": 0.75, "cached_input": 0.075, "output": 3.75},
            {"name": "Gemini 2.5 Flash-Lite", "provider": "Google", "input": 0.10, "cached_input": 0.01, "output": 0.40},
        ],
    },
    # Results snapshots on the Hugging Face Hub (`pharo-miner snapshot …`)
    "snapshot": {"repo": "pharo-llm/pharo-labels", "revision": "main", "include_api_cache": True,
                 "include_clones": False, "token_env": "HF_TOKEN"},
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
    def labelling_dir(self) -> Path:
        return self.resolve(self["project"]["labelling_dir"])

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
