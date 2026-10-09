"""SQLite schema of the evidence database.

Every conclusion row (domain label, operation label, recommendation) links back
to source evidence: repository → snapshot (commit) → file → class → method →
observed structure → candidate operation → taxonomy category.
"""

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS schema_info (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    config_hash TEXT,
    config_json TEXT,
    tool_version TEXT,
    code_hash TEXT,
    python_version TEXT,
    platform TEXT,
    arguments_json TEXT,
    counts_json TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS repositories (
    repo_id INTEGER PRIMARY KEY,
    full_name TEXT UNIQUE NOT NULL,
    owner TEXT,
    name TEXT,
    url TEXT,
    description TEXT,
    topics_json TEXT,
    language TEXT,
    languages_json TEXT,
    stars INTEGER,
    forks INTEGER,
    contributors INTEGER,
    contributors_json TEXT,
    created_at TEXT,
    updated_at TEXT,
    pushed_at TEXT,
    archived INTEGER,
    is_fork INTEGER,
    parent_full_name TEXT,
    license TEXT,
    default_branch TEXT,
    head_commit_sha TEXT,
    commit_count INTEGER,
    size_kb INTEGER,
    readme_text TEXT,
    readme_path TEXT,
    discovery_json TEXT,
    discovery_date TEXT,
    metadata_source TEXT,
    metadata_collected_at TEXT,
    pharo_status TEXT,
    pharo_evidence_json TEXT,
    pharo_counter_evidence_json TEXT,
    verification_source TEXT,
    tree_summary_json TEXT,
    attributes_json TEXT
);

CREATE TABLE IF NOT EXISTS repository_duplicates (
    repo_id INTEGER, other_repo_id INTEGER, kind TEXT, similarity REAL, detail TEXT,
    canonical_repo_id INTEGER, run_id TEXT,
    PRIMARY KEY (repo_id, other_repo_id, kind)
);

CREATE TABLE IF NOT EXISTS selection_decisions (
    run_id TEXT, repo_id INTEGER, strategy TEXT, decision TEXT, reason TEXT, stratum TEXT,
    PRIMARY KEY (run_id, repo_id)
);

CREATE TABLE IF NOT EXISTS repository_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id INTEGER NOT NULL,
    commit_sha TEXT,
    local_path TEXT,
    clone_status TEXT,
    clone_error TEXT,
    cloned_at TEXT,
    parsed_at TEXT,
    parser_backend TEXT,
    source_fingerprint TEXT,
    n_files INTEGER, n_packages INTEGER, n_classes INTEGER, n_methods INTEGER, n_parse_failures INTEGER,
    n_lines INTEGER,
    run_id TEXT,
    UNIQUE (repo_id, commit_sha)
);

CREATE TABLE IF NOT EXISTS source_files (
    snapshot_id INTEGER, path TEXT, format TEXT, status TEXT, n_methods INTEGER, n_classes INTEGER,
    package TEXT, warnings_json TEXT, dialect_hints_json TEXT,
    PRIMARY KEY (snapshot_id, path)
);

CREATE TABLE IF NOT EXISTS packages (
    package_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER, name TEXT, paths_json TEXT, fingerprint TEXT, n_methods INTEGER, is_test INTEGER,
    UNIQUE (snapshot_id, name)
);

CREATE TABLE IF NOT EXISTS package_dependencies (
    snapshot_id INTEGER, from_name TEXT, to_name TEXT, kind TEXT, evidence_json TEXT
);

CREATE TABLE IF NOT EXISTS classes (
    class_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER, package TEXT, name TEXT, kind TEXT, superclass TEXT, external_root_superclass TEXT,
    instance_variables_json TEXT, class_variables_json TEXT, class_instance_variables_json TEXT,
    traits TEXT, class_traits TEXT, category TEXT, source_path TEXT, line INTEGER, is_test INTEGER,
    comment TEXT
);

CREATE TABLE IF NOT EXISTS methods (
    method_id INTEGER PRIMARY KEY AUTOINCREMENT,
    method_key TEXT UNIQUE,
    snapshot_id INTEGER, repo_id INTEGER,
    package TEXT, class_name TEXT, side TEXT, selector TEXT, protocol TEXT,
    source_path TEXT, line INTEGER, container_format TEXT, source TEXT, source_hash TEXT,
    arguments_json TEXT, temporaries_json TEXT, block_arguments_json TEXT,
    literals_json TEXT, returns_json TEXT, assignments_json TEXT, blocks_json TEXT,
    referenced_globals_json TEXT, variables_read_json TEXT, pragmas_json TEXT,
    ast_json TEXT,
    parse_status TEXT, parse_errors_json TEXT, parser_backend TEXT,
    is_test INTEGER, is_extension INTEGER, in_trait INTEGER,
    duplicate_of_method_id INTEGER,
    features_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_methods_repo ON methods(repo_id);
CREATE INDEX IF NOT EXISTS idx_methods_hash ON methods(source_hash);

CREATE TABLE IF NOT EXISTS message_sends (
    method_id INTEGER, ordinal INTEGER, selector TEXT, kind TEXT, line INTEGER,
    receiver_kind TEXT, receiver_name TEXT, receiver_resolution TEXT, resolved_target TEXT,
    resolution_evidence TEXT, in_cascade INTEGER, block_depth INTEGER, polymorphic_candidate INTEGER,
    PRIMARY KEY (method_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_sends_selector ON message_sends(selector);

CREATE TABLE IF NOT EXISTS api_references (
    method_id INTEGER, name TEXT, kind TEXT, defined_in_repository INTEGER, framework TEXT, occurrences INTEGER,
    PRIMARY KEY (method_id, name)
);

CREATE TABLE IF NOT EXISTS taxonomy_categories (
    taxonomy_version TEXT, category_id TEXT, parent_id TEXT, level INTEGER, name TEXT, definition TEXT,
    include_json TEXT, exclude_json TEXT, concept_type TEXT, status TEXT, rules_json TEXT, related_json TEXT,
    PRIMARY KEY (taxonomy_version, category_id)
);

CREATE TABLE IF NOT EXISTS operation_evidence (
    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    method_id INTEGER, operation_id TEXT, evidence_kind TEXT, evidence_json TEXT, strength TEXT,
    classification TEXT, taxonomy_version TEXT, review_status TEXT, run_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_evidence_method ON operation_evidence(method_id);
CREATE INDEX IF NOT EXISTS idx_evidence_op ON operation_evidence(operation_id);

CREATE TABLE IF NOT EXISTS method_labels (
    method_id INTEGER, operation_id TEXT, label_source TEXT, label_status TEXT, score REAL, score_kind TEXT,
    taxonomy_version TEXT, review_status TEXT, model TEXT, prompt_version TEXT, created_at TEXT, run_id TEXT,
    PRIMARY KEY (method_id, operation_id, label_source)
);

CREATE TABLE IF NOT EXISTS method_status (
    method_id INTEGER, label_source TEXT, status TEXT, n_operations INTEGER, taxonomy_version TEXT, run_id TEXT,
    PRIMARY KEY (method_id, label_source)
);

CREATE TABLE IF NOT EXISTS method_descriptions (
    method_id INTEGER, summary TEXT, proposed_operations_json TEXT, accepted_operations_json TEXT,
    rejected_json TEXT, evidence_json TEXT, uncertainty_json TEXT, model TEXT, prompt_version TEXT,
    created_at TEXT, raw_response TEXT, run_id TEXT,
    PRIMARY KEY (method_id, model, prompt_version)
);

CREATE TABLE IF NOT EXISTS domain_labels (
    repo_id INTEGER, label TEXT, method TEXT, score REAL, score_kind TEXT, evidence_json TEXT,
    review_status TEXT, model TEXT, prompt_version TEXT, run_id TEXT, created_at TEXT,
    PRIMARY KEY (repo_id, label, method)
);

CREATE TABLE IF NOT EXISTS annotation_decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT, method_id INTEGER, annotator TEXT, kind TEXT, operations_json TEXT,
    flags_json TEXT, new_category TEXT, notes TEXT, decided_at TEXT, source_file TEXT, imported_at TEXT
);

CREATE TABLE IF NOT EXISTS developer_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    respondent TEXT, respondent_role TEXT, category_id TEXT, recognizable INTEGER, important_in_work INTEGER,
    deserves_benchmark INTEGER, examples_representative INTEGER, missing_capabilities TEXT, notes TEXT,
    source_file TEXT, imported_at TEXT
);

CREATE TABLE IF NOT EXISTS benchmark_tasks (
    benchmark TEXT, task_id TEXT, description TEXT, operations_json TEXT, structure_json TEXT,
    required_capabilities_json TEXT, transfer_status TEXT, difficulty TEXT, preserves_capability TEXT,
    origin TEXT, source_file TEXT, unknown_operations_json TEXT, imported_at TEXT,
    PRIMARY KEY (benchmark, task_id)
);
"""
