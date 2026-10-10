"""Generate Markdown reports, figures and tables from exported analysis data.

Reports state their basis (validated vs automatic) prominently and only fill
templates with computed numbers; no conclusion is written that the data does
not support.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from . import charts, tables
from ..sampling.corpus_filter import study_repo_ids
from ..sampling.repository_filters import repository_profiles
from ..storage.database import Database, loads
from ..storage.provenance import now_iso


def _pct(v):
    return "–" if v is None else f"{v:.1%}"


def _md_table(rows: list[dict], cols: list[tuple[str, str]], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    out = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k)
            cells.append(fmt[k](v) if k in fmt and v is not None else ("–" if v is None else str(v).replace("|", "/")))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _banner(stats: dict | None, allow_unvalidated: bool, cfg=None) -> str:
    basis = (stats or {}).get("uncertainty", {}).get("label_basis", "unknown")
    if basis.startswith("human"):
        return "> **Basis:** operation labels validated against adjudicated human annotations.\n"
    if basis.startswith("llm"):
        txt = ("> **Basis: LLM-annotated sample — not human-validated.** Rule-based operation labels were checked "
               "against a sample labelled by a panel of language models")
        man = (cfg.validation_dir / "llm" / "run_manifest.json") if cfg is not None else None
        if man is not None and man.exists():
            m = json.loads(man.read_text())
            models = ", ".join(p.get("name", p.get("model")) for p in m.get("panel", []))
            txt += f" ({models}; aggregation: {m.get('aggregation')}; prompt `{m.get('prompt_version')}`)"
            ag = m.get("agreement") or {}
            if ag.get("mean_jaccard") is not None:
                txt += f"; mean inter-model Jaccard {ag['mean_jaccard']}"
        txt += "."
        spot = (cfg.validation_dir / "llm" / "spotcheck_report.json") if cfg is not None else None
        if spot is not None and spot.exists():
            sp = json.loads(spot.read_text())
            fin = {k: v for k, v in sp.get("comparisons", {}).items() if k.endswith("vs llm:final_labels") and v.get("items")}
            for k, v in fin.items():
                txt += (f" Human spot-check ({v['items']} items, {k.split(' vs ')[0]}): mean Jaccard {v['mean_jaccard']} "
                        f"(95% CI {v['mean_jaccard_95ci']}), micro-F1 {v['micro_f1']}.")
        else:
            txt += " No human spot-check has been computed yet (see validation/llm/human_spotcheck.csv)."
        return txt + "\n"
    return ("> **PRELIMINARY — automatic, unvalidated labels.** Operation and domain labels come from rules and are "
            "`review_status = pending`. They are candidate evidence for open coding, not validated findings.\n")


def generate(cfg, db: Database, allow_unvalidated: bool = False) -> dict:
    out, figs, tdir = cfg.output_dir / "reports", cfg.output_dir / "figures", cfg.output_dir / "tables"
    out.mkdir(parents=True, exist_ok=True)
    stats_path = cfg.data_dir / "processed" / "statistics.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    counts = {}
    counts["repositories_csv"] = tables.repositories_csv(db, tdir / "repositories.csv")
    counts["repository_domains_csv"] = tables.repository_domains_csv(db, tdir / "repository_domains.csv")
    counts["methods_csv"] = tables.methods_csv(db, tdir / "methods.csv")
    counts["operation_evidence_csv"] = tables.operation_evidence_csv(db, tdir / "operation_evidence.csv")
    profiles = repository_profiles(db, cfg)
    analysed = {r[0] for r in db.query("SELECT DISTINCT repo_id FROM repository_snapshots WHERE n_methods > 0")}
    study = study_repo_ids(db)
    if study is not None:  # the results describe the study corpus (filter-corpus), not every mined repository
        analysed &= study
    prof_an = [p for p in profiles if p["repo_id"] in analysed]
    cats = [dict(r) for r in db.query("SELECT * FROM taxonomy_categories WHERE taxonomy_version="
                                      "(SELECT MAX(taxonomy_version) FROM taxonomy_categories)")]
    for c in cats:
        c["id"], c["parent"] = c["category_id"], c["parent_id"]
    if not (tdir / "taxonomy.csv").exists() and stats:
        prev = {r["operation"]: r for r in stats["operations"]}
        from ..stages import write_csv
        write_csv(tdir / "taxonomy.csv", [{"id": c["id"], "parent": c["parent"], "level": c["level"], "name": c["name"],
                                            "concept_type": c["concept_type"], "status": "provisional", "definition": c["definition"],
                                            "repositories": (prev.get(c["id"]) or {}).get("repositories"),
                                            "repo_prevalence": (prev.get(c["id"]) or {}).get("repo_prevalence")} for c in cats])
    # ---------------- figures
    if stats:
        n = stats["uncertainty"]["analysed_repositories"]
        charts.domain_distribution(figs / "domain_distribution.png", stats["domains"], n)
        charts.repository_characteristics(figs / "repository_characteristics.png", prof_an or profiles)
        charts.taxonomy_tree(figs / "taxonomy.png", cats, {r["operation"]: r["repo_prevalence"] for r in stats["operations"]})
        charts.operation_prevalence(figs / "operation_prevalence.png", stats["operations"])
        lvl3 = [r["operation"] for r in stats["operations"] if r.get("level") == 3][:35]
        doms = [d["domain"] for d in stats["domains"] if d["domain"] != "unknown" and d["repositories"] >= 3]
        if lvl3 and doms:
            charts.heatmap(figs / "domain_operation_heatmap.png", stats["domain_operation"], lvl3, doms,
                           "Figure 5 — Operations by application domain: P(operation | domain)")
            charts.heatmap(figs / "domain_operation_lift.png", stats["domain_operation"], lvl3, doms,
                           "Figure 5b — Domain association: share of a domain's methods vs. all repositories",
                           value_key="balanced_share_lift", diverging=True, label="Balanced share lift (log scale)")
        charts.pharo_capabilities(figs / "pharo_capabilities.png", stats["language_features"])
        charts.cross_domain(figs / "cross_domain_operations.png", stats["cross_domain"])
    bc_path = cfg.data_dir / "processed" / "benchmark_coverage.json"
    bc = json.loads(bc_path.read_text()) if bc_path.exists() else None
    if bc:
        charts.benchmark_coverage(figs / "benchmark_coverage.png", bc["coverage"])
    banner = _banner(stats, allow_unvalidated, cfg)
    _ecosystem_summary(cfg, db, out, stats, prof_an, banner, bc)
    _repository_selection(cfg, db, out, profiles)
    _operation_analysis(cfg, out, stats, banner)
    _taxonomy_report(cfg, db, out, stats, cats, banner)
    _benchmark_report(out, bc, banner)
    return counts


def _ecosystem_summary(cfg, db, out: Path, stats, prof, banner, bc):
    manifest_p = cfg.data_dir / "raw" / "search_manifest.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}
    status = dict(db.query("SELECT COALESCE(pharo_status,'not_verified'), COUNT(*) FROM repositories GROUP BY 1"))
    strat = Counter()
    for r in db.query("SELECT discovery_json FROM repositories"):
        for s in {d["strategy"] for d in loads(r["discovery_json"], []) or []}:
            strat[s] += 1
    L = [f"# PharoTaskMiner — Ecosystem summary\n\n_Generated {now_iso()}_\n", banner]
    L.append("## Corpus\n")
    L.append(f"- Discovered repositories: **{sum(status.values())}** "
             f"(by strategy: {', '.join(f'{k} {v}' for k, v in strat.most_common())}).")
    if manifest:
        L.append(f"- Search tasks: {len(manifest.get('tasks', {}))}; tasks with incomplete or truncated results: "
                 f"{manifest.get('incomplete_tasks', 0)}; failed tasks: {manifest.get('failed_tasks', 0)}.")
    L.append("- Pharo identification: " + ", ".join(f"{k} = {v}" for k, v in sorted(status.items())) + ".")
    if stats:
        u = stats["uncertainty"]
        L.append(f"- Analysed: **{u['analysed_repositories']} repositories**, **{u['analysed_methods']} methods** "
                 f"(variant `{u['main_variant']}`); parse failures: {u['parse_failures']}; unsupported/failed source files: "
                 f"{u['unsupported_or_failed_files']}.")
        if u.get("study_corpus"):
            L.append(f"- Study corpus: {u['study_corpus']}.")
        dom = [d for d in stats["domains"] if d["domain"] != "unknown"]
        ops3 = [r for r in stats["operations"] if r.get("level") == 3]
        feats = stats["language_features"]
        L.append("\n## Findings (computed)\n")
        L.append(f"**Finding A — application domains.** {u['analysed_repositories']} analysed repositories carry "
                 f"{len(dom)} automatic domain labels. Most represented: "
                 + ", ".join(f"{d['domain']} ({d['repositories']})" for d in dom[:5])
                 + f". Repositories without any domain label: {u['repositories_without_domain']}.\n")
        L.append(f"**Finding B — programming operations.** Across {u['analysed_methods']} methods, "
                 f"{sum(1 for r in ops3 if r['methods'])} level-3 operation categories were observed. Most widespread across "
                 "repositories: " + ", ".join(f"{r['operation']} ({_pct(r['repo_prevalence'])})" for r in ops3[:6]) + ".\n")
        cross = [r for r in stats["cross_domain"] if r.get("level") == 3]
        if cross:
            most = max(r["domains_considered"] for r in cross)
            everywhere = [r["operation"] for r in cross if r["domains_with_operation"] == most and most]
            narrow = [r["operation"] for r in cross if 0 < r["domains_with_operation"] <= 2]
            L.append(f"**Finding C — domain differences.** {len(everywhere)} level-3 operations occur in all {most} domains "
                     f"with enough repositories; {len(narrow)} occur in at most two. See `domain_operation_matrix.csv`.\n")
        ps = [f for f in feats if f.get("concept_type") == "pharo_specific"]
        L.append("**Finding D — Pharo-specific capabilities.** Repository prevalence of Pharo-specific language features: "
                 + ", ".join(f"{f['feature']} {_pct(f['repo_prevalence'])}" for f in ps[:8]) + ".\n")
        if bc:
            c = Counter(r["existing_benchmark_coverage"] for r in bc["coverage"] if r["level"] == 3)
            L.append(f"**Finding E — benchmark coverage.** Level-3 operations: full {c.get('full', 0)}, partial {c.get('partial', 0)}, "
                     f"none {c.get('none', 0)} (explicitly mapped tasks only).\n")
        else:
            L.append("**Finding E — benchmark coverage.** Not computed yet (needs imported benchmark tasks and a validated taxonomy).\n")
        L.append("\n## Interpretation limits\n")
        L.append(f"- {u['confidence_interval_note']}\n- {u['interpretation_note']}\n- GitHub stars are reported as popularity "
                 "only, not as quality. Domain labels are keyword/metadata evidence, not proof.")
    else:
        L.append("\nRun `pharo-miner statistics` to compute findings.")
    (out / "ecosystem_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def _repository_selection(cfg, db, out: Path, profiles):
    run = db.scalar("SELECT run_id FROM analysis_runs WHERE stage='sample' AND status='completed' ORDER BY started_at DESC LIMIT 1")
    L = ["# Repository selection\n", f"_Generated {now_iso()}_\n"]
    sample_p = cfg.data_dir / "processed" / "sample.json"
    info = json.loads(sample_p.read_text()) if sample_p.exists() else {}
    if info:
        L.append(f"Strategy **{info.get('strategy')}**, seed {info.get('seed')}, requested size {info.get('requested_size')}, "
                 f"eligible {info.get('eligible')}. Decision counts: {info.get('counts')}.\n")
        if info.get("allocation"):
            L.append("Stratum allocation (`" + " | ".join(cfg['sampling']['strata']) + "`):\n")
            L.append(_md_table([{"stratum": k, "quota": v} for k, v in sorted(info["allocation"].items())],
                               [("stratum", "Stratum"), ("quota", "Quota")]))
    if run:
        reasons = db.query("SELECT decision, reason, COUNT(*) n FROM selection_decisions WHERE run_id=? GROUP BY decision, reason ORDER BY n DESC", (run,))
        L.append("\n## Inclusion and exclusion reasons\n")
        L.append(_md_table([dict(r) for r in reasons], [("decision", "Decision"), ("reason", "Reason"), ("n", "Repositories")]))
        sel = {r[0] for r in db.query("SELECT repo_id FROM selection_decisions WHERE run_id=? AND decision='included'", (run,))}
        rows = [p for p in profiles if p["repo_id"] in sel]
        L.append("\n## Selected repositories (quality indicators reported separately, not combined)\n")
        L.append(_md_table(rows, [("full_name", "Repository"), ("pharo_status", "Pharo"), ("primary_domain", "Domain"),
                                  ("size_category", "Size"), ("project_kind", "Kind"), ("core_infrastructure", "Core"),
                                  ("has_tests", "Tests"), ("has_readme", "Docs"), ("license", "License"),
                                  ("parseable", "Parsed"), ("reproducible_revision", "Pinned rev."), ("stars", "Stars")]))
    dups = db.query("SELECT r.full_name, d.kind, d.detail FROM repository_duplicates d JOIN repositories r ON r.repo_id=d.repo_id ORDER BY d.kind")
    L.append(f"\n## Duplicates and forks ({len(dups)})\n")
    L.append(_md_table([dict(r) for r in dups[:200]], [("full_name", "Repository"), ("kind", "Kind"), ("detail", "Detail")]))
    (out / "repository_selection.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def _operation_analysis(cfg, out: Path, stats, banner):
    L = ["# Programming-operation analysis (RQ2–RQ5)\n", f"_Generated {now_iso()}_\n", banner]
    if not stats:
        (out / "operation_analysis.md").write_text("\n".join(L) + "\nRun `pharo-miner statistics` first.\n", encoding="utf-8")
        return
    u = stats["uncertainty"]
    L.append(f"Denominators: {u['analysed_repositories']} repositories, {u['analysed_methods']} methods "
             f"(variant `{u['main_variant']}`). Repository-balanced frequency averages each repository's share of methods, "
             "so large repositories do not dominate. *Dominant share* = fraction of the operation's methods coming from its "
             "single largest repository.\n")
    pct = {k: _pct for k in ("repo_prevalence", "method_prevalence", "repo_balanced_frequency", "dominant_repository_share")}
    for lvl in (1, 2, 3):
        rows = [r for r in stats["operations"] if r.get("level") == lvl]
        L.append(f"\n## Level-{lvl} categories\n")
        L.append(_md_table(rows[:60], [("operation", "Operation"), ("concept_type", "Concept"), ("repositories", "Repos"),
                                       ("repo_prevalence", "Repo prev."), ("methods", "Methods"), ("method_prevalence", "Method prev."),
                                       ("repo_balanced_frequency", "Balanced"), ("dominant_repository_share", "Dominant share"),
                                       ("domain_count", "Domains")], pct))
    L.append("\n## Method label status\n")
    L.append(_md_table(stats["statuses"], [("status", "Status"), ("methods", "Methods"), ("method_prevalence", "Share"),
                                           ("repositories", "Repos")], {"method_prevalence": _pct}))
    L.append("\n## Language features and Pharo-specific capabilities (RQ3)\n")
    L.append(_md_table(stats["language_features"], [("feature", "Feature"), ("concept_type", "Concept"),
                                                    ("repo_prevalence", "Repo prev."), ("method_prevalence", "Method prev."),
                                                    ("repo_balanced_frequency", "Balanced")], pct))
    L.append("\n## Frameworks and APIs (RQ3)\n")
    L.append(_md_table(stats["frameworks"][:30], [("framework", "Framework"), ("repositories", "Repos"),
                                                  ("repo_prevalence", "Repo prev."), ("referencing_methods", "Methods")],
                       {"repo_prevalence": _pct}))
    L.append("\n## Most widespread selectors (raw vocabulary, RQ4)\n")
    L.append(_md_table(stats["selectors"][:40], [("selector", "Selector"), ("repositories", "Repos"), ("repo_prevalence", "Repo prev."),
                                                 ("occurrences", "Occurrences"), ("dominant_repository_share", "Dominant share")],
                       {"repo_prevalence": _pct, "dominant_repository_share": _pct}))
    L.append("\n## Common selector combinations\n")
    L.append(_md_table(stats["selector_cooccurrence"][:30], [("selector_a", "A"), ("selector_b", "B"),
                                                             ("repositories", "Repos"), ("methods", "Methods"), ("lift", "Lift")]))
    L.append("\n## Domain-associated operations (RQ5; domains with ≥3 repositories)\n")
    L.append("Presence-based P(o|d) saturates for large repositories, so operations are ranked by the *balanced share lift*: "
             "the mean share of a domain's methods carrying the operation divided by the same mean over all repositories.\n")
    cells = [c for c in stats["domain_operation"] if c["sufficient_sample"] and c.get("balanced_share_lift")
             and c["domain"] != "unknown" and c["repositories_with_operation"] >= 2]
    cells.sort(key=lambda c: -c["balanced_share_lift"])
    L.append(_md_table(cells[:30], [("domain", "Domain"), ("operation", "Operation"), ("p_operation_given_domain", "P(o|d)"),
                                    ("balanced_share_in_domain", "Share in domain"), ("balanced_share_lift", "Share lift"),
                                    ("domain_repositories", "Domain repos")],
                       {"p_operation_given_domain": _pct, "balanced_share_in_domain": _pct}))
    L.append("\n## Sensitivity analysis\n")
    diffs = [s for s in stats["sensitivity"] if s["repo_prevalence_all"] is not None]
    big = sorted(diffs, key=lambda s: -abs((s["repo_prevalence_all"] or 0) - (s["repo_prevalence_variant"] or 0)))[:15]
    L.append("Largest changes in repository prevalence when removing duplicated code or test code:\n")
    L.append(_md_table(big, [("operation", "Operation"), ("variant", "Variant"), ("repo_prevalence_all", "All"),
                             ("repo_prevalence_variant", "Variant")], {"repo_prevalence_all": _pct, "repo_prevalence_variant": _pct}))
    L.append("\n## Uncertainty and missing data\n")
    for k, v in u.items():
        L.append(f"- **{k}**: {v}")
    disc_p = cfg.data_dir / "processed" / "discovery.json"
    if disc_p.exists():
        d = json.loads(disc_p.read_text())
        L.append("\n## Open discovery — vocabulary not covered by any rule\n")
        L.append(_md_table(d["uncovered_vocabulary"][:30], [("selector", "Selector"), ("repositories", "Repos"),
                                                            ("methods", "Methods"), ("example", "Example")]))
        L.append("\n## Open discovery — candidate operation groups among unknown/ambiguous methods\n")
        L.append(_md_table(d["signature_groups"][:25], [("signature", "Signature"), ("repositories", "Repos"), ("methods", "Methods")]))
    (out / "operation_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def _taxonomy_report(cfg, db, out: Path, stats, cats, banner):
    L = ["# Taxonomy report\n", f"_Generated {now_iso()}_\n", banner]
    llm_basis = ((stats or {}).get("uncertainty", {}).get("label_basis") or "").startswith("llm")
    files = (sorted(p for p in cfg.validation_dir.glob("taxonomy_v*.json") if p.name.endswith("-llm.json") == llm_basis)
             or sorted(cfg.validation_dir.glob("taxonomy_draft*.json")))
    doc = json.loads(files[-1].read_text()) if files else None
    if doc:
        L.append(f"Source: `{files[-1].name}` (basis: {doc['validation_basis']}).\n")
        rows = [{"id": c["id"], "level": c["level"], "parent": c["parent"], "status": c["status"], "concept_type": c["concept_type"],
                 "repo_prevalence": (c.get("prevalence") or {}).get("repo_prevalence"),
                 "precision": (c.get("reference_validation") or c.get("human_validation") or {}).get("precision"),
                 "example": (c["evidence_examples"][0]["repository"] + ": " + c["evidence_examples"][0]["class"] + ">>"
                             + c["evidence_examples"][0]["method"]) if c.get("evidence_examples") else None} for c in doc["categories"]]
        L.append(_md_table(rows, [("id", "Category"), ("level", "L"), ("parent", "Parent"), ("status", "Status"),
                                  ("concept_type", "Concept"), ("repo_prevalence", "Repo prev."), ("precision", "Precision vs reference"),
                                  ("example", "Example")], {"repo_prevalence": _pct}))
        L.append(f"\n## Proposed categories ({len(doc['proposed_categories'])})\n")
        L.append(_md_table(doc["proposed_categories"][:40], [("id", "Proposal"), ("source", "Source"), ("repositories", "Repos"),
                                                             ("methods", "Methods"), ("votes", "Votes")]))
    else:
        L.append("No taxonomy document built yet (`pharo-miner build-taxonomy`). Seed categories:\n")
        L.append(_md_table(cats, [("id", "Category"), ("level", "L"), ("parent", "Parent"), ("concept_type", "Concept"),
                                  ("definition", "Definition")]))
    ag = cfg.validation_dir / "agreement_report.json"
    if ag.exists():
        a = json.loads(ag.read_text()).get("agreement") or {}
        L.append("\n## Inter-annotator agreement\n")
        L.append(f"Items: {a.get('items')}; multiply annotated: {a.get('items_multiply_annotated')}; exact set agreement: "
                 f"{a.get('exact_set_agreement')}; mean Jaccard: {a.get('mean_jaccard')}.\n")
        pc = [{"category": k, **{x: v[x] for x in ("items_with_label", "krippendorff_alpha")}} for k, v in (a.get("per_category") or {}).items()]
        L.append(_md_table(pc, [("category", "Category"), ("items_with_label", "Items"), ("krippendorff_alpha", "α")]))
    llm_ag = cfg.validation_dir / "llm" / "agreement_report.json"
    if llm_ag.exists():
        a = json.loads(llm_ag.read_text())
        L.append("\n## Agreement between LLM annotators\n")
        L.append(f"Annotators: {', '.join(a.get('annotators') or [])}; items: {a.get('items')}; exact set agreement: "
                 f"{a.get('exact_set_agreement')}; mean Jaccard: {a.get('mean_jaccard')}. "
                 "Details: `validation/llm/run_manifest.json`, `validation/llm/spotcheck_report.json`.\n")
        pc = [{"category": k, **{x: v[x] for x in ("items_with_label", "krippendorff_alpha")}} for k, v in (a.get("per_category") or {}).items()]
        L.append(_md_table(pc, [("category", "Category"), ("items_with_label", "Items"), ("krippendorff_alpha", "α")]))
    (out / "taxonomy_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def _benchmark_report(out: Path, bc, banner):
    L = ["# Benchmark coverage (RQ6)\n", f"_Generated {now_iso()}_\n", banner]
    if not bc:
        L.append("Not computed. Import benchmark task metadata with explicit operation mappings and run "
                 "`pharo-miner benchmark-coverage` (requires a validated taxonomy).")
    else:
        L.append("Coverage counts only tasks whose operations were mapped explicitly; descriptions are never keyword-matched.\n")
        L.append("Recommendation rules:\n" + "\n".join(f"- {r}" for r in bc["rules"]) + "\n")
        rows = [r for r in bc["coverage"] if r["level"] == 3]
        L.append(_md_table(rows, [("operation", "Operation"), ("repo_prevalence", "Repo prev."),
                                  ("existing_benchmark_coverage", "Coverage"), ("covering_tasks", "Tasks"),
                                  ("native_or_validated_tasks", "Native"), ("portable_tasks", "Portable"),
                                  ("tasks_needing_adaptation", "Adapt"), ("difficulty_concentrated", "Diff. concentrated")],
                           {"repo_prevalence": _pct}))
        L.append("\n## Recommendations (review_status = pending)\n")
        L.append(_md_table(bc["recommendations"], [("operation", "Operation"), ("recommendation", "Recommendation"),
                                                   ("existing_benchmark_coverage", "Coverage"), ("basis", "Basis"),
                                                   ("reasons", "Reasons")], {"reasons": lambda v: "; ".join(v)}))
    (out / "benchmark_coverage.md").write_text("\n".join(L) + "\n", encoding="utf-8")
