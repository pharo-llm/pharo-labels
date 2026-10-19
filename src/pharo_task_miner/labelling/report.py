"""results/REPORT.md: the whole study from zero, one section per step.

1. GitHub mining and filter 1 (is this real Pharo code?)
2. Filter 2 (the study corpus) and the method population
3. Sample size and inductive labelling (open coding), with statistics
4. The saturation loop
5. The codebook, classified by three models: agreement, reliability gate, expert sample
6. The experts' answers: accuracy, agreement, corrected shares, final taxonomy
7. What the labelling would cost on a commercial API
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from ..storage.database import Database
from .classify import RULE

QUALITY_ORDER = [
    ("quality: no Pharo package source (Tonel/FileTree)", "Has Tonel or FileTree source files"),
    ("quality: coursework, tutorial or programming puzzle", "Not coursework, tutorial or kata"),
    ("quality: fewer than", "commits"),
    ("quality: fewer than", "Pharo source files"),
    ("quality: no tests", "Has test classes"),
    ("quality: single author, no star, no fork", "≥ 2 contributors, or ≥ 1 star, or ≥ 1 fork"),
]


def _n(x) -> str:
    return f"{x:,}" if isinstance(x, int) else str(x)


def _pct(x) -> str:
    return f"{x:.1%}"


def _usd(x) -> str:
    if x is None:
        return "–"
    return f"${x:,.2f}" if x < 10 else f"${x:,.0f}"


def mining_section(cfg, db: Database) -> tuple[list[str], dict]:
    raw = cfg.data_dir / "raw" / "search_manifest.json"
    L, nums = [], {}
    total = db.scalar("SELECT COUNT(*) FROM repositories")
    nums["discovered"] = total
    L += ["## 1. GitHub mining and filter 1: is this real Pharo code?\n"]
    if raw.exists():
        man = json.loads(raw.read_text())
        per = defaultdict(lambda: [0, 0])
        for t in man.get("tasks", {}).values():
            per[t["strategy"]][0] += 1
            per[t["strategy"]][1] += t.get("retrieved") or 0
        L += [f"GitHub was searched with {sum(v[0] for v in per.values())} queries (search started "
              f"{man.get('created_at', '')[:10]}). A repository found by several queries counts once.\n",
              "| Search strategy | Queries | Results (with overlaps) |", "|---|---:|---:|"]
        L += [f"| {k} | {_n(v[0])} | {_n(v[1])} |" for k, v in sorted(per.items())]
        L.append("")
    status = dict(db.query("SELECT pharo_status, COUNT(*) FROM repositories GROUP BY 1"))
    pharo = status.get("confirmed_pharo", 0) + status.get("probable_pharo", 0)
    L += [f"**{_n(total)} unique candidate repositories.** Each was checked for Pharo evidence "
          "(Tonel/FileTree files, `BaselineOf`, `.project`, Pharo CI) by reading its file tree:\n",
          "| Verification | Repositories |", "|---|---:|"]
    for k in ("confirmed_pharo", "probable_pharo", "uncertain", "non_pharo"):
        L.append(f"| {k} | {_n(status.get(k, 0))} |")
    L.append("")
    run = db.scalar("SELECT run_id FROM analysis_runs WHERE stage='sample' AND status='completed' ORDER BY started_at DESC LIMIT 1")
    reasons = Counter(dict(db.query("SELECT reason, COUNT(*) FROM selection_decisions WHERE run_id=? GROUP BY 1", (run,)))) if run else Counter()
    included = db.scalar("SELECT COUNT(*) FROM selection_decisions WHERE run_id=? AND decision='included'", (run,)) if run else 0
    # reasons recorded by sampling.repository_filters.eligibility before the quality criteria
    dup = sum(v for k, v in reasons.items() if not k.startswith(("quality:", "pharo_status", "selected by"))
              and (" of repository " in k or k.startswith(("fork excluded", "archived repository"))))
    q = cfg["repositories"]["quality"]
    remaining = pharo - dup
    L += ["Filter 1 keeps the Pharo repositories with real, maintained code. The criteria are applied in this "
          "order; each repository is counted at the first one it fails. Stars are not a criterion: half of the "
          "Pharo repositories have none.\n",
          "| # | Criterion | Removed | Remaining |", "|---|---|---:|---:|",
          f"| | Pharo verified (confirmed or probable) | {_n(total - pharo)} | {_n(pharo)} |",
          f"| | Not a fork, mirror or duplicate of another candidate | {_n(dup)} | {_n(remaining)} |"]
    labels = {"commits": f"≥ {q.get('min_commits')} commits", "Pharo source files": f"≥ {q.get('min_pharo_source_files')} Pharo source files"}
    i = 0
    for prefix, label in QUALITY_ORDER:
        n = sum(v for k, v in reasons.items() if k.startswith(prefix) and (prefix != "quality: fewer than" or k.endswith(label)))
        remaining -= n
        i += 1
        L.append(f"| {i} | {labels.get(label, label)} | {_n(n)} | {_n(remaining)} |")
    L.append("")
    snaps = dict(db.query("SELECT clone_status, COUNT(*) FROM repository_snapshots GROUP BY 1"))
    parsed = db.scalar("SELECT COUNT(DISTINCT repo_id) FROM repository_snapshots WHERE n_methods > 0")
    methods = db.scalar("SELECT SUM(n_methods) FROM repository_snapshots WHERE n_methods > 0 AND snapshot_id IN "
                        "(SELECT MAX(snapshot_id) FROM repository_snapshots GROUP BY repo_id)")
    nums.update({"pharo": pharo, "filter1": included, "parsed": parsed, "methods_mined": methods})
    L += [f"**{_n(included)} repositories pass filter 1.** They were cloned ({_n(snaps.get('cloned', 0))} cloned, "
          f"{_n(snaps.get('failed', 0))} failed) and every method was parsed: **{_n(parsed)} repositories with "
          f"{_n(methods or 0)} methods**. Every decision and its reason: `results/tables/repository_selection.csv`.\n"]
    return L, nums


def corpus_section(cfg, plan: dict | None) -> tuple[list[str], dict]:
    p = cfg.data_dir / "processed" / "corpus.json"
    if not p.exists():
        return ["## 2. Filter 2: the study corpus\n", "Not run yet: `./run.sh filter`.\n"], {}
    c = json.loads(p.read_text())
    L = ["## 2. Filter 2: the study corpus\n",
         f"Filter 2 keeps the repositories that are engineered projects: licensed, developed by several people over "
         f"time, loadable in a current Pharo, tested and original. Each criterion is a separate threshold (no global "
         f"score), applied in this order (mining date for the last-push criterion: {c.get('mining_date') or 'unknown'}):\n",
         "| # | Criterion | Removed | Remaining | Fails alone |", "|---|---|---:|---:|---:|"]
    for i, r in enumerate(c["funnel"], 1):
        L.append(f"| {i} | {r['requirement']} | {_n(r['removed'])} | {_n(r['remaining'])} | {_n(r['fail_alone'])} |")
    L.append(f"\n*Fails alone* counts every repository that fails the criterion, whatever the others say.\n")
    selected = len(c["selected"]) if isinstance(c["selected"], list) else c["selected"]
    L.append(f"**{selected} repositories form the study corpus**"
             + (f" (chosen for diversity among {c['passing']} that pass every criterion)." if c.get("capped")
                else f" (all {c['passing']} that pass every criterion; the maximum is {c['max_repositories']}).")
             + " Their list, domains and owners: `results/reports/corpus_filter.md`; every repository and the criteria "
               "it fails: `results/tables/corpus_filter.csv`.\n")
    nums = {"filter2": selected}
    if plan:
        pop = plan["population"]
        nums["population"] = pop["methods"]
        L.append(f"The **population** to label is every method of these {pop['repositories']} repositories that parses, "
                 f"is not a copy of a method of another repository, and is in the latest snapshot: "
                 f"**{_n(pop['methods'])} methods** ({_n(pop['test_methods'])} of them in tests"
                 + ("" if pop["include_tests"] else ", excluded by `labelling.include_tests`") + ").\n")
    return L, nums


def labelling_sections(cfg, plan: dict, summ: dict) -> list[str]:
    ss = plan["sample_size"]
    N = plan["population"]["methods"]
    L = ["## 3. Sample size and labelling\n",
         "**Sample size.** Cochran's formula for a proportion with the finite population correction "
         "(Cochran 1977, eq. 4.2-4.3):\n",
         "```", "n0 = z² · p(1−p) / e²        n = n0 / (1 + (n0 − 1) / N)", "```\n",
         f"With confidence {ss['confidence']:.0%} (z = {ss['z']}), margin of error e = ±{ss['margin_of_error']:.1%}, "
         f"p = {ss['proportion']} (the most conservative value) and N = {_n(N)}: n0 = {ss['n0']}, "
         f"**n = {_n(ss['n'])}** methods"
         + (f" (a sample size of {_n(plan['initial_sample_size'])} was set by hand: margin of error "
            f"±{plan['initial_margin']:.1%})" if plan["initial_sample_size"] != ss["n"] else "") + ".\n",
         "For comparison, other choices for the same population:\n",
         "| Confidence | Margin | Sample size |", "|---|---:|---:|"]
    for row in plan["alternatives"]:
        L.append(f"| {row['confidence']:.0%} | ±{row['margin_of_error']:.0%} | {_n(row['n'])} |")
    L += ["",
          "**Draw.** Stratified random sampling, the repository as stratum, proportional allocation (Cochran 1977, "
          "ch. 5): every repository contributes in proportion to its number of methods, so the sample represents the "
          f"corpus without weights. Seed {plan['settings']['random_seed']}.\n",
          f"**Model.** `{plan['model']['name']}` ({plan['model']['description'].get('backend')}, greedy decoding). "
          "The model completes a few-shot prompt: for each method it writes a one-sentence summary, then a short, "
          "general name for the programming task the method performs (open coding). Each method is coded on its "
          "own: the prompt does not list the categories found so far, because small models copy such a list even "
          "when nothing in it fits. Twelve worked examples carry "
          f"{len(summ['seed_categories'])} seed categories ({', '.join(summ['seed_categories'])}).\n",
          "**Codebook.** A new name joins an existing category when it is equal after normalisation (case, plural, "
          f"punctuation), a spelling variant (character ratio ≥ {plan['settings']['merge_similarity']}), or"
          + (f" the same meaning: cosine similarity ≥ {plan['settings']['semantic_threshold']} between the embeddings "
             f"of the two names by `{plan['settings']['embedding_model']}`, a 45 MB sentence-embedding model (every "
             "such merge is listed with its similarity in `codebook.json` and shown to the experts)."
             if plan['settings'].get('embedding_model') else " (merging by meaning was off).")
          + f" At the end of each round, a name becomes a category once at least {plan['settings']['min_support']} "
          f"methods and at least {plan['settings'].get('min_share', 0):.1%} of all methods labelled so far have it; until "
          "then it is a *candidate*, listed in `taxonomy.csv` but not in the taxonomy. The share is what makes saturation "
          "reachable with a small model, which gives a one-off name to many methods, and it gives saturation a "
          f"precise meaning: the taxonomy aims at every task with a prevalence of at least "
          f"{plan['settings'].get('min_share', 0):.1%}; such a task stays unseen in n random methods with probability "
          "(1 − p)ⁿ.\n",
          "**Statistics of the labelling.**\n", "| | |", "|---|---:|",
          f"| Methods labelled (initial sample + saturation rounds) | {_n(summ['methods_labelled'])} |",
          f"| of which in the initial sample | {_n(summ['initial_sample'])} |",
          f"| Repositories they come from | {summ['repositories_covered']} |",
          f"| Test methods among them | {_n(summ['labelled_tests'])} ({_pct(summ['labelled_tests'] / summ['methods_labelled'])}) |",
          f"| Model answers without a usable task name | {_n(summ['no_answer'])} |",
          f"| Methods whose task name stayed too rare to be a category | {_n(summ['rare_names'])} |",
          f"| Categories in the taxonomy | {summ['categories']} |",
          f"| of which still at ≥ {plan['settings'].get('min_share', 0):.1%} of the labelled methods at the end | "
          f"{summ.get('categories_at_min_share', '–')} |",
          f"| Candidate names (below the support threshold) | {summ['candidates']} |",
          f"| Seed categories never used | {', '.join(summ['unused_seed_categories']) or 'none'} |",
          f"| Margin of error with all labelled methods (95%) | ±{summ['margin_of_error_all_labelled']:.1%} |",
          f"| Model time per method | {summ['seconds_per_method']} s |"]
    L += ["", "**Task categories.** Share of the corpus estimated from all labelled methods, 95% Wilson interval "
          "with the finite population correction, and the estimated number of corpus methods:\n",
          "| Category | Methods | Share [95% CI] | ≈ methods in corpus | Repositories | Tests | Seed |",
          "|---|---:|---|---:|---:|---:|:---:|"]
    for r in summ["taxonomy"]:
        name = r["category"] if r["in_taxonomy"] else f"*{r['category']}*"
        if r.get("below_min_share"):
            name += " †"
        L.append(f"| {name} | {r['methods']} | {_pct(r['share'])} [{_pct(r['ci_low'])}, {_pct(r['ci_high'])}] | "
                 f"{_n(r['estimated_methods_in_corpus'])} | {r['repositories']} | {_pct(r['test_share'])} | "
                 f"{'yes' if r['seed'] else ''} |")
    low = [r["category"] for r in summ["taxonomy"] if r.get("below_min_share")]
    if low:
        L.append(f"\n† {len(low)} categories entered the taxonomy early (in a round where they reached the thresholds) "
                 f"but are below {plan['settings'].get('min_share', 0):.1%} of all labelled methods at the end; they are "
                 "flagged for the experts (`below_min_share` in `taxonomy.csv`).")
    if summ["figures"].get("category_distribution"):
        L.append("\n![Task categories](figures/category_distribution.png)\n")
    L += ["## 4. The saturation loop\n",
          f"After the initial sample, each round draws {plan['settings']['batch_size']} new methods from the methods "
          "not drawn yet (same proportional allocation), labels them, and counts the categories that appear: a "
          f"category appears in the round where its name reaches both thresholds. The loop stops when {plan['settings']['patience']} rounds in a row add no "
          f"category (at most {plan['settings']['max_rounds']} rounds)"
          + (f". The patience is derived, not chosen: {plan['settings']['patience']} rounds of "
             f"{plan['settings']['batch_size']} methods cover at least {_n(plan['settings']['quiet_methods_needed'])} "
             f"methods, the run length in which a task with a prevalence of {plan['settings']['min_share']:.1%} "
             f"appears with probability {plan['sample_size']['confidence']:.0%} (1 − (1 − p)ᴸ ≥ "
             f"{plan['sample_size']['confidence']:.0%}).\n"
             if plan['settings'].get('patience_rule') == 'derived' else ".\n"),
          "| Round | Methods | Labelled so far | New categories | Categories | Candidates | New names proposed |",
          "|---:|---:|---:|---|---:|---:|---:|"]
    for c in summ["curve"]:
        new = f"{c['new_categories']}" + (f" ({c['new_category_names']})" if c["new_category_names"] else "")
        L.append(f"| {c['round']}{' (initial)' if c['round'] == 0 else ''} | {c['methods_in_round']} | "
                 f"{_n(c['methods_total'])} | {new} | {c['categories_total']} | {c['candidates_total']} | "
                 f"{c['new_names_proposed']} |")
    stop = plan.get("result", {}).get("stop_reason")
    L.append("\n" + {"saturated": f"**Saturation reached** after {summ['rounds']} rounds: "
                                  f"{plan['settings']['patience']} rounds in a row added no category.",
                     "max_rounds": f"**Saturation not reached** within {plan['settings']['max_rounds']} rounds: "
                                   "increase `MAX_ROUNDS` (the labels done so far are kept).",
                     "population_exhausted": "Every method of the corpus was labelled."}.get(stop, "") + "\n")
    if summ.get("unseen_name_probability") is not None and summ["curve"]:
        L.append(("The categories saturate; the long tail of rare names does not" if stop == "saturated" else
                  "The long tail of rare names") + ": the candidates grew from "
                 f"{summ['curve'][0]['candidates_total']} to {summ['curve'][-1]['candidates_total']} over the loop. "
                 f"{summ['singleton_names']} names were given to a single method, so the probability that the next method "
                 f"gets a name never seen before is about {_pct(summ['unseen_name_probability'])} (Good–Turing estimate, "
                 f"Good 1953: methods with a name seen once / methods labelled). Such names are each below "
                 f"{plan['settings'].get('min_share', 0):.1%}: they are what the taxonomy leaves under Other by design.\n")
    if summ["figures"].get("saturation_curve"):
        L.append("![Saturation curve](figures/saturation_curve.png)\n")
    return L + classify_section(cfg)


def _f(x, pct=False) -> str:
    if x is None:
        return "–"
    return _pct(x) if pct else str(x)


def classify_section(cfg) -> list[str]:
    cdir = cfg.labelling_dir / "classify"
    L = ["## 5. The codebook, classified by three models\n",
         "Open codes cannot be compared across models (two names for one task), so the inductive categories become "
         "a codebook with definitions (`codebook_draft.csv`, consolidated by the researcher in `codebook.csv`), and "
         "models of different families classify a **new** random sample with it (deductive coding: Shah et al. 2025; "
         "taxonomy then classification: TnT-LLM, KDD 2024). Different families, because errors of related models are "
         "correlated (Kim et al., ICML 2025). Each codebook entry has a definition, when to use it and when not to "
         "(MacQueen et al. 1998); categories below the minimum share at the end of the loop are left to Other. The "
         f"models, like the experts, follow one rule: {RULE}\n"]
    p = cdir / "agreement.json"
    if not p.exists():
        return L + ["Not run yet: `./run.sh classify`.\n"]
    a = json.loads(p.read_text())
    L += [f"**Codebook**: {len(a['codebook'])} categories including Other ("
          + ("consolidated by the researcher" if a["codebook_kind"] == "consolidated" else
             "**the model's draft, not consolidated**: copy `codebook_draft.csv` to `codebook.csv` and edit it")
          + f"). **Sample**: {_n(a['sample'])} methods drawn like the initial sample, none of them used to build the codebook.\n",
          "| Agreement of the models | |", "|---|---:|",
          f"| **Krippendorff's α** (nominal, {len(a['models'])} models) | **{_f(a['krippendorff_alpha'])}** |",
          f"| Fleiss' κ | {_f(a['fleiss_kappa'])} |",
          f"| Gwet's AC1 (robust to a dominant category) | {_f(a['gwet_ac1'])} |"]
    for pr in a["pairs"]:
        L.append(f"| `{pr['models'][0]}` / `{pr['models'][1]}`: same category, Cohen's κ | "
                 f"{_pct(pr['agreement'])}, κ = {_f(pr['cohen_kappa'])} |")
    total = sum(a["levels"].values())
    for lv in sorted(a["levels"], reverse=True):
        L.append(f"| Methods where {lv if lv != 'none' else 'no majority'} models agree | "
                 f"{_n(a['levels'][lv])} ({_pct(a['levels'][lv] / total)}) |")
    sc = a["selfcheck"]
    if sc["methods"]:
        L.append(f"| Prompt sensitivity: `{sc['model']}` again on {sc['methods']} methods, codebook and examples "
                 f"shuffled | {_pct(sc['agreement'])}, κ = {_f(sc['cohen_kappa'])} |")
    for m, k in a["no_answer"].items():
        if k:
            L.append(f"| `{m}`: answers outside the codebook | {k} |")
    L.append("")
    revs = a.get("revisions") or []
    if len(revs) > 1:
        L += [f"**Revisions** of the instrument (codebook or models), each measured on its own sample, disjoint from "
              "the earlier ones; all are reported, the last one is the one above:\n",
              "| Revision | Models | Codebook | Categories | α | AC1 | Other, per model | Gate |", "|---:|---|---|---:|---:|---:|---|---|"]
        for r in revs:
            L.append(f"| {r['revision']} | " + ", ".join(f"`{m}`" for m in r["models"]) + f" | {r['codebook_kind']} | "
                     f"{r['categories']} | {_f(r['krippendorff_alpha'])} | {_f(r['gwet_ac1'])} | "
                     + ", ".join(_pct(v) for v in r["other_share"].values()) + f" | {'passed' if r['reliable'] else 'not passed'} |")
        L.append("")
    L.append(f"**Gate** (Krippendorff 2004: α ≥ 0.8 reliable, ≥ 0.667 tentative; threshold here {a['min_alpha']}): "
             + ("**passed**: the experts get the files below.\n" if a["reliable"] else
                "**not passed**: no expert files; what to change is in `classify/NOT_RELIABLE.md`.\n"))
    L += ["Share of each category among the classified methods, per model (not yet corrected by the experts):\n",
          "| Category | " + " | ".join(f"`{m}`" for m in a["models"]) + " | Majority |",
          "|---|" + "---:|" * (len(a["models"]) + 1)]
    for c, d in sorted(a["distribution"].items(), key=lambda kv: -kv[1]["majority"]):
        L.append(f"| {c} | " + " | ".join(_pct(d[m]) for m in a["models"]) + f" | {_pct(d['majority'])} |")
    e = a.get("experts")
    if e:
        L += ["", f"**Expert sample**: {e['methods']} methods, stratified by agreement level, more where the models "
              "disagree (Neyman allocation), every stratum checked, including unanimous methods where shared errors "
              "hide; the weights make the estimates unbiased (Gligorić et al., NAACL 2025). The experts classify "
              "blind: the models' answers are not shown.\n",
              "| Agreement level | Methods | Checked by the experts |", "|---|---:|---:|"]
        for h, st in sorted(e["strata"].items(), reverse=True):
            L.append(f"| {h} | {_n(st['N'])} | {st['m']} |")
        L += ["", "Files: `labelling/expert_validation/` (README.md, method_review.csv, codebook_review.csv, examples.md). "
              "Each expert's filled files go in `expert_validation/returned/<expert name>/`, the agreed answers after "
              "discussion in `returned/consensus/`; then `./run.sh validate`.\n"]
    return L


def validation_section(cfg) -> list[str]:
    p = cfg.labelling_dir / "validated" / "validation.json"
    L = ["## 6. The experts' answers\n"]
    if not p.exists():
        return L + ["No expert answers yet: put each expert's files in "
                    "`results/labelling/expert_validation/returned/<expert name>/`, then `./run.sh validate`.\n"]
    v = json.loads(p.read_text())

    def acc(a):
        return f"{_pct(a['estimate'])} [{_pct(a['ci_low'])}, {_pct(a['ci_high'])}]" if a["checked"] else "–"
    ref = ", ".join(f"{k}: {n}" for k, n in v["reference"].items())
    L += [f"Experts: {', '.join(v['experts'])}. Reference label of a method: the experts' common answer, or their "
          f"consensus after discussion ({ref}). Accuracies are stratified estimates over the agreement levels "
          "(95% intervals).\n",
          "| | |", "|---|---:|"]
    if v["human_alpha"] is not None:
        L.append(f"| **Experts' agreement** (Krippendorff's α, the human baseline) | **{v['human_alpha']}** |")
    for pr in v["expert_pairs"]:
        L.append(f"| {pr['experts'][0]} / {pr['experts'][1]}: Cohen's κ ({pr['methods']} methods) | {_f(pr['cohen_kappa'])} |")
    L.append(f"| **Accuracy of the majority label** | **{acc(v['accuracy']['majority'])}** |")
    for m in v["models"]:
        L.append(f"| Accuracy of `{m}` | {acc(v['accuracy'][m])} |")
    for who, al in v["model_vs_reference_alpha"].items():
        L.append(f"| α between {'the majority label' if who == 'majority' else f'`{who}`'} and the reference | {_f(al)} |")
    for h, b in v["accuracy"]["majority"]["by_level"].items():
        L.append(f"| Majority label correct when {h if h != 'none' else 'no majority'} models agree | "
                 f"{_pct(b['accuracy'])} ({b['correct']}/{b['checked']}) |")
    d = v["decisions"]
    L += [f"| Codebook: dropped / renamed / merged / definitions fixed | {d['dropped']} / {d['renamed']} / {d['merged']} / "
          f"{d['definitions_fixed']} |", f"| Items still to discuss (`to_discuss.csv`) | {d['to_discuss']} |", "",
          "A model can stand in for a human rater when its agreement with the reference is close to the experts' "
          "agreement with each other (Ahmed et al., MSR 2025).\n",
          "**Final taxonomy.** Share of each task in the corpus: the models' share on the classified sample corrected "
          "with the experts' answers (prediction-powered inference, Angelopoulos et al., Science 2023), so the model's "
          "errors do not bias it; precision and recall of the majority label:\n",
          "| Category | Definition | Share [95% CI] | Models only | ≈ methods in corpus | Precision | Recall |",
          "|---|---|---|---:|---:|---:|---:|"]
    for t in v["final_taxonomy"]:
        L.append(f"| {t['category']} | {t['definition']} | {_pct(t['share'])} [{_pct(t['ci_low'])}, {_pct(t['ci_high'])}] | "
                 f"{_pct(t['model_share'])} | {_n(t['estimated_methods_in_corpus'])} | {_f(t['precision'])} | {_f(t['recall'])} |")
    L.append("\nFiles: `results/labelling/validated/` (final_taxonomy.csv, reference_labels.csv, to_discuss.csv).\n")
    return L


def cost_section(cfg, db: Database, plan: dict | None, summ: dict | None) -> list[str]:
    from . import cost, sampling
    from ..sampling.corpus_filter import study_repo_ids
    study = study_repo_ids(db)
    if not study:
        return []
    e = cost.estimate(db, cfg, sampling.population(db, study, True), cfg.labelling_dir, plan, summ)
    c, t = cfg["api_cost"], e["tokens"]
    table = cfg.output_dir / "tables" / "api_cost.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(e["prices"][0]))
        w.writeheader()
        w.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in e["prices"])
    lo, hi = c["budget_factor"]
    L = ["## 7. What the labelling would cost on a commercial API\n",
         f"Tokens counted with {e['tokenizer']} on the prompts this labelling builds; prices of "
         f"{c['prices_date']} (USD per million tokens, in `config.yaml`, `api_cost`). Every price below is for "
         "one model; all numbers: `results/tables/api_cost.csv`.\n",
         "| Tokens per call | |", "|---|---:|",
         f"| Shared prefix: instructions and {len(cost.coding.EXAMPLES)} worked examples (+ chat wrapping) | "
         f"{_n(e['prefix'])} (+ {e['chat_overhead']}) |"]
    if e["method_mean_this_run"] is not None:
        L.append(f"| Method, mean over the calls of this run | {e['method_mean_this_run']:,.0f} |")
    L += [f"| Method, mean over the whole corpus | {e['method_mean_corpus']:,.0f} |",
          f"| Answer (summary and task), mean | {e['output_mean']:,.0f} |", "",
          f"{_n(len(t['cut_by_cap']))} methods of the corpus are cut by the character cap (`max_method_chars`); "
          f"without it, {_n(t['too_long_without_cap'])} prompts would not fit in the labeller's "
          f"{_n(cost.NUM_CTX)}-token context.\n"]
    for sc in e["scenarios"]:
        what = (f"every call of the inductive labelling ({_n(sc['calls'])} methods)"
                if sc["scenario"] == "This run" else f"every method of the corpus once ({_n(sc['calls'])} calls)")
        rows = [r for r in e["prices"] if r["scenario"] == sc["scenario"]]
        L += [f"**{sc['scenario']}**: {what}; {rows[0]['input_mtok']:,.2f}M input and "
              f"{rows[0]['output_mtok']:,.2f}M output tokens (GPT count).\n",
              "| Model | Standard | Batch API | Batch, reasoning model | Batch + prompt cache | **Total budget** |",
              "|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            L.append(f"| {r['model']} | {_usd(r['standard'])} | {_usd(r['batch'])} | {_usd(r['reasoning_batch'])} | "
                     f"{_usd(r['cached_batch'])} | **{_usd(r['budget_low'])} – {_usd(r['budget_high'])}** |")
        L.append("")
    std = e["levers"][0][1]
    L += [f"*Reasoning model*: + {c['reasoning_tokens_per_call']} hidden tokens per call billed as output (low "
          "effort; an assumption, measure it on 50 calls). *Prompt cache*: the shared prefix is cached only from "
          f"{c['cache_min_prefix_tokens']} tokens; it is assumed padded to that size with more worked examples and "
          "read from the cache on every call (best case). *Total budget*: Batch API without reasoning "
          f"× {lo} to Batch API with reasoning × {hi}, for a pilot, retries, the classification models and a rerun. Claude "
          "counts more tokens than GPT on code: × 1.25 assumed (`token_factor`).\n",
          f"**How to reduce it** ({e['reference']}, whole corpus):\n",
          "| What | Cost | Compared with the standard API |", "|---|---:|---:|"]
    for name, v in e["levers"]:
        d = None if v is None else v / std - 1
        L.append(f"| {name} | {_usd(v)} | {'–' if d is None else f'{d:+.1%}' if 0.99 < abs(d) < 1 else f'{d:+.0%}'} |")
    if e["local"]:
        L.append(f"\nThe local model of this run (`{e['local']['model']}`) labels the whole corpus in about "
                 f"{e['local']['hours']:,.0f} hours, with no API cost.")
    L.append("\nThe saturation loop sends one batch per round: with the Batch API, each of its rounds can wait up to "
             "24 hours. Labelling the whole corpus is a single batch.\n")
    return L


def write(cfg, db: Database) -> Path:
    out = cfg.output_dir / "REPORT.md"
    ldir = cfg.labelling_dir
    plan = json.loads((ldir / "plan.json").read_text()) if (ldir / "plan.json").exists() else None
    summ = json.loads((ldir / "summary.json").read_text()) if (ldir / "summary.json").exists() else None
    s1, n1 = mining_section(cfg, db)
    s2, n2 = corpus_section(cfg, plan)
    nums = {**n1, **n2}
    funnel = [("GitHub candidate repositories", nums.get("discovered")), ("Verified Pharo", nums.get("pharo")),
              ("Filter 1 (6 quality criteria)", nums.get("filter1")), ("Cloned and parsed", nums.get("parsed")),
              ("Filter 2 (13 criteria): study corpus", nums.get("filter2"))]
    L = ["# Pharo task taxonomy: from GitHub to expert validation\n",
         "Generated by `./run.sh`. Every number below is computed from `results/`.\n",
         "| Step | Repositories | Methods |", "|---|---:|---:|"]
    for name, v in funnel:
        L.append(f"| {name} | {_n(v) if v is not None else '–'} | "
                 f"{_n(nums['methods_mined']) if name == 'Cloned and parsed' and nums.get('methods_mined') else (_n(nums['population']) if name.startswith('Filter 2') and nums.get('population') else '')} |")
    if summ:
        L += [f"| Initial sample (sample-size calculator) | | {_n(summ['initial_sample'])} |",
              f"| Labelled after the saturation loop | | {_n(summ['methods_labelled'])} |",
              f"| **Task categories** (found during the loop; still at ≥ "
              f"{plan['settings'].get('min_share', 0):.1%} at the end) | | **{summ['categories']}** "
              f"({summ.get('categories_at_min_share', '–')}) |"]
    L.append("")
    L += s1 + s2
    if plan and summ:
        L += labelling_sections(cfg, plan, summ)
        L += validation_section(cfg)
    else:
        L += ["## 3-5. Labelling\n", "Not run yet: `./run.sh label`.\n"]
    L += cost_section(cfg, db, plan, summ)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out
