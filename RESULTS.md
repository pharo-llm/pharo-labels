# Results: what each phase does, what came out, and what the figures show

Full run (`PROFILE=full`) on Grid'5000 vianden (8× AMD MI300X), 26–27 October 2026.
All 13 phases finished. Three AI models were used: **qwen3.8 27B**, **gemma4 31B** and **mistral-small3.2 24B**.

The figure links point to `results/`, which is on Luxembourg (`~/pharo-labels-pbr/results/`).

**✓ done** = final. **◐ provisional** = complete, but based on the category list (taxonomy) the AIs built, which
the experts have not checked yet. When the expert files come back, a rerun updates these results (the cache makes it fast).

---

## The big picture in 6 lines

1. We took **1,087 good Pharo projects** (1.8 M methods) and **12 well-known coding benchmarks** (6,433 tasks, mostly Python).
2. Three AIs read samples of both and built a **list of "what this code does" behaviours**: 18 categories and 360 behaviours for Pharo.
3. The AIs then labelled new samples with that list. They **agree only weakly** (α = 0.25 on exact behaviours, 0.51 on the main category).
4. In real Pharo code, **"object state" (44 %) and "collections" (23 %)** dominate. Existing benchmarks are mostly collections, maths and strings.
5. From the gaps, the pipeline wrote **100 new Pharo tasks**. **25 run**, **19 pass all execution checks**, and **6 are fully auto-verified**. The other 13 wait for an expert decision.
6. On those 6 tasks (18 variants), **mistral 89 %, gemma 78 %, qwen 67 %**. That is too few tasks to rank the models.

---

## Phase 01: Repository selection ✓

**What it does.** It decides which Pharo projects the study uses. Each of the 1,585 projects gets a quality score
from 0 to 1, built from 11 criteria: contributors, commits, tests, maintenance, documentation, size, originality,
licence, issues, loadability and popularity. Projects that are forks, mirrors (mostly copied code) or have no
parsable code are removed first. Projects scoring below 0.5 are dropped.

**Result.** **1,087 kept**, 367 dropped by the score, 127 mirrors, 4 with no Pharo code. The median score of kept
projects is 0.63, and the kept projects hold 1,842,164 methods.

**Small analysis.** The scores form a bell shape around 0.55, so the 0.5 cut falls right in the crowded middle. A small
change of threshold moves a lot of projects: 0.45 keeps about 1,270 and 0.55 about 870. The kept projects mainly stand
out by **maintenance, documentation, loadability and issues**, not by popularity, which is what was intended.

| Figure | What it shows |
|---|---|
| [decisions.png](results/01_repositories/figures/decisions.png) | How many projects were kept or removed, and why |
| [score_distribution.png](results/01_repositories/figures/score_distribution.png) | Histogram of quality scores with the 0.5 cut line |
| [threshold_sensitivity.png](results/01_repositories/figures/threshold_sensitivity.png) | How many projects and methods another threshold would keep |
| [criteria_kept_vs_excluded.png](results/01_repositories/figures/criteria_kept_vs_excluded.png) | Average score of each criterion, kept vs dropped projects |
| [score_contributions.png](results/01_repositories/figures/score_contributions.png) | Which criteria contribute most to the score (history, tests, community) |
| [popularity_vs_score.png](results/01_repositories/figures/popularity_vs_score.png) | Stars vs score: many 0-star projects are kept, so popularity does not decide |

## Phase 02: Literature ingestion ✓

**What it does.** It loads every task of the existing AI coding benchmarks (HumanEval+, MBPP+, BigCodeBench,
LiveCodeBench, SWE-bench, Terminal-Bench, NL2Repo). These are what "the literature" tests today.

**Result.** **12 benchmarks, 6,433 tasks (4,372 unique problems)**. DeepSWE is missing: it needs a Hugging Face token (`HF_TOKEN`).

**Small analysis.** Existing benchmarks are **88 % Python**, and mostly **code generation** (2,677 tasks), then bug
repair (1,442) and code completion (1,304). None is in Pharo or Smalltalk, which is the motivation for the study.

| Figure | What it shows |
|---|---|
| [tasks_per_benchmark.png](results/02_literature/figures/tasks_per_benchmark.png) | Number of tasks per benchmark, coloured by task type |
| [task_types.png](results/02_literature/figures/task_types.png) | Tasks per type (generation, repair, completion, test generation, other) |
| [languages.png](results/02_literature/figures/languages.png) | Programming language of the tasks (Python: 5,639 of 6,433) |

## Phase 03: Pharo corpus preparation ✓

**What it does.** It extracts every method of the kept projects and records where it comes from (project, commit,
file, line) and its structure (loops, conditionals, and so on). It finds copies, and filters out methods that say
nothing about behaviour: duplicates, getters, setters, empty methods, constants, generated code.

**Result.** Of 1,842,164 methods: **571,454 kept**, 283,648 kept separately as tests, 121,971 "to review", and
865,091 excluded.

**Small analysis.** **31 % of all methods are copies**: about 309 k come from another project and 265 k repeat within
the same project. Most production methods are tiny (1–2 statements), and tests are usually around 5 statements. The
corpus is very uneven: the **top 10 % of projects hold 69 % of the methods**. That is why sampling (phase 4) caps the
number of methods per project.

| Figure | What it shows |
|---|---|
| [method_filters.png](results/03_corpus/figures/method_filters.png) | How many methods each filter rule excluded, flagged or kept |
| [duplication.png](results/03_corpus/figures/duplication.png) | Copies between projects vs inside a project (exact vs normalised) |
| [method_size.png](results/03_corpus/figures/method_size.png) | Statements per method, production vs test |
| [features.png](results/03_corpus/figures/features.png) | Share of methods with each structural feature (instance creation 24 %, cascade 18 %, conditional 15 %, …) |
| [methods_per_repository.png](results/03_corpus/figures/methods_per_repository.png) | Cumulative curve: a few large projects hold most methods |
| [repository_groups.png](results/03_corpus/figures/repository_groups.png) | Projects per application domain (data formats, web, graphics, UI, …) |

## Phase 04: Representative sampling ✓

**What it does.** It draws fair random samples, so that results describe all of Pharo and not just the biggest
projects. It picks projects at random inside each domain, then a few methods per project, and each method gets a
weight to correct the estimates. It draws three separate samples: **discovery** (to build the categories),
**annotation** (to test agreement) and **profile** (for the large analysis). It also adds a "targeted" sample of rare
areas (files, network, concurrency, …) so they are not missed.

**Result.** 1,908 discovery, 494 annotation and 1,786 profile Pharo methods. 2,077 discovery and 360 annotation
literature tasks.

**Small analysis.** The literature sample matches its population perfectly (13 of 13 checks). The Pharo start sample
matches **11 of 17** checks. It has somewhat **fewer conditionals, cascades, instance creations and non-local
returns** than the corpus, and slightly more very short methods. Several small domains (education, documentation,
networking, testing, games) **do not have enough methods** to fill their share. This is listed as warnings, and the
weights correct the estimates, but these domains are noisier.

| Figure | What it shows |
|---|---|
| [representativeness_pharo.png](results/04_sampling/figures/representativeness_pharo.png) | Sample vs population for 17 code features (orange bar inside the interval = OK) |
| [representativeness_literature.png](results/04_sampling/figures/representativeness_literature.png) | Same check for benchmark tasks (13/13 OK) |
| [groups_corpus_vs_sample.png](results/04_sampling/figures/groups_corpus_vs_sample.png) | Share of each domain: corpus vs raw sample vs weighted sample |
| [methods_per_sampled_repository.png](results/04_sampling/figures/methods_per_sampled_repository.png) | Methods drawn per project. Note: the title says "at most 2", but about 440 projects have 4; worth checking |
| [targeted_signals.png](results/04_sampling/figures/targeted_signals.png) | How rare each targeted area is (streams 3 %, … database 0 %) |

## Phase 05: Taxonomy discovery (3 AIs, separately) ✓

**What it does.** Each AI, alone, reads sampled items in rounds of 50 and writes for each one "what this code
does", on 3 levels (category › subcategory › behaviour). It continues until 19 rounds in a row bring nothing new
("saturation"), or until the items run out. Then each AI organises its own proposals into a tree. This produces
6 taxonomies: 3 AIs × (Pharo, literature).

**Result.** For Pharo: qwen 251, gemma 198, mistral 128 behaviours. For the literature: qwen 394, gemma 319, mistral 321.

**Small analysis.** **No taxonomy reached saturation**: every curve is still climbing when the items run out (Pharo)
or at the 30-round limit (literature). The AIs keep finding "new" behaviours, partly because they word the same
thing differently. **The models differ a lot in granularity**: qwen splits finely, and mistral produces half as many
behaviours. Qwen also gives 2 behaviours to most literature tasks, while the others usually give 1.

| Figure | What it shows |
|---|---|
| [saturation_pharo.png](results/05_discovery/figures/saturation_pharo.png) | Behaviours found vs items seen, for the 3 AIs on Pharo (still rising: no saturation) |
| [saturation_literature.png](results/05_discovery/figures/saturation_literature.png) | Same on benchmark tasks (stopped at 30 rounds, still rising) |
| [concepts_per_model.png](results/05_discovery/figures/concepts_per_model.png) | Size of each AI's taxonomy (categories, subcategories, behaviours) |
| [proposals_per_item.png](results/05_discovery/figures/proposals_per_item.png) | How many behaviours each AI gave per item (1, 2 or 3) |
| `05_discovery/<model>/<source>/taxonomy.png` | Each AI's tree for each source, with the number of items per behaviour |
| `05_discovery/<model>/<source>/categories.png` | That AI's categories sorted by how many items they got |
| `05_discovery/<model>/<source>/saturation.png` | That AI's own saturation curve |

## Phase 06: Two final taxonomies ◐

**What it does.** It merges the 3 AIs' lists into one list for Pharo and one for the literature. Similar concepts
are grouped by meaning and by "do they label the same items". An editor AI (qwen) then decides which ones really mean
the same thing, and writes one name and definition for each. The result is the input of phases 7–13. The experts
review it in `review/taxonomy_review.xlsx`.

**Result.** **Pharo: 18 categories, 85 subcategories, 360 behaviours. Literature: 17 categories, 118 subcategories,
507 behaviours.**

**Small analysis.** Only **73 of the 360 Pharo behaviours were found by all 3 AIs**. 91 were found by 2, and **196 (54 %)
by a single AI**. The AIs overlap by only about 30–50 %. The largest Pharo categories are "Object Lifecycle, State and
Reflection" (1,355 items) and "Collections" (721). The draft also contains near-duplicate names, for example
"Reflection" vs "Object Lifecycle, State, and Reflection", and "Parsing" vs "Text, Parsing, and Serialization"; the
merge retired some of them. **This is the most important file for the experts to check.**

| Figure | What it shows |
|---|---|
| [pharo/taxonomy_v1.png](results/06_taxonomy/pharo/figures/taxonomy_v1.png) | The final Pharo tree (●●○ = which AIs proposed it; bar = number of items) |
| [literature/taxonomy_v1.png](results/06_taxonomy/literature/figures/taxonomy_v1.png) | The final literature tree |
| [pharo/model_overlap.png](results/06_taxonomy/pharo/figures/model_overlap.png) | % of one AI's behaviours that another AI also found (32–50 %) |
| [literature/model_overlap.png](results/06_taxonomy/literature/figures/model_overlap.png) | Same for the literature (28–53 %) |
| [pharo/support_per_behavior.png](results/06_taxonomy/pharo/figures/support_per_behavior.png) | How many AIs proposed each behaviour (3: 73, 2: 91, 1: 196) |
| [literature/support_per_behavior.png](results/06_taxonomy/literature/figures/support_per_behavior.png) | Same for the literature |

## Phase 07: Structured annotation ◐

**What it does.** Each AI, alone, labels **854 new items** (494 Pharo methods and 360 benchmark tasks, none used to
build the taxonomy) with 1 to 3 behaviours from the list, plus a quote from the code as evidence.

**Result.** 2,490 valid answers, and 72 invalid ones (all mistral). The evidence quote is found in the code 98 % of the time.

**Small analysis.** Qwen and gemma always answered correctly; **mistral failed on 72 items (8 %)**. The models differ in
style: gemma usually gives 2 behaviours, while qwen and mistral usually give 1. **No model ever answered "unknown" or
"not enough context" (0 %)**. That may mean the models force every item into some category rather than admitting it
does not fit. Gemma puts 58 % of Pharo items under "Object state", against about 40 % for the others.

| Figure | What it shows |
|---|---|
| [validity.png](results/07_annotation/figures/validity.png) | Valid on first try / after retry / invalid, per AI |
| [categories_by_model_pharo.png](results/07_annotation/figures/categories_by_model_pharo.png) | % of Pharo items each AI put in each category |
| [categories_by_model_literature.png](results/07_annotation/figures/categories_by_model_literature.png) | Same for benchmark tasks |
| [behaviors_per_item.png](results/07_annotation/figures/behaviors_per_item.png) | How many behaviours each AI gave per item |
| [unknown_insufficient.png](results/07_annotation/figures/unknown_insufficient.png) | Share of "unknown" / "insufficient" answers (empty: 0 % for all) |

## Phase 08: AI agreement and expert validation ◐

**What it does.** It measures how often the 3 AIs gave the same labels (Krippendorff's α: 1 = perfect, 0 = chance;
**0.67** is the usual minimum, **0.80** reliable). It also prepares the **expert workbook** (`expert/expert_workbook.xlsx`,
200 items, AI answers hidden) to check the AIs against humans.

**Result.** **α = 0.25** on exact behaviour sets, **0.35** on the main behaviour, **0.51** on the main category. All 3 AIs
fully agree on only **7.7 %** of items.

**Small analysis.** This is **below the 0.67 minimum everywhere**. Agreement is better on Pharo (category 0.55) than on
the literature (0.40). Qwen and gemma agree moderately (κ = 0.53); **mistral agrees little with either (κ ≈ 0.25)**.
Only 1 behaviour of the 40 with the most items reaches 0.80 ("Return class reference"). Agreement between AIs is not
correctness: only the expert workbook can say which AI is right.

| Figure | What it shows |
|---|---|
| [alpha_summary.png](results/08_validation/figures/alpha_summary.png) | Agreement scores (α, AC1) overall, for the literature and for Pharo |
| [agreement_levels.png](results/08_validation/figures/agreement_levels.png) | Items with full / partial / no agreement (mostly partial) |
| [pairwise_kappa.png](results/08_validation/figures/pairwise_kappa.png) | Agreement between each pair of AIs (mistral is the outlier) |
| [per_behavior_alpha.png](results/08_validation/figures/per_behavior_alpha.png) | Agreement per behaviour (green ≥ 0.8, orange ≥ 0.67, red below) |

## Phase 09: Large-scale Pharo analysis ◐

**What it does.** One AI labels the big random sample (**1,786 methods from 895 projects**) and estimates, with the
sampling weights, **what share of all Pharo code** does each behaviour, with a 95 % interval. It chose **qwen**, the
model that agrees most with the other two, because there are no expert labels yet.

**Result.** By category: **Object Lifecycle, State and Reflection 44 %**, **Collections 23 %**, UI 10.5 %, Control flow 9.4 %,
Error handling 8.2 %, Text and parsing 5.3 %, … Network 0.1 %. 230 behaviours were seen, 200 of them rare (< 1 %).

**Small analysis.** Pharo code is mostly **managing objects**: reading a property is the single most common behaviour
(13.5 %), then creating an object (7.7 %), delegating a call (5.3 %) and setting a variable (4.7 %). Domains differ
strongly: UI dominates in education and graphics projects, maths in language implementation, files in documentation.
The intervals are wide (for example 44 % is really 37–51 %). The 0 % "unknown/uncertain" is the same forced-fit
concern as in phase 7.

| Figure | What it shows |
|---|---|
| [weighted_frequency_categories.png](results/09_profile/figures/weighted_frequency_categories.png) | % of all Pharo methods per category, with 95 % intervals |
| [weighted_frequency_behaviors.png](results/09_profile/figures/weighted_frequency_behaviors.png) | Top 30 behaviours in Pharo code |
| [groups_by_category.png](results/09_profile/figures/groups_by_category.png) | Categories per application domain (heat map) |
| [unknown_uncertain.png](results/09_profile/figures/unknown_uncertain.png) | Unknown / uncertain classifications (all 0 %) |

## Phase 10: Literature vs Pharo comparison ◐

**What it does.** It compares, behaviour by behaviour, how often it appears in Pharo code vs in existing benchmarks,
to find **gaps**: behaviours Pharo uses that benchmarks do not test. Each behaviour gets a status: *both*, *pharo_only*,
*literature_only*, or *investigate* (not enough data, or the AIs disagree on it).

**Result.** **both 1, pharo_only 48, literature_only 1, investigate 310.**

**Small analysis.** **86 % of behaviours end up in "investigate"**, mainly because the rule requires AI agreement α ≥ 0.67 on a
behaviour, and almost none reaches it (phase 8). So the low agreement directly blocks the comparison. Still, the picture
is clear in the scatter plot. Pharo is full of **object-state behaviours** (read property 13.5 % in Pharo vs 0.3 % in
benchmarks), while benchmarks are heavy on **arithmetic, sorting, filtering and file I/O**. The 48 Pharo-only gaps include
fixed-size arrays, lazy initialisation, UI widgets and windows, FFI native calls, and parser combinators.

| Figure | What it shows |
|---|---|
| [coverage_scatter.png](results/10_coverage/figures/coverage_scatter.png) | Each behaviour: % in benchmarks (x) vs % in Pharo (y). Far from the diagonal = different focus |
| [coverage_by_category.png](results/10_coverage/figures/coverage_by_category.png) | Status per category (grey "investigate" dominates) |
| [gaps_pharo_only.png](results/10_coverage/figures/gaps_pharo_only.png) | The Pharo behaviours not found in the benchmark sample, by frequency |
| [benchmark_by_category.png](results/10_coverage/figures/benchmark_by_category.png) | What each benchmark exercises (HumanEval 90 % collections, Terminal-Bench 73 % files, …) |

## Phase 11: Pharo benchmark construction ◐

**What it does.** For **50 selected behaviours** (Pharo-only gaps first), an AI first writes a **contract** (what any
correct solution must do). An author AI (qwen) then writes a **new Pharo task**: a class, a reference solution, SUnit
tests and a description. It starts from 2 real Pharo methods per behaviour as inspiration ("seeds"). Each task is
installed in a real Pharo image and run. If it fails, the author sees the error and gets up to 2 tries to fix it.

**Result.** 100 seeds → 95 valid AI answers → **39 install in Pharo** → **25 whose reference passes its own tests**.

**Small analysis.** **The biggest loss is installation: 56 of 95 tasks do not even load in Pharo.** The AI writes Pharo
code that does not compile or has missing dependencies. UI tasks are the worst (2 of 20 work), and "Object state" gets
6 of 22. Improving this step (more repair rounds, a better prompt, or a stronger author model) is the easiest way to
get more tasks.

| Figure | What it shows |
|---|---|
| [funnel.png](results/11_benchmark/figures/funnel.png) | How many tasks survive each step (100 → 95 → 39 → 25) |
| [candidates_per_category.png](results/11_benchmark/figures/candidates_per_category.png) | Working vs rejected tasks per category |

## Phase 12: Benchmark validation ◐

**What it does.** It checks that the 25 working tasks are real, well-tested tasks, in 5 levels:
1. the task runs cleanly and repeatably;
2. other AIs write **independent tests** from the contract, and these must agree with the reference;
3. **MuTalk mutation testing**: small bugs are injected into the solution, and the tests must catch them;
4. the tests must cover the contract;
5. expert audit (the file `expert/benchmark_review.xlsx`, still to fill).

Each task then gets 3 variants: write it from scratch (generation), finish it (completion), and fix a bug (repair).

**Result.** 25 → **19 pass all execution checks** (6 are rejected for tests too weak against mutants) → **6 auto-verified**,
and **13 "quarantined"**: the independent tests disagree with the reference, and only a human can tell which side is
wrong. Mean mutation score 0.73. Of 681 mutants, 514 were caught by the tests and only 20 escaped detection.

**Small analysis.** The final benchmark today is **6 tasks × 3 variants = 18**. The 13 quarantined tasks are **not lost**:
if the experts accept them, the benchmark grows to 19 tasks. The verified tasks cover only 4 categories (control flow,
error handling, object state, UI).

| Figure | What it shows |
|---|---|
| [gates_funnel.png](results/12_verification/figures/gates_funnel.png) | How many of the 25 tasks pass each execution check (mutation is the main filter) |
| [mutation_scores.png](results/12_verification/figures/mutation_scores.png) | Test strength per task (share of injected bugs caught; red line = minimum 0.6) |
| [mutant_classes.png](results/12_verification/figures/mutant_classes.png) | What happened to the 681 injected bugs (caught, equivalent, escaped) |
| [task_status.png](results/12_verification/figures/task_status.png) | Auto-verified (6) vs quarantined for expert review (13) |
| [verified_per_category.png](results/12_verification/figures/verified_per_category.png) | Verified variants per category and type |

## Phase 13: Model evaluation ◐

**What it does.** The 3 AIs try to solve the verified Pharo tasks. Each answer is run against the hidden tests in
Pharo. pass@1 is the share of variants solved on the first try.

**Result.** **mistral 89 %**, **gemma 78 %**, **qwen 67 %**. The reference solutions score 100 %, which confirms the tests work.

**Small analysis.** With **only 18 variants, the 95 % intervals overlap widely** (qwen is somewhere between about 44 %
and 84 %), so **no ranking can be claimed yet**. Most failures are **runtime errors** in Pharo rather than wrong results.
Qwen struggles on bug repair (33 %), and every model is weakest on UI tasks. The same 3 models also built the tasks
(qwen was the author), which can bias the scores. A final evaluation should add models that took no part in building
the benchmark.

| Figure | What it shows |
|---|---|
| [pass_at_1.png](results/13_evaluation/figures/pass_at_1.png) | Score per model with 95 % intervals |
| [outcomes.png](results/13_evaluation/figures/outcomes.png) | Pass / test failure / runtime error per model |
| [pass_by_task_type.png](results/13_evaluation/figures/pass_by_task_type.png) | Score per variant type (generation, completion, repair) |
| [pass_by_category.png](results/13_evaluation/figures/pass_by_category.png) | Score per behaviour category |

---

## Points to discuss before the next run

1. **Low AI agreement** (α 0.25 / 0.51). It also pushes 86 % of behaviours into "investigate" in phase 10. The expert workbook
   (phase 8) and the taxonomy review (phase 6) are the way to fix it.
2. **Taxonomy too fine and not saturated.** 54 % of behaviours come from a single AI. Merging near-duplicates in the review
   would raise agreement.
3. **Only 6 verified tasks.** The main loss is Pharo code that does not install (56 of 95). The 13 quarantined tasks need an
   expert decision.
4. **0 % "unknown".** The models never admit an item does not fit the categories. Check a few items by hand.
5. **Evaluation bias.** The evaluated models also built the benchmark. Add independent models.
6. Small items: DeepSWE is missing (needs `HF_TOKEN`). The Pharo sample has fewer conditionals and cascades than the corpus.
   One sampling figure title ("at most 2") does not match its data.

Files for the experts: `results/06_taxonomy/*/review/taxonomy_review.xlsx`, `results/08_validation/expert/expert_workbook.xlsx`,
`results/12_verification/expert/benchmark_review.xlsx`. Completed copies go into the matching `returned/` folder.
