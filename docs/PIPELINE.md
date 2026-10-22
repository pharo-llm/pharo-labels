# The pipeline

**The question.** Which programming behaviours does real Pharo code exhibit, which of them do the code benchmarks
LLMs are evaluated on actually test, and how well do LLMs handle the behaviours that matter in Pharo?

**The answer, in 13 phases.** Stage 1 builds **two taxonomies of programming behaviours**: one from real Pharo
methods, one from the tasks of the existing benchmarks. Stage 2 validates them, compares them, and turns the
behaviours that Pharo needs into a **validated, executable Pharo benchmark** on which models are evaluated.
This document follows the data through every phase: what goes in, what happens and why, what comes out.

![The pipeline](pipeline.png)

*Stage 1. Blue: the Pharo side. Orange: the existing benchmarks. The two sides are processed the same way and meet
only at the end, as two comparable taxonomies.*

## The whole pipeline at a glance

```
 STAGE 1 — two taxonomies (./run.sh)                    STAGE 2 — validate, compare, build, evaluate
 ───────────────────────────────────────                ───────────────────────────────────────────────
  1 repositories   1,585 → ≈ 1,087 selected               7 annotation     new samples, 3 annotators, alone
  2 literature     12 benchmarks → 4,372 problems         8 validation     agreement (α) + blinded experts
  3 corpus         ≈ 1.84 M methods, AST filters          9 profile        how often each behaviour occurs
  4 sampling       99 % ±5 % samples + pools             10 coverage       Pharo × benchmarks, per behaviour
  5 discovery      3 models × 2 sources, saturation      11 benchmark      contracts → new Pharo tasks
  6 taxonomy       merged → 2 final taxonomies           12 verification   5-level validation of each task
                                                         13 evaluation     models × task variants
```

| # | Phase | What goes in | What comes out | Humans |
|---|---|---|---|---|
| 0 | Setup | `config.yaml` | models, Pharo image, benchmarks, corpus in `data/` | |
| 1 | Repository selection | 1,585 mined repositories | ≈ 1,087 engineered projects (weighted quality score ≥ 0.50) | |
| 2 | Literature ingestion | 12 benchmarks at pinned revisions | 6,433 tasks = 4,372 distinct problems, with their task type | |
| 3 | Corpus and method filters | ≈ 1.84 M methods | a decision per method, taken on its syntax tree | |
| 4 | Representative samples | eligible methods, problems | 99 % ±5 % start samples (663 + 577) and ordered pools | |
| 5 | Discovery | the samples | 6 taxonomies (3 annotators × 2 sources), each until saturation | |
| 6 | Final taxonomies | the 6 taxonomies | 2 final taxonomies with stable IDs (C / S / B) | review |
| 7 | Annotation | NEW samples + a taxonomy | 3 independent classifications of every item | |
| 8 | Validation | the classifications | agreement (α, κ, AC1); expert labels; `taxonomy_v2` | experts |
| 9 | Profile | a large random sample | design-weighted frequency of every behaviour in Pharo | |
| 10 | Coverage | phases 7–9 | per behaviour: both · Pharo only · literature only · investigate | |
| 11 | Benchmark construction | selected behaviours, Pharo seeds | behaviour contracts; tasks that execute in the image | selection |
| 12 | Benchmark validation | the tasks | verified / quarantined / rejected tasks, 3 variants each | audit |
| 13 | Evaluation | verified tasks, the models | pass rates per model × category × task type | |

**Models** (all local, Ollama): the three annotators `qwen3.8:27b` (Alibaba), `gemma4:31b` (Google) and
`mistral-small3.2:24b` (Mistral AI), plus `nomic-embed-text` for similarity. qwen also edits the taxonomy (phase 6)
and writes the tasks (phase 11). gemma writes the behaviour contracts. gemma and mistral write the independent tests
(phase 12), so none of these checks comes from the family that wrote the tasks. Phase 13 evaluates the three.

**Status.** `./run.sh` runs all 13 phases. Stage 2 works with one of the two final taxonomies,
`taxonomy.stage2_source` (default: the Pharo taxonomy), with which it classifies both the Pharo methods and the
benchmark tasks, so phase 10 compares them on the same behaviour IDs; behaviours found only in the benchmarks
appear there as UNKNOWN. A smoke run (2026-10-22) went through every phase; its phase 11 author model wrote no task
that passed its own tests, so phases 12–13 had nothing to validate.

---

## How to run it

```bash
./run.sh                  # setup, then phases 1–13; what is already done with the same inputs is skipped
./run.sh 5                # one phase       ./run.sh 3-6   a range       ./run.sh from 4   phases 4..13
./run.sh status           # where every phase stands       ./run.sh logs 5   the log of a phase
PROFILE=smoke ./run.sh    # the same pipeline on a few items, in results-smoke/ (a check that takes minutes)
```

Every phase writes into its own folder `results/NN_<phase>/`, with a `README.md`, its files and its `figures/`.
A phase resumes where it stopped (every model answer is cached), and is redone only when something it reads has
changed: its settings in `config.yaml`, the result of an earlier phase, or an expert file it imports.

---

## Step 0 — Setup: every dependency before anything runs

Before the first phase, `./run.sh` makes sure everything the phases need is there, and stops with the reason if
something cannot be obtained. When everything is present it takes a few seconds.

| What | How |
|---|---|
| The four models (the three annotators, which are also the editor, author, contract, oracle and evaluated models; the embedding model) | pulled with Ollama when missing; the files are checked to be complete |
| The Pharo 13 image (with MuTalk and PharoBench-Support) and the list of its global names | built or copied into `data/pharo-image/`; the VM must evaluate `3 + 4` the first time |
| The 12 benchmarks | downloaded at pinned revisions into `data/existing-benchmarks/` |
| The Pharo corpus (the parsed methods of every repository) | downloaded once from Hugging Face into `data/pharo-corpus/` |

Setup checks that every dependency is **present**, not that it **works**: no model is asked a question, and an
existing image is not run again. A broken model or image shows up in the first phase that uses it.

---

# Stage 1 — two taxonomies (phases 1–6)

## Phase 1 — Repository selection

**Goal.** Study engineered Pharo projects, not toy, personal or copied repositories.

**Input.** The 1,585 repositories of the corpus snapshot: they passed the first mining filter (real Pharo source
code, not coursework, some history). They are listed in `repos.txt`.

**What happens.**

1. **Hard filters**, which no score can compensate: GitHub forks; *mirrors*, whose non-trivial methods are mostly
   (≥ 50 %) copies of an older repository; repositories without any parsed method.
2. **A quality score** for each remaining repository: eleven criteria, each turned into a score between 0 and 1,
   then a weighted sum. This is the *reaper* way of recognising engineered projects (Munaiah et al. 2017).

   | criterion | what is measured | weight |
   |---|---|---|
   | community | contributors | 0.15 |
   | history | commits | 0.15 |
   | unit testing | share of test methods | 0.15 |
   | maintenance | how long it was developed × how recently | 0.10 |
   | documentation | share of classes with a class comment | 0.10 |
   | size | non-test methods | 0.10 |
   | originality | share of its code not copied from an older repository | 0.05 |
   | licence | a licence is declared | 0.05 |
   | issues | the issue tracker is used | 0.05 |
   | loadability | a Metacello `BaselineOf` (it can be loaded) | 0.05 |
   | popularity | stars and forks | 0.05 |

   Counts become scores with saturating curves (the 2,000th method adds almost nothing), so a giant repository
   cannot dominate. Popularity has a deliberately small weight: stars measure popularity, not quality (Borges &
   Valente 2018), and half of the Pharo repositories have none.
3. **The threshold**: a repository is selected when its score is at least **0.50**, half of the maximum.

**Output.** `selected_repos.txt` (≈ 1,087 repositories), `repository_scores.csv` (every measure, every criterion
score, the score, the decision and its reason), `scoring_scheme.md`.

![Scores of the repositories](figures/01_score_distribution.png)

![What other thresholds would keep](figures/01_threshold_sensitivity.png)

![Popularity does not decide](figures/01_popularity_vs_score.png)

*The score only decides which repositories are studied. It is never used later to make a repository count more.*

**Settings.** `config.yaml` → `repositories` (hard filters, weights, saturation points, threshold).

---

## Phase 2 — Literature ingestion

**Goal.** Know exactly what the existing benchmarks ask a model to do.

**Input.** The code benchmarks frontier models are evaluated on, at pinned revisions: HumanEval+, MBPP+,
BigCodeBench (Complete, Instruct), LiveCodeBench (code generation, test-output prediction, code execution),
SWE-bench Verified, Pro and Multilingual, Terminal-Bench 2.0, NL2Repo (DeepSWE when a Hugging Face token gives
access).

**What happens.** Every task is read and recorded with what the evaluated model receives, the reference answer,
how the answer is checked (tests), the programming language and the evaluation task type (code generation,
completion, bug repair, test generation, other). A problem that appears in several scenarios is counted once.

**Output.** `literature_tasks.parquet` (6,433 tasks = 4,372 distinct problems), `literature_benchmarks.csv`.

---

## Phase 3 — Corpus and method filters (decided on the syntax tree)

**Goal.** Keep the methods that carry real behaviour, and know why every other method is left out.

**Input.** Every method of the selected repositories (≈ 1.84 M), with its provenance (repository, commit, file,
line, link), its class, the definition of its class, and its **abstract syntax tree**.

**What happens.**

1. **Copies** are found across repositories, exactly and once comments and layout are removed. The oldest copy is
   the original.

   ![Copies across repositories](figures/03_duplication.png)

2. **Each method gets a decision, decided on its syntax tree and on its class definition.** The names in a method
   are resolved like the Pharo compiler does: arguments and temporaries, then the variables of its class, then
   `self`, `super`, `true`, `false`, `nil`, then globals. A getter, for instance, is a method whose only statement
   returns one of its class's instance variables: not a method whose text looks like one.

   | method | decision |
   |---|---|
   | a copy of an older method · in a mirror repository · cannot be parsed | exclude |
   | simple getter `^ v` · simple setter `v := arg` (v a variable of its class) | exclude |
   | returns a constant (`^ 42`, `^ 'text'`, `^ true`, a literal array) | exclude |
   | empty · placeholder (`self subclassResponsibility`, `^ self`, `self error: 'not implemented'`) | exclude |
   | generated (package manifests, Metacello baselines, `<generated>`, embedded data) | exclude |
   | trivial `initialize` (`super initialize` and `v := literal` / `v := Class new` only) | exclude |
   | simple message forwarding · other initialisation · very long · uses classes neither its project nor Pharo defines | review: kept, marked with the reason |
   | test methods | kept separately |

**Output.** `methods.parquet`, `method_filters.parquet` (the decision and every reason of every method),
`method_filters_summary.csv`, figures.

**Settings.** `config.yaml` → `method_filters` (long-method limits, whether review methods are sampled).

---

## Phase 4 — Representative samples

**Goal.** Samples that really represent Pharo code and the benchmarks, large enough for 99 % confidence.

**Pharo methods.** A large repository must not dominate: a few projects hold most of the methods.

![Methods per repository](figures/03_methods_per_repository.png)

So the sample is **stratified by repository domain** and drawn in two stages: repositories first, then at most 4
methods per repository. Every method carries its design weight (how many methods of the corpus it stands for).

![Groups: corpus vs sample](figures/04_groups_corpus_vs_sample.png)

**Benchmarks.** Each benchmark contributes in proportion to its number of problems.

**How many.** The start sample gives **99 % confidence with a ±5 % margin** (Cochran's formula with the finite
population correction, p = 0.5): **663 Pharo methods** and **577 benchmark problems**. (100 % confidence would mean
taking every item — a census.) Targeted methods from rare areas (concurrency, FFI, reflection, network…) and test
methods (kept separately) are added.

**Representativeness check.** The 663 methods are compared with the whole population on what their syntax trees
show — loops, conditionals, iteration, exception handling, blocks, reflection, recursion, method size — each with
a 95 % interval; for the benchmarks, the share of every benchmark and task type.

**Output.** `sampling_design.json`, the samples, `representativeness.csv`, `representativeness_pharo.png`,
`representativeness_literature.png`. Each sample is an *ordered pool*: the start sample, then the items the
saturation loop of phase 5 will add, in order.

---

## Phase 5 — Discovery until saturation: three annotators, two sources

**Goal.** Find every behaviour that occurs, without imposing a list of categories in advance.

**Who.** Three instruction-tuned models of three different families — **qwen3.8:27b** (Alibaba), **gemma4:31b**
(Google), **mistral-small3.2:24b** (Mistral AI) — because related models make the same mistakes. Each works alone
and never sees another model's answers.

**What each annotator does, for each source (Pharo methods, then benchmark tasks).**

1. **Reads the 99 % start sample** and gives, for each item, up to three behaviours at three levels —
   **category › subcategory › behaviour** — in language-neutral words, with a quote of the item as evidence.
2. **Saturation loop.** It goes on with rounds of 50 new items from the same pool. After each round, its proposals
   are compared with everything it proposed before (embeddings): a group of similar proposals that reaches 2 items
   is a behaviour; a round is *new* if a behaviour appears.
3. **Stops at saturation**: when **19 rounds in a row bring no new behaviour**. At that point, a behaviour present in
   0.5 % of the items would have appeared with 99 % probability.
4. **Consolidates** everything it proposed into its own taxonomy, with a definition for every concept and rules
   saying when to use a behaviour and when not to.

**Output: six taxonomies**, each in `05_discovery/<annotator>/<pharo|literature>/`: the figure `taxonomy.png`,
`taxonomy.json`, `taxonomy.csv` (one row per behaviour), `items.csv` (every item with the category, subcategory and
behaviour this annotator gave it), the saturation curve (`saturation.png`, `saturation.csv`).

An annotator's taxonomy, as it is drawn (here, from the earlier run):

![One annotator's taxonomy](figures/05_annotator_taxonomy_example.png)

---

## Phase 6 — The two final taxonomies

**Goal.** One taxonomy per source that the three annotators support, with stable identifiers.

**What happens, for the Pharo methods and, separately, for the benchmark tasks.**

1. **How similar two concepts are**, for each level (category, subcategory, behaviour) — two pieces of evidence:
   - **their meaning**: the embeddings of their texts ("name: definition"), compared by cosine;
   - **their items**: the three annotators classified the *same* items, so two concepts that hold the same items
     very likely mean the same thing, even with different words, and two concepts with similar words but different
     items very likely do not. The overlap (Jaccard) is measured on the items both annotators saw; for a category or a
     subcategory, its items are every item under it.

   **similarity = 0.6 × embedding cosine + 0.4 × item overlap.** Concepts are grouped by this similarity
   (average-linkage clustering). The groups are only *candidates*.
2. **Judged merges.** An editor model reads each candidate group — every member with the number of items it holds,
   the largest first — and keeps together only the concepts that mean the same thing, naming them after the
   members that hold most items. Groups it splits are listed for the experts.
3. **One tree, weighted by items.** A behaviour goes under the subcategory where most of its members' *items* were
   placed (not most of its members), a subcategory under the category where most of its items were; categories are
   then merged down to at most 16 (the editor sees how many items each holds), duplicates merged. Every operation is
   recorded.
4. **Stable identifiers** (`C001` categories, `S001` subcategories, `B001` behaviours), kept across reruns; each
   concept keeps its definition, its rules, its aliases (the annotators' wordings), the annotators that proposed it
   and the items classified under it.
5. **Expert review.** `review/taxonomy_review.xlsx` lets an expert accept, rename, redefine, merge, split, reject,
   add or move any concept; the completed file in `review/returned/` gives the reviewed taxonomy at the next run.

**Output.** `06_taxonomy/pharo/taxonomy_v1.json` and `06_taxonomy/literature/taxonomy_v1.json`, with their figures,
aliases, merge candidates, annotation guidelines and review workbooks.

### How to read a taxonomy figure

![A final taxonomy](taxonomy_example.png)

*(The earlier run, drawn the way every taxonomy is drawn now. In that run the item lists were not yet kept, so the
category and subcategory counts here are sums; a new run counts distinct items.)*

- **A box** is a category; **grey bold** lines are its subcategories; the other lines are its behaviours, with their
  identifier.
- **●●○** before a behaviour: one ● for each annotator whose concept was merged into it by the embedding + item
  merge of this phase (●●● = the three annotators found this behaviour).
- **Under the ●●○**, a thin bar and a number: **how many items were classified under that behaviour**.
- **On the right of a category or subcategory**, the same bar and number: how many items were classified anywhere
  inside it.
- The bars are scaled per level, so the largest category, subcategory and behaviour have a full bar.

---

# Stage 2 — validate, compare, build, evaluate (phases 7–13)

*Stage 2 uses the Pharo taxonomy (`taxonomy.stage2_source`) for both sources.*

Phase 4 draws **disjoint** splits of the Pharo methods with the same design. *discovery* builds the taxonomy
(phase 5), *annotation* validates it (phases 7–8), and *profile* measures frequencies (phase 9). The benchmark tasks
get the first two splits. No item is used both to build the taxonomy and to judge it.

## Phase 7 — Independent annotation

**Goal.** Know whether the taxonomy can be applied consistently to items it was not built from.

**What happens.** Each of the three annotators, alone, classifies every item of the annotation split with the
taxonomy. It gives 1 to 3 behaviour IDs (its answer is constrained to the taxonomy's IDs, plus UNKNOWN and
INSUFFICIENT), each with evidence quoted from the item and a confidence, plus an explanation and an uncertainty
level. Every model gets the same prompt, one item at a time.

**Output.** `annotations_raw.jsonl` (every answer, valid or not), `annotations_validated.parquet` (one row per item
and model: IDs, evidence and whether it occurs in the item, model digest, taxonomy version, tokens).

## Phase 8 — Agreement and expert validation

**Goal.** Measure how reliable the classifications are, against each other and against people.

**What happens.**

1. **AI agreement**: full, partial or no agreement per item. Krippendorff's α (MASI distance on behaviour sets,
   nominal on the primary behaviour and category), Fleiss' κ, Gwet's AC1, pairwise Cohen's κ, and α per behaviour.
2. **A blinded expert workbook** (`expert/expert_workbook.xlsx`): items stratified by AI agreement, with more items
   where the models disagree. The AI answers are **not** shown.
3. **Import** of the completed workbooks: the experts' agreement among themselves (the human baseline), then
   humans against the AI (precision, recall, F1, accuracy, per agreement level). AI majority is never taken as
   correctness.
4. **Adjudication** of the disagreements, now with the AI answers revealed. The experts' taxonomy decisions produce
   `taxonomy_v2.json`, with the map v1 → v2.

Without returned workbooks, the phase computes AI agreement only and is *provisional*.

## Phase 9 — How often each behaviour occurs in Pharo

**Goal.** Estimate the frequency of every behaviour in real Pharo code, with honest intervals.

**What happens.** The profile split (a large random sample) is classified with the latest taxonomy by the annotator
closest to the experts (or, without experts, the one agreeing most with the other two). For each behaviour,
subcategory and category it reports:

- the **design-weighted share**, with a 95 % interval (repositories are the sampling clusters);
- the share of repositories where the behaviour appears;
- differences between repository domains;
- rare and important behaviours.

When expert labels exist, the shares are corrected for the model's errors (prediction-powered inference,
Angelopoulos et al. 2023).

**Output.** `pharo_behavior_profiles.parquet`, `weighted_frequency.csv`, `repository_prevalence.csv`,
`group_differences.csv`, `unknowns.csv`.

## Phase 10 — Pharo versus the benchmarks

**Goal.** For each behaviour, say whether the existing benchmarks test what Pharo code does.

**What happens.** The two classified datasets are joined on the behaviour IDs. Each behaviour gets a status:
**both** (enough support on each side), **Pharo only** (observed in Pharo, not identified in the benchmark sample),
**literature only**, or **investigate** (too little support, or the annotators do not agree on it, α < 0.667).
Not observing a behaviour is never read as its absence: every gap carries its sample size and the upper bound of
its share that the sample still allows (rule of three).

**Output.** `coverage_matrix.csv`, `behavior_gaps.csv`, `coverage_by_benchmark.csv`, each row with example items.

## Phase 11 — Benchmark construction

**Goal.** New, self-contained Pharo tasks for the behaviours the benchmarks miss or that Pharo needs most.

**What happens.**

1. **Behaviours**: chosen from the coverage matrix (Pharo only first, then both; the most frequent in Pharo first),
   or listed by the research team in `selected_behaviors.csv`.
2. **Behaviour contracts, before any task.** For each behaviour, gemma writes from the taxonomy definition alone
   what any implementation must do: its input and output, 3–6 observable properties (P1…), the edge cases, and what
   is excluded. Example, for "select the elements satisfying a predicate": every returned element satisfies the
   predicate; every element that satisfies it is returned, duplicates included; the order is kept; an empty input
   gives an empty output.
3. **Seeds**: real Pharo methods labelled with the behaviour, one per repository (and, when one exists, a
   literature task with the same behaviour, for inspiration only).
4. **Authoring**: qwen writes a NEW task that satisfies the contract: one class, the methods to implement (the
   reference solution), the given methods, SUnit tests, and the statement, written last from the code.
5. **Execute and repair**: the task is installed and run in a clean image. When it fails, the author is shown the
   image's error report and rewrites it (up to 2 rounds).

**Output.** `contracts.jsonl`, `contracts/<contract_id>.md`, `benchmark_candidates.jsonl`, `tasks/<task_id>/`.

## Phase 12 — Benchmark validation, in five levels

**Goal.** Accept a task only on evidence that it is correct, that its tests catch faults, and that it tests the
intended behaviour, and measure how often this automatic validation is still wrong.

The author wrote the reference **and** its tests, so both can share the same misunderstanding: all tests pass, and
the task is still wrong. Each level answers a different question:

| Level | Question | Check | A task that fails it |
|---|---|---|---|
| 1 Executability | does it run? | installs in a clean Pharo 13 image; touches nothing outside it; ≥ 3 tests; reference passes 3× (once in reverse order); each target method stubbed out makes a test fail; MuTalk mutation score ≥ 0.6 | rejected |
| 2 Independent oracle | is the reference correct? | gemma and mistral, seeing only what a solver sees plus the contract, write their own tests. Each is run on the reference: agrees, **counterexample**, flaky, vacuous (passes even with the method stubbed) or invalid. A task needs ≥ 3 agreeing tests and no counterexample | quarantined |
| 3 Valid mutants | do the tests catch faults? | every MuTalk mutant is re-run in a clean image on both test suites: killed, survived but detectable (a test gap), possibly equivalent, unstable, terminated or invalid. The mutation score is given as a lower and an upper bound, because equivalence is never assumed | (reported) |
| 4 Behaviour alignment | does it test the intended behaviour? | each contract property is covered, violated or untested by the agreeing independent tests | quarantined |
| 5 Audit | how often are levels 1–4 wrong? | experts review every quarantined task plus a random sample of the verified ones, stratified by category; the share they reject, with a Wilson 95 % interval, is the residual error | expert decision |

A quarantined task means the reference or the independent test is wrong, and only a person can tell which. It is
never evaluated unless an expert accepts it.

**Example, from a test run in the real image.** A reference that wrongly rejects the last valid index passed every
level-1 check, with a mutation score of 0.97, because the author's tests never asked for the last element. An
independent test did: `Got nil instead of 3`. The task was quarantined, with that counterexample shown to the
experts.

**Variants.** Every task gives three variants that share its hidden tests: *code generation* (statement + class +
method signatures), *code completion* (the first lines of the reference are given) and *bug repair* (a mutant that
compiles, is killed reproducibly by a failing assertion or an error, and violates the contract; the one the fewest
tests catch, such as an off-by-one). They are three variants of one problem, not three problems.

**Literature equivalence.** The tasks are new, built from Pharo seeds, and validated against their own contracts.
None is claimed to be equivalent to a task of an existing benchmark.

**Output.** `benchmark_verified/tasks.jsonl` (what a solver sees) and `tasks_private.jsonl` (reference, tests,
provenance). `validation_reports/<task_id>.json` + `validation_reports.csv` hold one report per task (contract,
independent tests, counterexamples, properties, mutation bounds, status). `mutants.jsonl` holds one record per
mutant (original and mutated code, operator, distinguishing tests, failing assertions). Also
`independent_tests.jsonl`, `quarantine.csv`, `validation_report.md` (by level) and `expert/benchmark_review.xlsx`.

## Phase 13 — Model evaluation

**Goal.** How well do LLMs handle the Pharo behaviours?

**What happens.** Each evaluated model (qwen3.8, gemma4, mistral) gets the same prompt for every verified task
variant, at temperature 0 with a fixed seed. Its answer is parsed into methods and run as untrusted code in the
isolated image: resource limits, no network. The reference solutions are submitted as a model too and must score
100 %, which checks the whole evaluation path.

**Output.** Pass or fail and tests passed for every (task variant, model) pair, with the error kind (no code,
syntax, compile, test failure, timeout). Pass rates per model × category × behaviour × task type, with 95 %
intervals. `reproducibility.json` (config hash, git commit, package versions, image and harness hashes, model
digests) and `release/`, the publishable benchmark.

*Caveat: the three evaluated models also built the benchmark (qwen wrote the tasks; gemma and mistral wrote the
contracts and the independent tests). Report this with the results, or add models that played no part in building
it.*

---

## Where everything is

```
results/
  SUMMARY.md                                       every phase: status, key numbers, figures
  logs/                                            the logs of every phase, readable
  01_repositories/                                 repos.txt, repository_scores.csv, selected_repos.txt
  02_literature/                                   the benchmark tasks
  03_corpus/                                       the methods and the decision on each
  04_sampling/                                     the samples and their representativeness
  05_discovery/<annotator>/<pharo|literature>/     the six annotators' taxonomies
  06_taxonomy/<pharo|literature>/                  the two final taxonomies
  07_annotation/                                   the three independent classifications
  08_validation/                                   agreement, expert workbook and its returned/ files, taxonomy_v2
  09_profile/                                      frequency of every behaviour in Pharo
  10_coverage/                                     the coverage matrix and the gaps
  11_benchmark/                                    contracts, task candidates
  12_verification/                                 verified tasks, validation reports, mutants, audit workbook
  13_evaluation/                                   results per model, release/
data/                                              downloads, Pharo image, LLM cache (not in git)
config.yaml                                        every setting
```
