# Pharo benchmark research pipeline

Which programming behaviours do existing code benchmarks evaluate, which ones does real Pharo code exhibit,
where do they differ — and a validated, executable Pharo benchmark built from the difference.
Thirteen resumable phases; every phase writes into its own folder `results/NN_<phase>/`, with figures.

```
Pharo corpus snapshot: 1,585 repositories (repos.txt)
        │ 1 repositories: hard filters (forks, mirrors) + weighted quality score (reaper criteria) → selected_repos.txt
        ▼
existing benchmarks (12, 6,433 tasks)          selected Pharo repositories
        │ 2 literature                                 │ 3 corpus: provenance, structure, deduplication
        └──────────────┬───────────────────────────────┘
                       │ 4 sampling: stratified two-stage (random = frequencies, targeted = rare behaviours)
                       │   three disjoint splits: discovery · annotation · profile
                       ▼
       5 discovery ── 3 LLMs (Alibaba, Google, Mistral), each alone: category › subcategory › behaviour
                       ▼
       6 taxonomy ─── embedding candidates → editor judges merges → harmonisation → stable IDs → expert review
                       ▼
       7 annotation ─ 3 LLMs classify NEW items with taxonomy v1 (multi-label, IDs only, evidence)
                       ▼
       8 validation ─ AI–AI α (MASI) · blinded expert workbook · human vs AI · adjudication · taxonomy v2
                       ▼
       9 profile ──── larger Pharo sample: design-weighted prevalence (95% CI), repositories, groups, PPI
                       ▼
      10 coverage ─── literature × Pharo matrix: both / Pharo-only / literature-only / investigate
                       ▼
      11 benchmark ── tasks for selected behaviours, seeded by real methods, execute-and-repair in Pharo
                       ▼
      12 verification  isolated Pharo image: tests ×3 (reversed), fail-to-pass, MuTalk mutation score;
                       │ code-generation / completion / bug-repair variants; expert review workbook
                       ▼
      13 evaluation ─ coding models (incl. the Pharo PT model and its base), pass@1 by category / behaviour /
                       task type; release + reproducibility
```

## Run

```bash
./run.sh                    # every phase in order; what is up to date is skipped, interrupted work resumes
./run.sh 7                  # one phase      ./run.sh 3-7   a range      ./run.sh from 7   phases 7..13
./run.sh status             # one line per phase: ✓ done · ◐ provisional · ⏸ waiting · ✗ failed · (stale)
./run.sh logs               # the whole pipeline's log, clean;  ./run.sh logs 7  one phase (its last run)
./run.sh cost               # LLM calls, tokens, time and cost per phase and model
./run.sh reset 6            # delete what phase 7 produced (expert files in returned/ are kept)
PROFILE=full ./run.sh       # the full-size study (sample sizes: config.yaml, profiles)
```

Needs only Python ≥ 3.10, git and [Ollama](https://ollama.com). **Before any phase, `./run.sh` runs a setup step**
(`./run.sh setup` alone; `pbr/setup.py`, log `results/logs/00_setup.log`) that obtains every dependency and stops
the pipeline if one cannot be obtained:

| Dependency | How setup gets it (once; then a check of a few seconds) |
|---|---|
| Ollama models of `config.yaml` | annotators qwen3.8:27b, **gemma4:31b**, mistral-small3.2:24b; editor, embedder, author; evaluated models — `ollama pull` |
| **Pharo PT model** `pharo-llm/Qwen2.5-Coder-PT:7b` | its LoRA adapter from Hugging Face `pharo-llm/Qwen2.5-Coder-7B-PT` at branch `checkpoint-15400` (the final checkpoint; `main` holds only the README) → GGUF with llama.cpp's converter (pinned release, own venv in `data/tools/`) → registered in Ollama on the same base weights (`qwen2.5-coder:7b-base-q8_0`, Modelfile `ADAPTER`) → smoke completion |
| Pharo validation image | Pharo 13 + MuTalk + PharoBench-Support (`pharo/support/`) in `data/pharo-image/`, copied from `../pharo-benchmarks/work/pharo` when it has the same versions, else downloaded and built |
| Benchmarks | every benchmark of `pbr/benchmarks.py` into `data/existing-benchmarks/` (DeepSWE is gated: `HF_TOKEN` in `.env`) |
| Pharo corpus | the parsed methods of the Hugging Face snapshot `pharo-llm/pharo-labels` into `data/pharo-corpus/` (1.3 GB) |

`data/` is git-ignored. The Pharo PT model is a pre-trained model, not a chat model: phase 13 does not give it
instructions but Pharo source to continue (`mode: completion` in `config.yaml`).

**Resuming.** A phase is skipped when its *fingerprint* — the configuration it reads, the sample sizes, the
run of every upstream phase, and the content of expert files it imports — has not changed; otherwise it
and the phases after it are redone. Every LLM request is cached by a hash of everything that determines
its answer (`results/cache/llm/`), so a rerun or an interrupted phase never asks the same thing twice.

**Logs.** `results/logs/pipeline.log` (all phases), `results/logs/NN_<phase>.log` (one phase; tracebacks
only here), `results/logs/events.jsonl` (start / end / waiting / failure, machine-readable),
`results/logs/llm_calls.jsonl` (one line per LLM call: tokens, latency, validity, cost). Progress lines are
written every 30–60 s with rate and ETA, never progress bars. `results/SUMMARY.md` shows every phase, its
key numbers and its figures.

## Expert files (human in the loop)

| Phase | File to fill | Put it in | What it changes |
|---|---|---|---|
| 5 | `results/06_taxonomy/review/taxonomy_review.xlsx` (accept / rename / redefine / merge / split / reject / add / move) | `review/returned/` | taxonomy_v1 becomes *reviewed* |
| 7 | `results/08_validation/expert/expert_workbook.xlsx` — Method Annotation (**AI answers hidden**), Taxonomy Review, Behavior Relevance | `expert/returned/` | human reference, human vs AI metrics, taxonomy_v2, relevance |
| 7 | `results/08_validation/adjudication/adjudication.xlsx` (AI answers revealed, after the blind round) | `adjudication/returned/` | final reference labels |
| 10 | `results/11_benchmark/selected_behaviors.csv` (optional, column `behavior_id`) | — | replaces the automatic choice |
| 11 | `results/12_verification/expert/benchmark_review.xlsx` — Benchmark Task Validation | `expert/returned/` | accepted / revise / rejected |

Reviewer identity (README!B3, else the file name), file sha256 and import time are kept with every
decision; unknown rows, invalid IDs and missing identity are reported, never fixed silently. Until an
expert file arrives the phase is **provisional** (◐): complete, but it will change. `taxonomy.require_review:
true` makes phase 6 stop and wait instead. `./run.sh simulate-expert 7` writes a clearly marked
SIMULATED workbook to test the import path (never a result; delete it before real use).

## Results (one folder per phase)

| Folder | Main outputs | Figures |
|---|---|---|
| `01_repositories/` | `repos.txt` (every repository of the snapshot), `repository_scores.csv` (measures, criterion scores, weighted score, hard filters, decision, reason), `selected_repos.txt`, `scoring_scheme.md` | score distribution, threshold sensitivity, criteria selected vs excluded, score contributions, popularity vs score, decisions |
| `02_literature/` | `literature_tasks.parquet`, `literature_benchmarks.csv` | tasks per benchmark, task types, languages |
| `03_corpus/` | `repositories.parquet`, `methods.parquet` (source, provenance URL, structure), `duplicates.parquet` | concentration, domains, duplication, size, features |
| `04_sampling/` | `sampling_design.json`, `pilot_sample.csv`, `literature_sample.csv` | groups corpus vs sample, methods per repository, targeted signals |
| `05_discovery/` | one folder per model, `<model>/`: `taxonomy.png` (its own candidate taxonomy as a tree), `categories.png`, `candidate_taxonomy.json`, `proposals.jsonl`, `raw_responses.jsonl`; `discovery_summary.csv` | `figures/`: the models compared (concepts per model, proposals per item) |
| `06_taxonomy/` | `taxonomy_v1.json`, `taxonomy_aliases.csv`, `merge_candidates.csv`, `annotation_guidelines_v1.md`, `review/` | taxonomy tree, model overlap, support |
| `07_annotation/` | `annotations_raw.jsonl`, `annotations_validated.parquet` | categories by model, unknown, behaviours per item, validity |
| `08_validation/` | `agreement.json`, `item_agreement.csv`, `disagreement_report.csv`, `expert/`, `human_labels.csv`, `human_vs_ai.json`, `adjudication/`, `taxonomy_v2.json`, `validation_report.md` | agreement levels, α summary, κ, per-behaviour α, human vs AI |
| `09_profile/` | `pharo_behavior_profiles.parquet`, `weighted_frequency.csv`, `repository_prevalence.csv`, `group_differences.csv`, `unknowns.csv` | weighted frequency (categories, behaviours), groups, unknown |
| `10_coverage/` | `coverage_matrix.csv`, `behavior_gaps.csv`, `coverage_by_benchmark.csv` | coverage scatter, status by category, benchmark × category, gaps |
| `11_benchmark/` | `benchmark_candidates.jsonl`, `tasks/<id>/`, `auto_selected_behaviors.csv` | funnel, candidates per category |
| `12_verification/` | `benchmark_verified/` (tasks.jsonl, tasks_private.jsonl, tasks/), `gates.csv`, `validation_report.md`, `expert/` | gates, mutation scores, variants per category |
| `13_evaluation/` | `evaluation_results.csv`, `evaluation_summary.csv`, `answers.jsonl`, `reproducibility.json`, `release/` | pass@1, by task type, by category, outcomes |

## Code (the modules of §4 of the brief)

| Module | File |
|---|---|
| Orchestration, resume, logs, status | `pbr/cli.py`, `pbr/common.py`, `run.sh` |
| Dependencies (models, PT conversion, Pharo image, data) | `pbr/setup.py` |
| Repository selection (weighted quality score) | `pbr/phases/repositories.py` |
| Data ingestion | `pbr/phases/literature.py` (benchmark loader: `pbr/benchmarks.py`), `pbr/phases/corpus.py` |
| Repository analysis: metadata, deduplication, parsing, sampling | `pbr/phases/corpus.py`, `pbr/phases/sampling.py` |
| LLM annotation engine | `pbr/llm.py` (structured output + validation, retries, cache of raw answers, rate limits, cost ledger; Ollama / OpenAI-compatible / Anthropic), `pbr/items.py` |
| Taxonomy manager | `pbr/taxonomy.py` (IDs, validation, versions, revisions, translation), `pbr/phases/discovery.py`, `pbr/phases/taxonomy.py` |
| Agreement analysis | `pbr/agreement.py`, `pbr/phases/validation.py` |
| Expert validation | `pbr/workbook.py`, `p05`, `p07`, `p11`, `pbr/phases/simulate.py` |
| Coverage analyser | `pbr/phases/profile.py`, `pbr/phases/coverage.py` |
| Benchmark builder | `pbr/phases/benchmark.py` |
| Pharo test runner | `pbr/pharo.py` (sandboxed, resource-limited), `pharo/harness.st` (from PharoBench) |
| Evaluation / reporting | `pbr/phases/evaluation.py`, `pbr/figures.py`, `results/SUMMARY.md` |

Tests: `.venv/bin/python -m unittest discover -s tests`.

## Design choices and their reasons

| Choice | Reason |
|---|---|
| Behaviour (3 levels) and evaluation task type are two separate classifications | one behaviour supports several task types (§2); the task type is a property of how a benchmark is built, set once per benchmark in `config.yaml` |
| Stable opaque IDs + registry, versions never edited in place, explicit `id_map` | labels of different models and versions are compared on IDs, not on text; earlier annotations never change silently |
| Three model families, never shown each other's answers | correlated errors of related models (Kim et al., ICML 2025) |
| Inductive discovery then deductive annotation on a new split | a taxonomy measured on the data it was built from looks better than it is |
| Merge candidates by embeddings, but every merge reviewed | similar names are not the same meaning (§5) |
| Krippendorff's α with MASI distance | multi-label answers, missing answers, any number of annotators (Krippendorff 2004; Passonneau 2006) |
| Blinded expert round, then adjudication with AI revealed | showing model labels anchors experts (Fort & Sagot 2010) |
| Expert items stratified by AI agreement, stratified estimators | expert time goes to hard cases without biasing accuracy (Cochran 1977) |
| AI majority ≠ correctness | accuracy of the AI majority is reported per agreement level |
| Repositories selected by a weighted quality score (reaper criteria), hard filters for forks and mirrors | most GitHub repositories are personal, inactive or copies (Kalliamvakou et al., MSR 2014; Lopes et al., OOPSLA 2017); reaper scores engineered projects on weighted dimensions (Munaiah et al., EMSE 2017); stars get a low weight because they measure popularity (Borges & Valente, JSS 2018); the score only selects, it is never an importance weight |
| Stratified two-stage cluster sampling, capped per repository, design weights | 1% of repositories hold 36% of methods: no repository dominates, corpus-wide shares stay estimable |
| Random vs targeted samples kept apart | targeted samples find rare behaviours but have no inclusion probability |
| Non-observation ≠ absence | every gap carries its sample size and a rule-of-three upper bound |
| PPI correction of shares | LLM labels bias prevalence even when accurate (Egami et al., NeurIPS 2023; Angelopoulos et al., Science 2023) |
| Repository popularity never used as importance | half the Pharo repositories have no star; importance = expert relevance ratings |
| Execution gates in a pinned image, MuTalk mutation score, sandbox | a task is kept only on observed facts; model code is untrusted |
