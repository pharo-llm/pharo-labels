# Pharo benchmark research

Which programming behaviours does real Pharo code exhibit, and which ones do the code benchmarks that LLMs are
evaluated on actually test? This repository answers the first half of that question with a reproducible pipeline:
from the Pharo repositories mined on GitHub to **two final taxonomies of programming behaviours** — one built from
real Pharo methods, one built from the tasks of the existing benchmarks — each discovered independently by three
LLM annotators of three model families until saturation, then merged into one shared taxonomy per source.

![The pipeline, from the repositories to the two final taxonomies](docs/pipeline.png)

**The whole pipeline, step by step, with its figures: [docs/PIPELINE.md](docs/PIPELINE.md).**

## Run

```bash
./run.sh                    # setup, then phases 1–6 in order; a phase already done with the same inputs is skipped,
                            # an interrupted phase resumes item by item (every LLM answer is cached)
./run.sh 5                  # one phase          ./run.sh 3-6   a range      ./run.sh from 4   phases 4..6
./run.sh setup              # only the dependencies (done automatically before any phase; SKIP_SETUP=1 skips it)
./run.sh status             # one line per phase: ✓ done · ◐ provisional · ⏸ waiting · ✗ failed · (stale)
./run.sh logs [5]           # the log of the whole pipeline, or of one phase (its last run), readable
./run.sh cost               # LLM calls, tokens, time and cost per phase and model
./run.sh reset 5            # delete what phase 5 produced (expert files in returned/ are kept)
PROFILE=smoke ./run.sh      # a few items only, in results-smoke/: checks that every phase works in minutes
python -m pbr diagram       # redraw docs/pipeline.png
```

Needs only Python ≥ 3.10, git and [Ollama](https://ollama.com). Every number of a run is in `results/SUMMARY.md`;
every phase folder has a `README.md` explaining its files.

## The pipeline (stage 1: phases 1–6)

### Setup — every dependency before any phase (`pbr/setup.py`)

`./run.sh` first obtains what the phases need and stops if one cannot be obtained (log `results/logs/00_setup.log`;
a complete setup takes seconds):

| Dependency | How |
|---|---|
| the three annotators, the editor, the embedding model | `ollama pull` (checked to load, not only to be listed) |
| the Pharo PT model `pharo-llm/Qwen2.5-Coder-PT:7b` (evaluation, stage 2) | its LoRA adapter (Hugging Face `pharo-llm/Qwen2.5-Coder-7B-PT`, branch `checkpoint-15400`) merged into its base, converted to GGUF (llama.cpp), registered in Ollama |
| the Pharo 13 validation image (+ MuTalk, PharoBench-Support) and its global names | built or copied into `data/pharo-image/` |
| the 12 benchmarks | downloaded at pinned revisions into `data/existing-benchmarks/` |
| the Pharo corpus | the parsed methods of the snapshot `pharo-llm/pharo-labels` into `data/pharo-corpus/` (1.3 GB) |

### 1 · Repository selection — `results/01_repositories/`

The corpus snapshot holds the **1,585 repositories** that passed mining filter 1 (real Pharo code, not
coursework); they are listed in `repos.txt`. Most GitHub repositories are personal, inactive or copies
(Kalliamvakou et al., MSR 2014), so each repository gets a **quality score** in [0, 1]: the weighted sum of eleven
criterion scores, after the *reaper* framework for engineered projects (Munaiah et al., EMSE 2017):

| criterion | measure | weight |
|---|---|---|
| community | contributors | 0.15 |
| history | commits | 0.15 |
| unit testing | share of test methods | 0.15 |
| maintenance | development span × recency of the last push | 0.10 |
| documentation | share of classes with a comment | 0.10 |
| size | non-test methods | 0.10 |
| originality | share of non-trivial methods not copied from an older repository (Lopes et al., OOPSLA 2017) | 0.05 |
| licence | a licence is declared | 0.05 |
| issues | the issue tracker is used | 0.05 |
| loadability | a Metacello `BaselineOf` (Pharo-specific) | 0.05 |
| popularity | stars + forks — low on purpose: stars measure popularity, not quality (Borges & Valente, JSS 2018) | 0.05 |

Raw values become scores by saturating transforms, so no giant repository dominates. **Hard filters** come first:
GitHub forks, mirrors (≥ 50 % of non-trivial methods copied from an older repository), repositories without
methods. A repository is kept when its score is **≥ 0.50** (half of the maximum): ≈ **1,087** repositories. The
score only selects; it is never used as an importance or a sampling weight.

Outputs: `repos.txt`, `repository_scores.csv` (measures, criterion scores, score, decision, reason),
`selected_repos.txt`, `scoring_scheme.md`. Figures: score distribution, threshold sensitivity (what other thresholds
keep), criteria of selected vs excluded, score contributions, popularity vs score, decisions.

### 2 · Literature ingestion — `results/02_literature/`

Every task of the code benchmarks frontier models are evaluated on: HumanEval+, MBPP+, BigCodeBench Complete /
Instruct, LiveCodeBench code generation / test-output prediction / code execution, SWE-bench Verified / Pro /
Multilingual, Terminal-Bench 2.0, NL2Repo (DeepSWE when `HF_TOKEN` gives access). **6,433 tasks = 4,372 distinct
problems** (a problem asked in several scenarios counts once). Each task keeps what the evaluated model receives,
the reference answer, how it is checked, its language and its evaluation task type (code generation, completion,
bug repair, test generation, other). Outputs: `literature_tasks.parquet`, `literature_benchmarks.csv`; figures.

### 3 · Corpus and method filters — `results/03_corpus/`

The ≈ **1.84 M methods** of the selected repositories, each with its provenance (repository, commit, file, line,
GitHub link), its class context and its **abstract syntax tree**. Copies are detected across repositories (the
canonical copy is the oldest). Then every method gets a decision, **decided on its AST and its class definition**
(variables resolved like the compiler: arguments and temporaries, the class's instance / class variables,
pseudo-variables, globals) — `pbr/method_filters.py`:

| method (AST pattern) | decision |
|---|---|
| duplicate of an older method · in a mirror repository · cannot be parsed | exclude |
| simple getter `^ v` · simple setter `v := arg` (v a variable of the class) | exclude |
| returns a constant (`^ literal`, `^ true`, literal arrays) · empty · placeholder (`self subclassResponsibility`, `^ self`, …) | exclude |
| generated boilerplate (manifests, Metacello baselines / configurations, `<generated>`, embedded data) | exclude |
| trivial initialisation (`initialize` = `super initialize` + `v := literal / Class new`) | exclude |
| simple message forwarding · non-trivial initialisation · very long (> 60 lines or > 40 statements) · requires dependencies (globals neither its project nor Pharo 13 defines) | review: kept, marked |
| test methods | keep separately |

Outputs: `repositories.parquet`, `methods.parquet`, `duplicates.parquet`, `method_filters.parquet` (decision and
every reason per method), `method_filters_summary.csv`. Figures: method filters, duplication, method size, features.

### 4 · Representative samples — `results/04_sampling/`

**Pharo methods.** Stratified two-stage sampling: the strata are repository domains, stage 1 draws repositories,
stage 2 at most 4 methods per repository, so no large repository dominates; every method carries its design weight.
**Benchmarks.** Proportional to each benchmark's number of problems. For each source the sampler draws an **ordered
pool**: the start sample, then the items of the saturation rounds.

The start sample is a **99 % confidence, ±5 %** sample (Cochran 1977, with finite population correction, p = 0.5):
**663 Pharo methods** of ≈ 690,000 eligible, **577 benchmark problems** of 4,372 (100 % confidence would be a
census of every item). Targeted methods (rare areas: concurrency, FFI, reflection, network…) and test methods
(kept separately) are added. The **representativeness check** compares the start sample with the population: for
Pharo, the share of methods with each AST feature (loops, conditionals, iteration, exceptions, blocks, reflection,
recursion…) and the method size, design-weighted, with 95 % intervals; for the benchmarks, the share of every
benchmark and task type. Outputs: `sampling_design.json`, `pilot_sample.csv`, `literature_sample.csv`,
`representativeness.csv`; figures `representativeness_pharo.png`, `representativeness_literature.png`, groups.

### 5 · Discovery until saturation, three annotators, two sources — `results/05_discovery/`

Three instruction-tuned models of three families — **qwen3.8:27b** (Alibaba), **gemma4:31b** (Google),
**mistral-small3.2:24b** (Mistral AI) — each alone, never shown another model's answers (related models make the
same mistakes: Kim et al., ICML 2025), read the items of one source and give, for each, up to three behaviours at
three levels — **category › subcategory › behaviour** — in language-neutral terms, with a quote as evidence.

**Saturation loop.** Each annotator starts with the 99 % sample, then goes on with rounds of 50 new items until
**19 rounds in a row bring no new behaviour**: then a behaviour of 0.5 % prevalence would have appeared with 99 %
probability (inductive thematic saturation; the rule of the earlier labelling study). A behaviour is new when a
group of similar proposals (embedding cosine ≥ 0.85) reaches 2 items, so one odd wording never counts as new. Each
annotator then consolidates its proposals into its own taxonomy, with definitions and inclusion / exclusion rules.

This gives **six taxonomies**, in `<annotator>/<pharo|literature>/`: `taxonomy.png` (the tree; under each ●●○ / on
the right of each category and subcategory, a bar and the number of items classified there), `taxonomy.json`,
`taxonomy.csv` (one row per behaviour), `items.csv` (each item and its category / subcategory / behaviour),
`categories.png`, `saturation.csv` / `saturation.png`, `proposals.jsonl`, `raw_responses.jsonl`; `figures/`
compares them (`saturation_pharo.png`, `saturation_literature.png`, `concepts_per_model.png`).

### 6 · The two final taxonomies — `results/06_taxonomy/pharo/` and `results/06_taxonomy/literature/`

For each source, the three annotators' taxonomies become **one shared taxonomy**:

1. **Similarity of the annotators' concepts** = 0.6 × cosine of their text embeddings (`nomic-embed-text`, "name:
   definition") + 0.4 × the overlap of the items they classify (Jaccard, on the items both annotators saw; a
   category or subcategory holds every item under it). The annotators classified the same items, so the same items
   are evidence of the same meaning; candidate groups by average-linkage clustering on this similarity.
2. **Judged merges**: an editor model reads each group — members with their number of items, largest first — and
   keeps together only the concepts that mean the same thing, named after the best-supported members.
3. **Hierarchy and harmonisation, weighted by items**: a behaviour goes under the subcategory where most of its
   members' items were placed; categories are merged down to ≤ 16 and duplicates merged, every operation recorded.
4. **Stable IDs** (`C001` / `S001` / `B001`), definitions, inclusion and exclusion rules, aliases, the models that
   proposed each concept and the items behind it.
5. **Expert review**: `review/taxonomy_review.xlsx` (accept / rename / redefine / merge / split / reject / add / move);
   completed files in `review/returned/` produce the reviewed version on the next run. Until then the taxonomy is
   *provisional* (◐).

Outputs per source: **`taxonomy_v1.json`**, `figures/taxonomy_v1.png` (●●○ = the annotators whose concepts the
merge joined into a behaviour; bar and number = items classified in each category, subcategory and behaviour), `taxonomy_aliases.csv`,
`merge_candidates.csv`, `annotation_guidelines_v1.md`, `review/`.

### Stage 2 (phases 7–13): not yet adapted to the two taxonomies

The code of the next stage is in the repository — annotation of NEW samples with the final taxonomies (their
validation), agreement and blinded expert validation, corpus-wide profile, literature × Pharo coverage, benchmark
construction, execution in an isolated Pharo image, model evaluation — but it still expects one shared taxonomy.
`./run.sh` stops after phase 6; asking for a later phase says so.

## Results

```
results/
  SUMMARY.md                       every phase: status, key numbers, figures
  logs/                            pipeline.log, NN_<phase>.log, events.jsonl, llm_calls.jsonl, 00_setup.log
  01_repositories/  02_literature/  03_corpus/  04_sampling/
  05_discovery/<annotator>/<pharo|literature>/       the six taxonomies
  06_taxonomy/<pharo|literature>/                    the two final taxonomies
data/                              downloads, Pharo image, conversion tools, LLM cache (git-ignored)
```

A phase is redone only when what it reads changed (its settings, the run of an upstream phase, or an expert file it
imports); `./run.sh status` marks it *stale*. Raw LLM answers (every attempt) are kept in `data/llm-cache/`.

## Settings (`config.yaml`)

| Section | What |
|---|---|
| `repositories` | hard filters, criterion weights and saturation caps, threshold |
| `method_filters` | long-method limits, whether review methods are sampled |
| `sampling` | strata, methods per repository, targeted signals |
| `discovery` | confidence (0.99), margin (±5 %), round size, min_share → patience, novelty threshold, min_support |
| `llm` | the annotators, the editor, the embedding model, decoding |
| `taxonomy` | candidate thresholds, maximum categories, review required or not |
| `profiles` | `pilot` (default), `full`, `smoke` (tiny, own results folder) |

## Code

| Module | Role |
|---|---|
| `run.sh`, `pbr/cli.py`, `pbr/common.py` | orchestration, resume, fingerprints, logs, status |
| `pbr/setup.py` | dependencies |
| `pbr/phases/<name>.py` | one module per phase (repositories, literature, corpus, sampling, discovery, taxonomy, …) |
| `pbr/method_filters.py` | the AST rules of phase 3 |
| `pbr/benchmarks.py` | the benchmarks: pinned sources, download, item cards |
| `pbr/llm.py` | LLM engine: structured output and validation, retries, cache of raw answers, cost ledger |
| `pbr/taxonomy.py` | taxonomy manager: stable IDs, validation, versions, revisions |
| `pbr/agreement.py` | statistics: sample size, saturation, agreement, weighted estimates |
| `pbr/figures.py`, `pbr/diagram.py` | figure style, taxonomy trees, the pipeline figure |
| `tests/` | `.venv/bin/python -m unittest discover -s tests` |

## References

Kalliamvakou et al., *The promises and perils of mining GitHub*, MSR 2014 · Munaiah et al., *Curating GitHub for
engineered software projects*, EMSE 2017 · Lopes et al., *DéjàVu: a map of code duplicates on GitHub*, OOPSLA 2017 ·
Borges & Valente, *What's in a GitHub star?*, JSS 2018 · Cochran, *Sampling Techniques*, 1977 · Wan et al.,
*TnT-LLM*, KDD 2024 (generate, then consolidate) · Kim et al., ICML 2025 (correlated errors of related models) ·
Krippendorff, *Content Analysis*, 2004.
