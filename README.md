# Pharo benchmark research

Which programming behaviours does real Pharo code exhibit, which of them do the code benchmarks LLMs are evaluated
on actually test, and how well do LLMs handle the ones that matter in Pharo? This repository answers these
questions with one reproducible pipeline of 13 phases:

1. **Stage 1 (phases 1–6):** from the Pharo repositories mined on GitHub and the tasks of 12
   existing benchmarks to **two final taxonomies of programming behaviours**. One is built from real Pharo methods,
   the other from the benchmark tasks. Three LLM annotators of three model families discover each one
   independently until saturation, then the three versions are merged.
2. **Stage 2 (phases 7–13):** with the Pharo taxonomy (`taxonomy.stage2_source`), validate it on new samples (AI
   agreement and experts), measure how often each behaviour occurs in Pharo, compare Pharo with the benchmarks
   (both sources are classified with the same behaviours), build executable Pharo benchmark tasks, validate them,
   and evaluate models on them.

**The whole pipeline, phase by phase, with its figures: [docs/PIPELINE.md](docs/PIPELINE.md).**

![The pipeline, from the repositories to the two final taxonomies](docs/pipeline.png)

## Run

```bash
./run.sh                    # setup, then phases 1–13 in order; a phase already done with the same inputs is skipped,
                            # an interrupted phase resumes item by item (every LLM answer is cached)
./run.sh 5                  # one phase          ./run.sh 3-6   a range      ./run.sh from 4   phases 4..13
./run.sh setup              # only the dependencies (done automatically before any phase; SKIP_SETUP=1 skips it)
./run.sh status             # one line per phase: ✓ done · ◐ provisional · ⏸ waiting · ✗ failed · (stale)
./run.sh logs [5]           # the log of the whole pipeline, or of one phase (its last run), readable
./run.sh cost               # LLM calls, tokens, time and cost per phase and model
./run.sh reset 5            # delete what phase 5 produced (expert files in returned/ are kept)
PROFILE=smoke ./run.sh      # a few items only, in results-smoke/: checks that every phase works in minutes
python -m pbr diagram       # redraw docs/pipeline.png
```

**On a GPU server (one H100 or H200, Linux).** The same command, with parallel LLM requests (`config.yaml` →
`compute`; they change the speed, never a result):

```bash
LLM_PARALLEL=8 MODELS_PARALLEL=3 ./run.sh
```

`run.sh` starts Ollama with `OLLAMA_NUM_PARALLEL` / `OLLAMA_MAX_LOADED_MODELS` set to match (an Ollama already
running must have been started with them). Do not copy `data/llm-cache/` from another machine: an answer made by
another build of a model is asked again anyway (the cache checks the model digest). Pharo runs without network
through `unshare --net` (util-linux); where user namespaces are disabled the log says so.

Requirements: Python ≥ 3.10, git, curl and [Ollama](https://ollama.com). Everything else (models, Pharo image,
benchmarks, corpus) is obtained by the setup. Expect a large disk and about 20 GB of free memory: the largest
model, gemma4:31b, takes 19 GB. Profiles: `pilot` (default, small samples), `full`, `smoke`.

## The 13 phases

| # | Phase | In → out | Humans |
|---|---|---|---|
| 1 | Repository selection | 1,585 repositories → hard filters + weighted quality score → ≈ 1,087 selected | |
| 2 | Literature ingestion | 12 benchmarks → 6,433 tasks (4,372 distinct problems) | |
| 3 | Corpus and method filters | ≈ 1.84 M methods → a decision per method, taken on its syntax tree | |
| 4 | Representative samples | 99 % confidence ±5 % start samples (663 methods, 577 problems) + ordered pools | |
| 5 | Discovery until saturation | 3 annotators × 2 sources → 6 taxonomies | |
| 6 | The two final taxonomies | per source, 3 taxonomies merged (embeddings + editor) → `taxonomy_v1.json` | review (optional) |
| 7 | Annotation *(stage 2)* | new samples classified with the taxonomy by the 3 annotators, independently | |
| 8 | Agreement and expert validation | Krippendorff's α, blinded expert workbook, adjudication → `taxonomy_v2` | experts |
| 9 | Pharo behaviour profile | large random sample → design-weighted frequency of every behaviour | |
| 10 | Coverage | Pharo × benchmarks per behaviour: both · Pharo only · literature only · investigate | |
| 11 | Benchmark construction | behaviour contracts, then new Pharo tasks (reference + SUnit tests) from real seeds | selection (optional) |
| 12 | Benchmark validation | 5 levels: executes · independent oracle · valid mutants · behaviour alignment · audit | audit |
| 13 | Model evaluation | every model × every task variant (generation, completion, bug repair), in the image | |

`./run.sh` runs all 13 phases. Stage 2 classifies both sources with ONE of the two final taxonomies,
`taxonomy.stage2_source` (default `pharo`): the coverage of phase 10 is then which Pharo behaviours the benchmarks
test; behaviours found only in the benchmarks appear as UNKNOWN there.

## Models

All local, through Ollama (`config.yaml` → `llm`, `benchmark`, `evaluation`):

| Role | Phase | Model |
|---|---|---|
| annotators (3 families, never shown each other's answers) | 5, 7, 9 | `qwen3.8:27b` (Alibaba) · `gemma4:31b` (Google) · `mistral-small3.2:24b` (Mistral AI) |
| taxonomy editor (judges merges, writes definitions) | 6 | `qwen3.8:27b` |
| concept similarity (embeddings) | 5, 6 | `nomic-embed-text` |
| task author (reference, tests, statement) | 11 | `qwen3.8:27b` |
| behaviour contracts | 11 | `gemma4:31b` |
| independent tests | 12 | `gemma4:31b` · `mistral-small3.2:24b` |
| evaluated models | 13 | `qwen3.8:27b` · `gemma4:31b` · `mistral-small3.2:24b` (+ the reference solutions, which must score 100 %) |

The contract and test models are of other families than the task author, so they cannot share its
misunderstandings. In phase 13, every evaluated model also helped build the benchmark: report that with the results.

## Setup (`pbr/setup.py`, log `results/logs/00_setup.log`)

`./run.sh` obtains every dependency before the first phase, and stops if one cannot be obtained:

| Dependency | What setup does |
|---|---|
| the four Ollama models | `ollama pull` when missing; checks that the files are complete (`/api/show`) |
| the Pharo 13 validation image (+ MuTalk, PharoBench-Support), its VM and its global names | built or copied into `data/pharo-image/`; on that first build, the VM must evaluate `3 + 4` |
| the 12 benchmarks | downloaded at pinned revisions into `data/existing-benchmarks/`, each one loaded |
| the Pharo corpus | the parsed methods of the snapshot `pharo-llm/pharo-labels` into `data/pharo-corpus/` (1.3 GB) |

Setup checks that the dependencies are present. It does not check that they work: no model is asked a question,
and an existing image is not run again. A broken model or image shows up only in the phase that uses it.

## Results

```
results/
  SUMMARY.md                                  every phase: status, key numbers, figures
  logs/                                       pipeline.log, NN_<phase>.log, events.jsonl, llm_calls.jsonl, 00_setup.log
  01_repositories/                            repository_scores.csv, selected_repos.txt
  02_literature/                              literature_tasks.parquet
  03_corpus/                                  methods.parquet, method_filters.parquet
  04_sampling/                                the samples, representativeness
  05_discovery/<annotator>/<pharo|literature>/  the six taxonomies
  06_taxonomy/<pharo|literature>/             the two final taxonomies
  07_annotation/ … 13_evaluation/             stage 2 (see docs/PIPELINE.md)
data/                                         downloads, Pharo image, LLM cache (git-ignored)
```

Every phase folder has a `README.md` explaining its files, and a `figures/` folder. A phase is redone only when
something it reads has changed: its settings, the run of an upstream phase, or an expert file it imports.
`./run.sh status` marks such a phase *stale*. Raw LLM answers, every attempt included, are kept in
`data/llm-cache/`. Expert files: `06_taxonomy/<source>/review/`, `08_validation/expert/` and
`12_verification/expert/`; completed files go in their `returned/` folder.

## Settings (`config.yaml`)

| Section | What |
|---|---|
| `compute` | parallel LLM requests per model and annotators at once (speed only; env `LLM_PARALLEL`, `MODELS_PARALLEL`) |
| `profiles` | sample sizes: `pilot` (default), `full`, `smoke` (tiny, own results folder) |
| `repositories` | hard filters, criterion weights and saturation caps, threshold |
| `literature` | the evaluation task type of each benchmark |
| `corpus`, `method_filters` | corpus snapshot; long-method limits, whether review methods are sampled |
| `sampling` | strata, methods per repository, targeted signals |
| `llm` | annotators, editor, embedding model, decoding, context window (grown when a call needs it), retries, prices |
| `discovery` | confidence (0.99), margin (±5 %), round size, min_share → patience, novelty threshold, min_support, consolidation caps |
| `taxonomy` | candidate thresholds, similarity weights, maximum categories, review required or not |
| `annotation`, `validation`, `profile`, `coverage` | stage 2: behaviours per item, expert strata, profile models, support |
| `benchmark` | task author, contract model, oracles, validation thresholds, task types |
| `pharo` | image, time and resource limits, sandbox, repeats, minimum mutation score and tests |
| `evaluation` | evaluated models, samples per task, max tokens |

## Code

| Module | Role |
|---|---|
| `run.sh`, `pbr/cli.py`, `pbr/common.py` | orchestration, resume, fingerprints, logs, status |
| `pbr/setup.py` | dependencies |
| `pbr/phases/<name>.py` | one module per phase |
| `pbr/method_filters.py` | the AST rules of phase 3 |
| `pbr/benchmarks.py` | the benchmarks: pinned sources, download, item cards |
| `pbr/llm.py` | LLM engine: structured output and validation, retries, cache of raw answers, cost ledger |
| `pbr/taxonomy.py` | taxonomy manager: stable IDs, validation, versions, revisions |
| `pbr/agreement.py` | statistics: sample size, saturation, agreement, weighted estimates, Wilson, PPI |
| `pbr/pharo.py`, `pharo/harness.st` | isolated Pharo runner and harness (install, tests, stubs, candidates, MuTalk) |
| `pbr/oracle.py` | independent validation of tasks: contracts, independent tests, alignment, mutant validity, audit sample |
| `pbr/workbook.py` | expert workbooks (Excel): writing and importing with reviewer provenance |
| `pbr/figures.py`, `pbr/diagram.py` | figure style, taxonomy trees, the pipeline figure |
| `tests/` | `.venv/bin/python -m unittest discover -s tests` |

## References

Kalliamvakou et al., *The promises and perils of mining GitHub*, MSR 2014 · Munaiah et al., *Curating GitHub for
engineered software projects*, EMSE 2017 · Lopes et al., *DéjàVu: a map of code duplicates on GitHub*, OOPSLA 2017 ·
Borges & Valente, *What's in a GitHub star?*, JSS 2018 · Cochran, *Sampling Techniques*, 1977 · Wan et al.,
*TnT-LLM*, KDD 2024 (generate, then consolidate) · Kim et al., ICML 2025 (correlated errors of related models) ·
Krippendorff, *Content Analysis*, 2004 · Angelopoulos et al., *Prediction-powered inference*, Science 2023 ·
Barr et al., *The oracle problem in software testing: a survey*, TSE 2015 · Jia & Harman, *An analysis and survey
of the development of mutation testing*, TSE 2011 · Liu et al., *Is your code generated by ChatGPT really correct?*
(EvalPlus), NeurIPS 2023.
