# PharoTaskMiner

Which programming tasks does real Pharo code perform? This repository answers that from zero:
it mines GitHub, keeps the engineered Pharo projects, and builds a taxonomy of the tasks their
methods perform: a local model finds the categories (inductive coding, until saturation), three
models of three families classify a new sample with the resulting codebook (deductive coding),
and Pharo experts classify a stratified subsample blind, which gives each model's accuracy, the
experts' own agreement, and the share of each task in the corpus corrected for the models' errors. Benchmark generation lives in another repository.

```
~20,000 GitHub candidates
   │  Pharo verification + FILTER 1 (6 criteria: is this real Pharo code?)
   ▼
~1,585 repositories ── cloned, every method parsed (2.4 M methods)
   │  FILTER 2 (13 criteria: an engineered project worth studying)
   ▼
135 repositories = study corpus ── ~280,000 methods
   │  sample-size calculator (Cochran + finite population correction): 384 at 95% / ±5%
   ▼
initial sample ── a model writes a summary and names the task of each method (open coding)
   │  SATURATION LOOP: draw 50 new methods, label, count new categories;
   │  stop when enough rounds in a row add no category
   ▼
codebook_draft.csv ── categories + definitions written by the model
   │  you consolidate it: results/labelling/codebook.csv (merge, rename, define; + Other)
   ▼
NEW random sample (384) ── classified by qwen3.8 (Alibaba), gemma4 (Google), granite4.2 (IBM)
   │  Krippendorff's α, Fleiss' κ, Gwet's AC1; majority label; level 3/3, 2/3, none;
   │  prompt-sensitivity check (codebook and examples shuffled)
   ▼
α < 0.667 ── NOT_RELIABLE.md: sharpen the codebook, classify again
α ≥ 0.667 ── EXPERT FILES: 200 methods stratified by level (more where models disagree),
   │         classified blind by each expert, alone (expert_validation/returned/<expert>/)
   ▼
accuracy of each model (stratified, 95% CI) · experts' α (human baseline) · precision/recall
share of each task in the corpus, corrected by the experts (PPI) · final taxonomy
```

## Run

```bash
./run.sh            # everything; each step resumes where it stopped
./run.sh mine       # 1. GitHub mining + filter 1 (or download the Hugging Face snapshot, ~16 GB)
./run.sh filter     # 2. filter 2: the study corpus
./run.sh label      # 3. inductive: sample size, open coding, saturation loop, codebook draft
./run.sh classify   # 4. deductive: 3 models classify a new sample; agreement; gate; expert files
./run.sh validate   # 5. the experts' answers: accuracy, agreement, corrected shares, final taxonomy
./run.sh report     # 6. results/REPORT.md
./run.sh status | pull | push
```

All settings are at the top of `run.sh` (model, confidence, margin, batch size, patience, …);
the thresholds of the two filters are in `config.yaml`. Needs Python ≥ 3.10 and
[Ollama](https://ollama.com) for the default models (started and pulled automatically).

### Models

Open-weight models on your own machine (an M4 Max with 36 GB), no API and no GPU cluster:

| Model | Role |
|---|---|
| `qwen3.8:27b` (Alibaba, 18 GB) | `MODEL`: inductive coding and the codebook definitions; first of `CLASSIFY_MODELS` |
| `gemma4:12b` (Google, 8 GB) | second of `CLASSIFY_MODELS` |
| `granite4.2:8b` (IBM, 5 GB) | third of `CLASSIFY_MODELS` |

Labelling is describing code, so the three models are instruction-tuned models from three
companies, asked through Ollama's chat API (thinking off, deterministic decoding). In step 3 each
method is coded on its own, without the list of categories found so far: models copy such a list
even when nothing in it fits.

### Methodology, and the problem each step answers

The full explanation, with every problem, its solution and its sources: [METHODOLOGY.md](METHODOLOGY.md).

| Step | Problem | Answer (source) |
|---|---|---|
| Saturation loop | when have all the tasks been seen? | stop when a task of ≥ 0.5% prevalence would have appeared with 95% probability; inductive thematic saturation (De Paoli & Mathis, *Quality & Quantity*) |
| Inductive, then deductive | open codes of different models cannot be compared ("Setter" vs "Accessor") | build a codebook, then classify with it: LLMs agree with humans far better deductively (Shah et al. 2025; TnT-LLM, KDD 2024; Nakashima et al., ASE 2026) |
| New sample for step 4 | a codebook tested on the data it was built from looks better than it is | the classified sample excludes every method of step 3 |
| Three families | related models make the same mistakes (Kim et al., ICML 2025); a majority vote amplifies shared errors | Alibaba, Google, IBM; experts also check methods where all three agree |
| α, AC1 | kappa collapses when one category dominates (kappa paradox); more than two annotators | Krippendorff's α (≥ 0.667 tentative, ≥ 0.8 reliable), Gwet's AC1, Fleiss' κ |
| Gate | experts' time spent on a codebook the models cannot apply | no expert files below α = 0.667 (model-model agreement as the signal: Ahmed et al., MSR 2025) |
| Prompt check | answers can depend on the prompt (Camuffo et al. 2026) | the first model again with the codebook and examples shuffled |
| Stratified expert sample | a random sample wastes expert time on easy cases; mixing in hard cases biases a naive accuracy | more methods where models disagree, known weights, stratified estimators (Cochran 1977, ch. 12; Gligorić et al., NAACL 2025) |
| Blind experts, two of them | showing the model's label anchors the expert; no human baseline | experts classify without the models' answers; their α is the baseline a model must approach (Ahmed et al.) |
| Corrected shares | LLM labels bias prevalence estimates even at 80-90% accuracy (Egami et al., NeurIPS 2023) | prediction-powered inference: model shares corrected with the expert sample, valid 95% intervals (Angelopoulos et al., *Science* 2023) |
| Reporting | reproducibility | model versions, prompts, every answer stored (Baltes et al., *Empirical Software Engineering*) |

## Results

| Path | Contents |
|---|---|
| `results/REPORT.md` | **start here**: every step from GitHub to the expert export, with numbers and figures, and what the labelling would cost on commercial APIs |
| `results/labelling/taxonomy.csv`, `taxonomy.md`, `labels.csv`, `curve.csv` | inductive phase: categories, labelled methods, saturation curve |
| `results/labelling/codebook_draft.csv` → `codebook.csv` | the model's codebook draft; your consolidated codebook |
| `results/labelling/classify/` | deductive phase: every model's answers, `labels.csv`, `agreement.json`, `gold_selection.csv` (strata and weights, not for the experts) |
| `results/labelling/expert_validation/` | **what Pharo experts fill in** (`README.md` explains it); their files go back in `returned/<expert>/`, the agreed answers after discussion in `returned/consensus/` |
| `results/labelling/validated/` | **the final numbers**: `final_taxonomy.md`, `validation.json`, `to_discuss.csv` |
| `results/labelling/labels.jsonl`, `codebook.json`, `plan.json`, `rounds/` | every inductive answer, the open-coding codebook, the settings and the methods drawn per round |
| `results/figures/` | saturation curve, category distribution |
| `results/reports/corpus_filter.md`, `results/tables/` | filter 2 and filter 1 decisions for every repository |
| `results/data/` | GitHub cache, clones, parsed methods, SQLite database |

## Code

| Path | Contents |
|---|---|
| `run.sh` | the whole pipeline (settings at the top) |
| `config.yaml` | filter thresholds and defaults |
| `src/pharo_task_miner/discovery/`, `extraction/`, `analysis/`, `sampling/` | mining, parsing, filters 1 and 2 |
| `src/pharo_task_miner/labelling/` | sample size, models, open coding, saturation loop (step 3); `classify.py` (step 4); `agreement.py` (α, κ, AC1, stratified and PPI estimators); `validation.py` (step 5); report |
| `src/pharo_task_miner/snapshot.py` | saving / restoring all results on Hugging Face (`pharo-llm/pharo-labels`) |
| `taxonomy/` | application domains and framework mappings used by mining |
| `pharo/ASTExporter/` | optional parser backend using Pharo's own `RBParser` |

Individual stages: `.venv/bin/pharo-miner --help`. Tests: `.venv/bin/python -m unittest discover -s tests`.
