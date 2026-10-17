# PharoTaskMiner

Which programming tasks does real Pharo code perform? This repository answers that from zero:
it mines GitHub, keeps the engineered Pharo projects, and builds a taxonomy of the tasks their
methods perform, labelled by small (Pharo-trained) language models, checked for saturation, and
exported for validation by Pharo experts. Benchmark generation lives in another repository.

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
initial sample ── a small model writes a summary and names the task of each method (open coding)
   │  SATURATION LOOP: draw 50 new methods, label, count new categories;
   │  stop after 3 rounds in a row without a new category
   ▼
taxonomy + statistics ── check model agreement ── EXPERT VALIDATION export
```

## Run

```bash
./run.sh            # everything; each step resumes where it stopped
./run.sh mine       # 1. GitHub mining + filter 1 (or download the Hugging Face snapshot, ~16 GB)
./run.sh filter     # 2. filter 2: the study corpus
./run.sh label      # 3. sample size, labelling, saturation loop, check model, expert export
./run.sh report     # 4. results/REPORT.md
./run.sh status | pull | push
```

All settings are at the top of `run.sh` (model, confidence, margin, batch size, patience, …);
the thresholds of the two filters are in `config.yaml`. Needs Python ≥ 3.10 and
[Ollama](https://ollama.com) for the default models (started and pulled automatically).

### Models

Small models on your own machine, no API and no GPU cluster:

| `MODEL=` | What |
|---|---|
| `pharo-llm/Qwen2.5-Coder-SFT:1.5B` (default) | Pharo-trained Qwen2.5-Coder through Ollama; ~0.2 s per method on a laptop |
| `pharo-llm/Qwen2.5-Coder-SFT:0.5B`, `:3b`, `:7b` | the same family, smaller or larger (`:7b` is the default check model) |
| `hf:pharo-llm/Qwen2.5-Coder-1.5B-PT` | the continued-pretraining LoRA adapters (0.5B, 1.5B, 3B, 7B) through transformers; the last `checkpoint-N` branch is used, torch/peft are installed on first use |
| any Ollama tag, e.g. `qwen2.5-coder:7b` | a general code model, for comparison |

The Pharo models are code-completion models, not chat models, so they complete a few-shot
prompt (summary, then task name). Each method is coded on its own, without the list of
categories found so far: small models copy such a list even when nothing in it fits.

## Results

| Path | Contents |
|---|---|
| `results/REPORT.md` | **start here**: every step from GitHub to the expert export, with numbers and figures |
| `results/labelling/taxonomy.csv`, `taxonomy.md` | the categories, their share of the corpus (95% intervals) and examples |
| `results/labelling/labels.csv` | every labelled method, its summary, its category and a GitHub link |
| `results/labelling/curve.csv` | the saturation curve |
| `results/labelling/expert_validation/` | **what Pharo experts fill in** (`README.md` explains it) |
| `results/labelling/labels.jsonl`, `check.jsonl`, `codebook.json`, `plan.json`, `rounds/` | every model answer, the codebook, the settings and the methods drawn per round |
| `results/figures/` | saturation curve, category distribution |
| `results/reports/corpus_filter.md`, `results/tables/` | filter 2 and filter 1 decisions for every repository |
| `results/data/` | GitHub cache, clones, parsed methods, SQLite database |

## Code

| Path | Contents |
|---|---|
| `run.sh` | the whole pipeline (settings at the top) |
| `config.yaml` | filter thresholds and defaults |
| `src/pharo_task_miner/discovery/`, `extraction/`, `analysis/`, `sampling/` | mining, parsing, filters 1 and 2 |
| `src/pharo_task_miner/labelling/` | sample size, models, open coding and codebook, saturation loop, export, report |
| `src/pharo_task_miner/snapshot.py` | saving / restoring all results on Hugging Face (`pharo-llm/pharo-labels`) |
| `taxonomy/` | application domains and framework mappings used by mining |
| `pharo/ASTExporter/` | optional parser backend using Pharo's own `RBParser` |

Individual stages: `.venv/bin/pharo-miner --help`. Tests: `.venv/bin/python -m unittest discover -s tests`.
