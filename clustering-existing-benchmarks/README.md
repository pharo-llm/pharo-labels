# Clustering existing benchmarks

What do the code benchmarks LLMs are evaluated on really ask? Each of three local models, on its own, puts every
item of every benchmark in a **category** (what its code is about: GUI, databases, dates, …), then a
**subcategory** (what it does or fixes inside it), and writes its **actual task** ("fix OuterRef resolution in
exclude and negated Q objects"). Then the three are compared: do they group the same items together?
On its own: its own `run.sh`, code and `results/`; only the downloads are shared, in `../data/existing-benchmarks/`.

It looks at the **problems**, so each problem counts once: 10 benchmarks (HumanEval+, MBPP+, BigCodeBench,
LiveCodeBench, SWE-bench Verified / Pro / Multilingual, Terminal-Bench 2.0, NL2Repo-Bench, DeepSWE if `HF_TOKEN` is
set). Left out, because they only ask something else about problems already here: BigCodeBench-Instruct (the same
1,140 problems, reworded), LiveCodeBench test output prediction and code execution (182 and 92 problems of
LiveCodeBench code generation). The target of each benchmark is what `../labeling-existing-benchmarks` names.

```bash
./run.sh                              # everything: each model -> results/<model>/, then results/comparison/
SAMPLE=1 OUT=results/pilot ./run.sh   # pilot: 1 item per benchmark, in a folder of its own
./run.sh figures                      # redraw every figure and the comparison from the items.csv files, no model
./run.sh download                     # only download the benchmarks
oarsub -S ./job.oar.sh                # everything, from the beginning, on larochette: the 3 models at once on the GPUs
```

Models (`MODELS` at the top of `run.sh`): `qwen3.8:27b` (Alibaba), `gemma4:31b` (Google), `muse-glimmer:30b`.
Needs Python ≥ 3.10 and [Ollama](https://ollama.com); the models are pulled automatically (~57 GB in all).
DeepSWE is gated: accept its terms on https://huggingface.co/datasets/datacurve/deep-swe and set `HF_TOKEN`
(in `run.sh`, or better `export HF_TOKEN=hf_...` in `../.env`, which is never committed); without it, it is skipped.

### On larochette (4 × AMD MI210, 64 GB each)

`job.oar.sh` does everything from the beginning: installs Ollama and its ROCm add-on (`.ollama-bin/`), starts it on
the GPUs, downloads the benchmarks and the models (`.ollama-models/`, kept for the next jobs), runs the three models
**at the same time** (~20 GB each, one per GPU, 4 requests each), then the comparison and the figures.

```bash
cd ~/Desktop/pharo-labels
rsync -av --exclude .venv --exclude __pycache__ clustering-existing-benchmarks data .env luxembourg.g5k:pharo-labels/
ssh luxembourg.g5k 'cd pharo-labels/clustering-existing-benchmarks && SAMPLE=1 OUT=results/pilot oarsub -S ./job.oar.sh'   # pilot
ssh luxembourg.g5k 'cd pharo-labels/clustering-existing-benchmarks && oarsub -S ./job.oar.sh'                             # every item
rsync -av luxembourg.g5k:pharo-labels/clustering-existing-benchmarks/results/ clustering-existing-benchmarks/results/
```

The job's log is `job.<jobid>.out` in this folder. A job that runs out of walltime (55 h) is resubmitted: every
answer already saved is reused.

## How (each model on its own: `cluster.py`)

| Step | What |
|---|---|
| 1. describe | for every item: its *subject* (noun phrase) and its *actual task* (5–15 words; for a fix, what is wrong) |
| 2. categories | the subjects grouped into ≤ 20 categories with definitions → `categories.csv` |
| 3. classify | every item into one category (or Other) |
| 4. subcategories | inside each category, the actual tasks grouped into ≤ 8 subcategories → `subcategories.csv` |
| 5. classify | every item into one subcategory of its category |

Generate, then classify (TnT-LLM, Wan et al., KDD 2024). The model sees what the evaluated model receives, the
reference answer and how it is checked. `categories.csv` and `subcategories.csv` are kept once written: edit them
(merge, rename, define) and rerun, every item is classified again with your lists (the descriptions are reused);
delete them to let the model propose new ones. Every answer is saved in `answers/` and reused by a rerun.

## Comparison (`compare.py`, no model needed)

Each model has its own category names, so the comparison asks whether two models **group the same items
together**, for every pair, overall and per benchmark, at the category and the subcategory level:
**ARI** (adjusted Rand index, Hubert & Arabie 1985: 1 = same grouping, 0 = chance; ≥ 0.90 excellent, ≥ 0.80 good,
≥ 0.65 moderate, Steinley 2004) and **NMI** (normalized mutual information, Strehl & Ghosh 2002). The mapping shows
where the items of each category of one model went in the other.

## Results

| Folder / file | Contents |
|---|---|
| `results/<model>/clusters.md` | that model's view: per benchmark, category › subcategory › every actual task |
| `results/<model>/items.csv` | one row per item: category, subcategory, actual task, subject |
| `results/<model>/categories.csv`, `subcategories.csv` | its categories and subcategories (editable) |
| `results/<model>/figures/` | `overview.png` (benchmark × category), one PNG per benchmark (categories › subcategories › examples of actual tasks), `all_benchmarks.pdf` |
| `results/comparison/comparison.md` | how much the models agree: ARI and NMI per pair, per benchmark, and each benchmark's main categories by model |
| `results/comparison/*.csv` | `agreement.csv`, `agreement_per_benchmark.csv`, `category_mapping.csv`, `benchmark_profiles.csv` |
| `results/comparison/figures/` | `agreement.png` (pairwise matrices), `agreement_per_benchmark_*.png`, `profiles.png`, `mapping_<a>_to_<b>.png`, `comparison.pdf` |

Every figure is drawn from the `items.csv` files only: change `figures.py` or `compare.py`, then `./run.sh figures`.
