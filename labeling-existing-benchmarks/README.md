# Labeling existing benchmarks

The code benchmarks LLMs are evaluated on, downloaded once into one shared folder, and two separate analyses of
them, each with its own code folder and its own results folder:

```
benchmarks.py            the benchmarks (pinned sources), their download, item cards, Ollama client
../data/already-benchmarks/  every benchmark, downloaded once, shared with ../clustering-existing-benchmarks/
tasks/                   the TASK each benchmark asks (input -> output, e.g. resolve an issue with a patch)
results/tasks/           results of tasks/
run.sh                   download + tasks
oar.sh                   any of the steps on larochette
```

```bash
./run.sh                          # download, then tasks
./run.sh download                 # benchmarks -> ../data/already-benchmarks/ (no model)
./run.sh tasks                    # -> results/tasks/
SAMPLE=5 ./run.sh tasks           # pilot: 5 items per benchmark
```

Needs Python ≥ 3.10 and [Ollama](https://ollama.com) with `qwen3.8:27b` (pulled automatically). Every step
resumes: an item whose answer is already saved is never asked again.

### On an OAR cluster

`oar.sh` runs `./run.sh` on one larochette node (4 × AMD MI210; `STEPS="tasks"` for only some steps), installing Ollama and its ROCm
add-on in `.ollama-bin/` and keeping the model in `.ollama-models/` for the next jobs. Send the folder with what is
already labelled, but without the macOS `.venv`, then submit from the frontend:

```bash
rsync -av --exclude .venv --exclude __pycache__ ../labeling-existing-benchmarks <frontend>:pharo-labels/
rsync -av ../data <frontend>:pharo-labels/        # the shared downloads (../data/already-benchmarks/)
rsync -av ../.env <frontend>:pharo-labels/        # only for DeepSWE (HF_TOKEN)
ssh <frontend> 'cd pharo-labels/labeling-existing-benchmarks && oarsub -S ./oar.sh'
```

Items already answered in `results/` are reused, never relabelled; a job that runs out of walltime is resubmitted.
Copy `results/` back with `rsync -av <frontend>:pharo-labels/labeling-existing-benchmarks/results/ results/`.
`results/tasks/`: `tasks.csv`, `tasks.md`, `benchmarks.csv` and `consolidation.json` are rewritten at the end of each run from the
benchmarks of that run only, so finish with a run of all of them (not `ONLY=…`).

## Benchmarks

| Benchmark | Group | Source (pinned revision in `benchmarks.py`) |
|---|---|---|
| HumanEval+, MBPP+ | classic | `evalplus/humanevalplus`, `evalplus/mbppplus` |
| BigCodeBench Complete, Instruct | classic | `bigcode/bigcodebench` v0.1.4 |
| LiveCodeBench code generation, test output prediction, code execution | frontier | `livecodebench/*` |
| SWE-bench Verified, Pro (public), Multilingual | frontier | `princeton-nlp/…`, `ScaleAI/…`, `SWE-bench/…` |
| Terminal-Bench 2.0 | frontier | `harborframework/terminal-bench-2.0` (official mirror) |
| NL2Repo-Bench | frontier | GitHub `multimodal-art-projection/NL2RepoBench` |
| DeepSWE | frontier | `datacurve/deep-swe`: gated, needs `HF_TOKEN` in `../.env` and its terms accepted |

*frontier* = reported in 2026 model cards of large AI companies; *classic* = reference benchmarks of the
code-generation surveys. A benchmark with several scenarios counts each scenario as its own benchmark. LiveCodeBench code
generation: all six releases (1,055 problems), streamed (4.5 GB with tests; only the problems are kept).

## How the task is named (`tasks/`)

1. For every item of every benchmark, `qwen3.8:27b` sees what the evaluated model receives, the
   reference answer and how it is checked, and names the task (an imperative phrase, no topic), its input
   and its output. The examples in the prompt are tasks of none of these benchmarks.
2. Per benchmark, the model reads the names given to all its items and writes the one task they share, and the items that
   do not fit: **agreement** = share that fits.
3. The model groups the benchmarks whose input and output are of the same kind: one row per task.

## Results of `tasks/`

| File | Contents |
|---|---|
| `results/tasks/tasks.csv`, `tasks.md` | **the list of tasks**: input, output, which benchmarks ask it |
| `results/tasks/benchmarks.csv` | each benchmark: its task, agreement, size, source and revision |
| `results/tasks/answers.jsonl` | every item's answer (a rerun reuses them) |
| `results/tasks/consolidation.json` | the model's answers to steps 2 and 3 |
