# pharo-labels

Two separate studies, each in its own folder with its own `run.sh`:

| Folder | What it does | Start with |
|---|---|---|
| [`labeling-existing-benchmarks/`](labeling-existing-benchmarks/README.md) | downloads the code benchmarks LLMs are evaluated on (HumanEval+, MBPP+, BigCodeBench, LiveCodeBench, SWE-bench Verified / Pro / Multilingual, Terminal-Bench, NL2Repo, DeepSWE) and names the **task** each one asks the model to do | `results/tasks.csv` |
| [`labeling-pharo-repos/`](labeling-pharo-repos/README.md) | mines GitHub for Pharo projects, filters them to the study corpus, labels the programming task of their methods (inductive, then 3 models, then experts) | `results/REPORT.md` |

```bash
cd labeling-existing-benchmarks && ./run.sh
cd labeling-pharo-repos && ./run.sh
```

Tokens (`GITHUB_TOKEN`, `HF_TOKEN`) are kept once in `.env` here (git-ignored) and read by both.
