# Documentation

The project runs as five scripts, one after the other. Each page below explains one script:
what it does, its settings, what it writes, and what can go wrong.

| Page | Script | In one sentence |
|---|---|---|
| [1-mining.md](1-mining.md) | `./1-mining.sh` | Finds Pharo repositories on GitHub, keeps the serious ones, parses every method and labels its operations with rules |
| [1-2-filtering.md](1-2-filtering.md) | `./1-2-filtering.sh` | Keeps the repositories worth labelling and benchmarking: the study corpus, 150 at most |
| [2-ai-labelling.md](2-ai-labelling.md) | `./2-ai-labelling.sh` | Language models label a sample of 300 methods of the study corpus, to check the rule labels of step 1 |
| [3-results.md](3-results.md) | `./3-results.sh` | Builds the taxonomy, the statistics, the figures and the reports |
| [4-benchmark.md](4-benchmark.md) | `./4-benchmark.sh` | Writes benchmark tasks for the operations found, and keeps only those a Pharo image proves correct |
| [cluster-job.md](cluster-job.md) | `job.oar.sh` | Runs steps 1, 2 and 3 as one job on a Grid'5000 node |
| [snapshots.md](snapshots.md) | `./snapshot.sh` | Saves all results on Hugging Face and gets them back |

## How the steps fit together

```
1-mining.sh        GitHub → Pharo repositories → quality filter → clone → parse → rule labels
      │
1-2-filtering.sh   1,581 mined repositories → 13 criteria → study corpus (150 at most)
      │                                                       └→ 300-method sample
2-ai-labelling.sh  models label the 300 methods independently → reviewed labels
      │
3-results.sh       taxonomy → statistics → benchmark coverage → figures and reports
      │
4-benchmark.sh     real methods of the study corpus as seeds → a model writes tasks → Pharo image accepts or rejects
```

- Step 1 produces the data. Its labels are automatic, so they are not trusted yet.
- Step 1-2 decides which repositories the rest is based on. Steps 2, 3 and 4 use only those.
- Step 2 produces labels to compare them with. People can do this instead of models.
- Step 3 needs the labels of step 2 (or of people) and produces everything for a paper.
- Step 4 needs only steps 1 and 1-2. It works better after step 3, because it then knows which
  operations existing benchmarks miss.

## What all scripts have in common

- **Settings are at the top of each script**, in a block named `SETTINGS`. A variable of the
  same name set in the environment wins: `LIMIT=10 ./2-ai-labelling.sh`.
- **Everything is written to `results/`**, which is git-ignored. Each script also has a log
  file there.
- **Every script can be interrupted and rerun.** It continues where it stopped.
- **The first run creates `.venv/`** and installs the Python package. Python 3.10 or newer is
  needed.
- **Secrets are asked once and kept in `.env`** (git-ignored): `GITHUB_TOKEN`, `HF_TOKEN`, and
  the API key of any model you choose. A cluster job cannot ask, so put them in `.env` first.
- The other settings (thresholds, filters, sample sizes) are in `config.yaml`.
- Single stages can be run by hand: `.venv/bin/pharo-miner --help`.

## Other files

| File | Contents |
|---|---|
| `config.yaml` | All settings that are not in a script |
| `models.yaml` | The models that can be used: local (Ollama), on a server, or through an API |
| `taxonomy/` | The operation categories and their rules, the domains, the annotation guidelines |
| `benchmarks/README.md` | The file format for importing existing benchmark tasks (used by step 3) |

Tests: `.venv/bin/python -m unittest discover -s tests`
