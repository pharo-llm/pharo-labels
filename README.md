# PharoTaskMiner

Find out which programming operations real Pharo code uses, so that LLM
benchmarks for Pharo can be built from evidence instead of guesses.

```
GitHub repos → Pharo check → sample → clone → parse methods → detect operations
   → study corpus (150 repositories at most) → validation (human or LLM panel) → taxonomy → statistics → benchmark coverage → reports
```

Every label keeps its evidence: repository, commit, file, class, method, line,
and the selector or AST node that triggered it.

## Run

```bash
./1-mining.sh        # GitHub → Pharo repos → quality filter → parse methods → rule labels
./1-2-filtering.sh   # keeps the repositories worth labelling and benchmarking (150 at most) → 300-method sample
./2-ai-labelling.sh  # language models label the 300 methods (choose the models at the top of the file)
./3-results.sh       # taxonomy, statistics, benchmark coverage, reports
./4-benchmark.sh     # write benchmark tasks and keep those a Pharo image proves correct
```

Each step resumes where it stopped when rerun. **Everything goes to `results/`.**
Settings are at the top of each script. The first run asks for the tokens it
needs and keeps them in `.env`, which is git-ignored.

## Documentation

Everything is explained in **[docs/](docs/README.md)**, one page per script:

| Page | Explains |
|---|---|
| [docs/1-mining.md](docs/1-mining.md) | `1-mining.sh`: discovery, the repository quality filter, parsing, rule labels |
| [docs/1-2-filtering.md](docs/1-2-filtering.md) | `1-2-filtering.sh`: the study corpus, its thirteen criteria and the papers behind them |
| [docs/2-ai-labelling.md](docs/2-ai-labelling.md) | `2-ai-labelling.sh`: the models, how their labels are combined and checked |
| [docs/3-results.md](docs/3-results.md) | `3-results.sh`: taxonomy, statistics, figures, reports |
| [docs/4-benchmark.md](docs/4-benchmark.md) | `4-benchmark.sh`: task generation, the ten checks, the papers behind them |
| [docs/cluster-job.md](docs/cluster-job.md) | `job.oar.sh`: all steps as one Grid'5000 job |
| [docs/snapshots.md](docs/snapshots.md) | `snapshot.sh`: saving and restoring results on Hugging Face |

## Where the results are

| Path | Contents |
|---|---|
| `results/reports/` | Markdown reports (start with `ecosystem_summary.md`) |
| `results/figures/` | PNG figures |
| `results/tables/` | CSV tables |
| `results/validation/` | Annotation sample, human decisions, agreement, taxonomy |
| `results/validation/llm/` | AI labelling: every model's answers, final labels, agreement, settings, spot-check |
| `results/benchmark/` | Generated benchmark: accepted and rejected tasks, the installable `PharoTaskBench` packages |
| `results/data/` | GitHub cache, clones, parsed methods, SQLite database |
| `results/*.log` | Console output of every run |

Individual stages: `.venv/bin/pharo-miner --help`.

## Where the code is

| Path | Contents |
|---|---|
| `config.yaml` | All settings |
| `taxonomy/` | Operation categories and rules, domains, framework mappings, annotation guidelines (editable YAML) |
| `src/pharo_task_miner/` | The code: discovery, extraction, analysis, validation, statistics, benchmarks, reporting |
| `pharo/ASTExporter/` | Optional parser backend that uses Pharo's own `RBParser` (untested here) |
| `benchmarks/` | Format for importing existing benchmark tasks |
| `docs/` | The documentation, one page per script |
| `1-mining.sh`, `1-2-filtering.sh`, `2-ai-labelling.sh`, `3-results.sh` | Steps 1 to 3 (settings at the top of each file) |
| `job.oar.sh` | All steps as one OAR cluster job (installs and runs Ollama on the node) |
| `4-benchmark.sh` | Step 4: generate and validate benchmark tasks (see `docs/4-benchmark.md`) |
| `pharo/BenchmarkHarness/` | The scripts that validate a task inside the Pharo image |
| `snapshot.sh` | Save/restore all results on Hugging Face |
| `models.yaml` | Models: local (Ollama, llama.cpp, LM Studio) and APIs (Claude, OpenAI, OpenRouter) |
| `results/` | Everything a run produces (git-ignored) |

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```
