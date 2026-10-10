# Step 3: results (`./3-results.sh`)

Turns the data of step 1 and the reviewed labels of step 2 into the taxonomy, the statistics,
the figures and the reports. All of them describe the study corpus: the repositories kept by
[step 1-2](1-2-filtering.md), not every mined repository.

```bash
./3-results.sh
```

Output: `results/reports/`, `results/tables/`, `results/figures/` (log:
`results/results.log`). Start reading with `results/reports/ecosystem_summary.md`.

## What it does

| Stage | What happens |
|---|---|
| `llm-spotcheck` | Only if `human_spotcheck.csv` was filled: compares the AI labels with the human answers |
| `build-taxonomy` | Builds the versioned taxonomy of operations, each category backed by evidence |
| `statistics` | Prevalence of each operation, association with domains, language features, uncertainty, sensitivity |
| `benchmark-coverage` | Only if `BENCHMARKS` is set: compares existing benchmark tasks with the taxonomy |
| `report` | Writes the figures, the tables and the Markdown reports |

## Which labels it uses

`validation.label_source` in `config.yaml`:

| Value | Labels used |
|---|---|
| `auto` (default) | Human annotations when they were imported, otherwise the AI labels of step 2 |
| `human` | Human annotations only |
| `llm` | AI labels only |

The reports state which one was used. Without any reviewed label the script stops, because
the statistics would rest on unchecked rule labels.

## Settings

At the top of `3-results.sh`:

| Setting | Meaning |
|---|---|
| `BENCHMARKS` | Benchmark task files to compare with the taxonomy, separated by spaces. Empty = skip. The format is in `benchmarks/README.md` |
| `PUSH_SNAPSHOT` | `yes`: upload all results to Hugging Face at the end |

## What it writes

| Path | Contents |
|---|---|
| `results/reports/ecosystem_summary.md` | The overview. Start here |
| `results/reports/taxonomy_report.md` | The taxonomy, with the evidence for each category |
| `results/reports/operation_analysis.md` | The operations: how common, in which domains |
| `results/reports/repository_selection.md` | How many repositories each quality criterion removed |
| `results/reports/benchmark_coverage.md` | Which operations existing benchmarks cover and miss (only with `BENCHMARKS`) |
| `results/figures/taxonomy.png` | The category hierarchy, with a bar for how many repositories use each |
| `results/figures/*.png` | Operation prevalence, domain distribution, domain and operation heatmaps, and others |
| `results/tables/*.csv` | The numbers behind the figures |

On macOS: `open results/figures/taxonomy.png`

## Link with step 4

The benchmark coverage analysis recommends the operations that existing benchmarks miss.
[Step 4](4-benchmark.md) writes tasks for exactly those when its `OPERATIONS` setting is
`auto`. When the analysis was not run, step 4 covers every level-3 operation of the taxonomy.

## If something goes wrong

- **"Stopped: see the message above".** Usually no reviewed labels exist yet: run
  `./2-ai-labelling.sh`, or import human annotations.
- **"Spot-check comparison failed".** Only that comparison is skipped; the rest continues.
- **"Upload to Hugging Face failed".** The results are safe locally. Retry with
  `./snapshot.sh push`.
