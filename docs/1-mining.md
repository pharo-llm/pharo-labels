# Step 1: mining (`./1-mining.sh`)

Finds Pharo repositories on GitHub, keeps the ones worth analysing, parses every method in
them and labels the programming operations each method performs.

```bash
./1-mining.sh
```

Output: `results/` (log: `results/run.log`). Next: [step 1-2](1-2-filtering.md).

## What it does

| Stage | What happens |
|---|---|
| `discover` | Searches GitHub by topic, language, organisation and Pharo indicators (about 20,000 candidates) |
| `verify` | Decides for each one whether it really is Pharo, with the evidence |
| `metadata` | Collects commits, contributors, languages, licence and the revision |
| `sample` | Applies the quality filter below and selects the repositories to analyse |
| `clone` | Shallow-clones the selected repositories |
| `parse` | Reads the Tonel and FileTree sources and extracts every method |
| `analyze` | Resolves receivers, finds API and framework use, dependencies and duplicated code |
| `classify-domains` | Labels each repository with its application domains |
| `discover-operations` | Labels each method with operations, using the rules in `taxonomy/` |
| `export-annotations` | Draws a first sample of 300 methods; [step 1-2](1-2-filtering.md) draws it again from the study corpus |

Every label keeps its evidence: repository, commit, file, class, method, line, and the
selector or AST node that triggered it.

## It does not mine twice

Mining takes hours, so the script first looks for results it can reuse:

1. Results already on this computer: they are used as they are.
2. Otherwise, a snapshot on Hugging Face: it is downloaded (minutes). See
   [snapshots.md](snapshots.md).
3. Otherwise: mining from GitHub, then the results are uploaded as a snapshot.

## Which repositories are analysed

The GitHub search is wide on purpose, so a filter decides which repositories are worth
analysing. The criteria are in `config.yaml` under `repositories.quality`. Each one is
separate; there is no combined score.

| Criterion | Removes |
|---|---|
| Has Tonel or FileTree source | Repositories that only mention Pharo in a README |
| Not coursework, a tutorial or a puzzle | Homework, katas, Advent of Code, classroom repositories |
| At least 10 commits | One-off uploads |
| At least 10 Pharo source files | Toys |
| Has tests | Code without any test class |
| 2 contributors, or a star, or a fork | Purely personal repositories |

Stars are not a threshold: half of the Pharo repositories have none. With
`sampling.strategy: all`, every repository that passes is analysed. On the current data that
is 1,585 of 19,894 candidates.

This filter only removes what is not worth parsing. Which of the remaining repositories are
labelled, analysed and used for the benchmark is decided by [step 1-2](1-2-filtering.md),
with stricter criteria that need the parsed code.

Every excluded repository is listed with the criterion it failed in
`results/tables/repository_selection.csv`. The counts per criterion are in
`results/reports/repository_selection.md`.

**Changing the criteria needs no manual step.** The script notices that the results were made
with other settings, selects again, and redoes only cloning, parsing and the sample. The
GitHub search is not repeated and no GitHub token is needed. Data of repositories that are no
longer selected is removed from the database. The previous annotation sample and its labels
are moved to `results/validation.before-reselection-<date>`, not deleted.

## Settings

At the top of `1-mining.sh`:

| Setting | Meaning |
|---|---|
| `USE_SNAPSHOT` | `auto`: local results, else the snapshot, else mine. `no`: always mine |
| `SNAPSHOT_VERSION` | Snapshot to download. Empty = latest, or a tag such as `snapshot-20261009-153000` |
| `PUSH_SNAPSHOT` | `yes`: upload the results to Hugging Face when mining finishes |

Mining options can be given on the command line, for example `./1-mining.sh --size 300`.
They apply only when mining really runs.

## What it needs

- A GitHub token (`GITHUB_TOKEN`), only when it really mines. Read-only public access is
  enough.
- A Hugging Face token with write access (`HF_TOKEN`), only to upload.
- Disk space: the database is about 14 GB for 1,585 repositories, plus the clones.

## What it writes

| Path | Contents |
|---|---|
| `results/data/processed/pharo_task_miner.sqlite` | The database: repositories, methods, labels, evidence |
| `results/data/repositories/` | The clones |
| `results/data/raw/api_cache/` | Raw GitHub responses, kept for auditing and resuming |
| `results/validation/annotation_sample.csv` | A first sample of 300 methods, replaced by step 1-2 |
| `results/validation/annotation_guidelines.md` | The guidelines for whoever labels them |
| `results/tables/repository_selection.csv` | Each repository, selected or not, and why |

## Optional: descriptions by a local model

The labels are rule-based by default. With `--semantic`, a local open-weight model also
describes a capped sample of methods and repositories. No API key is needed and no code
leaves the machine.

```bash
.venv/bin/pharo-miner models                      # the models in models.yaml, and which are installed
./1-mining.sh --semantic --semantic-limit 50      # default model
./1-mining.sh --semantic --model qwen3-8b         # a model from the list, or any Ollama tag
```

A model's label is kept only if it uses a taxonomy id and cites evidence found in the method.

## If something goes wrong

- **It stopped in the middle.** Run it again. It resumes.
- **"Upload to Hugging Face failed".** The results are safe locally. Retry with
  `./snapshot.sh push`.
- **It reuses old results and you want to mine again.** Set `USE_SNAPSHOT="no"`, or move
  `results/` away.
