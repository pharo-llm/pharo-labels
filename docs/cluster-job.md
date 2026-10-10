# All steps as one cluster job (`job.oar.sh`)

Runs [step 1](1-mining.md), [step 1-2](1-2-filtering.md), [step 2](2-ai-labelling.md),
[step 3](3-results.md) and [step 4](4-benchmark.md) in one OAR job on one Grid'5000 node,
starting from an empty `results/` folder. It is written for `chirop`: 64 cores, 512 GiB of memory, no GPU,
and no Ollama installed.

```bash
oarsub -S ./job.oar.sh          # submit, from the repository folder
bash job.oar.sh                 # or run it directly on a reserved node
```

Everything is set **inside the file**, in the `SETTINGS` block at the top: write the value
you want, then submit. For a short trial first, write `LIMIT=10` and `BENCHMARK_LIMIT=5`
there; for the real run, put both back to `0`.

```bash
nano job.oar.sh                 # edit the SETTINGS block
```

## On a GPU node: `./submit-gpu.sh`

The job is the same on a GPU node, and much faster: Ollama uses the GPUs by itself.
`submit-gpu.sh` shows which GPU nodes are free on every Grid'5000 site and asks where to
submit. Run it on any frontend, from the repository folder:

```bash
./submit-gpu.sh                  # show the free GPUs of every site, then ask where to submit
./submit-gpu.sh --list           # only show them
./submit-gpu.sh --yes            # submit on the strongest free one without asking
./submit-gpu.sh --cluster chuc   # submit on this cluster (the job waits if it is busy)
```

For each site it says how many GPU clusters are free, and for each free one: the GPUs and
their memory, how many nodes are free (and for how long, when a reservation is coming), the
processor and memory of the node, the queue and the longest job it allows. You answer with
the number of the one you want, or `n` for nothing.

The file is self-contained: it does not use anything else from this project, and can be
copied into another one. It submits the script named in `JOB_SCRIPT` (or given as argument).

Each site has its own home directory. When the cluster you choose is on another site, the
folder is copied there first (without `results/` and `.venv/`), and **the job runs and writes
its results on that site**. The script prints the commands to follow and stop the job.

| Setting (top of `submit-gpu.sh`) | Meaning |
|---|---|
| `JOB_SCRIPT` | The script to submit (`job.oar.sh`) |
| `WALLTIME_HOURS` | 72. A cluster that allows less gets its own maximum |
| `SITES` | Sites to look at. Empty = all of Grid'5000 |
| `MIN_GPU_GB` | Only show nodes with at least this much GPU memory in total. `0` = all |
| `DO_NOT_COPY` | What is not copied to another site |
| `START_TIMEOUT` | Seconds to wait to see whether the job started |

Things to know:

- The biggest model of the job (`llama3.3:70b`) needs 43 GB of GPU memory. On a node with
  less, it still runs, but partly on the processor and more slowly. The list shows the
  memory of each node so that you can choose.
- Old cards (Tesla P100 and V100) are listed after the others: they run an older driver
  environment. The job prints whether a GPU was detected.
- Clusters of the `default` queue are subject to the Grid'5000 usage policy, which limits
  long jobs during working hours. The script says so when you choose one and gives the
  command that checks it (`usagepolicycheck -t`).
- A home directory holds 25 GB by default on each site; this job needs more for `results/`.
  Ask for a quota extension on the site you use.

## Before submitting

A batch job has no terminal, so it cannot ask for a token. Put the secrets in `.env` first:

```bash
echo 'export HF_TOKEN=...' >> .env        # write access: without it, nothing is uploaded
echo 'export GITHUB_TOKEN=...' >> .env    # only if step 1 really has to mine GitHub
```

The node gets the mined data from the Hugging Face snapshot (see
[snapshots.md](snapshots.md)), so the latest results must have been uploaded from the machine
that mined them. The filtering is done again on the node, so it does not matter whether the
snapshot was uploaded before or after `1-2-filtering.sh`.

The node needs `curl` and `unzip`, and access to `get.pharo.org` and `github.com`: step 4
downloads Pharo and loads MuTalk into it. The job checks the tools before anything else.

## What it does

| Part | What happens |
|---|---|
| 1. Mining | Runs `1-mining.sh`. It comes first, so a missing token fails within seconds. Then `1-2-filtering.sh` keeps the study corpus |
| 2. Install Ollama | Installs Ollama on the node's local disk. No sudo, no system service |
| 3. Start Ollama | Starts the server on the default address and waits until it answers |
| 4. Pull the models | Downloads the models that `MODELS`, `AUTHOR` and `SOLVERS` need (about 80 GB) |
| 5. AI labelling | Runs `2-ai-labelling.sh`, the models one after the other |
| 6. Results | Runs `3-results.sh`. It needs no model; Ollama stays up for the next part |
| 7. Benchmark | Runs `4-benchmark.sh`: downloads Pharo, builds the validation image, generates and validates the tasks |
| 8. Stop Ollama | Frees the memory |
| 9. Generated files | Lists what was produced |

Parts 1 to 6 upload their results to Hugging Face when `HF_TOKEN` is set. The benchmark is
**not** uploaded: it stays in `results/benchmark/`.

## The models

The annotators are the `cluster-…` entries of `models.yaml`: three families, combined by
majority vote, so a kept label was found independently by at least two of them.

| Name | Model | Size |
|---|---|---|
| `cluster-qwen-coder-32b` | `qwen2.5-coder:32b` (Alibaba, code-specialised) | 20 GB |
| `cluster-llama-3.3-70b` | `llama3.3:70b` (Meta, the slowest of the three) | 43 GB |
| `cluster-gemma3-27b` | `gemma3:27b` (Google) | 17 GB |

Step 4 reuses two of them: `cluster-qwen-coder-32b` writes the tasks and
`cluster-gemma3-27b` must solve each one from its statement alone. They are from different
families, so that check is independent. Whether models of this size write enough valid Pharo
is not known yet: try `BENCHMARK_LIMIT=5` first and read the rejection reasons in
`results/reports/benchmark_generation.md`. A stronger author through an API also works: set
`AUTHOR` to its name in `models.yaml` and put its key in `.env`.

There is no adjudicator: it would ask one big model at a time, which is too slow on CPU, and
three annotators already give a majority.

## Settings

In the `SETTINGS` block at the top of `job.oar.sh`. The values written there are the ones
used; nothing is read from the command line.

| Setting | Meaning |
|---|---|
| `MAX_REPOSITORIES` | Most repositories that go on to steps 3 and 4 (150; the filter currently keeps 135) |
| `LABEL_REPOSITORIES` | How many of them the AI labels in step 2, every method of each (5). `0` = a sample of methods instead |
| `SAMPLE_SIZE` | Only when `LABEL_REPOSITORIES=0`: methods sampled across all repositories (300) |
| `MODELS`, `ADJUDICATOR`, `CONSENSUS`, `EVIDENCE_POLICY`, `SPOTCHECK_SIZE` | As in step 2 |
| `LIMIT` | `0` = label all the chosen methods; a small number for a trial |
| `OLLAMA_PARALLEL` | Requests Ollama answers at once. Not lower than the `concurrency` of the models in `models.yaml` (4) |
| `BENCHMARKS` | As in step 3 |
| `RUN_BENCHMARK` | `yes` runs step 4; `no` stops after step 3 |
| `AUTHOR`, `SOLVERS`, `OPERATIONS`, `TASKS_PER_OPERATION` | As in step 4 |
| `BENCHMARK_LIMIT` | How many candidate tasks are attempted. `0` = all (operations × `TASKS_PER_OPERATION`) |
| `BENCHMARK_CONCURRENCY` | Tasks processed at once, each with its own copy of the image (`CONCURRENCY` of step 4) |
| `OLLAMA_MAX_LOADED` | Models kept in memory together (3). Step 4 alternates between author and solver |
| `PUSH_SNAPSHOT` | `auto`: upload only when `HF_TOKEN` is in `.env` |
| `SCRATCH_DIR`, `OLLAMA_MODELS_DIR` | Where Ollama and its models go. The default is the node's local disk, because home quotas are too small |

The node, the queue and the walltime (24 hours) are in the `#OAR` lines at the top.

## How long the labelling takes

Labelling whole repositories is thousands of methods: about 20,000 for 5 repositories, 26,000
for 10, 70,000 for 20. Each of the three models reads every one of them, and on a CPU node a
big model needs up to 16 seconds per method. That is days, not hours, and more than one
24-hour job. Resubmit the job until step 2 is finished: answers are kept and it continues
where it stopped. To go faster: fewer repositories, fewer models in `MODELS`, or a node with
GPUs. The repositories chosen and their method counts are written to
`results/validation/labelled_repositories.csv` before any model is called.

## Where things go

| Path | Contents |
|---|---|
| `results/` | The same results as when running the steps by hand |
| `pharo-labels.<jobid>.out`, `.err` | The job's console output |
| `oar-runs/<jobid>/ollama.log` | The log of the Ollama server |

## If something goes wrong

- **Step 4 accepted few or no tasks.** The author model is too weak for Pharo. Use a stronger
  `AUTHOR`; the other results are not affected.
- **The walltime ran out.** Submit the job again. `results/` is kept, model answers are
  cached, and every step resumes.
- **A model failed to download.** Usually the disk is full; the job prints the free space.
  Point `OLLAMA_MODELS_DIR` at storage with enough room.
- **The local disk was emptied between jobs.** The models are downloaded again (minutes).
- **"… is required but there is no terminal to ask on".** Add the token to `.env`.
