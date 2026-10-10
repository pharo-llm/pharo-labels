# All steps as one cluster job (`job.oar.sh`)

Runs [step 1](1-mining.md), [step 1-2](1-2-filtering.md), [step 2](2-ai-labelling.md),
[step 3](3-results.md) and [step 4](4-benchmark.md) in one OAR job on one Grid'5000 node,
starting from an empty `results/` folder. It is written for `chirop`: 64 cores, 512 GiB of memory, no GPU,
and no Ollama installed.

```bash
oarsub -S ./job.oar.sh          # submit, from the repository folder
bash job.oar.sh                 # or run it directly on a reserved node
LIMIT=10 BENCHMARK_LIMIT=5 bash job.oar.sh    # a short trial first
RUN_BENCHMARK=no oarsub -S ./job.oar.sh       # steps 1 to 3 only
```

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

At the top of `job.oar.sh`. A variable of the same name set when submitting wins.

| Setting | Meaning |
|---|---|
| `MODELS`, `ADJUDICATOR`, `CONSENSUS`, `EVIDENCE_POLICY`, `SPOTCHECK_SIZE` | As in step 2 |
| `LIMIT` | `0` = all 300 methods (the default here) |
| `OLLAMA_PARALLEL` | Requests Ollama answers at once. Not lower than the `concurrency` of the models in `models.yaml` (4) |
| `BENCHMARKS` | As in step 3 |
| `RUN_BENCHMARK` | `yes` (default) runs step 4; `no` stops after step 3 |
| `AUTHOR`, `SOLVERS`, `OPERATIONS`, `TASKS_PER_OPERATION` | As in step 4 |
| `BENCHMARK_LIMIT` | `0` = all seed methods; a small number for a trial (`LIMIT` of step 4) |
| `BENCHMARK_CONCURRENCY` | Tasks processed at once, each with its own copy of the image (`CONCURRENCY` of step 4) |
| `OLLAMA_MAX_LOADED` | Models kept in memory together (3). Step 4 alternates between author and solver |
| `PUSH_SNAPSHOT` | `auto`: upload only when `HF_TOKEN` is in `.env` |
| `SCRATCH_DIR`, `OLLAMA_MODELS_DIR` | Where Ollama and its models go. The default is the node's local disk, because home quotas are too small |

The node, the queue and the walltime (24 hours) are in the `#OAR` lines at the top.

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
