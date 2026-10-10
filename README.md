# PharoTaskMiner

Find out which programming operations real Pharo code uses, so that LLM
benchmarks for Pharo can be built from evidence instead of guesses.

```
GitHub repos → Pharo check → sample → clone → parse methods → detect operations
   → validation (human or LLM panel) → taxonomy → statistics → benchmark coverage → reports
```

Every label keeps its evidence: repository, commit, file, class, method, line,
and the selector or AST node that triggered it.

## Run: three steps

```bash
./1-mining.sh        # GitHub → Pharo repos → sample → parse methods → rule labels → 300-method annotation sample
./2-ai-labelling.sh  # language models label the 300 methods (choose the models at the top of the file)
./3-results.sh       # taxonomy, statistics, benchmark coverage, reports
```

Each step resumes where it stopped when rerun. **Everything goes to `results/`.**
The first run asks for a GitHub token (and, in step 2, any API key the chosen
models need) and keeps it in `.env`, which is git-ignored. `./run.sh` still
works and runs step 1.

| Path | Contents |
|---|---|
| `results/reports/` | Markdown reports (start with `ecosystem_summary.md`) |
| `results/figures/` | PNG figures |
| `results/tables/` | CSV tables |
| `results/validation/` | Annotation sample, human decisions, agreement, taxonomy |
| `results/validation/llm/` | AI labelling: every model's answers, final labels, agreement, settings, spot-check |
| `results/data/` | GitHub cache, clones, parsed methods, SQLite database |
| `results/*.log` | Console output of every run |

Individual stages: `.venv/bin/pharo-miner --help`.

Step 3 also generates **`results/figures/taxonomy.png`**, showing the category
hierarchy with repository prevalence bars. After running `./3-results.sh`,
open the figure on macOS with:

```bash
open results/figures/taxonomy.png
```

### Snapshots on Hugging Face (no need to mine twice)

All results are saved to the Hugging Face dataset set in `config.yaml`
(`snapshot.repo`, default [`pharo-llm/pharo-labels`](https://huggingface.co/datasets/pharo-llm/pharo-labels)):

- `./1-mining.sh` uses the results on this computer if they exist, otherwise
  **downloads the latest snapshot** (minutes) instead of mining GitHub (hours),
  and mines only when there is no snapshot. Set `USE_SNAPSHOT="no"` in the
  script to always mine, or `SNAPSHOT_VERSION` to use an older version.
- Each step uploads a new version when it finishes (`PUSH_SNAPSHOT="yes"` at the
  top of each script). Every version is tagged `snapshot-YYYYMMDD-HHMMSS`.
- By hand: `./snapshot.sh push`, `./snapshot.sh pull`, `./snapshot.sh info`
  (versions), `./snapshot.sh push --dry-run` (what would be uploaded).

A snapshot holds the evidence database, parsed methods, discovery data, raw
GitHub responses, the annotation sample, human and AI annotations (every model
answer), tables, reports and logs, with a sha256 for every file in
`snapshot/manifest.json`. A pull verifies every checksum before installing
anything and never deletes local results: with `--force`, the old `results/`
is renamed to `results.before-pull-<date>`. **E-mail addresses in raw GitHub
responses are replaced by `<redacted>`** before upload, and cloned repositories
(third-party code) are left out unless `snapshot.include_clones: true`. Uploading
needs a Hugging Face token with write access (`HF_TOKEN`, asked once, kept in
`.env`); downloading a public dataset needs none.

### Why step 2 exists

The rule-based labels of step 1 are automatic and may be wrong, so the final
statistics wait until a sample of 300 methods has been labelled independently.
That reference can come from **people** (fill
`results/validation/reviewer_decisions.csv`, then `pharo-miner import-annotations`)
or from **language models** (step 2). `validation.label_source` in
`config.yaml` decides which one is used (`auto`: human labels when imported,
otherwise the AI labels). Reports always state the basis, and AI labels are
never presented as human-validated.

## Step 2 in detail: AI labelling

Settings are at the top of `2-ai-labelling.sh`:

| Setting | Meaning |
|---|---|
| `MODELS` | The annotators: names from `models.yaml` (local Ollama models, Claude, OpenAI, OpenRouter, any OpenAI-compatible server) or an Ollama tag |
| `ADJUDICATOR` | Optional model that settles disagreements (preferably not one of `MODELS`) |
| `CONSENSUS` | Without adjudicator: `majority` or `unanimous` |
| `EVIDENCE_POLICY` | `enforce`: a label is dropped unless the model quotes code from the method that shows it |
| `LIMIT` | Label only the first N methods: a cheap trial before a full run |
| `SPOTCHECK_SIZE` | Methods exported for a blind human check |

**Big models on your own server.** `models.yaml` has `server-…` entries for Ollama
or vLLM running on a GPU machine. Replace `SERVER-ADDRESS` with the server's
address (or open an SSH tunnel and use `localhost`), then put the entry's name
in `MODELS`. Step 2 runs on your laptop; only the model runs on the server, and
all results are still written to `results/validation/llm/` locally.

How it works, in the terms a methods section needs:

- **Same task as humans.** Each model receives the same information as a human
  annotator (repository, class, selector, protocol, source) and the same
  guidelines (`taxonomy/annotation_guidelines.md`), plus the definition,
  include and exclude notes of every category. Rule-based labels are hidden.
- **Constrained output.** Answers follow a JSON schema whose category field
  only accepts taxonomy ids. Labels must cite code from the method; with
  `enforce`, uncited or invented labels are rejected and counted per model.
- **Independent annotators, measured agreement.** Every model labels every
  method without seeing the others. Agreement between models uses the same
  measures as for humans (Cohen's κ, Krippendorff's α, Jaccard).
- **Aggregation.** Majority or unanimous vote, or an adjudicator model that
  sees the anonymised, shuffled answers of the disagreeing models.
- **Deterministic and auditable.** Temperature 0 and a fixed seed where the
  provider allows it. Every raw answer, the model that actually served it,
  token usage, latency and the prompt hash are kept. Answers are cached per
  model, prompt and method, so an interrupted run resumes and a finished run
  costs nothing to repeat. Changing the prompt or the taxonomy relabels.
- **Human calibration.** `human_spotcheck.csv` holds a random subset without
  the AI labels. When a person fills its `operations` column, `3-results.sh`
  (or `pharo-miner llm-spotcheck`) reports how well the AI labels agree with
  them: exact agreement, Jaccard with a bootstrap 95% CI, micro precision,
  recall and F1, and per-category κ. If two people fill it (`human_1`,
  `human_2`), human–human agreement is reported too, as the baseline.

Outputs in `results/validation/llm/`:

| File | Contents |
|---|---|
| `llm_adjudicated_labels.csv` | Final labels per method, with status (agreed, majority, adjudicated, no_consensus) and votes |
| `llm_decisions.csv` | Every model's answer, in the human `reviewer_decisions.csv` format |
| `agreement_report.json` | Agreement between the models |
| `run_manifest.json` | Models, served model ids, settings, prompt hash, tokens, rejection rates, agreement |
| `prompt_system.md`, `prompt_user_example.md`, `output_schema.json` | The exact prompt and schema (for the paper's appendix) |
| `raw/*.jsonl` | Every raw response (also the resume cache) |
| `failures.csv` | Methods a model could not label, with the reason |
| `human_spotcheck.csv`, `spotcheck_report.json` | Blind human check and its results |

Recommendations:

- Run `LIMIT=10` first to check a model's answers and cost.
- Use **2–3 strong models from different families**, or two plus an adjudicator.
  With only two models, `majority` means both must agree. In a trial on this
  corpus, two local 7B models (`qwen2.5-coder-7b`, `pharo-qwen-sft-7b`) agreed
  on 1 of 8 methods. Small models, and especially models fine-tuned only for
  code generation, are not reliable annotators.
- Fill the spot-check (30 methods is about an hour) and report its numbers next
  to the AI labels.

## All three steps as one cluster job (OAR)

`job.oar.sh` runs steps 1, 2 and 3 on one cluster node that has no GPU and no
Ollama: it installs Ollama on the node's local disk (no sudo), starts it, pulls
the models, labels all 300 methods and builds the reports.

```bash
echo 'export HF_TOKEN=...' >> .env    # optional: upload the results (a batch job cannot ask for it)
oarsub -S ./job.oar.sh
LIMIT=10 bash job.oar.sh              # a short trial, directly on a reserved node
```

The annotators are the `cluster-…` entries of `models.yaml`: `qwen2.5-coder:32b`,
`llama3.3:70b` and `gemma3:27b`, three families combined by majority vote. They
need about 80 GB of disk and run on CPU, so the job takes hours; if the walltime
runs out, submit it again and it continues from the cached answers. Settings are
at the top of the file and can be overridden by variables of the same name.

## Optional: local LLM

The pipeline is rule-based by default. Adding `--semantic` makes a **local
open-weight model** describe a capped sample of methods and repositories. No
API key is needed and no code leaves your machine.

```bash
.venv/bin/pharo-miner models                      # the list in models.yaml, and which are installed
ollama pull qwen2.5-coder:7b                      # the default (~4.7 GB)
./1-mining.sh --semantic --semantic-limit 50      # default model
./1-mining.sh --semantic --model qwen3-8b         # a model from the list
./1-mining.sh --semantic --model mistral:7b       # any Ollama tag
```

`models.yaml` lists the available models: Qwen2.5-Coder 3B/7B/14B, Qwen3-8B,
DeepSeek-Coder-V2, Llama 3.1, Gemma 3, Pharo fine-tuned models, and llama.cpp
or LM Studio servers. Add your own entries there. LLM labels are only kept if
they use a taxonomy id **and** cite evidence found in the method.

## Where things are

| Path | Contents |
|---|---|
| `config.yaml` | All settings |
| `taxonomy/` | Operation categories and rules, domains, framework mappings, annotation guidelines (editable YAML) |
| `src/pharo_task_miner/` | The code: discovery, extraction, analysis, validation, statistics, benchmarks, reporting |
| `pharo/ASTExporter/` | Optional parser backend that uses Pharo's own `RBParser` (untested here) |
| `benchmarks/` | Format for importing existing benchmark tasks |
| `1-mining.sh`, `2-ai-labelling.sh`, `3-results.sh` | The three steps (settings at the top of each file) |
| `job.oar.sh` | The three steps as one OAR cluster job (installs and runs Ollama on the node) |
| `snapshot.sh` | Save/restore all results on Hugging Face |
| `models.yaml` | Models: local (Ollama, llama.cpp, LM Studio) and APIs (Claude, OpenAI, OpenRouter) |
| `results/` | Everything a run produces (git-ignored) |

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```
