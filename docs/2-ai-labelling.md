# Step 2: AI labelling (`./2-ai-labelling.sh`)

Language models label the sample of 300 methods drawn from the study corpus (the
repositories kept by [step 1-2](1-2-filtering.md)). Their labels are the
reference that the rule labels of step 1 are compared with.

```bash
./2-ai-labelling.sh      # choose the models at the top of the file first
```

Output: `results/validation/llm/` (log: `results/ai-labelling.log`). Next:
[step 3](3-results.md).

## Why this step exists

The rule labels of step 1 are automatic and may be wrong. The final statistics therefore wait
until a sample has been labelled independently. That reference can come from:

- **people**: fill `results/validation/reviewer_decisions.csv`, then run
  `.venv/bin/pharo-miner import-annotations`;
- **language models**: this step.

`validation.label_source` in `config.yaml` decides which one is used. With `auto`, human
labels are used when they exist, otherwise the AI labels. Reports always say which, and AI
labels are never presented as human-validated.

## What it does

1. Checks what the chosen models need: an API key, a reachable server, or a local Ollama
   model (downloaded if missing).
2. Sends each method to each model, with the same guidelines a human annotator gets.
3. Combines the answers of the models into one set of labels per method.
4. Measures how much the models agree.
5. Exports a small random subset for a blind human check.

## Settings

At the top of `2-ai-labelling.sh`:

| Setting | Meaning |
|---|---|
| `MODELS` | The annotators: names from `models.yaml`, or any Ollama tag |
| `ADJUDICATOR` | Optional model that settles disagreements. Preferably not one of `MODELS` |
| `CONSENSUS` | Without adjudicator: `majority` or `unanimous` |
| `EVIDENCE_POLICY` | `enforce`: a label is dropped unless the model quotes code from the method that shows it |
| `LIMIT` | Label only the first N methods. **The default in the file is 50; set `0` for all 300** |
| `CONCURRENCY` | Parallel requests per model. Empty = automatic |
| `SPOTCHECK_SIZE` | Methods exported for a blind human check |
| `PUSH_SNAPSHOT` | `yes`: upload the results to Hugging Face at the end |

**Big models on your own server.** `models.yaml` has `server-…` entries for Ollama or vLLM on
a GPU machine. Replace `SERVER-ADDRESS` with the server's address (or open an SSH tunnel and
use `localhost`), then put the entry's name in `MODELS`. The script still runs on your
computer; only the model runs on the server.

## How it works

- **Same task as humans.** Each model gets what a human annotator gets (repository, class,
  selector, protocol, source), the same guidelines (`taxonomy/annotation_guidelines.md`), and
  the definition of every category. The rule labels are hidden.
- **Constrained output.** Answers follow a JSON schema whose category field accepts only
  taxonomy ids. Labels must cite code from the method. With `enforce`, uncited or invented
  labels are rejected and counted per model.
- **Independent annotators.** Every model labels every method without seeing the others.
  Agreement uses the same measures as for humans: Cohen's κ, Krippendorff's α, Jaccard.
- **Aggregation.** Majority or unanimous vote, or an adjudicator model that sees the
  anonymised, shuffled answers of the models that disagree.
- **Repeatable.** Temperature 0, and a fixed seed where the provider allows it. Every raw
  answer is kept, with the model that served it, token usage, latency and the prompt hash.
- **Cached.** Answers are stored per model, prompt and method. An interrupted run resumes, and
  repeating a finished run costs nothing. Changing the prompt or the taxonomy relabels.
- **Human calibration.** `human_spotcheck.csv` holds a random subset without the AI labels.
  When a person fills its `operations` column, step 3 reports how well the AI labels agree
  with that person. If two people fill it (`human_1`, `human_2`), agreement between them is
  reported too, as the baseline.

## What it writes

In `results/validation/llm/`:

| File | Contents |
|---|---|
| `llm_adjudicated_labels.csv` | Final labels per method, with status (agreed, majority, adjudicated, no_consensus) and votes |
| `llm_decisions.csv` | Every model's answer, in the same format as human decisions |
| `agreement_report.json` | Agreement between the models |
| `run_manifest.json` | Models, served model ids, settings, prompt hash, tokens, rejection rates |
| `prompt_system.md`, `prompt_user_example.md`, `output_schema.json` | The exact prompt and schema, for a paper's appendix |
| `raw/*.jsonl` | Every raw response (also the resume cache) |
| `failures.csv` | Methods a model could not label, with the reason |
| `human_spotcheck.csv`, `spotcheck_report.json` | The blind human check and its results |

## Recommendations

- Run `LIMIT=10` first, to see a model's answers and its cost.
- Use two or three strong models from different families, or two plus an adjudicator. With
  only two models, `majority` means both must agree.
- Do not use small models. In a trial on this corpus, two local 7B models agreed on 1 of 8
  methods. Models fine-tuned only for code generation are not reliable annotators either.
- Fill the spot-check (30 methods is about an hour) and report its numbers next to the AI
  labels.

## Labelling more than the sample

This step labels the 300-method sample, not the whole study corpus, which has about 280,000
methods. On a CPU node a big model needs about 16 seconds per method, so labelling
everything is not feasible there; it needs GPUs. A larger sample is the practical option:
`.venv/bin/pharo-miner export-annotations --size 3000`, then this step with `LIMIT=0`.

## If something goes wrong

- **"The study corpus is not filtered yet"** or **"No annotation sample yet".** Run
  `./1-2-filtering.sh` first.
- **"Ollama is not running".** Open the Ollama app or run `ollama serve`, then rerun.
- **"Cannot reach the model server".** Check the address in `models.yaml`, or open the SSH
  tunnel.
- **It stopped in the middle.** Run it again. Finished answers are kept.
