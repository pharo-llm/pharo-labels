#!/usr/bin/env bash
# STEP 2 — AI labelling: language models label the 300 sampled methods, in place of (or before)
# human annotators. Each model works independently on the same guidelines humans use; their
# labels are combined, agreement between models is measured, and a small random subset is
# exported for a human spot-check.
#
# Choose the models in the SETTINGS block below, then run:   ./2-ai-labelling.sh
# Safe to interrupt and rerun: answers are cached, nothing is paid for twice.
# Output: results/validation/llm/   (log: results/ai-labelling.log). Next: ./3-results.sh

# ============================== SETTINGS ===================================================
# Annotators: names from models.yaml, separated by spaces. Any installed Ollama tag also
# works (e.g. "qwen2.5-coder:14b"). For a paper, use 2-3 DIFFERENT model families, e.g.
#   MODELS="claude-opus-5-5 openrouter-qwen-coder-32b openrouter-llama-3.3-70b"
# Bigger models on your own GPU server: entries starting with "server-" in models.yaml
# (put the server's address there once). List all models with:  .venv/bin/pharo-miner models
MODELS="${MODELS:-qwen2.5-coder-7b}"

# Optional model that settles disagreements between the annotators. Leave empty to use a
# vote instead (CONSENSUS). Best practice: a strong model that is NOT in MODELS.
ADJUDICATOR="${ADJUDICATOR:-}"

# How labels are combined when there is no adjudicator:
#   majority  = keep a category when more than half of the models chose it
#   unanimous = keep a category only when every model chose it
CONSENSUS="${CONSENSUS:-majority}"

# enforce = drop a label if the model cannot quote code from the method that shows it
#           (protects against invented labels; recommended)
# record  = keep every label
EVIDENCE_POLICY="${EVIDENCE_POLICY:-enforce}"

# 0 = label all 300 methods. Use e.g. 10 for a quick, cheap trial of a model first.
LIMIT="${LIMIT:-50}"

# Parallel requests per model. Empty = automatic (1 for local models, 4 for APIs).
CONCURRENCY="${CONCURRENCY:-}"

# Methods exported for a blind human check of the AI labels (0 = none).
SPOTCHECK_SIZE="${SPOTCHECK_SIZE:-30}"

# yes = when labelling succeeds, upload all results (including every AI answer) to Hugging Face
PUSH_SNAPSHOT="${PUSH_SNAPSHOT:-yes}"
# Each setting keeps the value above unless a variable of the same name is already set in the
# environment (this is how job.oar.sh configures a cluster run).
# ===========================================================================================

source "$(dirname "$0")/scripts/_common.sh"
LOG=results/ai-labelling.log

if [ ! -s results/validation/annotation_sample.csv ]; then
  echo "No annotation sample yet: run ./1-mining.sh first."; exit 1
fi

# What the chosen models need: API keys and/or local Ollama models
reqs=$("$MINER" llm-annotate --print-requirements --models "$MODELS" --adjudicator "${ADJUDICATOR:-none}") || exit $?
req_lines=()
while IFS= read -r line; do [ -n "$line" ] && req_lines+=("$line"); done <<< "$reqs"
for line in ${req_lines[@]+"${req_lines[@]}"}; do
  kind=${line%% *}; value=${line#* }
  case "$kind" in
    env) ask_secret "$value" "API key for the selected model(s)" ;;
    server)
      url=${value%% *}; model=${value#* }
      if ! curl -sf --max-time 10 "$url" >/dev/null; then
        echo "Cannot reach the model server at ${url%/api/tags} (model $model)."
        echo "Check that the server is running and the address in models.yaml is right, or open an SSH tunnel."
        exit 1
      fi ;;
    ollama)
      command -v ollama >/dev/null || { echo "Ollama is not installed (https://ollama.com), but $value needs it."; exit 1; }
      if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
        echo "Ollama is not running. Open the Ollama app or run 'ollama serve' in another terminal, then rerun."; exit 1
      fi
      if ! ollama list | awk 'NR>1 {print $1}' | grep -qx -e "$value" -e "$value:latest"; then
        echo "Downloading local model $value ..."
        ollama pull "$value"
      fi ;;
  esac
done

args=(--models "$MODELS" --consensus "$CONSENSUS" --evidence-policy "$EVIDENCE_POLICY"
      --limit "$LIMIT" --spotcheck-size "$SPOTCHECK_SIZE" --adjudicator "${ADJUDICATOR:-none}")
[ -n "$CONCURRENCY" ] && args+=(--concurrency "$CONCURRENCY")

echo "== AI labelling $(date '+%Y-%m-%d %H:%M') — models: $MODELS${ADJUDICATOR:+, adjudicator: $ADJUDICATOR}" >> "$LOG"
status=0
run_logged "$LOG" "$MINER" llm-annotate "${args[@]}" || status=$?
echo
if [ "$status" -eq 0 ]; then
  echo "AI labelling done. Files: results/validation/llm/"
  echo "  llm_adjudicated_labels.csv  final labels per method      run_manifest.json  models, settings, costs, agreement"
  echo "  llm_decisions.csv           each model's answers         failures.csv       items a model could not label"
  if [ "$SPOTCHECK_SIZE" != "0" ]; then
    echo "Recommended: fill the 'operations' column of results/validation/llm/human_spotcheck.csv"
    echo "  (ids from results/validation/annotation_guidelines.md; do not look at the AI labels first)."
  fi
  if [ "$PUSH_SNAPSHOT" = "yes" ]; then
    ask_secret HF_TOKEN "Hugging Face token with write access (https://huggingface.co/settings/tokens)"
    run_logged "$LOG" "$MINER" snapshot push --message "AI labelling: $MODELS" \
      || echo "Upload to Hugging Face failed (results are safe locally). Retry later with: ./snapshot.sh push"
  fi
  echo "Next: ./3-results.sh"
else
  echo "Failed (exit $status). Fix the message above and rerun; finished answers are kept. Log: $LOG"
fi
exit "$status"
