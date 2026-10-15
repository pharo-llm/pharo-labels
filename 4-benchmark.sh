#!/usr/bin/env bash
# STEP 4 — Benchmark generation: write programming tasks for the operations found in steps 1-3
# and keep only the tasks that a clean Pharo image proves correct. No manual validation.
#
# For every operation, real methods from the study corpus (the repositories kept by
# ./1-2-filtering.sh) are used as seeds. A model
# writes a new self-contained task from each seed (a class, a reference solution, SUnit tests).
# The task is then installed in a pinned Pharo image and must pass every check: it compiles,
# the tests pass on the solution, fail without it, kill the mutants of the solution (MuTalk),
# the solution really performs the operation, and a second model solves the task from its
# statement alone. Failing tasks are sent back for repair a few times, then discarded.
# The method and the papers behind it: docs/4-benchmark.md
#
#   ./4-benchmark.sh
# Safe to interrupt and rerun: model answers and decided tasks are kept.
# Output: results/benchmark/   (log: results/benchmark.log)

# ============================== SETTINGS ===================================================
# The model that writes the tasks: a name from models.yaml or an Ollama tag. Use the strongest
# model you have: small models (7B) cannot write valid Pharo and all their tasks are rejected.
#   AUTHOR="claude-opus-5-5"        AUTHOR="cluster-qwen-coder-32b"
AUTHOR="${AUTHOR:-}"

# Models that must solve each task from its statement alone (separated by spaces). Use a
# model DIFFERENT from AUTHOR, so the check is independent. "none" skips the check: the
# statements are then not verified, and the report says so.
SOLVERS="${SOLVERS:-}"

# Operations to cover: "auto" = those the benchmark coverage analysis of step 3 recommends,
# or every level-3 operation of the taxonomy when that analysis was not run. Or a list of ids.
OPERATIONS="${OPERATIONS:-auto}"

# Seed methods, and so candidate tasks, per operation. Expect a part of them to be rejected.
TASKS_PER_OPERATION="${TASKS_PER_OPERATION:-3}"

# 0 = all seeds. Use e.g. 5 for a quick trial of a model first.
LIMIT="${LIMIT:-0}"

# Tasks processed in parallel (each one runs its own copy of the Pharo image).
CONCURRENCY="${CONCURRENCY:-2}"

# Each setting keeps the value above unless a variable of the same name is already set in the
# environment. The thresholds (mutation score, number of tests, repair rounds) are in
# config.yaml under benchmark_generation.
# ===========================================================================================

source "$(dirname "$0")/scripts/_common.sh"
LOG=results/benchmark.log

[ -n "$AUTHOR" ] || { echo "Set AUTHOR at the top of 4-benchmark.sh (the model that writes the tasks)."; exit 1; }
[ -n "$SOLVERS" ] || { echo "Set SOLVERS at the top of 4-benchmark.sh (a model other than AUTHOR, or none)."; exit 1; }
if [ ! -s results/data/processed/pharo_task_miner.sqlite ]; then
  echo "No mined data yet: run ./1-mining.sh first."; exit 1
fi
if [ ! -s results/data/processed/corpus.json ]; then
  echo "The study corpus is not filtered yet: run ./1-2-filtering.sh first."; exit 1
fi
for tool in curl unzip; do
  command -v "$tool" >/dev/null || { echo "$tool is required to download the Pharo image."; exit 1; }
done

# What the chosen models need: API keys and/or local Ollama models
reqs=$("$MINER" benchmark-generate --print-requirements --author "$AUTHOR" --solvers "$SOLVERS") || exit $?
req_lines=()
while IFS= read -r line; do [ -n "$line" ] && req_lines+=("$line"); done <<< "$reqs"
for line in ${req_lines[@]+"${req_lines[@]}"}; do
  kind=${line%%$'\t'*}; value=${line#*$'\t'}
  case "$kind" in
    env) ask_secret "$value" "API key for the selected model(s)" ;;
    server)
      url=${value%% *}; model=${value#* }
      if ! curl -sf --max-time 10 "$url" >/dev/null; then
        echo "Cannot reach the model server at ${url%/api/tags} (model $model)."; exit 1
      fi ;;
    ollama)
      command -v ollama >/dev/null || { echo "Ollama is not installed (https://ollama.com), but $value needs it."; exit 1; }
      if ! curl -sf "http://${OLLAMA_HOST:-localhost:11434}/api/tags" >/dev/null; then
        echo "Ollama is not running. Open the Ollama app or run 'ollama serve' in another terminal, then rerun."; exit 1
      fi
      if ! ollama list | awk 'NR>1 {print $1}' | grep -qx -e "$value" -e "$value:latest"; then
        echo "Downloading local model $value ..."
        ollama pull "$value"
      fi ;;
  esac
done

args=(--author "$AUTHOR" --solvers "$SOLVERS" --tasks-per-operation "$TASKS_PER_OPERATION"
      --limit "$LIMIT" --concurrency "$CONCURRENCY")
[ "$OPERATIONS" != "auto" ] && args+=(--operations "$OPERATIONS")

echo "== Benchmark generation $(date '+%Y-%m-%d %H:%M') — author: $AUTHOR, solvers: $SOLVERS" >> "$LOG"
status=0
run_logged "$LOG" "$MINER" benchmark-generate "${args[@]}" || status=$?
echo
if [ "$status" -eq 0 ]; then
  echo "Benchmark generation done. Files: results/benchmark/"
  echo "  tasks.jsonl       accepted tasks: statement, solution, tests, provenance, validation evidence"
  echo "  rejected.jsonl    discarded candidates and the check each one failed"
  echo "  PharoTaskBench/   the benchmark as Tonel packages with a Metacello baseline"
  echo "  manifest.json     image, models, thresholds, filtering funnel"
  echo "Report: results/reports/benchmark_generation.md"
  echo "Check it again at any time, without any model:  $MINER benchmark-validate"
else
  echo "Failed (exit $status). Fix the message above and rerun; finished tasks are kept. Log: $LOG"
fi
exit "$status"
