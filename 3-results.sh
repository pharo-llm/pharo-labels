#!/usr/bin/env bash
# STEP 3 — Results: human spot-check agreement (if filled), taxonomy, statistics, benchmark
# coverage (if benchmark files are given) and the Markdown reports with figures.
#
#   ./3-results.sh
# Output: results/reports/, results/tables/, results/figures/   (log: results/results.log)
#
# Which reviewed labels are used is set by validation.label_source in config.yaml:
#   auto (default) = human annotations when imported, otherwise the AI labels; human | llm.

# ============================== SETTINGS ===================================================
# Benchmark task files to compare with the taxonomy (see benchmarks/README.md), separated by
# spaces, e.g. BENCHMARKS="benchmarks/humaneval_pharo.json". Empty = skip that step.
BENCHMARKS=""

# yes = upload all results (taxonomy, statistics, reports) to Hugging Face at the end
PUSH_SNAPSHOT="yes"
# ===========================================================================================

source "$(dirname "$0")/scripts/_common.sh"
LOG=results/results.log
echo "== Results $(date '+%Y-%m-%d %H:%M')" >> "$LOG"

if [ -s results/validation/llm/human_spotcheck.csv ]; then
  run_logged "$LOG" "$MINER" llm-spotcheck || echo "Spot-check comparison failed (see above); continuing."
fi
run_logged "$LOG" "$MINER" build-taxonomy || { echo "Stopped: see the message above (log: $LOG)"; exit 1; }
run_logged "$LOG" "$MINER" statistics || { echo "Stopped: see the message above (log: $LOG)"; exit 1; }
if [ -n "$BENCHMARKS" ]; then
  inputs=()
  for f in $BENCHMARKS; do inputs+=(--input "$f"); done
  run_logged "$LOG" "$MINER" benchmark-coverage "${inputs[@]}" || { echo "Stopped: see the message above (log: $LOG)"; exit 1; }
else
  echo "Benchmark coverage skipped (set BENCHMARKS at the top of 3-results.sh)."
fi
run_logged "$LOG" "$MINER" report || { echo "Stopped: see the message above (log: $LOG)"; exit 1; }
if [ "$PUSH_SNAPSHOT" = "yes" ]; then
  ask_secret HF_TOKEN "Hugging Face token with write access (https://huggingface.co/settings/tokens)"
  run_logged "$LOG" "$MINER" snapshot push --message "Results: taxonomy, statistics, reports" \
    || echo "Upload to Hugging Face failed (results are safe locally). Retry later with: ./snapshot.sh push"
fi
echo
echo "Done. Start with results/reports/ecosystem_summary.md"
