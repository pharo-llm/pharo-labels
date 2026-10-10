#!/usr/bin/env bash
# STEP 3 — Results: human spot-check agreement (if filled), taxonomy, statistics, benchmark
# coverage (if benchmark files are given) and the Markdown reports with figures. All of them
# describe the study corpus: the repositories kept by ./1-2-filtering.sh.
#
#   ./3-results.sh
# Output: results/reports/, results/tables/, results/figures/   (log: results/results.log)
#
# Which reviewed labels are used is set by validation.label_source in config.yaml:
#   auto (default) = human annotations when imported, otherwise the AI labels; human | llm.

# ============================== SETTINGS ===================================================
# Benchmark task files to compare with the taxonomy (see benchmarks/README.md), separated by
# spaces, e.g. BENCHMARKS="benchmarks/humaneval_pharo.json". Empty = skip that step.
BENCHMARKS="${BENCHMARKS:-}"

# yes = upload all results (taxonomy, statistics, reports) to Hugging Face at the end
PUSH_SNAPSHOT="${PUSH_SNAPSHOT:-yes}"
# Each setting keeps the value above unless a variable of the same name is already set in the
# environment (this is how job.oar.sh configures a cluster run).
# ===========================================================================================

source "$(dirname "$0")/scripts/_common.sh"
LOG=results/results.log
if [ ! -s results/data/processed/corpus.json ]; then
  echo "The study corpus is not filtered yet: run ./1-2-filtering.sh first."; exit 1
fi
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
echo
echo "Taxonomy figure: results/figures/taxonomy.png"
echo "  Shows the category hierarchy with repository prevalence bars."
echo "  Open on macOS: open results/figures/taxonomy.png"
if [ "$PUSH_SNAPSHOT" = "yes" ]; then
  ask_secret HF_TOKEN "Hugging Face token with write access (https://huggingface.co/settings/tokens)"
  run_logged "$LOG" "$MINER" snapshot push --message "Results: taxonomy, statistics, reports" \
    || echo "Upload to Hugging Face failed (results are safe locally). Retry later with: ./snapshot.sh push"
fi
echo
echo "Done. Start with results/reports/ecosystem_summary.md"
