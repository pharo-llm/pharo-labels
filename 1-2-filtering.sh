#!/usr/bin/env bash
# STEP 1-2 — Filtering: keep the repositories worth labelling and benchmarking (150 at most).
#
# Mining keeps every repository that is real Pharo code (about 1,600). Most are small, inactive
# or personal. This step measures each mined repository and keeps the ones that are engineered
# projects whose code can serve a benchmark for writing and fixing Pharo code: licensed,
# developed by several people over time, loadable in a current Pharo, tested, original.
# These repositories are the "study corpus". Steps 2, 3 and 4 use nothing else:
#   2-ai-labelling.sh   the 300 methods to label are drawn from them
#   3-results.sh        the taxonomy and the statistics describe them
#   4-benchmark.sh      the seed methods of the benchmark tasks come from them
# Every criterion is a separate threshold taken from the literature: docs/1-2-filtering.md
#
#   ./1-2-filtering.sh
# Takes about two minutes, needs no token and no model, and removes nothing from the database.
# Rerun it after changing the thresholds: when the corpus changes, the previous annotation
# sample and its labels are moved aside (not deleted) and a new sample is drawn.
# Output: results/reports/corpus_filter.md, results/tables/corpus_filter.csv
#         (log: results/filtering.log). Next: ./2-ai-labelling.sh

# ============================== SETTINGS ===================================================
# Most repositories the study corpus may hold. When more pass every criterion, they are chosen
# for diversity of domain, size and kind (never for stars). Empty = corpus_filter.max_repositories
# in config.yaml (150).
MAX_REPOSITORIES="${MAX_REPOSITORIES:-}"

# The thresholds themselves (licence, contributors, commits, tests, duplication, ...) are in
# config.yaml under corpus_filter. Each setting keeps the value above unless a variable of the
# same name is already set in the environment.
# ===========================================================================================

source "$(dirname "$0")/scripts/_common.sh"
LOG=results/filtering.log

if [ ! -s results/data/processed/pharo_task_miner.sqlite ]; then
  echo "No mined data yet: run ./1-mining.sh first."; exit 1
fi

args=()
[ -n "$MAX_REPOSITORIES" ] && args+=(--max-repositories "$MAX_REPOSITORIES")

echo "== Filtering $(date '+%Y-%m-%d %H:%M')" >> "$LOG"
run_logged "$LOG" "$MINER" filter-corpus ${args[@]+"${args[@]}"} || { echo "Stopped: see the message above (log: $LOG)"; exit 1; }
# Draws the 300-method sample from the study corpus (an existing sample of the same corpus is kept)
run_logged "$LOG" "$MINER" export-annotations || { echo "Stopped: see the message above (log: $LOG)"; exit 1; }
echo
echo "Study corpus: results/reports/corpus_filter.md   (every repository and why: results/tables/corpus_filter.csv)"
echo "The 300-method sample is in results/validation/annotation_sample.csv."
echo "Next: ./2-ai-labelling.sh   (or have people fill results/validation/reviewer_decisions.csv)"
