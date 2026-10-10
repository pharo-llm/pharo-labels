#!/usr/bin/env bash
# STEP 1-2 — Filtering: keep the repositories worth labelling and benchmarking (150 at most).
#
# Mining keeps every repository that is real Pharo code (about 1,600). Most are small, inactive
# or personal. This step measures each mined repository and keeps the ones that are engineered
# projects whose code can serve a benchmark for writing and fixing Pharo code: licensed,
# developed by several people over time, loadable in a current Pharo, tested, original.
# These repositories are the "study corpus". Steps 2, 3 and 4 use nothing else:
#   2-ai-labelling.sh   the methods to label come from them (a sample, or whole repositories)
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

# What step 2 labels. Two ways:
#   LABEL_REPOSITORIES=10   every method of 10 repositories of the study corpus. The
#                           repositories are chosen for diversity of domain, size and kind.
#                           A repository has about 1,300 methods (some more than 10,000), so
#                           this is thousands of methods per repository: plan the model time.
#   LABEL_REPOSITORIES=0    a sample of SAMPLE_SIZE methods taken across the whole study corpus.
# Empty = validation.label_repositories / annotation_sample_size in config.yaml (0 / 300).
# When the methods to label change, the previous ones are moved aside with their labels
# (not deleted).
LABEL_REPOSITORIES="${LABEL_REPOSITORIES:-}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"

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
# Writes the methods step 2 will label (when they are the same as before, nothing changes)
size_args=()
[ -n "$SAMPLE_SIZE" ] && size_args+=(--size "$SAMPLE_SIZE")
[ -n "$LABEL_REPOSITORIES" ] && size_args+=(--label-repositories "$LABEL_REPOSITORIES")
run_logged "$LOG" "$MINER" export-annotations ${size_args[@]+"${size_args[@]}"} || { echo "Stopped: see the message above (log: $LOG)"; exit 1; }
echo
echo "Study corpus: results/reports/corpus_filter.md   (every repository and why: results/tables/corpus_filter.csv)"
echo "The methods to label are in results/validation/annotation_sample.csv."
echo "Next: ./2-ai-labelling.sh   (or have people fill results/validation/reviewer_decisions.csv)"
