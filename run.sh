#!/usr/bin/env bash
# Pharo benchmark research — stage 1: from the 1,585 Pharo repositories and the existing code benchmarks to TWO
# final taxonomies of programming behaviours (Pharo methods, benchmark tasks). Every phase writes results/NN_<phase>/.
#
#   ./run.sh                    setup, then phases 1-6; a phase already done with the same inputs is skipped,
#                               an interrupted phase resumes item by item (LLM answers are cached)
#   ./run.sh 5                  one phase (re-run)          ./run.sh 3-6      a range
#   ./run.sh from 4             phases 4 to 6               ./run.sh discovery    by name
#   ./run.sh setup              only the dependencies (done automatically before any phase; SKIP_SETUP=1 skips)
#   ./run.sh status             what is done, provisional, waiting, failed or stale
#   ./run.sh logs [5] [-n 100]  the logs (all phases, or one), readable
#   ./run.sh cost               LLM calls, tokens, time and cost per phase and model
#   ./run.sh reset 5            delete what phase 5 produced (expert files kept)
#   PROFILE=smoke ./run.sh      a few items only, in results-smoke/ (checks every phase in minutes)
#
#    1 repositories  repos.txt (1,585) -> hard filters + weighted quality score (11 criteria) -> selected_repos.txt
#    2 literature    12 existing benchmarks -> 6,433 tasks (4,372 distinct problems)
#    3 corpus        methods of the selected repositories -> method filters decided on the AST
#    4 sampling      99% confidence start samples (Cochran) + ordered pools, representativeness check
#    5 discovery     3 annotators, alone, per source: category > subcategory > behaviour until saturation
#                    -> 6 taxonomies (05_discovery/<annotator>/<pharo|literature>/)
#    6 taxonomy      per source, the 3 taxonomies merged (embeddings + editor) -> the 2 final taxonomies
#   (7-13: stage 2 — validation on new samples, profile, coverage, benchmark, evaluation — not yet adapted)
#
# Results: results/SUMMARY.md (all phases, key numbers, figures). Expert files: results/06_taxonomy/review/,
# results/08_validation/expert/, results/12_verification/expert/ (completed files go in their returned/ folder).

# ============================== SETTINGS ===================================================
PROFILE="${PROFILE:-pilot}"        # pilot = first deliverable (small samples) | full (config.yaml: profiles)
# Models, sample sizes, thresholds: config.yaml
# ===========================================================================================

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if [ -f .env ]; then . ./.env; fi          # tokens (HF_TOKEN, GITHUB_TOKEN), never committed
export PROFILE

if [ ! -x .venv/bin/python ] || [ requirements.txt -nt .venv/.installed ]; then
  py=${PYTHON:-}
  if [ -z "$py" ]; then
    for c in python3.13 python3.12 python3.11 python3; do command -v "$c" >/dev/null && { py=$c; break; }; done
  fi
  "$py" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "Python >= 3.10 is required."; exit 1; }
  [ -x .venv/bin/python ] || "$py" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
  touch .venv/.installed
fi

needs_setup() {   # does this command run phases? (setup first: every dependency before any phase)
  case "${1:-run}" in status|logs|cost|reset|help|-h|--help|simulate-expert) return 1 ;; esac
  return 0
}
if needs_setup "${1:-}"; then
  mkdir -p data
  command -v ollama >/dev/null || { echo "Ollama is needed: https://ollama.com/download"; exit 1; }
  host="${OLLAMA_HOST:-localhost:11434}"
  if ! curl -sf "http://${host#http://}/api/tags" >/dev/null; then
    echo "Starting Ollama (log: data/ollama.log) …"
    nohup ollama serve > data/ollama.log 2>&1 &
    for _ in $(seq 30); do curl -sf "http://${host#http://}/api/tags" >/dev/null && break; sleep 1; done
  fi
  # models (Ollama library), Pharo image, benchmarks, corpus
  if [ "${SKIP_SETUP:-0}" != 1 ]; then
    .venv/bin/python -m pbr setup || { echo "Setup failed: nothing was run (log: results/logs/00_setup.log)."; exit 1; }
  fi
  [ "${1:-}" = setup ] && exit 0
fi

case "${1:-}" in
  ""|run)                         shift || true; exec .venv/bin/python -m pbr run "$@" ;;
  status|logs|cost|reset|simulate-expert) exec .venv/bin/python -m pbr "$@" ;;
  help|-h|--help)                 sed -n '2,27p' "$0" ;;
  *)                              exec .venv/bin/python -m pbr run "$@" ;;
esac
