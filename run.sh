#!/usr/bin/env bash
# Pharo benchmark research pipeline — from existing benchmarks and 1,585 Pharo repositories to a validated,
# executable Pharo benchmark, in 12 resumable phases. Every phase writes into results/NN_<phase>/.
#
#   ./run.sh                    every phase in order; a phase already done with the same inputs is skipped,
#                               an interrupted phase resumes item by item (LLM answers are cached)
#   ./run.sh 6                  one phase (re-run)          ./run.sh 3-7      a range
#   ./run.sh from 6             phases 6 to 12              ./run.sh annotation   by name
#   ./run.sh setup              only the dependencies (done automatically before any phase; SKIP_SETUP=1 skips)
#   ./run.sh status             what is done, provisional, waiting, failed or stale
#   ./run.sh logs [6] [-n 100]  the logs (all phases, or one), readable
#   ./run.sh cost               LLM calls, tokens, time and cost per phase and model
#   ./run.sh reset 6            delete what phase 6 produced (human inputs kept)
#   ./run.sh simulate-expert 7  a SIMULATED expert workbook, only to test the import path
#
#    1 literature    existing benchmarks -> literature_tasks.parquet, literature_benchmarks.csv
#    2 corpus        Pharo methods (HF snapshot of ../labeling-pharo-repos) -> repositories / methods / duplicates
#    3 sampling      stratified two-stage samples (random for frequencies, targeted for rare behaviours)
#    4 discovery     3 LLMs (3 families), independently, propose categories > subcategories > behaviours
#    5 taxonomy      merge candidates across models -> taxonomy_v1.json + review workbook (expert decisions)
#    6 annotation    3 LLMs classify literature tasks and Pharo methods with taxonomy v1 (multi-label)
#    7 validation    AI agreement; blinded expert workbook; human vs AI metrics; adjudication; taxonomy v2
#    8 profile       larger Pharo sample: prevalence per behaviour (design-weighted), per repository and group
#    9 coverage      literature vs Pharo coverage matrix and gaps
#   10 benchmark     executable Pharo task candidates for the selected behaviours
#   11 verification  run references + tests in an isolated Pharo image; expert review workbook
#   12 evaluation    coding models on the verified benchmark; results by category / behaviour / task type
#
# Results: results/SUMMARY.md (all phases, key numbers, figures). Expert files: results/05_taxonomy/review/,
# results/07_validation/expert/, results/11_verification/expert/ (completed files go in their returned/ folder).

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
  mkdir -p results/logs
  command -v ollama >/dev/null || { echo "Ollama is needed: https://ollama.com/download"; exit 1; }
  host="${OLLAMA_HOST:-localhost:11434}"
  if ! curl -sf "http://${host#http://}/api/tags" >/dev/null; then
    echo "Starting Ollama (log: results/logs/ollama.log) …"
    nohup ollama serve > results/logs/ollama.log 2>&1 &
    for _ in $(seq 30); do curl -sf "http://${host#http://}/api/tags" >/dev/null && break; sleep 1; done
  fi
  # models (library + the Pharo PT model built from its Hugging Face adapter), Pharo image, benchmarks, corpus
  if [ "${SKIP_SETUP:-0}" != 1 ]; then
    .venv/bin/python -m pbr setup || { echo "Setup failed: nothing was run (log: results/logs/00_setup.log)."; exit 1; }
  fi
  [ "${1:-}" = setup ] && exit 0
fi

case "${1:-}" in
  ""|run)                         shift || true; exec .venv/bin/python -m pbr run "$@" ;;
  status|logs|cost|reset|simulate-expert) exec .venv/bin/python -m pbr "$@" ;;
  help|-h|--help)                 sed -n '2,32p' "$0" ;;
  *)                              exec .venv/bin/python -m pbr run "$@" ;;
esac
