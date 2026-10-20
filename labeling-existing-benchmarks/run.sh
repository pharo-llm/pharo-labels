#!/usr/bin/env bash
# The code benchmarks LLMs are evaluated on: download them once into ../data/already-benchmarks/, and name the TASK
# each one asks (input -> output). The clustering is its own project: ../clustering-existing-benchmarks/.
#
#   ./run.sh                          download, then tasks
#   ./run.sh download                 benchmarks -> data/already-benchmarks/ (no model)
#   ./run.sh tasks                    the TASK each benchmark asks -> results/tasks/
#   SAMPLE=5 ./run.sh tasks           pilot: 5 items per benchmark
#   ONLY="swebench_pro nl2repo" ./run.sh
#
# Every step resumes: an item whose answer is already saved is never asked again.
# DeepSWE is gated on Hugging Face: accept its terms on https://huggingface.co/datasets/datacurve/deep-swe and put
# 'export HF_TOKEN=...' in ../.env; without it, it is listed as not downloaded.

# ============================== SETTINGS ===================================================
MODEL="${MODEL:-qwen3.8:27b}"                # the labelling model of pharo-labels, for every step
SAMPLE="${SAMPLE:-0}"                        # 0 = every item of every benchmark (~6,400); n = pilot
ONLY="${ONLY:-}"                             # "" = all benchmarks
# ===========================================================================================

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if [ -f ../.env ]; then . ../.env; fi
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(download tasks)
for s in "${STEPS[@]}"; do
  case "$s" in download|tasks) ;; *) echo "Unknown step: $s (download | tasks)"; exit 1 ;; esac
done
mkdir -p ../data/already-benchmarks results/tasks

if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c 'import pyarrow' 2>/dev/null; then
  py=${PYTHON:-}
  if [ -z "$py" ]; then
    for c in python3.13 python3.12 python3.11 python3; do command -v "$c" >/dev/null && { py=$c; break; }; done
  fi
  "$py" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "Python >= 3.10 is required."; exit 1; }
  "$py" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
# shellcheck disable=SC2206
only=(); [ -n "$ONLY" ] && only=(--only $ONLY)

if [[ " ${STEPS[*]} " == *" tasks "* ]]; then
  host="${OLLAMA_HOST:-localhost:11434}"
  command -v ollama >/dev/null || { echo "Ollama is needed: https://ollama.com/download"; exit 1; }
  if ! curl -sf "http://${host#http://}/api/tags" >/dev/null; then
    echo "Starting Ollama (log: results/ollama.log) …"
    nohup ollama serve > results/ollama.log 2>&1 &
    for _ in $(seq 30); do curl -sf "http://${host#http://}/api/tags" >/dev/null && break; sleep 1; done
  fi
  ollama list | awk 'NR>1 {print $1}' | grep -qx -e "$MODEL" -e "$MODEL:latest" || { echo "Downloading $MODEL …"; ollama pull "$MODEL"; }
fi

for s in "${STEPS[@]}"; do
  echo "=== $s ==="
  case "$s" in
    download)
      .venv/bin/python -I benchmarks.py ${only[@]+"${only[@]}"} ;;
    tasks)
      .venv/bin/python -I tasks/label_tasks.py --model "$MODEL" --sample "$SAMPLE" ${only[@]+"${only[@]}"} 2>&1 | tee -a results/tasks/run.log ;;
  esac
done
