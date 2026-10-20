#!/usr/bin/env bash
# Download the code benchmarks LLMs are evaluated on and name the TASK each one asks the model to do
# (what it is given and what it must produce), not the topic of its code.
#
#   ./run.sh                    every item of every benchmark -> results/tasks.csv (the one list) + tasks.md + benchmarks.csv
#   SAMPLE=5 ./run.sh           pilot: 5 items per benchmark
#   ONLY="swebench_pro nl2repo" ./run.sh
#
# DeepSWE is gated on Hugging Face: accept its terms on https://huggingface.co/datasets/datacurve/deep-swe and put
# 'export HF_TOKEN=...' in ../.env; without it, it is listed as not downloaded.

# ============================== SETTINGS ===================================================
MODEL="${MODEL:-qwen3.8:27b}"                # names the task of every item (the labelling model of pharo-labels)
SAMPLE="${SAMPLE:-0}"                        # 0 = every item of every benchmark (~6,400, ~7 h); n = pilot
ONLY="${ONLY:-}"                             # "" = all benchmarks
# ===========================================================================================

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if [ -f ../.env ]; then . ../.env; fi
mkdir -p results

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

host="${OLLAMA_HOST:-localhost:11434}"
command -v ollama >/dev/null || { echo "Ollama is needed: https://ollama.com/download"; exit 1; }
if ! curl -sf "http://${host#http://}/api/tags" >/dev/null; then
  echo "Starting Ollama (log: results/ollama.log) …"
  nohup ollama serve > results/ollama.log 2>&1 &
  for _ in $(seq 30); do curl -sf "http://${host#http://}/api/tags" >/dev/null && break; sleep 1; done
fi
for m in $MODEL; do
  ollama list | awk 'NR>1 {print $1}' | grep -qx -e "$m" -e "$m:latest" || { echo "Downloading $m …"; ollama pull "$m"; }
done

args=(--model "$MODEL")
args+=(--sample "$SAMPLE")
# shellcheck disable=SC2206
[ -n "$ONLY" ] && args+=(--only $ONLY)
.venv/bin/python -I label_benchmarks.py "${args[@]}" 2>&1 | tee -a results/run.log
