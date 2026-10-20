#!/usr/bin/env bash
# What the existing code benchmarks really ask, according to each of three models on its own: every item in a
# category, then a subcategory, and its actual task; then how much the three models agree. Benchmarks from
# ../data/already-benchmarks/ (shared).
#
#   ./run.sh                              everything: each model -> OUT/<model>/, then OUT/comparison/
#   SAMPLE=1 OUT=results/pilot ./run.sh   pilot: 1 item per benchmark, in a folder of its own
#   ./run.sh figures                      redraw every figure and the comparison from the items.csv files, no model
#   ./run.sh download                     only download the benchmarks into ../data/already-benchmarks/
#   ONLY="swebench_pro nl2repo" ./run.sh
#
# OUT/<model>/: clusters.md, items.csv, categories.csv, subcategories.csv, figures/ (one model's view)
# OUT/comparison/: comparison.md, agreement.csv, agreement_per_benchmark.csv, category_mapping.csv, figures/
# Every answer is saved in OUT/<model>/answers/ and reused by a rerun. OUT/<model>/categories.csv and
# subcategories.csv are kept once written: edit them and rerun to classify again; delete them for new proposals.

# ============================== SETTINGS ===================================================
MODELS="${MODELS:-qwen3.8:27b gemma4:31b muse-glimmer:30b}"   # each one labels every item on its own
OUT="${OUT:-results}"                        # results folder
SAMPLE="${SAMPLE:-0}"                        # 0 = every item (~6,400); n = pilot of n items per benchmark
ONLY="${ONLY:-}"                             # "" = all benchmarks
MAX_CATEGORIES="${MAX_CATEGORIES:-20}"
MAX_SUBCATEGORIES="${MAX_SUBCATEGORIES:-8}"  # per category
MIN_ITEMS="${MIN_ITEMS:-5}"                  # a category with fewer items is not split
WORKERS="${WORKERS:-1}"                      # requests sent to Ollama at the same time (4 on larochette)
# ===========================================================================================

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if [ -f ../.env ]; then . ../.env; fi
STEP="${1:-cluster}"
case "$STEP" in cluster|figures|download) ;; *) echo "Unknown step: $STEP (cluster | figures | download)"; exit 1 ;; esac
folder() { echo "$1" | tr ':/' '__'; }
mkdir -p "$OUT"

if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c 'import pyarrow, matplotlib' 2>/dev/null; then
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

case "$STEP" in
  figures)
    for m in $MODELS; do [ -f "$OUT/$(folder "$m")/items.csv" ] && .venv/bin/python -I figures.py --out "$OUT/$(folder "$m")"; done
    # shellcheck disable=SC2086
    .venv/bin/python -I compare.py --models $MODELS --out "$OUT"; exit ;;
  download) .venv/bin/python -I benchmarks.py ${only[@]+"${only[@]}"}; exit ;;
esac

host="${OLLAMA_HOST:-localhost:11434}"
command -v ollama >/dev/null || { echo "Ollama is needed: https://ollama.com/download"; exit 1; }
if ! curl -sf "http://${host#http://}/api/tags" >/dev/null; then
  echo "Starting Ollama (log: $OUT/ollama.log) …"
  OLLAMA_NUM_PARALLEL="$WORKERS" nohup ollama serve > "$OUT/ollama.log" 2>&1 &
  for _ in $(seq 30); do curl -sf "http://${host#http://}/api/tags" >/dev/null && break; sleep 1; done
fi
for m in $MODELS; do
  ollama list | awk 'NR>1 {print $1}' | grep -qx -e "$m" -e "$m:latest" || { echo "Downloading $m …"; ollama pull "$m"; }
done

for m in $MODELS; do
  echo "=== $m -> $OUT/$(folder "$m")/ ==="
  .venv/bin/python -I cluster.py --model "$m" --out "$OUT/$(folder "$m")" --sample "$SAMPLE" --workers "$WORKERS" \
    --max-categories "$MAX_CATEGORIES" --max-subcategories "$MAX_SUBCATEGORIES" --min-items "$MIN_ITEMS" \
    ${only[@]+"${only[@]}"} 2>&1 | tee -a "$OUT/run.log" || echo "FAILED: $m (see $OUT/run.log); the others go on"
done
echo "=== comparison -> $OUT/comparison/ ==="
# shellcheck disable=SC2086
.venv/bin/python -I compare.py --models $MODELS --out "$OUT" 2>&1 | tee -a "$OUT/run.log"
