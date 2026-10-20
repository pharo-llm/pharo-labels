#!/usr/bin/env bash
# OAR job: everything ./run.sh does, from the beginning, on one larochette node (4 x AMD MI210, 64 GiB each):
# installs Ollama with its ROCm add-on, starts it on the GPUs, downloads the benchmarks and the three models, then
# the three models cluster every item AT THE SAME TIME (about 20 GB each: one per GPU), then the comparison and
# every figure.
#
#   oarsub -S ./job.oar.sh                                  from this folder on the frontend: every item -> results/
#   SAMPLE=1 OUT=results/pilot oarsub -S ./job.oar.sh       pilot: 1 item per benchmark (run.sh settings pass through)
#
# Log of the job: job.<jobid>.out / .err here. Ollama is installed in .ollama-bin/ and the models kept in
# .ollama-models/, both reused by the next jobs. Resumes: every answer already saved in results/ is reused, so a job
# that runs out of walltime is simply resubmitted. DeepSWE needs HF_TOKEN (in run.sh or ../.env).
#
#OAR -q default
#OAR -p larochette
#OAR -l host=1,walltime=55:00:00
#OAR -n benchmark-clusters
#OAR -O job.%jobid%.out
#OAR -E job.%jobid%.err

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p results
OLLAMA_DIR="$HERE/.ollama-bin"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HERE/.ollama-models}"
export OLLAMA_HOST="127.0.0.1:11434"      # host=1: the node is this job's alone
export OLLAMA_NOPRUNE=1
export PARALLEL="${PARALLEL:-1}"          # the three models at once
export WORKERS="${WORKERS:-4}"            # requests at the same time per model, one Ollama slot each
export OLLAMA_NUM_PARALLEL="$WORKERS"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-3}"     # the three stay loaded, one GPU each
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"                  # never unloaded during the job
OLLAMA_LOG="results/ollama.${OAR_JOB_ID:-manual}.log"
OLLAMA_PID=""

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
cleanup() { [ -n "$OLLAMA_PID" ] && kill "$OLLAMA_PID" 2>/dev/null && wait "$OLLAMA_PID" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 143' TERM

extract_zst() {       # extract_zst <archive>, into the current directory
  if tar --help 2>/dev/null | grep -q zstd; then tar --zstd -xf "$1"
  elif command -v zstd >/dev/null; then zstd -dc "$1" | tar -xf -
  else log "ERROR: zstd is needed to extract Ollama"; exit 1; fi
}

log "Job ${OAR_JOB_ID:-manual} on $(hostname -f 2>/dev/null || hostname)"
if command -v rocm-smi >/dev/null; then rocm-smi || true; elif [ -e /dev/kfd ]; then log "AMD GPU found (/dev/kfd)"; fi

# 1. Ollama in this folder, no sudo; on an AMD node it also needs the ROCm add-on of the same release
#    (without it the model silently runs on the CPU)
mkdir -p "$OLLAMA_DIR" "$OLLAMA_MODELS"
if [ ! -x "$OLLAMA_DIR/bin/ollama" ] || { [ -e /dev/kfd ] && ! compgen -G "$OLLAMA_DIR/lib/ollama/rocm*" >/dev/null; }; then
  log "Installing Ollama in $OLLAMA_DIR"
  (
    cd "$OLLAMA_DIR"
    curl -fL --retry 5 --retry-delay 10 https://ollama.com/download/ollama-linux-amd64.tar.zst -o ollama.tar.zst
    extract_zst ollama.tar.zst
    if [ -e /dev/kfd ]; then
      version="$(OLLAMA_HOST=127.0.0.1:1 bin/ollama --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -n 1 || true)"
      curl -fL --retry 5 --retry-delay 10 \
        "https://ollama.com/download/ollama-linux-amd64-rocm.tar.zst${version:+?version=$version}" -o rocm.tar.zst
      extract_zst rocm.tar.zst
    fi
    rm -f ollama.tar.zst rocm.tar.zst
  )
fi
export PATH="$OLLAMA_DIR/bin:$PATH"
ollama --version || true

# 2. Start the server here, on every GPU, so run.sh finds it running and uses it
ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!
for _ in $(seq 120); do
  curl -fsS "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1 && break
  kill -0 "$OLLAMA_PID" 2>/dev/null || { log "ERROR: Ollama crashed"; cat "$OLLAMA_LOG"; exit 1; }
  sleep 1
done
curl -fsS "http://$OLLAMA_HOST/api/tags" >/dev/null || { log "ERROR: Ollama did not start in 120 s"; cat "$OLLAMA_LOG"; exit 1; }
for _ in $(seq 60); do grep -q 'inference compute' "$OLLAMA_LOG" && break; sleep 1; done
if grep 'inference compute' "$OLLAMA_LOG" | grep -qiE 'library=(rocm|cuda)'; then
  log "Ollama uses these GPUs:"; grep 'inference compute' "$OLLAMA_LOG"
else
  log "WARNING: Ollama found no GPU: the model runs on the CPU (much slower)"
  grep -iE 'inference compute|gpu|rocm|amdgpu' "$OLLAMA_LOG" | head -n 30 || true
fi

# 3. The benchmarks (into ../data/existing-benchmarks/, skipped when already there), then everything of run.sh:
#    the models pulled, the three clusterings at once, the comparison, the figures
./run.sh download
./run.sh
rocm-smi --showmemuse 2>/dev/null || true
log "Job finished: results in $HERE/results"
