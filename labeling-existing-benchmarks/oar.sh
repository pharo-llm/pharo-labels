#!/usr/bin/env bash
# OAR job: ./run.sh on one larochette node (4 x AMD MI210, 64 GiB each).
#
#   oarsub -S ./oar.sh                               from this folder on the frontend
#   SAMPLE=5 ONLY="nl2repo" oarsub -S ./oar.sh       (run.sh settings pass through)
#
# Ollama (with its ROCm add-on) is installed in .ollama-bin/ and the model kept in .ollama-models/, both reused by
# the next jobs. Resumes: results/answers.jsonl is never truncated and every item already in it is reused, so a job
# that runs out of walltime is simply resubmitted.
#
#OAR -q default
#OAR -p larochette
#OAR -l host=1,walltime=14:00:00
#OAR -n benchmark-tasks
#OAR -O results/oar.%jobid%.out
#OAR -E results/oar.%jobid%.err

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p results
OLLAMA_DIR="$HERE/.ollama-bin"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HERE/.ollama-models}"
export OLLAMA_HOST="127.0.0.1:11434"      # host=1: the node is this job's alone
export OLLAMA_NOPRUNE=1
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

# 2. Start the server here, so run.sh finds it running and uses it
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

# 3. The labelling (pulls the model on the first job, then reuses every answer already in results/answers.jsonl)
./run.sh
log "Job finished: results in $HERE/results"
