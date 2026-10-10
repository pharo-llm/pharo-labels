#!/usr/bin/env bash

#OAR -q default
#OAR -p chirop
#OAR -l host=1,walltime=24:00:00
#OAR -n pharo-labels
#OAR -O pharo-labels.%jobid%.out
#OAR -E pharo-labels.%jobid%.err

# The whole workflow in one cluster job, on one CPU-only node (chirop: 2 x Xeon Platinum 8358,
# 64 cores, 512 GiB RAM, no GPU):
#   1. ./1-mining.sh        results from the Hugging Face snapshot, or mined from GitHub
#      ./1-2-filtering.sh   keeps the repositories worth labelling (150 at most)
#   2. ./2-ai-labelling.sh  three big open-weight models label the sample, on an Ollama server
#                           that this job installs and starts on the node (no sudo)
#   3. ./3-results.sh       taxonomy, statistics, reports
#
#   oarsub -S ./job.oar.sh          submit (from the repository folder)
#   bash job.oar.sh                 or run it directly on a reserved node
#
# There is no terminal in a batch job, so secrets must be in .env before submitting:
#   export HF_TOKEN=...             write access: upload the results (without it, nothing is uploaded)
#   export GITHUB_TOKEN=...         only needed if step 1 really has to mine GitHub
#
# Safe to resubmit: results/ is kept, model answers are cached, and every step resumes. If the
# walltime runs out during step 2, submit again and it continues where it stopped.

source "$(dirname "${BASH_SOURCE[0]}")/scripts/_common.sh"   # set -euo pipefail, cd to the repository, .env, .venv
ROOT_DIR="$ROOT"


###############################################################################
# Configuration
#
# Edit the defaults here. Variables of the same name set at submission time
# override them.
###############################################################################

# The annotators (names from models.yaml). Chosen for the best labels this node can produce
# without a GPU: the strongest practical model of three different families, so that a label
# kept by the majority vote was found independently by at least two of them.
#   cluster-qwen-coder-32b   qwen2.5-coder:32b   20 GB   Alibaba, code-specialised
#   cluster-llama-3.3-70b    llama3.3:70b        43 GB   Meta, strongest general model
#   cluster-gemma3-27b       gemma3:27b          17 GB   Google
export MODELS="${MODELS:-cluster-qwen-coder-32b cluster-llama-3.3-70b cluster-gemma3-27b}"

# No adjudicator: adjudication asks one model at a time, which is too slow on CPU for a big
# model, and three annotators already give a majority.
export ADJUDICATOR="${ADJUDICATOR:-}"
export CONSENSUS="${CONSENSUS:-majority}"
export EVIDENCE_POLICY="${EVIDENCE_POLICY:-enforce}"

# 0 = all 300 methods. For a first trial of the job: LIMIT=10
export LIMIT="${LIMIT:-0}"
export SPOTCHECK_SIZE="${SPOTCHECK_SIZE:-30}"

# Requests Ollama answers at the same time. Must not be lower than `concurrency` of the
# cluster-* models in models.yaml (4), or requests queue and may time out.
OLLAMA_PARALLEL="${OLLAMA_PARALLEL:-4}"

# Step 3: benchmark task files to compare with the taxonomy ("" = skip).
export BENCHMARKS="${BENCHMARKS:-}"

# Upload results to Hugging Face after each step: auto = only when HF_TOKEN is in .env.
PUSH_SNAPSHOT="${PUSH_SNAPSHOT:-auto}"
if [ "$PUSH_SNAPSHOT" = "auto" ]; then
    if [ -n "${HF_TOKEN:-}" ]; then PUSH_SNAPSHOT="yes"; else PUSH_SNAPSHOT="no"; fi
fi
export PUSH_SNAPSHOT

RUN_ID="${OAR_JOB_ID:-manual-$(date '+%Y%m%d-%H%M%S')}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/oar-runs/$RUN_ID}"
RESULTS_DIR="$ROOT_DIR/results"

# Ollama and its models (~80 GB) go on the node's local disk, not in the home directory:
# home quotas are usually far smaller than that. The local disk may be emptied between jobs;
# models are then downloaded again (minutes on this network). To keep them, point
# OLLAMA_MODELS_DIR at persistent storage with enough room.
SCRATCH_DIR="${SCRATCH_DIR:-/tmp/${USER:-$(id -un)}/pharo-labels}"
OLLAMA_INSTALL_DIR="${OLLAMA_INSTALL_DIR:-$SCRATCH_DIR/ollama-bin}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$SCRATCH_DIR/ollama-models}"

OLLAMA_LOG="$RUN_DIR/ollama.log"
OLLAMA_PID=""
STEP_PID=""


###############################################################################
# Logging
###############################################################################

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}


###############################################################################
# Cleanup
###############################################################################

cleanup() {
    if [ -n "${STEP_PID:-}" ]; then
        kill "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" 2>/dev/null || true
    fi

    if [ -n "${OLLAMA_PID:-}" ]; then
        if kill -0 "$OLLAMA_PID" 2>/dev/null; then
            log "Stopping Ollama (PID $OLLAMA_PID)"
            kill "$OLLAMA_PID" || true
            wait "$OLLAMA_PID" 2>/dev/null || true
        fi
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM


# run_step NAME script...: run one step script in the background and wait for it, so that
# the traps above still fire while it runs (OAR sends TERM at the end of the walltime).
run_step() {
    local name="$1" status=0
    shift
    bash "$@" &
    STEP_PID=$!
    wait "$STEP_PID" || status=$?
    STEP_PID=""
    if [ "$status" -ne 0 ]; then
        log "ERROR: $name failed (exit $status). Fix the message above and resubmit: finished work is kept."
        exit "$status"
    fi
    log "$name OK"
}


###############################################################################
# Create directories
###############################################################################

mkdir -p "$RUN_DIR"
mkdir -p "$OLLAMA_INSTALL_DIR"
mkdir -p "$OLLAMA_MODELS_DIR"


###############################################################################
# Job information
###############################################################################

log "============================================================"
log "JOB STARTED"
log "============================================================"

log "Job ID: ${OAR_JOB_ID:-manual}"
log "Host: $(hostname -f 2>/dev/null || hostname)"
log "Root directory: $ROOT_DIR"
log "Run directory: $RUN_DIR"
log "Results directory: $RESULTS_DIR"
log "Ollama install directory: $OLLAMA_INSTALL_DIR"
log "Ollama models directory: $OLLAMA_MODELS_DIR"
log "Models: $MODELS${ADJUDICATOR:+; adjudicator: $ADJUDICATOR}"
log "Methods to label: $([ "$LIMIT" = "0" ] && echo all || echo "first $LIMIT")"
log "Upload to Hugging Face: $PUSH_SNAPSHOT"


###############################################################################
# System information
###############################################################################

log "============================================================"
log "SYSTEM INFORMATION"
log "============================================================"

uname -a || true

if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E 'Model name|Socket|Core|Thread|CPU\(s\)' || true
fi

free -g 2>/dev/null || true
df -h "$OLLAMA_MODELS_DIR" "$ROOT_DIR" || true

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA GPU detected"
    nvidia-smi || true
else
    log "No NVIDIA GPU detected: models run on CPU"
fi


###############################################################################
# 1. Mining (needs no Ollama; first, so a missing token fails within seconds)
###############################################################################

log "============================================================"
log "1. MINING"
log "============================================================"

run_step "1-MINING" "$ROOT_DIR/1-mining.sh" "$@"

# The study corpus: needs no token and no model, about two minutes
run_step "1-2-FILTERING" "$ROOT_DIR/1-2-filtering.sh"


###############################################################################
# 2. Install Ollama locally
#
# NO sudo
# NO systemctl
# NO /usr/local/bin
###############################################################################

log "============================================================"
log "2. INSTALLING OLLAMA LOCALLY"
log "============================================================"

find_ollama() {
    find "$OLLAMA_INSTALL_DIR" -type f -name ollama -perm -u+x | head -n 1
}

OLLAMA_BIN="$(find_ollama)"

if [ -n "$OLLAMA_BIN" ]; then

    log "Ollama is already installed on this node"

else

    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64|amd64)  OLLAMA_ARCH="amd64" ;;
        aarch64|arm64) OLLAMA_ARCH="arm64" ;;
        *)
            log "ERROR: Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac

    OLLAMA_ARCHIVE="$OLLAMA_INSTALL_DIR/ollama-linux-${OLLAMA_ARCH}.tar.zst"

    log "Architecture: $ARCH"
    log "Downloading Ollama: $(basename "$OLLAMA_ARCHIVE")"

    curl -fL "https://ollama.com/download/$(basename "$OLLAMA_ARCHIVE")" -o "$OLLAMA_ARCHIVE"

    log "Extracting Ollama"

    if tar --help 2>/dev/null | grep -q zstd; then
        tar --zstd -xf "$OLLAMA_ARCHIVE" -C "$OLLAMA_INSTALL_DIR"
    elif command -v unzstd >/dev/null 2>&1; then
        unzstd -c "$OLLAMA_ARCHIVE" | tar -xf - -C "$OLLAMA_INSTALL_DIR"
    elif command -v zstd >/dev/null 2>&1; then
        zstd -dc "$OLLAMA_ARCHIVE" | tar -xf - -C "$OLLAMA_INSTALL_DIR"
    else
        log "ERROR: Cannot extract Ollama: zstd support was not found."
        exit 1
    fi

    rm -f "$OLLAMA_ARCHIVE"
    OLLAMA_BIN="$(find_ollama)"

fi

if [ -z "${OLLAMA_BIN:-}" ]; then
    log "ERROR: Ollama executable not found"
    find "$OLLAMA_INSTALL_DIR" -maxdepth 4 -type f -print || true
    exit 1
fi

log "Ollama executable: $OLLAMA_BIN"
"$OLLAMA_BIN" --version || true

# 2-ai-labelling.sh calls `ollama` by name.
export PATH="$(dirname "$OLLAMA_BIN"):$PATH"

log "2-INSTALL OLLAMA OK"


###############################################################################
# 3. Configure and start Ollama
###############################################################################

log "============================================================"
log "3. STARTING OLLAMA SERVER"
log "============================================================"

export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
# The step scripts and models.yaml expect the default address.
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_NUM_PARALLEL="$OLLAMA_PARALLEL"
# The annotators run one after the other: keep one model in memory, and keep it loaded.
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE="24h"
# A 43 GB model can take minutes to load.
export OLLAMA_LOAD_TIMEOUT="30m"

log "OLLAMA_MODELS=$OLLAMA_MODELS"
log "OLLAMA_HOST=$OLLAMA_HOST"
log "OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL"

if curl -fsS "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
    log "ERROR: something already answers on $OLLAMA_HOST (another Ollama on this node?). Stop it and resubmit."
    exit 1
fi

"$OLLAMA_BIN" serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

log "Ollama PID: $OLLAMA_PID"
log "Ollama log: $OLLAMA_LOG"

show_ollama_log() {
    echo
    echo "================ OLLAMA LOG ================="
    cat "$OLLAMA_LOG" || true
    echo "============================================="
}

log "Waiting for Ollama API..."
OLLAMA_READY=0

for i in $(seq 1 120); do

    if curl -fsS "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
        OLLAMA_READY=1
        log "Ollama API is ready"
        break
    fi

    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        log "ERROR: Ollama server crashed"
        show_ollama_log
        exit 1
    fi

    sleep 1

done

if [ "$OLLAMA_READY" -ne 1 ]; then
    log "ERROR: Ollama did not start within 120 seconds"
    show_ollama_log
    exit 1
fi

log "3-START OLLAMA OK"


###############################################################################
# 4. Pull the annotation models
###############################################################################

log "============================================================"
log "4. PULLING OLLAMA MODELS"
log "============================================================"

# The Ollama tags behind MODELS (and ADJUDICATOR), as models.yaml defines them.
OLLAMA_MODELS_TO_PULL=()
requirements="$("$MINER" llm-annotate --print-requirements --models "$MODELS" --adjudicator "${ADJUDICATOR:-none}")"
while read -r kind value; do
    if [ "$kind" = "ollama" ]; then OLLAMA_MODELS_TO_PULL+=("$value"); fi
done <<< "$requirements"

if [ "${#OLLAMA_MODELS_TO_PULL[@]}" -eq 0 ]; then
    log "None of the chosen models runs on this node's Ollama: nothing to pull"
fi

for model in ${OLLAMA_MODELS_TO_PULL[@]+"${OLLAMA_MODELS_TO_PULL[@]}"}; do

    log "------------------------------------------------------------"
    log "Model: $model"
    log "------------------------------------------------------------"

    if "$OLLAMA_BIN" list | awk 'NR > 1 { print $1 }' | grep -Fxq -e "$model" -e "$model:latest"; then
        log "Model already downloaded: $model"
        continue
    fi

    log "Pulling: $model"

    if "$OLLAMA_BIN" pull "$model"; then
        log "Successfully pulled: $model"
    else
        log "ERROR: Failed to pull model: $model"
        df -h "$OLLAMA_MODELS_DIR" || true
        exit 1
    fi

done

log "INSTALLED OLLAMA MODELS"
"$OLLAMA_BIN" list

log "4-PULL MODELS OK"


###############################################################################
# 5. AI labelling
###############################################################################

log "============================================================"
log "5. AI LABELLING"
log "============================================================"

if ! curl -fsS "http://$OLLAMA_HOST/api/tags" >/dev/null; then
    log "ERROR: Ollama API is no longer responding"
    show_ollama_log
    exit 1
fi

# The models run one after the other. On CPU the 70B model is by far the slowest; progress
# and the estimated remaining time are printed as it goes (also in results/ai-labelling.log).
run_step "5-AI LABELLING" "$ROOT_DIR/2-ai-labelling.sh"


###############################################################################
# 6. Stop Ollama (step 3 needs no model; this frees the memory)
###############################################################################

log "============================================================"
log "6. STOPPING OLLAMA"
log "============================================================"

"$OLLAMA_BIN" ps || true

log "LAST 100 LINES OF OLLAMA LOG"
tail -n 100 "$OLLAMA_LOG" || true

kill "$OLLAMA_PID" 2>/dev/null || true
wait "$OLLAMA_PID" 2>/dev/null || true
OLLAMA_PID=""

log "6-STOP OLLAMA OK"


###############################################################################
# 7. Results
###############################################################################

log "============================================================"
log "7. RESULTS"
log "============================================================"

run_step "7-RESULTS" "$ROOT_DIR/3-results.sh"


###############################################################################
# 8. Show results
###############################################################################

log "============================================================"
log "8. GENERATED FILES"
log "============================================================"

for dir in validation/llm reports tables figures; do
    find "$RESULTS_DIR/$dir" -maxdepth 1 -type f -print 2>/dev/null || true
done


###############################################################################
# Finished
###############################################################################

log "============================================================"
log "JOB FINISHED SUCCESSFULLY"
log "============================================================"

log "Job ID: ${OAR_JOB_ID:-manual}"
log "Results: $RESULTS_DIR (start with reports/ecosystem_summary.md)"
log "AI labels: $RESULTS_DIR/validation/llm (agreement between the models: run_manifest.json)"
log "Ollama log: $OLLAMA_LOG"
log "Ollama models: $OLLAMA_MODELS_DIR"
