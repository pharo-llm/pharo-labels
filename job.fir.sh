#!/usr/bin/env bash

# The whole pharo-labels pipeline as ONE Slurm job on the Fir cluster (Digital Research
# Alliance of Canada), on one full NVIDIA H100 80 GB. The Fir version of job.oar.sh.
#
#   1.   ./1-mining.sh        mined data from the Hugging Face snapshot (or mined from GitHub)
#   1-2. ./1-2-filtering.sh   the study corpus (150 repositories at most) and the methods to label
#        Ollama               installed and started by the job (no sudo), models pulled
#   2.   ./2-ai-labelling.sh  three open-weight models of three families label the methods
#   3.   ./3-results.sh       taxonomy, statistics, figures, reports
#   4.   ./4-benchmark.sh     one model writes tasks, a Pharo image validates them, another solves them
#
# ---- Run it (from the repository folder on Fir) ---------------------------------------------
#   sbatch job.fir.sh                               the real run
#   sbatch --time=3:00:00 --export=ALL,TRIAL=yes job.fir.sh
#                                                   a short trial: 10 methods, 5 benchmark tasks
#   bash job.fir.sh --prepare                       optional, on the LOGIN node, before sbatch:
#                                                   downloads everything (Python packages, the 15 GB
#                                                   snapshot, Ollama, ~80 GB of models) so the GPU
#                                                   job does not spend its hours downloading
# Follow it:   sq                                   (PD = waiting, R = running)
#              tail -f pharo-labels.<jobid>.log     EVERYTHING is in this one log file
# Cancel it:   scancel <jobid>
#
# ---- What you get ---------------------------------------------------------------------------
#   pharo-labels.<jobid>.log        the whole story: settings, every step, progress, key numbers,
#                                   a summary table at the end (also when it fails)
#   fir-runs/<jobid>/               the important results, copied to your HOME (safe):
#                                   reports, tables, figures, AI labels, benchmark tasks, logs,
#                                   settings, GPU usage. Updated after every step.
#   /scratch/$USER/pharo-labels/    the big things (full results/ with the 15 GB database and
#                                   clones, Ollama, models). Home is too small for them.
#                                   Scratch is purged after 60 days without use.
#
# Safe to resubmit at any time: every step resumes, model answers are cached. When the time
# limit comes, the job submits its own continuation (RESUBMIT below) and stops cleanly.
#
# Secrets (a batch job cannot ask for them) go in .env before submitting:
#   echo 'export HF_TOKEN=...' >> .env       write access: upload results to Hugging Face (optional)
#   echo 'export GITHUB_TOKEN=...' >> .env   only if step 1 really has to mine GitHub

#SBATCH --job-name=pharo-labels
#SBATCH --account=def-foutsekh
# A full H100: llama3.3:70b alone needs 43 GB, so a 40 GB MIG slice is not enough.
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# One log for everything (stderr goes into the same file).
#SBATCH --output=pharo-labels.%j.log
# 15 minutes before the time limit, the job saves its results and submits its continuation.
#SBATCH --signal=B:USR1@900
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80


###############################################################################
# SETTINGS
#
# Every value can also be given at submission without editing this file:
#   sbatch --export=ALL,LABEL_REPOSITORIES=10,RUN_BENCHMARK=no job.fir.sh
###############################################################################

# yes = a quick trial of the whole chain: only 10 methods labelled and 5 benchmark tasks,
# nothing uploaded. Use with --time=3:00:00.
TRIAL="${TRIAL:-no}"

# ---- Step 1-2: study corpus and what step 2 labels --------------------------------------
# Repositories kept for the results and the benchmark, at most.
MAX_REPOSITORIES="${MAX_REPOSITORIES:-150}"
# Repositories whose EVERY method is labelled (about 1,300 methods each; 5 = about 20,000).
# 0 = label a sample of SAMPLE_SIZE methods across the whole corpus instead.
LABEL_REPOSITORIES="${LABEL_REPOSITORIES:-5}"
SAMPLE_SIZE="${SAMPLE_SIZE:-300}"

# ---- Step 2: AI labelling ---------------------------------------------------------------
# Three families, so a label kept by the majority was found by two models independently.
#   cluster-qwen-coder-32b  qwen2.5-coder:32b  20 GB  Alibaba, code-specialised
#   cluster-llama-3.3-70b   llama3.3:70b       43 GB  Meta, strongest general model
#   cluster-gemma3-27b      gemma3:27b         17 GB  Google
MODELS="${MODELS:-cluster-qwen-coder-32b cluster-llama-3.3-70b cluster-gemma3-27b}"
# 0 = all the methods chosen above.
LIMIT="${LIMIT:-0}"
# Requests sent to the model at the same time. The H100 holds 8 for every model above
# (with the q8_0 KV cache set below).
LABEL_CONCURRENCY="${LABEL_CONCURRENCY:-8}"
ADJUDICATOR="${ADJUDICATOR:-}"
CONSENSUS="${CONSENSUS:-majority}"
EVIDENCE_POLICY="${EVIDENCE_POLICY:-enforce}"
SPOTCHECK_SIZE="${SPOTCHECK_SIZE:-30}"

# ---- Step 3: results --------------------------------------------------------------------
BENCHMARKS="${BENCHMARKS:-}"

# ---- Step 4: benchmark generation -------------------------------------------------------
RUN_BENCHMARK="${RUN_BENCHMARK:-yes}"
AUTHOR="${AUTHOR:-cluster-qwen-coder-32b}"
SOLVERS="${SOLVERS:-cluster-gemma3-27b}"
OPERATIONS="${OPERATIONS:-auto}"
TASKS_PER_OPERATION="${TASKS_PER_OPERATION:-3}"
# 0 = all candidate tasks.
BENCHMARK_LIMIT="${BENCHMARK_LIMIT:-0}"
# Tasks validated at the same time, each in its own Pharo image (16 CPUs requested).
BENCHMARK_CONCURRENCY="${BENCHMARK_CONCURRENCY:-4}"

# ---- Storage ----------------------------------------------------------------------------
# Big files go to scratch (home quota is 48 GB, the snapshot alone is 15 GB).
STORE_DIR="${STORE_DIR:-/scratch/${USER:-$(id -un)}/pharo-labels}"
# Files bigger than this are not copied to fir-runs/ (they stay on scratch; the log lists them).
SAVE_MAX_FILE_MB="${SAVE_MAX_FILE_MB:-200}"

# ---- Hugging Face, Ollama, time limit ---------------------------------------------------
# Upload results to Hugging Face after steps 1-3: auto = only when HF_TOKEN is in .env.
PUSH_SNAPSHOT="${PUSH_SNAPSHOT:-auto}"
OLLAMA_PARALLEL="${OLLAMA_PARALLEL:-8}"
OLLAMA_MAX_LOADED="${OLLAMA_MAX_LOADED:-2}"
# yes = when the time limit comes, submit the same job again (it continues where it stopped).
RESUBMIT="${RESUBMIT:-yes}"
MAX_RESUBMITS="${MAX_RESUBMITS:-5}"
# Minutes between two status lines (GPU use, current step) in the log. 0 = none.
HEARTBEAT_MINUTES="${HEARTBEAT_MINUTES:-15}"

###############################################################################
# End of SETTINGS
###############################################################################

MODE="job"
if [ "${1:-}" = "--prepare" ]; then MODE="prepare"; shift; fi
RESUBMIT_COUNT="${RESUBMIT_COUNT:-0}"
ORIGINAL_ARGS=("$@")

if [ "$TRIAL" = "yes" ]; then
    LIMIT="${TRIAL_LIMIT:-10}"
    BENCHMARK_LIMIT="${TRIAL_BENCHMARK_LIMIT:-5}"
    PUSH_SNAPSHOT="no"
    RESUBMIT="no"
fi

# Slurm runs a copy of this script from its spool directory: the repository is where sbatch was called.
if [ -n "${SLURM_JOB_ID:-}" ] && [ "$MODE" = "job" ]; then
    ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [ ! -f "$ROOT_DIR/2-ai-labelling.sh" ] || [ ! -f "$ROOT_DIR/scripts/_common.sh" ]; then
    echo "Run sbatch from the pharo-labels folder (2-ai-labelling.sh not found in $ROOT_DIR)." >&2
    exit 1
fi

RUN_ID="${SLURM_JOB_ID:-$MODE-$(date '+%Y%m%d-%H%M%S')}"
SAVE_DIR="$ROOT_DIR/fir-runs/$RUN_ID"
RUN_DIR="$STORE_DIR/runs/$RUN_ID"
RESULTS_DIR="$ROOT_DIR/results"
OLLAMA_INSTALL_DIR="$STORE_DIR/ollama-bin"
OLLAMA_MODELS_DIR="$STORE_DIR/ollama-models"
OLLAMA_LOG="$RUN_DIR/ollama.log"
JOB_START=$(date +%s)


###############################################################################
# Logging
###############################################################################

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
hms() { local s=$1; printf '%dh%02dm%02ds' $((s / 3600)) $((s % 3600 / 60)) $((s % 60)); }
banner() {
    echo
    log "######################################################################"
    log "## $*"
    log "######################################################################"
}

# Summary table printed at the end, whatever happens
STEP_NAMES=()
STEP_STATES=()
STEP_TIMES=()
CURRENT_STEP="starting"
FINAL_STATE="FAILED"
FINAL_NOTE=""

record_step() { STEP_NAMES+=("$1"); STEP_STATES+=("$2"); STEP_TIMES+=("$3"); }

print_summary() {
    banner "SUMMARY ($RUN_ID, $MODE mode)"
    printf '    %-30s %-12s %s\n' "STEP" "STATE" "TIME"
    local i
    for i in "${!STEP_NAMES[@]}"; do
        printf '    %-30s %-12s %s\n' "${STEP_NAMES[$i]}" "${STEP_STATES[$i]}" "$(hms "${STEP_TIMES[$i]}")"
    done
    echo
    log "Result: $FINAL_STATE after $(hms $(($(date +%s) - JOB_START)))${FINAL_NOTE:+ — $FINAL_NOTE}"
    if [ -d "$RESULTS_DIR" ] && [ -x "$ROOT_DIR/.venv/bin/python" ]; then
        log "Key numbers:"
        "$ROOT_DIR/.venv/bin/python" -I "$ROOT_DIR/scripts/job_summary.py" "$RESULTS_DIR" all || true
    fi
    echo
    log "Important results (home, kept): $SAVE_DIR"
    log "Everything (scratch):           $STORE_DIR"
    log "This log:                       $ROOT_DIR/pharo-labels.${SLURM_JOB_ID:-<jobid>}.log"
}


###############################################################################
# Saving the important results to home
###############################################################################

save_outputs() {
    [ -d "$RESULTS_DIR" ] || return 0
    mkdir -p "$SAVE_DIR/results" "$SAVE_DIR/run"
    # Everything a person reads or a paper cites; not the database, clones, Pharo images or caches.
    rsync -a --prune-empty-dirs --max-size="${SAVE_MAX_FILE_MB}m" \
        --exclude 'data/' --exclude '.snapshot_staging/' --exclude 'benchmark/work/' \
        --exclude 'benchmark/pharo*/' --exclude '*.image' --exclude '*.changes' --exclude '*.sources' \
        "$RESULTS_DIR"/ "$SAVE_DIR/results/" 2>/dev/null || true
    [ -f "$RESULTS_DIR/data/processed/corpus.json" ] && cp -p "$RESULTS_DIR/data/processed/corpus.json" "$SAVE_DIR/results/" || true
    cp -p "$RUN_DIR"/*.csv "$RUN_DIR"/*.txt "$RUN_DIR"/pull-*.log "$SAVE_DIR/run/" 2>/dev/null || true
    # The Ollama log can reach many GB (one line per request): only its end goes to home.
    [ -f "$OLLAMA_LOG" ] && tail -n 3000 "$OLLAMA_LOG" > "$SAVE_DIR/run/ollama.last-3000-lines.log" || true
    [ -n "${SLURM_JOB_ID:-}" ] && cp -p "$ROOT_DIR/pharo-labels.$SLURM_JOB_ID.log" "$SAVE_DIR/" 2>/dev/null || true
    return 0
}

report_skipped_big_files() {
    [ -d "$RESULTS_DIR" ] || return 0
    local big
    big="$(find -L "$RESULTS_DIR" -path "$RESULTS_DIR/data" -prune -o -path '*/benchmark/pharo*' -prune -o \
        -path '*/benchmark/work' -prune -o -type f -size +"${SAVE_MAX_FILE_MB}"M -print 2>/dev/null || true)"
    if [ -n "$big" ]; then
        log "Not copied to home (bigger than ${SAVE_MAX_FILE_MB} MB, still on scratch):"
        printf '    %s\n' $big
    fi
}


###############################################################################
# Cleanup, time limit, resubmission
###############################################################################

OLLAMA_PID=""
STEP_PID=""
HEARTBEAT_PID=""
GPU_MON_PID=""

stop_children() {
    for pid in "$STEP_PID" "$HEARTBEAT_PID" "$GPU_MON_PID"; do
        if [ -n "$pid" ]; then
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    STEP_PID="" HEARTBEAT_PID="" GPU_MON_PID=""
    if [ -n "$OLLAMA_PID" ] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
        log "Stopping Ollama (PID $OLLAMA_PID)"
        kill "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
    fi
    OLLAMA_PID=""
}

on_exit() {
    local status=$?
    trap - EXIT
    set +e
    stop_children
    if [ "$FINAL_STATE" = "FAILED" ] && [ -z "$FINAL_NOTE" ]; then
        FINAL_NOTE="stopped during '$CURRENT_STEP' (exit $status). Fix the message above and resubmit: finished work is kept"
    fi
    save_outputs
    report_skipped_big_files
    print_summary
    save_outputs   # once more, so the copy of this log in fir-runs/ has the summary
    exit "$status"
}

on_time_limit() {
    trap '' USR1
    echo
    log "!!!!! The time limit is 15 minutes away: saving and stopping cleanly (during '$CURRENT_STEP')."
    FINAL_STATE="TIME LIMIT"
    if [ "$RESUBMIT" = "yes" ] && [ "$RESUBMIT_COUNT" -lt "$MAX_RESUBMITS" ]; then
        local next
        if next="$(cd "$ROOT_DIR" && sbatch --parsable ${SLURM_JOB_ACCOUNT:+--account="$SLURM_JOB_ACCOUNT"} \
                --dependency="afterany:$SLURM_JOB_ID" \
                --export=ALL,RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) \
                "$ROOT_DIR/job.fir.sh" ${ORIGINAL_ARGS[@]+"${ORIGINAL_ARGS[@]}"})"; then
            FINAL_NOTE="continuation submitted as job $next (resubmission $((RESUBMIT_COUNT + 1))/$MAX_RESUBMITS): follow pharo-labels.${next%%;*}.log"
        else
            FINAL_NOTE="could not submit the continuation: run 'sbatch job.fir.sh' yourself, it resumes"
        fi
    else
        FINAL_NOTE="not resubmitted (RESUBMIT=$RESUBMIT, $RESUBMIT_COUNT/$MAX_RESUBMITS done): run 'sbatch job.fir.sh' to continue"
    fi
    log "$FINAL_NOTE"
    exit 0
}

trap on_exit EXIT
trap on_time_limit USR1
trap 'FINAL_NOTE="interrupted"; exit 130' INT
trap 'FINAL_NOTE="killed by Slurm (time limit or scancel)"; exit 143' TERM

# run_step NAME script [args...]: run one step in the background and wait, so the traps still fire.
run_step() {
    local name="$1" status=0 start
    shift
    CURRENT_STEP="$name"
    start=$(date +%s)
    bash "$@" &
    STEP_PID=$!
    wait "$STEP_PID" || status=$?
    STEP_PID=""
    if [ "$status" -ne 0 ]; then
        record_step "$name" "FAILED" $(($(date +%s) - start))
        log "ERROR: $name failed (exit $status), see the lines above."
        exit "$status"
    fi
    record_step "$name" "OK" $(($(date +%s) - start))
    log "$name OK in $(hms $(($(date +%s) - start)))"
    save_outputs
}

# timed NAME function: same bookkeeping for the steps written in this file
timed() {
    local name="$1" start
    shift
    CURRENT_STEP="$name"
    start=$(date +%s)
    "$@"
    record_step "$name" "OK" $(($(date +%s) - start))
    log "$name OK in $(hms $(($(date +%s) - start)))"
}

key_numbers() {
    "$ROOT_DIR/.venv/bin/python" -I "$ROOT_DIR/scripts/job_summary.py" "$RESULTS_DIR" "$1" || true
}


###############################################################################
# Storage: big folders live on scratch, linked from the repository
###############################################################################

mkdir -p "$STORE_DIR" "$RUN_DIR" "$OLLAMA_INSTALL_DIR" "$OLLAMA_MODELS_DIR"

# link NAME: make $ROOT_DIR/NAME a link to $STORE_DIR/NAME (an existing real folder is left alone)
link_to_scratch() {
    local name="$1"
    if [ -L "$ROOT_DIR/$name" ] || [ ! -e "$ROOT_DIR/$name" ]; then
        mkdir -p "$STORE_DIR/$name"
        ln -sfn "$STORE_DIR/$name" "$ROOT_DIR/$name"
    else
        log "WARNING: $ROOT_DIR/$name is a real folder in home, not on scratch: mind the home quota (48 GB)."
    fi
}
link_to_scratch results                # the 15 GB snapshot, clones, Pharo images
link_to_scratch .snapshot_download     # where the snapshot is downloaded before being installed
export HF_HOME="$STORE_DIR/hf-home"    # Hugging Face cache
export HF_HUB_DISABLE_PROGRESS_BARS=1  # progress bars become one unreadable line in a log file
export PIP_CACHE_DIR="$STORE_DIR/pip-cache"

# The step scripts read these
export MAX_REPOSITORIES LABEL_REPOSITORIES SAMPLE_SIZE MODELS LIMIT ADJUDICATOR CONSENSUS EVIDENCE_POLICY
export SPOTCHECK_SIZE BENCHMARKS AUTHOR SOLVERS OPERATIONS TASKS_PER_OPERATION


###############################################################################
# START
###############################################################################

banner "PHARO-LABELS PIPELINE — $( [ "$MODE" = prepare ] && echo "PREPARATION (login node, no GPU)" || echo "FIR JOB" )"
log "Job:            ${SLURM_JOB_ID:-none} on $(hostname -f 2>/dev/null || hostname)$( [ "$RESUBMIT_COUNT" -gt 0 ] && echo " (continuation $RESUBMIT_COUNT/$MAX_RESUBMITS)")"
log "Account:        ${SLURM_JOB_ACCOUNT:-n/a}   time limit: $(squeue -h -j "${SLURM_JOB_ID:-0}" -o %l 2>/dev/null || echo n/a)"
log "Repository:     $ROOT_DIR (commit $(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo ?)$(git -C "$ROOT_DIR" diff --quiet 2>/dev/null || echo ', with local changes'))"
log "Results:        $RESULTS_DIR -> $(readlink -f "$RESULTS_DIR")"
log "Saved to home:  $SAVE_DIR"

# Python environment and .env (creates .venv on the first run: ~15 minutes on Fir, better done with --prepare)
CURRENT_STEP="python environment"
if [ ! -x "$ROOT_DIR/.venv/bin/pharo-miner" ]; then
    log "Creating the Python environment (.venv): this takes ~15 minutes on Fir (pydantic is compiled)"
fi
module load python/3.11 >/dev/null 2>&1 || true
source "$ROOT_DIR/scripts/_common.sh"   # set -euo pipefail, cd to the repository, .env, .venv, $MINER

if [ "$PUSH_SNAPSHOT" = "auto" ]; then
    if [ -n "${HF_TOKEN:-}" ]; then PUSH_SNAPSHOT="yes"; else PUSH_SNAPSHOT="no"; fi
fi
export PUSH_SNAPSHOT

banner "SETTINGS"
{
    echo "MODE=$MODE TRIAL=$TRIAL"
    echo "MAX_REPOSITORIES=$MAX_REPOSITORIES LABEL_REPOSITORIES=$LABEL_REPOSITORIES SAMPLE_SIZE=$SAMPLE_SIZE"
    echo "MODELS=\"$MODELS\" LIMIT=$LIMIT LABEL_CONCURRENCY=$LABEL_CONCURRENCY"
    echo "ADJUDICATOR=\"$ADJUDICATOR\" CONSENSUS=$CONSENSUS EVIDENCE_POLICY=$EVIDENCE_POLICY SPOTCHECK_SIZE=$SPOTCHECK_SIZE"
    echo "BENCHMARKS=\"$BENCHMARKS\""
    echo "RUN_BENCHMARK=$RUN_BENCHMARK AUTHOR=$AUTHOR SOLVERS=\"$SOLVERS\" OPERATIONS=$OPERATIONS"
    echo "TASKS_PER_OPERATION=$TASKS_PER_OPERATION BENCHMARK_LIMIT=$BENCHMARK_LIMIT BENCHMARK_CONCURRENCY=$BENCHMARK_CONCURRENCY"
    echo "PUSH_SNAPSHOT=$PUSH_SNAPSHOT RESUBMIT=$RESUBMIT MAX_RESUBMITS=$MAX_RESUBMITS"
    echo "OLLAMA_PARALLEL=$OLLAMA_PARALLEL OLLAMA_MAX_LOADED=$OLLAMA_MAX_LOADED STORE_DIR=$STORE_DIR"
    echo "HF_TOKEN=$([ -n "${HF_TOKEN:-}" ] && echo set || echo 'not set (nothing is uploaded)')  GITHUB_TOKEN=$([ -n "${GITHUB_TOKEN:-}" ] && echo set || echo 'not set')"
} | tee "$RUN_DIR/settings.txt" | sed 's/^/    /'
echo "    AI labelling: $([ "$LABEL_REPOSITORIES" = "0" ] && echo "a sample of $SAMPLE_SIZE methods" || echo "every method of $LABEL_REPOSITORIES repositories")$([ "$LIMIT" = "0" ] || echo " (only the first $LIMIT)")"
echo "    Benchmark:    $([ "$RUN_BENCHMARK" = yes ] && echo "author $AUTHOR, solvers $SOLVERS, $([ "$BENCHMARK_LIMIT" = 0 ] && echo 'all candidates' || echo "first $BENCHMARK_LIMIT candidates")" || echo skipped)"


###############################################################################
# 0. Checks: fail in seconds, not after hours
###############################################################################

preflight() {
    banner "0. SYSTEM AND CHECKS"
    echo "    $(uname -srm)"
    lscpu 2>/dev/null | grep -E '^Model name|^CPU\(s\)' | sed 's/^/    /' || true
    echo "    CPUs for this job: ${SLURM_CPUS_PER_TASK:-$(nproc)}   memory: ${SLURM_MEM_PER_NODE:+$((SLURM_MEM_PER_NODE / 1024)) GB}"
    df -h "$STORE_DIR" "$ROOT_DIR" 2>/dev/null | sed 's/^/    /' || true
    timeout 60 diskusage_report 2>/dev/null | grep -E 'home|scratch' | sed 's/^/    quota /' || true

    if [ "$MODE" = "job" ]; then
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/    GPU: /'
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$RUN_DIR/gpu.txt"
        else
            log "ERROR: no GPU visible. The models would run on the CPU and take days: check --gpus-per-node."
            exit 1
        fi
    fi

    for tool in curl unzip rsync; do
        command -v "$tool" >/dev/null || { log "ERROR: $tool is required"; exit 1; }
    done
    "$MINER" llm-annotate --print-requirements --models "$MODELS" --adjudicator "${ADJUDICATOR:-none}" >/dev/null \
        || { log "ERROR: MODELS/ADJUDICATOR are not usable (see above)"; exit 1; }
    if [ "$RUN_BENCHMARK" = "yes" ]; then
        "$MINER" benchmark-generate --print-requirements --author "$AUTHOR" --solvers "$SOLVERS" >/dev/null \
            || { log "ERROR: AUTHOR/SOLVERS are not usable (see above)"; exit 1; }
    fi
    log "Models and tools: OK"
}
timed "0. checks" preflight


###############################################################################
# GPU monitoring and heartbeat (in the background)
###############################################################################

if [ "$MODE" = "job" ]; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv -l 30 > "$RUN_DIR/gpu-usage.csv" 2>/dev/null &
    GPU_MON_PID=$!
fi

if [ "$HEARTBEAT_MINUTES" != "0" ]; then
    (
        trap - EXIT USR1 INT TERM
        while sleep $((HEARTBEAT_MINUTES * 60)); do
            gpu=""
            if [ "$MODE" = "job" ]; then
                gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
                    | awk -F', ' '{printf "GPU %s%%, %.0f/%.0f GB", $1, $2/1024, $3/1024}' || true)"
            fi
            log "[heartbeat] running $(hms $(($(date +%s) - JOB_START)))${gpu:+ | $gpu} | load $(cut -d' ' -f1 /proc/loadavg) | scratch $(du -sh "$OLLAMA_MODELS_DIR" 2>/dev/null | cut -f1) models"
        done
    ) &
    HEARTBEAT_PID=$!
fi


###############################################################################
# 1. Mining and 1-2. Filtering (no model needed)
###############################################################################

banner "1. MINING (Hugging Face snapshot, else GitHub)"
log "Uses the local results if they exist, else downloads the snapshot (15 GB), else mines GitHub."
run_step "1. mining" "$ROOT_DIR/1-mining.sh" ${ORIGINAL_ARGS[@]+"${ORIGINAL_ARGS[@]}"}
# The downloaded snapshot is installed in results/ by now: its download copy is no longer needed.
find "$STORE_DIR/.snapshot_download" -mindepth 1 -delete 2>/dev/null || true
log "Mined data: $(du -sh "$(readlink -f "$RESULTS_DIR")" 2>/dev/null | cut -f1) in results/ ($("$MINER" snapshot state 2>/dev/null))"

banner "1-2. FILTERING: the study corpus"
run_step "1-2. filtering" "$ROOT_DIR/1-2-filtering.sh"
log "Key numbers:"
key_numbers corpus


###############################################################################
# Ollama: install (reused from scratch when present), start, pull the models
###############################################################################

install_ollama() {
    banner "OLLAMA: INSTALL"
    OLLAMA_BIN="$(find "$OLLAMA_INSTALL_DIR" -type f -name ollama -perm -u+x 2>/dev/null | head -n 1)"
    if [ -n "$OLLAMA_BIN" ]; then
        log "Already installed on scratch: $OLLAMA_BIN"
    else
        local archive="$OLLAMA_INSTALL_DIR/ollama-linux-amd64.tar.zst"
        log "Downloading Ollama (1.4 GB; can take 30 minutes on Fir, done once and kept on scratch)"
        curl -fsSL --retry 3 "https://ollama.com/download/ollama-linux-amd64.tar.zst" -o "$archive"
        log "Extracting"
        zstd -dc "$archive" | tar -xf - -C "$OLLAMA_INSTALL_DIR"
        rm -f "$archive"
        OLLAMA_BIN="$(find "$OLLAMA_INSTALL_DIR" -type f -name ollama -perm -u+x | head -n 1)"
        [ -n "$OLLAMA_BIN" ] || { log "ERROR: Ollama executable not found after extraction"; exit 1; }
    fi
    export PATH="$(dirname "$OLLAMA_BIN"):$PATH"
    log "Ollama $("$OLLAMA_BIN" --version 2>&1 | grep -o '[0-9][0-9.]*' | tail -n 1)"
}

show_ollama_log() { log "Last lines of the Ollama log ($OLLAMA_LOG):"; tail -n 40 "$OLLAMA_LOG" | sed 's/^/    /' || true; }

start_ollama() {
    banner "OLLAMA: START"
    # GPU nodes are shared: use the default port unless someone else's Ollama is on it.
    local port=11434 p
    if (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null; then
        for p in $(shuf -i 20000-60000 -n 50); do
            if ! (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null; then port=$p; break; fi
        done
        log "Port 11434 is taken on this node (another user?): using $port"
    fi
    export OLLAMA_HOST="127.0.0.1:$port"
    export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
    export OLLAMA_NUM_PARALLEL="$OLLAMA_PARALLEL"
    export OLLAMA_MAX_LOADED_MODELS="$OLLAMA_MAX_LOADED"
    export OLLAMA_KEEP_ALIVE="24h"
    export OLLAMA_LOAD_TIMEOUT="30m"
    # Flash attention and an 8-bit KV cache halve the memory of 8 parallel 16k-token contexts:
    # llama3.3:70b then fits on the H100 with 8 requests at once.
    export OLLAMA_FLASH_ATTENTION=1
    export OLLAMA_KV_CACHE_TYPE=q8_0
    env | grep '^OLLAMA_' | sort | sed 's/^/    /' | tee "$RUN_DIR/ollama-env.txt"

    # Ollama writes several lines per request (13 GB in two hours in heuristics-vs-llm): the filter
    # drops them unless they report a problem and caps the log at 200 MB.
    "$OLLAMA_BIN" serve > >(awk -v limit_mb="${OLLAMA_LOG_MAX_MB:-200}" \
        -f "$ROOT_DIR/scripts/ollama-log-filter.awk" > "$OLLAMA_LOG") 2>&1 &
    OLLAMA_PID=$!
    local i
    for i in $(seq 1 120); do
        if curl -fsS "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
            log "Ollama is ready (PID $OLLAMA_PID, log $OLLAMA_LOG)"
            [ "$MODE" = "job" ] && { grep -iE 'inference compute|library=cuda|gpu' "$OLLAMA_LOG" | tail -n 3 | sed 's/^/    /' || true; }
            return 0
        fi
        kill -0 "$OLLAMA_PID" 2>/dev/null || { log "ERROR: Ollama crashed at start"; show_ollama_log; exit 1; }
        sleep 1
    done
    log "ERROR: Ollama did not answer within 120 s"; show_ollama_log; exit 1
}

pull_models() {
    banner "OLLAMA: MODELS"
    local tags=() kind value
    while read -r kind value; do
        [ "$kind" = "ollama" ] && tags+=("$value")
    done <<< "$("$MINER" llm-annotate --print-requirements --models "$MODELS" --adjudicator "${ADJUDICATOR:-none}")"
    if [ "$RUN_BENCHMARK" = "yes" ]; then
        while IFS=$'\t ' read -r kind value; do
            [ "$kind" = "ollama" ] && [[ " ${tags[*]-} " != *" $value "* ]] && tags+=("$value")
        done <<< "$("$MINER" benchmark-generate --print-requirements --author "$AUTHOR" --solvers "$SOLVERS")"
    fi
    local m pid status waited
    for m in ${tags[@]+"${tags[@]}"}; do
        if "$OLLAMA_BIN" list | awk 'NR > 1 { print $1 }' | grep -Fxq -e "$m" -e "$m:latest"; then
            log "Already downloaded: $m"
            continue
        fi
        log "Downloading $m (progress every 2 minutes)"
        local pull_log="$RUN_DIR/pull-${m//[:\/]/_}.log"
        "$OLLAMA_BIN" pull "$m" > "$pull_log" 2>&1 &
        pid=$!
        waited=0
        while kill -0 "$pid" 2>/dev/null; do
            sleep 10; waited=$((waited + 10))
            if [ $((waited % 120)) -eq 0 ]; then
                log "  $m: $(tr '\r' '\n' < "$pull_log" | grep -E '[0-9]+%' | tail -n 1 | sed $'s/\x1b\\[[0-9;?]*[A-Za-z]//g; s/  */ /g')"
            fi
        done
        status=0; wait "$pid" || status=$?
        if [ "$status" -ne 0 ]; then
            log "ERROR: downloading $m failed:"; tr '\r' '\n' < "$pull_log" | grep -v '^\s*$' | tail -n 5 | sed 's/^/    /'
            df -h "$OLLAMA_MODELS_DIR" | sed 's/^/    /'
            exit 1
        fi
        log "  $m downloaded in $(hms "$waited")"
    done
    log "Models on scratch ($OLLAMA_MODELS_DIR):"
    "$OLLAMA_BIN" list | sed 's/^/    /' | tee "$RUN_DIR/ollama-models.txt"
}

timed "ollama install" install_ollama
timed "ollama start" start_ollama
timed "ollama models" pull_models


###############################################################################
# Preparation mode stops here
###############################################################################

if [ "$MODE" = "prepare" ]; then
    FINAL_STATE="PREPARED"
    FINAL_NOTE="everything is downloaded: now run   sbatch job.fir.sh"
    exit 0
fi

# Load the first annotator now so step 2 does not start with a long silent wait, and show where it runs.
warm_up() {
    banner "OLLAMA: GPU CHECK"
    local first; first="$("$MINER" llm-annotate --print-requirements --models "${MODELS%% *}" --adjudicator none | awk '$1 == "ollama" { print $2; exit }')"
    [ -n "$first" ] || return 0
    log "Loading $first and asking one question"
    local t0; t0=$(date +%s)
    curl -fsS --max-time 1800 "http://$OLLAMA_HOST/api/generate" \
        -d "{\"model\": \"$first\", \"prompt\": \"Reply with OK.\", \"stream\": false, \"options\": {\"num_predict\": 5}}" \
        | "$ROOT_DIR/.venv/bin/python" -I -c 'import json,sys; r=json.load(sys.stdin); print("    answer:", r.get("response","").strip()[:40], "| tokens/s: %.1f" % (r["eval_count"]/max(r["eval_duration"],1)*1e9) if r.get("eval_count") else "")' \
        || { log "ERROR: the model did not answer"; show_ollama_log; exit 1; }
    log "Loaded and answered in $(hms $(($(date +%s) - t0)))"
    "$OLLAMA_BIN" ps | sed 's/^/    /'
    if "$OLLAMA_BIN" ps | grep -q 'CPU'; then
        log "WARNING: part of the model runs on the CPU (see PROCESSOR above): labelling will be much slower."
    fi
}
timed "gpu check" warm_up


###############################################################################
# 2. AI labelling
###############################################################################

banner "2. AI LABELLING"
log "The models run one after the other; progress and remaining time are printed as they go."
CONCURRENCY="$LABEL_CONCURRENCY" run_step "2. AI labelling" "$ROOT_DIR/2-ai-labelling.sh"
log "Key numbers:"
key_numbers labelling


###############################################################################
# 3. Results
###############################################################################

banner "3. RESULTS: taxonomy, statistics, reports"
run_step "3. results" "$ROOT_DIR/3-results.sh"
key_numbers results


###############################################################################
# 4. Benchmark
###############################################################################

if [ "$RUN_BENCHMARK" = "yes" ]; then
    banner "4. BENCHMARK GENERATION"
    log "The first run downloads Pharo and builds the validation image; then one line per task."
    LIMIT="$BENCHMARK_LIMIT" CONCURRENCY="$BENCHMARK_CONCURRENCY" run_step "4. benchmark" "$ROOT_DIR/4-benchmark.sh"
    log "Key numbers:"
    key_numbers benchmark
else
    record_step "4. benchmark" "SKIPPED" 0
fi


###############################################################################
# Done
###############################################################################

banner "OLLAMA: FINAL STATE"
"$OLLAMA_BIN" ps | sed 's/^/    /' || true
grep -ciE 'error|panic' "$OLLAMA_LOG" | xargs -I{} echo "    {} error lines in the Ollama log ($OLLAMA_LOG)" || true

FINAL_STATE="SUCCESS"
FINAL_NOTE="start with $SAVE_DIR/results/reports/ecosystem_summary.md"
exit 0
