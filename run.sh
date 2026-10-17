#!/usr/bin/env bash
# PharoTaskMiner — from GitHub to a validated taxonomy of the programming tasks of Pharo code.
#
#   ./run.sh              everything, in order (each step resumes where it stopped):
#     1. mine     GitHub search -> Pharo verification -> FILTER 1 (6 criteria) -> clone -> parse
#                 (downloads the Hugging Face snapshot instead when there are no local results)
#     2. filter   FILTER 2 (13 criteria): the study corpus, ~135 repositories / ~280,000 methods
#     3. label    sample-size calculator -> small model labels the sample (open coding) ->
#                 saturation loop over new methods -> check model -> expert-validation export
#     4. report   results/REPORT.md: every number of steps 1-3, with figures
#   ./run.sh mine | filter | label | report     one step only
#   ./run.sh status                             what the database holds
#   ./run.sh pull | push                        get / save all results on Hugging Face
#
# Everything goes to results/ (start with results/REPORT.md); the experts' files are in
# results/labelling/expert_validation/. Tokens are asked once and kept in .env (git-ignored).

# ============================== SETTINGS ===================================================
# --- step 1: mining
USE_SNAPSHOT="${USE_SNAPSHOT:-auto}"        # auto = local results, else the Hugging Face snapshot, else mine
                                            # no   = always mine from GitHub (needs a GitHub token)
SNAPSHOT_VERSION="${SNAPSHOT_VERSION:-}"    # "" = latest, or a tag like snapshot-20261010-182354
ALLOW_GITHUB_MINING="${ALLOW_GITHUB_MINING:-yes}"   # no = never clone or search GitHub

# --- step 3: labelling
# The model that labels. An Ollama tag (Pharo models: pharo-llm/Qwen2.5-Coder-SFT:0.5B, :1.5B,
# :3b, :7b), or a Hugging Face model as hf:<repo>[@revision], e.g. the continued-pretraining
# adapters hf:pharo-llm/Qwen2.5-Coder-1.5B-PT (0.5B, 1.5B, 3B, 7B; the last checkpoint is used).
MODEL="${MODEL:-pharo-llm/Qwen2.5-Coder-SFT:1.5B}"
# A second model codes the review subset on its own; their agreement is in the report. none = skip.
CHECK_MODEL="${CHECK_MODEL:-pharo-llm/Qwen2.5-Coder-SFT:7b}"
CONFIDENCE="${CONFIDENCE:-0.95}"            # sample-size calculator: confidence level
MARGIN="${MARGIN:-0.05}"                    #                         margin of error
SAMPLE_SIZE="${SAMPLE_SIZE:-}"              # "" = computed (384 for 280,000 methods at 95% / ±5%)
BATCH_SIZE="${BATCH_SIZE:-50}"              # methods drawn per saturation round
PATIENCE="${PATIENCE:-}"                    # rounds in a row without a new category = saturation; "" = derived:
                                            # long enough that a task of MIN_SHARE prevalence would have appeared
                                            # with 95% probability (12 rounds of 50 methods for 0.5%)
MAX_ROUNDS="${MAX_ROUNDS:-40}"              # safety cap on saturation rounds
MIN_SUPPORT="${MIN_SUPPORT:-2}"             # methods a name needs to become a category ...
MIN_SHARE="${MIN_SHARE:-0.005}"             # ... and its share of the labelled methods (0.5%): every task
                                            # with at least this prevalence is in the taxonomy
# Small embedding model that merges task names with the same meaning ("Calculation" = "Computation").
# Without it, the open codes of a small model never saturate. none = merge by spelling only.
EMBED_MODEL="${EMBED_MODEL:-all-minilm}"

# --- sharing
PUSH_SNAPSHOT="${PUSH_SNAPSHOT:-no}"        # yes = upload all results to Hugging Face at the end
# Every setting keeps the value above unless a variable of the same name is set in the environment.
# The thresholds of the two filters are in config.yaml (repositories.quality, corpus_filter).
# ===========================================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
MINER="$ROOT/.venv/bin/pharo-miner"
LOG=results/run.log
if [ -f .env ]; then . ./.env; fi

# ---------------------------------------------------------------------------- helpers
ask_secret() {   # ask_secret VAR "description": ask once (hidden), keep it in .env
  local var="$1" what="$2" val
  if [ -n "${!var:-}" ]; then export "$var"; return 0; fi
  if ! { : < /dev/tty; } 2>/dev/null; then
    echo "$var is required ($what) but there is no terminal to ask on. Add 'export $var=...' to $ROOT/.env"; exit 1
  fi
  read -rsp "$what — $var (input hidden): " val < /dev/tty; echo
  val=$(printf '%s' "$val" | LC_ALL=C sed $'s/\x1b\\[[0-9;?]*[A-Za-z~]//g' | LC_ALL=C tr -d '[:cntrl:][:space:]')
  [ -n "$val" ] || { echo "$var is required."; exit 1; }
  umask 077; printf 'export %s=%q\n' "$var" "$val" >> .env; export "$var=$val"; echo "Saved $var to .env"
}

run_logged() {   # run, show, and append the output to the log; returns the command's status
  set +e; "$@" 2>&1 | tee -a "$LOG"; local status=${PIPESTATUS[0]}; set -e; return "$status"
}

step() { echo; echo "=== $* ($(date '+%Y-%m-%d %H:%M'))" | tee -a "$LOG"; }

setup_python() {   # created on first run, refreshed when pyproject.toml changes
  if [ ! -x .venv/bin/pharo-miner ] || [ pyproject.toml -nt .venv/.installed ]; then
    local py=${PYTHON:-}
    if [ -z "$py" ]; then
      for c in python3.13 python3.12 python3.11 python3; do command -v "$c" >/dev/null && { py=$c; break; }; done
    fi
    "$py" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "Python >= 3.10 is required."; exit 1; }
    [ -x .venv/bin/python ] || { rm -rf .venv; "$py" -m venv .venv; }
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -e .
    touch .venv/.installed
  fi
}

ollama_up() {
  local host="${OLLAMA_HOST:-localhost:11434}"
  command -v ollama >/dev/null || { echo "Ollama is needed for $1: https://ollama.com/download"; exit 1; }
  if ! curl -sf "http://${host#http://}/api/tags" >/dev/null; then
    echo "Starting Ollama (log: results/ollama.log) …"
    nohup ollama serve > results/ollama.log 2>&1 &
    for _ in $(seq 30); do curl -sf "http://${host#http://}/api/tags" >/dev/null && break; sleep 1; done
  fi
  curl -sf "http://${host#http://}/api/tags" >/dev/null || { echo "Ollama does not answer at $host."; exit 1; }
}

prepare_model() {   # make the model usable: Ollama pull, or the Hugging Face packages
  local m="$1"
  case "$m" in
    ""|none|no|off) return 0 ;;
    hf:*)
      .venv/bin/python -c 'import torch, transformers, peft' 2>/dev/null || {
        echo "Installing torch, transformers and peft for $m …"; .venv/bin/pip install -q -e '.[hf]'; } ;;
    *)
      ollama_up "$m"
      ollama list | awk 'NR>1 {print $1}' | grep -qx -e "$m" -e "$m:latest" || { echo "Downloading $m …"; ollama pull "$m"; } ;;
  esac
}

push_snapshot() {
  [ "$PUSH_SNAPSHOT" = "yes" ] || return 0
  ask_secret HF_TOKEN "Hugging Face token with write access (https://huggingface.co/settings/tokens)"
  run_logged "$MINER" snapshot push --message "$1" || echo "Upload failed (results are safe locally): ./run.sh push"
}

no_github() { echo "ERROR: $1"; echo "ALLOW_GITHUB_MINING=no: nothing is mined or cloned from GitHub."; exit 1; }

# ---------------------------------------------------------------------------- steps
do_mine() {
  step "1. Mining and filter 1"
  local state; state=$("$MINER" snapshot state)
  if [ "$USE_SNAPSHOT" = "auto" ]; then
    if [ "$state" = "none" ]; then
      local rev=(); [ -n "$SNAPSHOT_VERSION" ] && rev=(--revision "$SNAPSHOT_VERSION")
      echo "No local results: downloading the mining snapshot from Hugging Face (about 16 GB) …"
      if run_logged "$MINER" snapshot pull ${rev[@]+"${rev[@]}"}; then state=$("$MINER" snapshot state)
      else echo "The download failed: mining from GitHub instead."; fi
    fi
    if [ "$state" = "stale" ]; then
      echo "Filter 1 settings in config.yaml changed: selecting the repositories again."
      run_logged "$MINER" sample || exit 1
      state=$("$MINER" snapshot state)
    fi
    if [ "$state" = "complete" ]; then echo "Mining results are here (results/): nothing to mine."; return 0; fi
    if [ "$state" = "selected" ]; then
      [ "$ALLOW_GITHUB_MINING" = "no" ] && no_github "the selected repositories would have to be cloned."
      for s in clone parse analyze classify-domains; do
        run_logged "$MINER" "$s" || { echo "Failed at '$s': rerun ./run.sh to continue (log: $LOG)"; exit 1; }
      done
      push_snapshot "Mining results (new repository selection)"; return 0
    fi
  fi
  [ "$ALLOW_GITHUB_MINING" = "no" ] && no_github "no usable results (state: $state)."
  ask_secret GITHUB_TOKEN "GitHub token (https://github.com/settings/tokens, read-only public access is enough)"
  run_logged "$MINER" pipeline || { echo "Mining stopped: rerun ./run.sh to resume (log: $LOG)"; exit 1; }
  push_snapshot "Mining results"
}

do_filter() {
  step "2. Filter 2: the study corpus"
  run_logged "$MINER" filter-corpus || { echo "Stopped (log: $LOG)"; exit 1; }
}

do_label() {
  step "3. Labelling: sample, saturation loop, expert-validation export"
  prepare_model "$MODEL"
  prepare_model "$CHECK_MODEL"
  prepare_model "$EMBED_MODEL"
  local args=(--model "$MODEL" --check-model "${CHECK_MODEL:-none}" --confidence "$CONFIDENCE" --margin "$MARGIN"
              --batch-size "$BATCH_SIZE" --max-rounds "$MAX_ROUNDS" --min-support "$MIN_SUPPORT" --min-share "$MIN_SHARE"
              --embedding-model "${EMBED_MODEL:-none}")
  [ -n "$SAMPLE_SIZE" ] && args+=(--sample-size "$SAMPLE_SIZE")
  [ -n "$PATIENCE" ] && args+=(--patience "$PATIENCE")
  run_logged "$MINER" label "${args[@]}" || { echo "Stopped: rerun ./run.sh label to continue (log: $LOG)"; exit 1; }
}

do_report() {
  step "4. Report"
  run_logged "$MINER" report
  echo "Open: results/REPORT.md    Experts: results/labelling/expert_validation/README.md"
}

# ---------------------------------------------------------------------------- main
setup_python
mkdir -p results
case "${1:-all}" in
  all)    do_mine; do_filter; do_label; do_report; push_snapshot "Labelling: $MODEL" ;;
  mine)   do_mine ;;
  filter) do_filter ;;
  label)  do_label ;;
  report) do_report ;;
  status) "$MINER" status ;;
  pull)   shift; "$MINER" snapshot pull "$@" ;;
  push)   PUSH_SNAPSHOT=yes; push_snapshot "Results" ;;
  -h|--help|help) sed -n '2,17p' "$0" ;;
  *) echo "Unknown step '$1'."; sed -n '2,17p' "$0"; exit 1 ;;
esac
