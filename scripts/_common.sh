# Shared setup for 1-mining.sh, 2-ai-labelling.sh and 3-results.sh (sourced, not run).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MINER="$ROOT/.venv/bin/pharo-miner"

# Secrets live in .env (git-ignored, readable only by you).
if [ -f .env ]; then . ./.env; fi

# ask_secret VAR "description": ask once (hidden input), then keep it in .env
ask_secret() {
  local var="$1" what="$2" val
  if [ -n "${!var:-}" ]; then export "$var"; return 0; fi
  # Batch jobs (job.oar.sh) have no terminal to ask on: the secret must already be in .env.
  if ! { : < /dev/tty; } 2>/dev/null; then
    echo "$var is required ($what) but there is no terminal to ask on."
    echo "Add this line to $ROOT/.env and rerun:   export $var=..."
    exit 1
  fi
  read -rsp "$what — $var (input hidden): " val < /dev/tty; echo
  # Hidden input also captures stray key presses (arrow keys, bracketed paste): drop escape sequences and control characters.
  val=$(printf '%s' "$val" | LC_ALL=C sed $'s/\x1b\\[[0-9;?]*[A-Za-z~]//g' | LC_ALL=C tr -d '[:cntrl:][:space:]')
  [ -n "$val" ] || { echo "$var is required."; exit 1; }
  umask 077
  printf 'export %s=%q\n' "$var" "$val" >> .env
  export "$var=$val"
  echo "Saved $var to .env"
}

# Python environment (created on first run, refreshed when pyproject.toml changes)
if [ ! -x .venv/bin/pharo-miner ] || [ pyproject.toml -nt .venv/.installed ]; then
  PY=${PYTHON:-python3}
  "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "Python >= 3.10 is required."; exit 1; }
  [ -d .venv ] || "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -e .
  touch .venv/.installed
fi
mkdir -p results

# run_logged LOGFILE cmd...: run, show and append the output to LOGFILE; returns the command's status
run_logged() {
  local logfile="$1"; shift
  set +e
  "$@" 2>&1 | tee -a "$logfile"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}
