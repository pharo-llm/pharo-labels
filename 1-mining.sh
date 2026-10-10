#!/usr/bin/env bash
# STEP 1 — Mining: find Pharo repositories on GitHub, verify them, collect metadata, sample,
# clone, parse every method, classify operations with rules, and export the 300-method
# annotation sample.
#
# It does NOT mine again when it does not have to:
#   1. results already on this computer        -> they are used as they are
#   2. otherwise, a snapshot on Hugging Face   -> it is downloaded and used (minutes, not hours)
#   3. otherwise                               -> mining from GitHub (hours), then uploaded as a snapshot
# In cases 1 and 2, if the results were made with other selection settings than config.yaml has
# now (repositories.quality, sampling), the repositories are selected again and only cloning,
# parsing and the annotation sample are redone: no GitHub search, no GitHub token.
# The Hugging Face dataset is set in config.yaml (snapshot.repo, default pharo-llm/pharo-labels).
#
#   ./1-mining.sh                      as above
#   ./1-mining.sh --size 300           mining options (applied only when mining actually runs)
#
# Output: results/  (log: results/run.log). Next: ./2-ai-labelling.sh

# ============================== SETTINGS ===================================================
# auto = use local results, else the Hugging Face snapshot, else mine
# no   = always mine (resumes an interrupted mining run; never downloads a snapshot)
USE_SNAPSHOT="${USE_SNAPSHOT:-auto}"

# Snapshot version to download: "" = latest; or a tag such as snapshot-20261009-153000
# (all versions: ./snapshot.sh info)
SNAPSHOT_VERSION="${SNAPSHOT_VERSION:-}"

# yes = after mining, upload the results to Hugging Face so the next run (or a colleague) can
#       skip mining. Needs a Hugging Face token with write access (asked once, kept in .env).
PUSH_SNAPSHOT="${PUSH_SNAPSHOT:-yes}"
# Each setting keeps the value above unless a variable of the same name is already set in the
# environment (this is how job.oar.sh configures a cluster run).
# ===========================================================================================

source "$(dirname "$0")/scripts/_common.sh"
LOG=results/run.log

finish_ok() {
  echo "The 300-method sample is in results/validation/annotation_sample.csv."
  echo "Next: ./2-ai-labelling.sh   (or have people fill results/validation/reviewer_decisions.csv)"
  exit 0
}

push_snapshot() {
  [ "$PUSH_SNAPSHOT" = "yes" ] || return 0
  ask_secret HF_TOKEN "Hugging Face token with write access (https://huggingface.co/settings/tokens)"
  run_logged "$LOG" "$MINER" snapshot push --message "$1" \
    || echo "Upload to Hugging Face failed (results are safe locally). Retry later with: ./snapshot.sh push"
}

state=$("$MINER" snapshot state)
if [ "$USE_SNAPSHOT" = "auto" ]; then
  if [ "$state" = "none" ]; then
    echo "No local results: looking for a snapshot on Hugging Face …"
    rev_args=()
    [ -n "$SNAPSHOT_VERSION" ] && rev_args=(--revision "$SNAPSHOT_VERSION")
    if "$MINER" snapshot info ${rev_args[@]+"${rev_args[@]}"} 2>/dev/null | grep -q '"remote_snapshot": true'; then
      if run_logged "$LOG" "$MINER" snapshot pull ${rev_args[@]+"${rev_args[@]}"}; then
        echo
        echo "Snapshot downloaded and installed: no GitHub search needed."
        state=$("$MINER" snapshot state)
      else
        echo "Downloading the snapshot failed (see above); falling back to mining."
      fi
    else
      echo "No snapshot found: mining from GitHub."
    fi
  fi
  if [ "$state" = "stale" ]; then
    echo "The selection settings in config.yaml differ from the ones these results were made with: selecting the repositories again."
    run_logged "$LOG" "$MINER" sample || { echo "Failed: see $LOG"; exit 1; }
    state=$("$MINER" snapshot state)
    [ "$state" = "complete" ] && echo "The same repositories are selected: nothing to redo."
  fi
  if [ "$state" = "complete" ]; then
    echo "Mining results exist on this computer (results/): using them, nothing to mine."
    [ $# -gt 0 ] && echo "Note: options '$*' were ignored. To mine again: USE_SNAPSHOT=\"no\" in this file (or move results/ away)."
    finish_ok
  fi
  if [ "$state" = "selected" ]; then
    echo "Cloning and parsing the selected repositories (clones that already exist are reused) …"
    for stage in clone parse analyze classify-domains discover-operations export-annotations; do
      run_logged "$LOG" "$MINER" "$stage" || { echo "Failed at '$stage'. Rerun ./1-mining.sh to continue; see $LOG"; exit 1; }
    done
    push_snapshot "Mining results (new repository selection)"
    finish_ok
  fi
  [ "$state" = "partial" ] && echo "An interrupted mining run was found in results/: resuming it."
fi

ask_secret GITHUB_TOKEN "GitHub token (https://github.com/settings/tokens, read-only public access is enough)"
status=0
run_logged "$LOG" "$MINER" pipeline "$@" || status=$?
echo
case $status in
  0|3)
    push_snapshot "Mining results"
    finish_ok ;;
  *) echo "Failed (exit $status). Rerun ./1-mining.sh to resume; see $LOG"; exit "$status" ;;
esac
