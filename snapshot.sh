#!/usr/bin/env bash
# Save or restore ALL results (database, parsed methods, annotations, AI labels, reports) on
# Hugging Face. The dataset is set in config.yaml (snapshot.repo, default pharo-llm/pharo-labels).
#
#   ./snapshot.sh push              upload the current results (a new version, tagged snapshot-<date>)
#   ./snapshot.sh push --dry-run    show what would be uploaded, upload nothing
#   ./snapshot.sh pull              download the latest snapshot (only when there are no local results)
#   ./snapshot.sh pull --force      replace local results (they are kept as results.before-pull-<date>)
#   ./snapshot.sh pull --revision snapshot-20261009-153000    a specific older version
#   ./snapshot.sh info              what is local, what is online, list of versions
source "$(dirname "$0")/scripts/_common.sh"
[ $# -ge 1 ] || { sed -n '2,11p' "$0"; exit 1; }
if [ "$1" = "push" ] && [[ " $* " != *" --dry-run "* ]]; then
  ask_secret HF_TOKEN "Hugging Face token with write access (https://huggingface.co/settings/tokens)"
fi
"$MINER" snapshot "$@"
