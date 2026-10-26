#!/bin/bash
# Run in a job on Lille (or Nancy): their internet is ~100 MB/s, Luxembourg's a few MB/s (since 2026-10-21 the
# Luxembourg site is on its backup network link). Downloads every dependency of ./run.sh setup here, fast, then
# copies it to the Luxembourg home (resumable: a stopped copy restarts where it stopped). Submit from Lille:
#   oarsub -l core=2,walltime=10:00:00 "bash $HOME/pharo-labels-pbr/g5k-fetch.sh"
set -eu
V=v0.40.7
OL=/tmp/$USER-ollama                                 # the binary (no GPU libraries needed to download)
export OLLAMA_MODELS=$HOME/ollama-transit OLLAMA_HOST=127.0.0.1:11555
export RSYNC_RSH="ssh -o StrictHostKeyChecking=accept-new"
mkdir -p $OL $OLLAMA_MODELS
[ -x $OL/bin/ollama ] || curl -fsSL https://github.com/ollama/ollama/releases/download/$V/ollama-linux-amd64.tar.zst | zstd -d | tar -x -C $OL
curl -sf http://$OLLAMA_HOST/api/tags >/dev/null || { nohup $OL/bin/ollama serve > /tmp/$USER-ollama-serve.log 2>&1 & sleep 5; }
for m in qwen3.8:27b gemma4:31b mistral-small3.2:24b nomic-embed-text; do
  echo "== pull $m"; $OL/bin/ollama pull $m >/dev/null
done
# Pharo image, image globals, benchmarks, corpus snapshot: the setup steps after the models
cd $HOME/pharo-labels-pbr
[ -x .venv/bin/python ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
.venv/bin/python - <<"PY"
from pbr.common import load_config, get_logger
from pbr import setup as S
cfg = load_config(); log = get_logger(cfg, "00 setup", "00_setup.log")
for fn in (S.pharo_image, S.image_globals, S.data):
    fn(cfg, log)
PY
echo "== downloads done: copying to Luxembourg"
copy() { for _ in $(seq 20); do rsync -a --partial --append-verify "$@" && return 0; sleep 30; done; return 1; }
copy data/ luxembourg:pharo-labels-pbr/data/
echo "== data copied"
copy $OLLAMA_MODELS/blobs/ luxembourg:ollama/models/blobs/
copy $OLLAMA_MODELS/manifests/ luxembourg:ollama/models/manifests/   # last: a model is listed only once complete
$RSYNC_RSH luxembourg "touch ~/pharo-labels-pbr/.deps-ready"   # g5k-job.sh waits for it
echo "== ALL DONE"
