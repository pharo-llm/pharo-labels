#!/bin/bash
# Frontend only: Ollama (with its AMD ROCm libraries, for the vianden MI300X GPUs) and every dependency, then stop.
set -eu
cd $HOME/pharo-labels-pbr && source g5k.env
V=v0.40.7                                            # same Ollama as Trillium
if [ ! -x $HOME/ollama/bin/ollama ]; then
  mkdir -p $HOME/ollama
  for f in ollama-linux-amd64.tar.zst ollama-linux-amd64-rocm.tar.zst; do
    curl -fsSL https://github.com/ollama/ollama/releases/download/$V/$f | zstd -d | tar -x -C $HOME/ollama
  done
fi
ollama -v || true
export OLLAMA_HOST=127.0.0.1:11555                   # a private port: the frontend is shared
curl -sf http://$OLLAMA_HOST/api/tags >/dev/null || { nohup ollama serve > $HOME/ollama/serve-frontend.log 2>&1 & sleep 5; }
./run.sh setup
