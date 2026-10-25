#!/bin/bash
# Login node only (compute nodes have no internet): every dependency, then stop.
cd $SCRATCH/pharo-labels && source trillium.env
export OLLAMA_HOST=127.0.0.1:11555                   # a private port: login nodes are shared
curl -sf http://$OLLAMA_HOST/api/tags >/dev/null || { nohup ollama serve > $SCRATCH/ollama/serve-login.log 2>&1 & sleep 5; }
./run.sh setup
