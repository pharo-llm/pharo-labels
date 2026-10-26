#!/bin/bash
# The whole pipeline (PROFILE=full) on vianden (Grid5000 Luxembourg, 8 AMD MI300X). Its dependencies come from
# Lille (g5k-fetch.sh: Luxembourg's own internet is too slow), the Ollama ROCm build from ./g5k-setup.sh. Reserved with:
#   oarsub -t exotic -p "cluster='vianden'" -l host=1,walltime=14:00:00 -r "YYYY-MM-DD 19:00:00" "bash $HOME/pharo-labels-pbr/g5k-job.sh"
# Stopped by the walltime, it resumes where it stopped at the next run (every LLM answer is cached).
set -u
cd $HOME/pharo-labels-pbr && source g5k.env
# the dependencies come from Lille (g5k-fetch.sh): wait for them rather than fail (it copies the model manifests last)
M=$OLLAMA_MODELS/manifests/registry.ollama.ai/library
until [ -f .deps-ready ] || [ -f $M/qwen3.8/27b -a -f $M/gemma4/31b -a -f $M/mistral-small3.2/24b -a -f $M/nomic-embed-text/latest ]; do echo "$(date +%H:%M) waiting for the dependencies (g5k-fetch.sh)"; sleep 300; done
# one Ollama server per GPU, one model each (a single server puts every model on GPU 0 and leaves the others idle)
pkill -u $USER -x ollama; pkill -u $USER -f lib/ollama/llama-server; sleep 3      # an earlier server of this job
P=${LLM_PARALLEL:-16}
serve() {   # serve <gpu> <port>
  HIP_VISIBLE_DEVICES=$1 OLLAMA_HOST=127.0.0.1:$2 OLLAMA_NUM_PARALLEL=$P OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=-1 \
    nohup ollama serve > $HOME/ollama/serve-$OAR_JOB_ID-gpu$1.log 2>&1 &
}
serve 0 11434; serve 1 11435; serve 2 11436; serve 3 11437
export OLLAMA_HOST=127.0.0.1:11434                                 # qwen (annotator and editor), setup checks
export OLLAMA_HOST_GEMMA4_31B=127.0.0.1:11435 OLLAMA_HOST_MISTRAL_SMALL3_2_24B=127.0.0.1:11436 \
       OLLAMA_HOST_NOMIC_EMBED_TEXT=127.0.0.1:11437                # pbr/llm.py: one server per model
export LLM_PARALLEL=$P MODELS_PARALLEL=3
for p in 11434 11435 11436 11437; do
  for _ in $(seq 60); do curl -sf http://127.0.0.1:$p/api/tags >/dev/null && break; sleep 2; done
done
rocm-smi --showproductname 2>/dev/null | grep -i "card series" || ls /dev/dri
grep -ihE "inference compute|no compatible GPUs" $HOME/ollama/serve-$OAR_JOB_ID-gpu*.log
./run.sh status
./run.sh
code=$?
echo "pipeline exit code $code"
./run.sh status
exit $code
