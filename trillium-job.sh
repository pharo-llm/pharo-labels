#!/bin/bash
# The whole pipeline (phases 1-13, PROFILE=full) on Trillium GPUs. Setup must have run on a login node first
# (./trillium-setup.sh: compute nodes have no internet). Submit: sbatch trillium-job.sh
# A job lasts at most 24 h: each job queues its follow-up at start (it resumes where this one stopped; every
# LLM answer is cached) and cancels it when the pipeline ends by itself (done, failed or waiting for experts).
#SBATCH --job-name=pharo-labels
#SBATCH --account=def-foutsekh
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/oabedelk/pharo-labels/slurm-%j.out
set -u
cd $SCRATCH/pharo-labels && source trillium.env
CHAIN=${CHAIN:-1}
if [ "$CHAIN" -lt 6 ]; then
  NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID --export=ALL,CHAIN=$((CHAIN + 1)) trillium-job.sh)
  echo "follow-up job $NEXT queued (runs only if this one is stopped by the time limit)"
fi
export OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=${LLM_PARALLEL:-8} OLLAMA_MAX_LOADED_MODELS=3 OLLAMA_KEEP_ALIVE=-1
export LLM_PARALLEL=${LLM_PARALLEL:-8} MODELS_PARALLEL=3 SKIP_SETUP=1
nvidia-smi -L
ollama serve > $SCRATCH/ollama/serve-$SLURM_JOB_ID.log 2>&1 &
for _ in $(seq 60); do curl -sf http://$OLLAMA_HOST/api/tags >/dev/null && break; sleep 2; done
./run.sh status
./run.sh
code=$?
echo "pipeline exit code $code"
./run.sh status
[ -n "${NEXT:-}" ] && scancel "$NEXT" && echo "follow-up $NEXT cancelled: the pipeline ended by itself"
exit $code
