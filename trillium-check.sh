#!/bin/bash
# 30-minute check on one GPU before the real run: Ollama on the GPU, the four models, parallel requests, the
# Pharo VM and its network sandbox on a compute node. Submit: sbatch trillium-check.sh
#SBATCH --job-name=pharo-labels-check
#SBATCH --account=def-foutsekh
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/oabedelk/pharo-labels/slurm-check-%j.out
cd $SCRATCH/pharo-labels && source trillium.env
export OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=8 OLLAMA_MAX_LOADED_MODELS=3 OLLAMA_KEEP_ALIVE=-1 PROFILE=smoke
nvidia-smi -L
ollama serve > $SCRATCH/ollama/serve-check-$SLURM_JOB_ID.log 2>&1 &
for _ in $(seq 60); do curl -sf http://$OLLAMA_HOST/api/tags >/dev/null && break; sleep 2; done
.venv/bin/python - <<"PY"
import time
from pbr.common import load_config, pmap
from pbr.llm import Client, annotators
cfg = load_config()
cfg["project"]["cache_dir"] = "/tmp/pbr-check-cache"           # never pollutes the real cache
sch = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
for c in annotators(cfg):
    i = c.info(); t = time.time()
    a = pmap(lambda k: c.ask(system="Answer with JSON only.", user=f"Give x = {k} + 2", schema=sch, max_tokens=30,
                             item_id=f"check{k}"), range(8), 8)
    print(c.name, i.get("digest"), i.get("quantization"), "8 parallel answers:", [x.data["x"] for x in a],
          f"{time.time() - t:.1f}s", flush=True)
e = Client({"name": "embedder", **cfg["llm"]["embedder"]}, cfg).embed(["a test"])
print("embedder dim", len(e[0]))
from pbr.pharo import Runner
r = Runner(cfg, print)
print("pharo sandbox:", r.sandbox, "| image", r.info.get("system_version"))
PY
cd data/pharo-image && ./pharo PharoBench.image eval "3 + 4"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
