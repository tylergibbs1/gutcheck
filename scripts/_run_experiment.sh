#!/usr/bin/env bash
# One experiment iteration: sync eval script to pod, run it, echo the METRICS line.
set -e
HOST=103.207.149.102
PORT=11425
scp -P "$PORT" -q scripts/eval_sam31_zeroshot.py "root@${HOST}:/workspace/gutcheck/scripts/eval_sam31_zeroshot.py"
ssh -p "$PORT" "root@${HOST}" \
  "source /workspace/sam31env/bin/activate && source /root/.hf_env && cd /workspace/gutcheck && python scripts/eval_sam31_zeroshot.py 2>&1 | tail -50"
