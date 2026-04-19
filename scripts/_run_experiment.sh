#!/usr/bin/env bash
# One experiment iteration: sync eval script to pod, run it, echo the METRICS line.
# Expects GUTCHECK_POD_HOST and GUTCHECK_POD_PORT (SSH) in the environment.
set -e
HOST="${GUTCHECK_POD_HOST:?set GUTCHECK_POD_HOST to your pod IP or hostname}"
PORT="${GUTCHECK_POD_PORT:-22}"
REMOTE_ROOT="${GUTCHECK_POD_REMOTE_ROOT:-/workspace/gutcheck}"
VENV_ACTIVATE="${GUTCHECK_POD_VENV:-/workspace/sam31env/bin/activate}"
scp -P "$PORT" -q scripts/eval_sam31_zeroshot.py "root@${HOST}:${REMOTE_ROOT}/scripts/eval_sam31_zeroshot.py"
ssh -p "$PORT" "root@${HOST}" \
  "source '${VENV_ACTIVATE}' && source /root/.hf_env && cd '${REMOTE_ROOT}' && python scripts/eval_sam31_zeroshot.py 2>&1 | tail -50"
