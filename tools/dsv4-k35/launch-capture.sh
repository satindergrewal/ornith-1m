#!/bin/bash
# launch-capture.sh - (re)start the full DSV4 capture on this pod.
# Exists so the ssh command line never contains the process pattern
# (pkill -f from an ssh one-liner kills the ssh's own remote shell).
set -u
PORT=11827
KEY="$HOME/.runpod/ssh/runpodctl-ssh-key"
for pid in $(pgrep -f "dsv4_capture.py" 2>/dev/null); do
  [ "$pid" != "$$" ] && kill -9 "$pid" 2>/dev/null
done
sleep 1
rm -rf /workspace/calibration/main-full
cd /wd
TC_BATCH_ROWS="${TC_BATCH_ROWS:-8000}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup /usr/local/bin/python -u dsv4_capture.py \
    --corpus /wd/data/corpus-v1.jsonl \
    --out /workspace/calibration/main-full \
    --target-tokens "${TARGET_TOKENS:-250000}" \
    > /wd/logs/capture-full.log 2>&1 &
echo "LAUNCHED pid $!"
