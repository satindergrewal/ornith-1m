#!/bin/bash
# launch-supplement.sh - supplemental capture run (fills dead-expert coverage)
# then merge with the main store and re-seal ONE manifest.
set -u
for pid in $(pgrep -f "dsv4_capture.py" 2>/dev/null); do
  [ "$pid" != "$$" ] && kill -9 "$pid" 2>/dev/null
done
sleep 1
rm -rf /workspace/calibration/main-supplement
cd /wd
TC_BATCH_ROWS="${TC_BATCH_ROWS:-8000}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup /usr/local/bin/python -u dsv4_capture.py \
    --corpus /wd/data/corpus-supplement.jsonl \
    --out /workspace/calibration/main-supplement \
    --target-tokens "${TARGET_TOKENS:-120000}" \
    > /wd/logs/capture-supplement.log 2>&1 &
echo "SUPPLEMENT-LAUNCHED pid $!"
