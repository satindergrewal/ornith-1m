#!/bin/bash
# launch-probe-range.sh START END - sequential probe over a layer range
set -u
LO="${1:?start}"
HI="${2:?end}"
cd /wd
export PYTHONPATH=/workspace/campaign/src:/workspace/campaign/r10:/wd
export DSV4_WORK_ROOT=/workspace/dsv4-work
export K35_EXLLAMAV3_EXT=/workspace/campaign/exllamav3-src/exllamav3_ext.cpython-312-x86_64-linux-gnu.so
nohup bash -c '
for L in $(seq '"$LO"' '"$HI"'); do
  echo "=== L$L $(date -u +%H:%M) ==="
  /usr/local/bin/python -u dsv4_probe_driver.py --layer "$L" \
    --repo-root /workspace/campaign/r10 >> /wd/logs/probe-sweep.log 2>&1 || {
    echo "PROBE-L$L-FAIL" >> /wd/logs/probe-sweep.log
    break
  }
done
echo "SWEEP-DONE" >> /wd/logs/probe-sweep.log
' > /dev/null 2>&1 &
echo "SWEEP $LO-$HI pid $!"
