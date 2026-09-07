#!/bin/bash
# launch-gss-range.sh START END - sequential phase-6 GSS over a layer range.
# Both rate halves per layer. Requires --transform-seed-sha256 on a fresh
# work root; the seed is fixed campaign-wide (recorded in the runbook).
set -u
LO="${1:?start}"
HI="${2:?end}"
SEED="${GSS_SEED:?set GSS_SEED to the campaign transform seed (64-hex)}"
cd /wd
export PYTHONPATH=/workspace/campaign/src:/workspace/campaign/r10:/wd
export DSV4_WORK_ROOT=/workspace/dsv4-work
export K35_EXLLAMAV3_EXT=/workspace/campaign/exllamav3-src/exllamav3_ext.cpython-312-x86_64-linux-gnu.so
nohup bash -c '
for L in $(seq '"$LO"' '"$HI"'); do
  echo "=== GSS L$L $(date -u +%H:%M) ==="
  /usr/local/bin/python -u dsv4_phase6_gss.py --layers "$L" \
    --repo-root /workspace/campaign/r10 \
    --transform-seed-sha256 '"$SEED"' \
    >> /wd/logs/gss-sweep.log 2>&1 || {
    echo "GSS-L$L-FAIL" >> /wd/logs/gss-sweep.log
    break
  }
done
echo "GSS-SWEEP-DONE" >> /wd/logs/gss-sweep.log
' > /dev/null 2>&1 &
echo "GSS $LO-$HI pid $!"
