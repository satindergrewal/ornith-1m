#!/bin/bash
# launch-merge-rescan.sh - merge supplement into main capture, re-seal,
# then rescan dead-expert coverage across all 43 layers.
set -u
cd /wd
export PYTHONPATH=/wd
rm -rf /workspace/calibration/main-full-merged
/usr/local/bin/python -u dsv4_capture_merge.py \
  --shards /workspace/calibration/main-full /workspace/calibration/main-supplement \
  --out /workspace/calibration/main-full-merged > /wd/logs/merge.log 2>&1 || { echo MERGE-FAIL; exit 1; }
tail -1 /wd/logs/merge.log
/usr/local/bin/python - <<'PYEOF' >> /wd/logs/merge.log 2>&1
import sys
sys.path.insert(0, "/wd")
import dsv4_common as C
import numpy as np
bad = {}
mins = []
for L in range(43):
    cap = C.open_capture(C.DEFAULT_CALIBRATION_ROOT if False else
                         __import__("pathlib").Path("/workspace/calibration/main-full-merged"),
                         L, verify_hashes=False)
    counts = np.zeros(C.NUM_EXPERTS, dtype=np.int64)
    m = (cap.row_roles == "fit")
    ids = cap.ids[m]
    for e in range(C.NUM_EXPERTS):
        counts[e] = int(((ids == e).any(axis=1)).sum())
    z = np.flatnonzero(counts == 0)
    if len(z):
        bad[L] = [int(x) for x in z]
    mins.append(int(counts.min()))
print("MERGED zero-fit experts:", bad if bad else "NONE")
print("fit-min per layer:", mins)
PYEOF
tail -2 /wd/logs/merge.log
