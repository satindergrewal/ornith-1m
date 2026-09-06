#!/bin/bash
# fs2_resume.sh - resume the GLM cap chain from regen (workers died to
# unbounded pack OOM; glm_tc_batch_gen.py now chunks forwards at
# TC_MAX_PACK_TOKENS). Existing fs_prob shards reused; merge/train/gate
# replicates glm_fs2.sh tail verbatim.
set -uo pipefail
cd /root/tc-glm
PY=/usr/local/bin/python
echo "[gfs2r] start $(date -Is)"

for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TC_MAX_PACK_TOKENS=32768 $PY -u glm_tc_batch_gen.py \
    --problems data/fs_prob_s$i.jsonl \
    --samples 3 --max-new 160 --sample-seed $((20260220 + i)) \
    --out data/fs_s$i.jsonl --tag fs_s$i > logs/fs_regen_s$i.log 2>&1 &
done
wait
echo "[gfs2r] regen shards done $(date -Is)"
for i in 0 1 2 3; do tail -2 logs/fs_regen_s$i.log; done

cat data/fs_s0.jsonl data/fs_s1.jsonl data/fs_s2.jsonl data/fs_s3.jsonl > data/regen.jsonl
echo "[gfs2r] merged regen rows: $(grep -c '' data/regen.jsonl)"

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -u glm_tc_lora_train.py > logs/fs_train.log 2>&1
TR=$?
echo "[gfs2r] train exit $TR $(date -Is)"
tail -8 logs/fs_train.log
if [ $TR -ne 0 ]; then echo "[gfs2r] TRAIN-FAIL"; exit 1; fi

CUDA_VISIBLE_DEVICES=0 $PY -u glm_tc_holdout_gate.py base > logs/fs_gate_base.log 2>&1
echo "[gfs2r] gate base done $(date -Is)"; tail -3 logs/fs_gate_base.log
CUDA_VISIBLE_DEVICES=0 $PY -u glm_tc_holdout_gate.py capped > logs/fs_gate_capped.log 2>&1
echo "[gfs2r] gate capped done $(date -Is)"; tail -3 logs/fs_gate_capped.log
$PY -u glm_tc_holdout_gate.py compare > logs/fs_gate_compare.log 2>&1
echo "[gfs2r] gate compare done $(date -Is)"; tail -8 logs/fs_gate_compare.log
echo "[gfs2r] COMPLETE $(date -Is)"
