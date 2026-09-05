#!/bin/bash
# run_remaining.sh — coordinator order: train -> capped gate arm -> compare.
# Merge/parity/base-gate receipts already stand; do NOT rerun them.
set -euo pipefail
mkdir -p /wd/logs
cd /wd
echo "[chain2] start $(date -Is)"
python3 -u tc_lora_train.py >> logs/train.log 2>&1
echo "[chain2] train done $(date -Is)"
python3 -u tc_holdout_gate.py capped >> logs/gate_capped.log 2>&1
echo "[chain2] gate capped done $(date -Is)"
python3 tc_holdout_gate.py compare > logs/gate_compare.log 2>&1
echo "[chain2] compare done $(date -Is)"
cat logs/gate_compare.log
echo "[chain2] ALL DONE $(date -Is)"
