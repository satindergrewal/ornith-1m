#!/bin/bash
# run_night.sh — coordinator-order chain after kind2 regen completes.
# Every step tee'd under /wd/logs/; set -e stops the chain on first failure.
set -euo pipefail
mkdir -p /wd/logs
cd /wd

echo "[chain] start $(date -Is)"
# 0. preserve the 48-budget run as the merge source
if [ ! -f data/regen_run48.jsonl ]; then
  cp data/regen.jsonl data/regen_run48.jsonl
fi

# 1. merged regen + report + sft view
python3 build_merged_report.py > logs/merge.log 2>&1
echo "[chain] merge done $(date -Is)"
tail -3 logs/merge.log

# 2. engine parity after LoRA-hook edits (base path must be unchanged)
python3 - > logs/parity.log 2>&1 <<'EOF'
import torch
from tokenizers import Tokenizer
from full_loader import StreamingDSV4
from dsv4_full import DSV4Full
tok = Tokenizer.from_file("/model/tokenizer.json")
L = StreamingDSV4(); eng = DSV4Full(L)
ids = tok.encode("The capital of France is").ids
lg = eng.forward(ids)
top = lg[-1].topk(3)
print("top3:", [(tok.decode([int(i)]), round(v, 2)) for v, i in
                zip(top.values.tolist(), top.indices.tolist())])
assert int(top.indices[0]) == 11111, "top-1 is not ' Paris' — engine edit broke the base path"
print("PARITY OK")
EOF
echo "[chain] parity done $(date -Is)"
tail -2 logs/parity.log

# 3. holdout gate BASE arm (also validates the gate harness)
python3 -u tc_holdout_gate.py base > logs/gate_base.log 2>&1
echo "[chain] gate base done $(date -Is)"

# 4. 04-real LoRA training
python3 -u tc_lora_train.py > logs/train.log 2>&1
echo "[chain] train done $(date -Is)"

# 5. holdout gate CAPPED arm
python3 -u tc_holdout_gate.py capped > logs/gate_capped.log 2>&1
echo "[chain] gate capped done $(date -Is)"

# 6. compare + gate verdict
python3 tc_holdout_gate.py compare > logs/gate_compare.log 2>&1
echo "[chain] compare done $(date -Is)"
cat logs/gate_compare.log
echo "[chain] ALL DONE $(date -Is)"
