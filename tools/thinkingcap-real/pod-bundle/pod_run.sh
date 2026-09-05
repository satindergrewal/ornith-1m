#!/bin/sh
# Pod-side runner: both arms in parallel, one per GPU, then merge.
# Inside container with /model (checkpoint) and /wd (bundle) mounted.
cd /wd
echo "=== GPU check ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || { echo NO-GPU; exit 1; }
echo "=== model check ==="
ls /model/*.safetensors 2>/dev/null | wc -l
[ -f /model/config.json ] || { echo MODEL-MISSING; exit 1; }
PY=/usr/local/bin/python
echo "=== smoke: engine loads + 8 greedy tokens ==="
$PY - <<'EOF'
import torch
from tokenizers import Tokenizer
from full_loader import StreamingDSV4
from dsv4_full import DSV4Full
tok = Tokenizer.from_file("/model/tokenizer.json")
eng = DSV4Full(StreamingDSV4(model_dir="/model", meta_path="/wd/tensor_meta.json"))
ids = tok.encode("<｜begin▁of▁sentence｜><｜User｜>What is 6*7?<｜Assistant｜>").ids
out, gen = eng.generate(ids, max_new_tokens=8)
print("SMOKE-OK:", repr(tok.decode(gen, skip_special_tokens=True))[:120])
EOF
[ $? -ne 0 ] && { echo SMOKE-FAIL; exit 1; }
echo "=== bench: arm base on GPU0, arm cap on GPU1 (parallel) ==="
CUDA_VISIBLE_DEVICES=0 $PY pod_bench.py --arm base --n 48 --budget 512 --open-budget 768 --out /wd/results-base.json > /wd/bench-base.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 $PY pod_bench.py --arm cap --n 48 --budget 512 --open-budget 768 --out /wd/results-cap.json > /wd/bench-cap.log 2>&1 &
P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
echo "base exit=$R1 cap exit=$R2"
$PY - <<'EOF'
import json
b = json.load(open("/wd/results-base.json"))
c = json.load(open("/wd/results-cap.json"))
def line(tag, r):
    t = r["math"]["total"]
    print(f"{tag}: correct {t['correct']}/{t['n']} | mean tok all "
          f"{t['mean_tokens_all']} | correct-only {t['mean_tokens_correct']} "
          f"| hit_cap {t['hit_cap']}")
line("BASE", b)
line("CAP ", c)
d = round(c["math"]["total"]["mean_tokens_all"] - b["math"]["total"]["mean_tokens_all"], 1)
print(f"delta mean_tokens_all (cap - base): {d}")
print(f"open: base {[o['n_new'] for o in b['open']]} cap {[o['n_new'] for o in c['open']]}")
EOF
echo BENCH-ALL-DONE
