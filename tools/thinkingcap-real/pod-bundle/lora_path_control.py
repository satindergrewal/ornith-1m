#!/usr/bin/env python3
"""Control: do the attached live adapters actually enter the BATCHED path?
One forward_batch, base vs attach_lora, same input. Logit maxdiff decides:
  diff > 0  -> adapters ARE live; identical outputs = real cap verdict.
  diff == 0 -> batched path bypasses adapters; bench result is an artifact.
"""
import torch
from tokenizers import Tokenizer
from full_loader import StreamingDSV4
from dsv4_full import DSV4Full
from tc_lora_train import SCALE

tok = Tokenizer.from_file("/model/tokenizer.json")
eng = DSV4Full(StreamingDSV4(model_dir="/model", meta_path="/wd/tensor_meta.json"))

ids = tok.encode("<｜begin▁of▁sentence｜><｜User｜>What is 6 + 7 * 3?<｜Assistant｜>").ids
seqs = [ids, ids + [100], ids + [200]]

with torch.no_grad():
    base = eng.forward_batch(seqs).clone()

n = eng.attach_lora("/wd/lora.safetensors", SCALE, fold=False)
print(f"attached {n} adapters", flush=True)

with torch.no_grad():
    cap = eng.forward_batch(seqs)

d = (base - cap).abs()
print(f"BATCHED-PATH LOGIT MAXDIFF: {float(d.max()):.6f}")
print(f"argmax same: {[bool(a == b) for a, b in zip(base.argmax(-1), cap.argmax(-1))]}")

# also the single-sequence path for comparison
with torch.no_grad():
    b1 = eng.forward(ids)
eng2 = DSV4Full(StreamingDSV4(model_dir="/model", meta_path="/wd/tensor_meta.json"))
with torch.no_grad():
    s1 = eng2.forward(ids)
print(f"single-path no-lora vs base sanity: {float((s1 - b1).abs().max()):.6f}")
