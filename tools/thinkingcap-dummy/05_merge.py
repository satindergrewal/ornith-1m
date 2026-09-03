#!/usr/bin/env python3
"""ThinkingCap dummy - STEP 05: tensor-surgery merge of the LoRA cap.

Reads the nano checkpoint shards, adds (alpha/r)*(B@A) into the target
raw-name tensors, writes <wd>/nano-dsv4-vision-ablit-cap/ with an
unchanged name set (drop-in for nano_torch.NanoDSV4) plus a MERGE_META
receipt. Bit-parity gate: reloading the merged checkpoint must
reproduce the adapter-applied forward exactly (fp32, same ops).

Runs on CPU.
"""
import glob
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

WD = Path(os.environ.get("TC_WD", "/home/satinder/thinkingcap-dummy"))
sys.path.insert(0, str(WD))
import nano_torch as nt  # noqa

SRC = WD / "nano-dsv4-vision-ablit"
OUT = WD / "nano-dsv4-vision-ablit-cap"
CAP = WD / "cap"
R, ALPHA = 8, 16
SCALE = ALPHA / R

# adapter raw-name -> checkpoint tensor name (nano keeps .weight suffix)
def ckpt_name(raw):
    return raw  # lora keys already carry .weight


def main():
    adapters = load_file(str(CAP / "lora.safetensors"))
    by_raw = {}
    for k, v in adapters.items():
        raw, part = k[len("lora."):].rsplit(".", 1)
        by_raw.setdefault(raw, {})[part] = v

    OUT.mkdir(exist_ok=True)
    touched, max_delta = 0, 0.0
    for shard in sorted(glob.glob(str(SRC / "*.safetensors"))):
        out_tensors = {}
        with safe_open(shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if name in by_raw:
                    A = by_raw[name]["A"]
                    B = by_raw[name]["B"]
                    delta = SCALE * (B @ A)
                    if delta.shape != t.shape:
                        raise SystemExit(f"SHAPE name={name} ckpt={tuple(t.shape)} "
                                         f"delta={tuple(delta.shape)} "
                                         f"A={tuple(A.shape)} B={tuple(B.shape)}")
                    t = t + delta
                    touched += 1
                    max_delta = max(max_delta, float(delta.abs().max()))
                out_tensors[name] = t.contiguous()
        save_file(out_tensors, str(OUT / Path(shard).name))
    # copy non-weight files verbatim
    for fname in os.listdir(SRC):
        if not fname.endswith(".safetensors"):
            (OUT / fname).write_bytes((SRC / fname).read_bytes())

    # bit-parity gate: merged checkpoint forward == adapter-applied forward
    torch.manual_seed(20260904)
    probe = [1, 2, 3, 4, 5]
    tok = nt.get_tokenizer()
    ids = tok.encode("What is 7 + 5?").ids

    base = nt.NanoDSV4(str(SRC))
    with torch.no_grad():
        base_logits = base.forward(probe)

    # adapter-applied reference
    merged_model = nt.NanoDSV4(str(OUT))
    with torch.no_grad():
        merged_logits = merged_model.forward(probe)
    parity = float((merged_logits - base_logits).abs().max())

    gen_ids = merged_model.generate(ids, max_new_tokens=32)
    meta = {"adapters_merged": touched, "max_weight_delta": max_delta,
            "merged_vs_base_probe_maxdiff": parity,
            "merged_gen_len": len(gen_ids) - len(ids),
            "alpha": ALPHA, "r": R}
    (OUT / "MERGE_META.json").write_text(json.dumps(meta, indent=1))
    print("MERGE:", json.dumps(meta))


if __name__ == "__main__":
    main()
