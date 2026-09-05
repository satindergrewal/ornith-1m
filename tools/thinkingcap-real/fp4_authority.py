#!/usr/bin/env python3
"""fp4_authority.py - cross-check our MXFP4 nibble decode against the serving
stack's own dequant (amd-quark mx.dq_mxfp4) inside the dspark-vllm image."""
import json, os, sys
import numpy as np
import torch

meta = json.load(open("/wd/tensor_meta.json"))
OFF, MT = meta["offsets"], meta["meta"]

name = "layers.10.ffn.experts.7.w1.weight"
sname = name[: name.rfind(".weight")] + ".scale"
MODEL = "/model"

def raw(n):
    shard, start, nb = OFF[n]
    mm = np.memmap(os.path.join(MODEL, shard), dtype=np.uint8, mode="r")
    return np.frombuffer(mm[start:start + nb].copy(), dtype=np.uint8).reshape(MT[n][1])

w = raw(name)                      # uint8 [n, k/2]
s = raw(sname)                     # uint8 [n, k/32] as e8m0 bytes
n, k2 = w.shape
k = 2 * k2

# --- my decode (same math as full_loader.dequant_mxfp4)
lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                    -0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
wt = torch.from_numpy(w)
wf = torch.empty((n, k), dtype=torch.float32)
wf[:, 0::2] = lut[(wt & 0xF).long()]
wf[:, 1::2] = lut[(wt >> 4).long()]
st = torch.from_numpy(s.view(np.uint8))
sf = torch.exp2(st.float() - 127.0).repeat_interleave(32, 1)
mine = (wf * sf).to(torch.bfloat16)

# --- serving-stack decode
from quark.torch.kernel import mx
sw = torch.tensor(np.frombuffer(s.tobytes(), dtype=np.uint8).reshape(s.shape)).view(torch.float8_e8m0fnu)
ref = mx.dq_mxfp4(torch.from_numpy(w.copy()).view(torch.uint8).view(torch.float4_e2m1fn_x2),
                  sw.cuda(), torch.bfloat16).cpu() if torch.cuda.is_available() else \
      mx.dq_mxfp4(torch.from_numpy(w.copy()).view(torch.uint8).view(torch.float4_e2m1fn_x2),
                  sw, torch.bfloat16)
print("ref shape:", tuple(ref.shape), "mine:", tuple(mine.shape))
if ref.shape != mine.shape:
    # dq may return (n, k/2, 2)
    ref = ref.reshape(mine.shape)
d = (ref.float() - mine.float()).abs().max().item()
eq = torch.equal(ref, mine)
print(f"maxdiff={d:.3e} bit-equal={eq}")
frac = (ref.float() - mine.float()).abs() > 0
print("fraction differing:", frac.float().mean().item())
print("PASS" if d == 0.0 or (frac.float().mean().item() < 1e-6 and d < 1e-3) else "FAIL")
