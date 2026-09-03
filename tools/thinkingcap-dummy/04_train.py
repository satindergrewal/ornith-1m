#!/usr/bin/env python3
"""ThinkingCap dummy - STEP 04: LoRA via nano_torch autograd (no transformers).

Transformers cannot load raw DSV4 checkpoints (see DESIGN + memory
dsv4-transformers-param-mismatch), so the cap trains directly on the
pure-torch nano engine: adapters on language-layer projections and the
shared expert, everything else frozen (vision/MTP absent from the nano
slice by construction; the real run freezes them the same way -
drafter-anchor lesson).

LoRA semantics are kept exact for the 05 merge rehearsal:
    W_eff = W + (alpha/r) * (B @ A),  A (r,k) init kaiming, B (n,r) zero
so step 0 is bit-identical to the base. Save format is raw checkpoint
names: lora.layers.{i}.attn.wq_b.weight.{A,B} etc.

Runs on CPU. Output: <wd>/cap/{lora.safetensors,report.json}
"""
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

WD = Path(os.environ.get("TC_WD", "/home/satinder/thinkingcap-dummy"))
sys.path.insert(0, str(WD))
import nano_torch as nt  # noqa

from nano_torch import hc_apply, hc_head, hc_mix, layer_attn, moe, rms  # noqa

OUT = WD / "cap"
R = 8
ALPHA = 16
SCALE = ALPHA / R
LR = 3e-4
EPOCHS = 3
SEED = 20260904


def train_forward(model, input_ids):
    """Copy of NanoDSV4.forward without @torch.inference_mode."""
    M = model.M
    ids = torch.as_tensor(input_ids, dtype=torch.long)
    h = M.embed[ids]
    st = h.unsqueeze(0).unsqueeze(2).expand(1, -1, M.hc_mult, -1).contiguous()
    for L in M.layers:
        post, comb, coll = hc_mix(st, L.attn_hc, M.eps, M.hc_eps, M.hc_iters)
        y = layer_attn(rms(coll[0], L.attn_norm, M.eps), L, M)
        st = hc_apply(st, y, post, comb)
        post, comb, coll = hc_mix(st, L.ffn_hc, M.eps, M.hc_eps, M.hc_iters)
        y = moe(rms(coll[0], L.ffn_norm, M.eps), L.ffn, M)
        st = hc_apply(st, y, post, comb)
    h = rms(hc_head(st, M.hh, M.eps, M.hc_eps), M.final_norm, M.eps)
    model.hidden_at_last_layer = h
    return (h @ M.head.T)[0]


def collect_targets(model):
    """(container, attr, raw_name) for every adapter site."""
    targets = []
    for i, L in enumerate(model.M.layers):
        for attr in ("wq_b", "wkv", "wo_b"):
            targets.append((L.attn, attr, f"layers.{i}.attn.{attr}.weight"))
        for attr in ("sw1", "sw2", "sw3"):
            name = {"sw1": "w1", "sw2": "w2", "sw3": "w3"}[attr]
            targets.append((L.ffn, attr,
                            f"layers.{i}.ffn.shared_experts.{name}.weight"))
    return targets


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = nt.NanoDSV4()
    tok = nt.get_tokenizer()
    sft = [json.loads(line) for line in
           (WD / "regen" / "sft.jsonl").read_text().splitlines() if line]

    targets = collect_targets(model)
    adapters, frozen = [], []
    for container, attr, raw in targets:
        W = getattr(container, attr).detach().float()
        n, k = W.shape
        A = torch.empty(R, k).normal_(0, 1.0 / k ** 0.5)
        B = torch.zeros(n, R)
        A.requires_grad_(True)
        B.requires_grad_(True)
        adapters.append({"container": container, "attr": attr, "raw": raw,
                         "W": W, "A": A, "B": B})
        frozen.append(W.numel())

    opt = torch.optim.AdamW(
        [p for a in adapters for p in (a["A"], a["B"])], lr=LR)
    n_adapter = sum(a["A"].numel() + a["B"].numel() for a in adapters)

    def apply_effective():
        for a in adapters:
            setattr(a["container"], a["attr"],
                    a["W"] + SCALE * (a["B"] @ a["A"]))

    report = {"r": R, "alpha": ALPHA, "lr": LR, "epochs": EPOCHS,
              "targets": [a["raw"] for a in adapters],
              "adapter_params": n_adapter,
              "frozen_params": sum(frozen),
              "n_samples": len(sft), "epoch_loss": []}

    # step-0 bit-identity check: B=zero init must reproduce base logits
    probe = [1, 2, 3]
    apply_effective()
    with torch.no_grad():
        base_logits = train_forward(model, probe).detach().clone()

    step0_logits = None
    for epoch in range(EPOCHS):
        order = list(range(len(sft)))
        random.shuffle(order)
        total, ntok = 0.0, 0
        for si in order:
            row = sft[si]
            pids = tok.encode(row["prompt"]).ids
            cids = tok.encode(row["completion"]).ids
            if not cids:
                continue
            ids = pids + cids
            apply_effective()
            logits = train_forward(model, ids)
            shift = logits[:-1]
            tgt = torch.as_tensor(ids[1:], dtype=torch.long)
            mask = torch.zeros(len(ids) - 1, dtype=torch.bool)
            mask[len(pids) - 1:] = True
            loss = F.cross_entropy(shift[mask], tgt[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * int(mask.sum())
            ntok += int(mask.sum())
        avg = total / max(ntok, 1)
        report["epoch_loss"].append(round(avg, 4))
        print(f"epoch {epoch}: loss={avg:.4f} tokens={ntok}", flush=True)

    # cap effect probe: trained adapters vs base logits on a fixed input
    apply_effective()
    with torch.no_grad():
        trained_logits = train_forward(model, probe).detach()
        drift = float((trained_logits - base_logits).abs().max())
    report["adapter_drift_vs_base"] = drift

    OUT.mkdir(exist_ok=True)
    from safetensors.torch import save_file
    tensors = {}
    for a in adapters:
        tensors[f"lora.{a['raw']}.A"] = a["A"].detach().contiguous()
        tensors[f"lora.{a['raw']}.B"] = a["B"].detach().contiguous()
    save_file(tensors, str(OUT / "lora.safetensors"))
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print("CAP:", json.dumps({k: report[k] for k in
                              ("epoch_loss", "adapter_params",
                               "adapter_drift_vs_base")}))


if __name__ == "__main__":
    main()
