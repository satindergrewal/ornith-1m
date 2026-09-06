#!/usr/bin/env python3
"""tc_lora_train.py — ThinkingCap REAL STEP 3 (04-real): LoRA cap training
through the layer-streaming engine.

Target set (DOCUMENTED BEFORE TRAINING, PROGRESS.md): per language layer
i in 0..42, six sites — attn.wq_b, attn.wkv, attn.wo_b + ffn.shared_experts.
{w1,w2,w3}. This is the dummy 04_train.py precedent (24 adapters on the 4-layer
nano: same six sites) which itself follows the official ThinkingCap lineage:
cap the language-stack projections; routed experts, hc, norms untouched; vision
absent; MTP/DSpark untouched (drafter-anchor lesson). R=8, ALPHA=16, LR=3e-4,
AdamW, CE on completion tokens only, eos appended to every target completion
(termination is the signal the cap trains). Adapter save format = dummy 05
merge format: lora.{layers.i.site.weight}.{A,B}.

Training-through-streaming method: layer-wise rematerialization.
  Pass A (no grad): full packed forward, saving the 43+1 stream-stack
    boundaries (fp32; ~22 GB at S=3000) and computing the CE loss with grad
    attached ONLY to the final boundary -> g_final.
  Pass B (reverse, one layer at a time): recompute layer i from its saved
    input boundary under autograd (adapters active as a separate fp32 delta
    path — bf16 weight folding would round the small deltas away), then
    dot(st_out, g_i).sum().backward() gives exact grads for that layer's
    adapters and g_{i-1} = input.grad. Peak = one layer + boundaries.
Both passes run with engine.act_dtype=fp32 (tf32 matmuls allowed) so the
adapter delta reaches the loss un-quantized.

Output: /wd/cap/{lora.safetensors, report_train.json}
"""

import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from full_loader import StreamingDSV4
from dsv4_full import DSV4Full, rms_w
from tc_batch_gen import build_prompt

R = 8
ALPHA = 16
SCALE = ALPHA / R
LR = 3e-4
EPOCHS = 30
SEED = 20260904
OUT = "/wd/cap"

SITES = ["attn.wq_b.weight", "attn.wkv.weight", "attn.wo_b.weight",
         "ffn.shared_experts.w1.weight", "ffn.shared_experts.w2.weight",
         "ffn.shared_experts.w3.weight"]


def collect_targets(loader):
    """Adapter sites with shapes read from the checkpoint meta (no hardcoding)."""
    out = []
    for i in range(loader.n_layers):
        for site in SITES:
            name = f"layers.{i}.{site}"
            shape = tuple(loader.meta[name][1])
            out.append((i, site, name, shape))
    return out


def build_sft(tok, rows, eos):
    """Shortest-correct completion per problem; completion ids + eos (termination)."""
    best = {}
    for r in rows:
        if not (r["correct"] or r.get("correct_first_answer")):
            continue
        cur = best.get(r["pid"])
        if cur is None or r["n_new"] < cur["n_new"]:
            best[r["pid"]] = r
    sft = []
    for pid, r in sorted(best.items()):
        pids = tok.encode(build_prompt(r["prompt"])).ids
        cids = tok.encode(r["completion"]).ids + [eos]
        sft.append({"pid": pid, "kind": r["kind"], "pids": pids, "cids": cids,
                    "text": r["completion"]})
    return sft


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("/model/tokenizer.json")
    L = StreamingDSV4()
    eng = DSV4Full(L)
    eng.act_dtype = torch.float32
    eos = L.cfg.get("eos_token_id", 1)

    rows = [json.loads(l) for l in open("/wd/data/regen.jsonl")]
    sft = build_sft(tok, rows, eos)
    print(f"[train] SFT rows: {len(sft)} (shortest-correct per problem), "
          f"tokens: prompt+completion = "
          f"{sum(len(r['pids']) + len(r['cids']) for r in sft)}", flush=True)

    targets = collect_targets(L)
    adapters = {}
    n_params = 0
    for i, site, name, (n, k) in targets:
        A = torch.empty(R, k, device=eng.dev, dtype=torch.float32).normal_(
            0.0, 1.0 / k ** 0.5)
        B = torch.zeros(n, R, device=eng.dev, dtype=torch.float32)
        A.requires_grad_(True)
        B.requires_grad_(True)
        adapters[(i, site)] = (A, B)
        n_params += A.numel() + B.numel()
    eng.lora = adapters
    eng.lora_scale = SCALE
    eng.lora_fold = False
    frozen = sum(n * k for _, _, _, (n, k) in targets)
    print(f"[train] adapters: {len(adapters)} ({n_params:,} params; frozen "
          f"target-site weights {frozen:,}); r={R} alpha={ALPHA} lr={LR} "
          f"epochs={EPOCHS}", flush=True)

    # pack SFT rows once. Full-scale windowing: block_batch attends within a
    # row only (starts/lens/seq_of isolate rows), so row-aligned windows are
    # an EXACT re-chunking of the single-pack forward; gradients accumulate
    # across windows before each optimizer step. Window peak memory stays in
    # the pilot's ~9 GB class instead of scaling with the whole SFT set.
    # fp32 activation sims on routed-expert graphs make pass-B memory scale
    # ~linearly with window S (12K windows OOM'd a 95GB card; the pilot's
    # ~9GB peak was at S=3000). 4000 keeps peak in the pilot class.
    WINDOW_S = 4000

    def pack_rows(rows):
        flat, starts, lens = [], [], []
        for r in rows:
            starts.append(len(flat))
            ids = r["pids"] + r["cids"]
            lens.append(len(ids))
            flat.extend(ids)
        S = len(flat)
        # global predictor mask, length S-1: predictor j (0-based) predicts
        # flat[j+1]; True iff flat[j+1] is a completion token of the SAME row
        # (row-boundary predictors stay False; their targets belong to the next
        # row's prompt)
        mask_t = torch.zeros(S - 1, dtype=torch.bool, device=eng.dev)
        for r, s in zip(rows, starts):
            ln = len(r["pids"]) + len(r["cids"])
            mask_t[s + len(r["pids"]) - 1: s + ln - 1] = True
        seq_of = torch.repeat_interleave(
            torch.arange(len(rows), device=eng.dev),
            torch.tensor(lens, device=eng.dev))
        pos_of = torch.arange(S, device=eng.dev) - \
            torch.tensor(starts, device=eng.dev, dtype=torch.long)[seq_of]
        return {"ids": torch.tensor(flat, dtype=torch.long, device=eng.dev),
                "mask": mask_t, "seq_of": seq_of, "pos_of": pos_of,
                "starts": starts, "lens": lens, "S": S,
                "n_ce": int(mask_t.sum())}

    windows, cur, cur_tok = [], [], 0
    for r in sft:
        ln = len(r["pids"]) + len(r["cids"])
        if cur and cur_tok + ln > WINDOW_S:
            windows.append(cur)
            cur, cur_tok = [], 0
        cur.append(r)
        cur_tok += ln
    if cur:
        windows.append(cur)
    packs = [pack_rows(w) for w in windows]
    S = sum(p["S"] for p in packs)
    n_ce_total = sum(p["n_ce"] for p in packs)
    print(f"[train] packed S={S} tokens in {len(packs)} window(s) of "
          f"<={WINDOW_S}, CE targets {n_ce_total}", flush=True)

    opt = torch.optim.AdamW([p for ab in adapters.values() for p in ab], lr=LR)

    def top_loss_from(st_leaf, p):
        h = eng.hc_head(st_leaf)[0]
        h = rms_w(h, eng.top["norm"], eng.eps)
        logits = h.float() @ eng.top["head_f32"].T
        tgt = p["ids"][1:]
        return F.cross_entropy(logits[:-1][p["mask"]], tgt[p["mask"]]), logits

    def forward_collect(p):
        """Pass A: boundaries + loss/grad at the top, one window."""
        ids_t, S = p["ids"], p["S"]
        bounds = []
        with torch.no_grad():
            h = eng.top["embed"][ids_t].to(eng.act_dtype)
            st = h.unsqueeze(0).unsqueeze(2).expand(
                1, S, eng.hc, eng.H).contiguous()
            for i in range(eng.n_layers):
                bounds.append(st)
                Lw = eng._load_layer(i)
                st = eng.block_batch(i, Lw, st, ids_t, p["seq_of"], p["pos_of"],
                                     p["starts"], p["lens"])
                del Lw
        bounds.append(st)
        st_leaf = bounds[-1].detach().requires_grad_(True)
        loss, logits = top_loss_from(st_leaf, p)
        return bounds, st_leaf, loss, logits

    # ---- step-0 bit-identity check (B=0 -> engine output identical to base)
    bounds0, st_leaf0, loss0, _ = forward_collect(packs[0])
    loss0.backward()
    g0 = st_leaf0.grad.detach()
    eng.lora = None
    with torch.no_grad():
        b_base, _, loss_base, _ = forward_collect(packs[0])
    eng.lora = adapters
    ident = (loss0.item() == loss_base.item())
    print(f"[train] step-0 identity (B=0 loss == base loss): {ident} "
          f"({loss0.item():.6f} vs {loss_base.item():.6f})", flush=True)
    assert ident, "adapter path changed the base forward"
    del bounds0, b_base

    report = {"r": R, "alpha": ALPHA, "scale": SCALE, "lr": LR, "epochs": EPOCHS,
              "seed": SEED, "n_targets": len(adapters), "adapter_params": n_params,
              "frozen_target_site_params": frozen,
              "target_sites": [t[2] for t in targets[:6]] + ["... x 43 layers"],
              "n_sft_rows": len(sft), "sft_tokens": S,
              "n_windows": len(packs), "window_s": WINDOW_S,
              "ce_targets": n_ce_total,
              "step0_identity": ident, "epoch_loss": [], "epoch_wall_s": [],
              "grad_norm": [], "act_dtype": "fp32 (tf32 matmuls)",
              "method": "layer-wise rematerialization: no-grad boundary pass + "
                        "reverse per-layer dot-product backward; row-aligned "
                        "exact windows with cross-window gradient accumulation"}
    os.makedirs(OUT, exist_ok=True)

    def save_cap(final=False):
        from safetensors.torch import save_file
        tensors = {}
        for (i, site), (A, B) in adapters.items():
            tensors[f"lora.layers.{i}.{site}.A"] = A.detach().contiguous()
            tensors[f"lora.layers.{i}.{site}.B"] = B.detach().contiguous()
        save_file(tensors, os.path.join(OUT, "lora.safetensors"))
        report["final"] = final
        json.dump(report, open(os.path.join(OUT, "report_train.json"), "w"),
                  indent=1)

    t_all = time.time()
    for ep in range(EPOCHS):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        ep_loss_sum = 0.0
        for p in packs:
            bounds, st_leaf, loss, _ = forward_collect(p)
            # window CE is a mean over its own targets; rescale so the
            # accumulated gradient equals the single-pack full-batch gradient
            scaled = loss * (p["n_ce"] / n_ce_total)
            scaled.backward()
            g = st_leaf.grad.detach()
            for i in reversed(range(eng.n_layers)):
                st_in = bounds[i].detach().requires_grad_(True)
                Lw = eng._load_layer(i)
                st_out = eng.block_batch(i, Lw, st_in, p["ids"], p["seq_of"],
                                         p["pos_of"], p["starts"], p["lens"])
                (st_out * g).sum().backward()
                g = st_in.grad.detach()
                del Lw, st_out, st_in
            ep_loss_sum += loss.item() * p["n_ce"]
            del bounds, st_leaf
        ep_loss = ep_loss_sum / n_ce_total
        gn = math.sqrt(sum((p.grad ** 2).sum().item()
                           for p in opt.param_groups[0]["params"]
                           if p.grad is not None))
        opt.step()
        dt = time.time() - t0
        report["epoch_loss"].append(round(ep_loss, 4))
        report["epoch_wall_s"].append(round(dt, 1))
        report["grad_norm"].append(round(gn, 4))
        print(f"[train] epoch {ep:02d}: loss={ep_loss:.4f} grad={gn:.3f} "
              f"t={dt:.0f}s ({len(packs)} windows)", flush=True)
        if (ep + 1) % 5 == 0:
            save_cap()
    save_cap(final=True)

    wall = time.time() - t_all
    report["wall_s"] = round(wall, 1)
    report["training_tokens_per_s"] = round(
        EPOCHS * S / wall, 1)   # forward tokens per wall second (both passes)
    save_cap(final=True)
    print(f"[train] CAP saved: {OUT}/lora.safetensors; "
          f"loss {report['epoch_loss'][0]} -> {report['epoch_loss'][-1]} "
          f"over {EPOCHS} epochs, wall {wall/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
