#!/usr/bin/env python3
"""glm_tc_lora_train.py — ThinkingCap LoRA cap training for GLM-5.3-Flash
through the layer-streaming pure-torch engine (port of tc_lora_train.py,
windowed variant: row-aligned windows with cross-window gradient
accumulation — KDA rows are cu_seqlens-isolated so windows are exact).

Target set (GLM class analogue of the DSV4 six-site set, per layer class):
  KDA layers: self_attn.{q,k,v,b,o}_proj.weight + shared_experts x3
  DSA layers: self_attn.{q_a,q_b,kv_a_proj_with_mqa,kv_b,o}_proj.weight
              + shared_experts x3
  dense L0-2: mlp.{gate,up,down}_proj.weight
Routed experts, hc, norms untouched; vision absent; MTP/DSpark untouched
(drafter-anchor). R=8 ALPHA=16 LR 3e-4, AdamW, CE on completion tokens only,
<|endoftext|> (154820) appended to every target completion.

Method: layer-wise rematerialization (pass A no-grad boundaries + top CE;
pass B reverse per-layer recompute with (st_out*g).sum().backward()) — same
as the DSV4 trainer; windowed at WINDOW_S with exact cross-window gradient
accumulation (loss_w * n_ce_w / n_ce_total). Incremental save every 5
epochs. Output: /root/tc-glm/cap/{lora.safetensors, report_train.json}."""

import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from glm_full_loader import StreamingGLM
from glm_full import GLMFull, rms_w
from glm_tc_batch_gen import get_tok, special_ids, build_prompt_ids

R = 8
ALPHA = 16
SCALE = ALPHA / R
LR = 3e-4
EPOCHS = 30
SEED = 20260904
OUT = "/root/tc-glm/cap"
EOS_ID = 154820                                    # <|endoftext|>
WINDOW_S = 4096

SITES_KDA = ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
             "self_attn.v_proj.weight", "self_attn.b_proj.weight",
             "self_attn.o_proj.weight"]
SITES_DSA = ["self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight",
             "self_attn.kv_a_proj_with_mqa.weight",
             "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight"]
SITES_SHARED = ["mlp.shared_experts.gate_proj.weight",
                "mlp.shared_experts.up_proj.weight",
                "mlp.shared_experts.down_proj.weight"]
SITES_DENSE = ["mlp.gate_proj.weight", "mlp.up_proj.weight",
               "mlp.down_proj.weight"]


def collect_targets(loader):
    """Adapter sites with shapes from the checkpoint meta (no hardcoding)."""
    eng_types = loader.cfg.get("layer_types") or []
    out = []
    for i in range(loader.n_layers):
        lt = eng_types[i] if i < len(eng_types) else "linear_attention"
        base = "model.language_model.layers.%d." % i
        sites = (SITES_KDA if lt == "linear_attention" else SITES_DSA) \
            + SITES_SHARED
        if f"{base}mlp.gate.weight" not in loader.meta:
            sites = sites + SITES_DENSE            # dense L0..first_k-1
        for site in sites:
            name = base + site
            if name not in loader.meta:
                continue
            shape = tuple(loader.meta[name][1])
            out.append((i, site, name, shape))
    return out


def build_sft(tok, sp, rows, eos):
    """Shortest-correct completion per problem; completion ids + eos."""
    best = {}
    for r in rows:
        if not (r["correct"] or r.get("correct_first_answer")):
            continue
        cur = best.get(r["pid"])
        if cur is None or r["n_new"] < cur["n_new"]:
            best[r["pid"]] = r
    sft = []
    for pid, r in sorted(best.items()):
        pids = build_prompt_ids(tok, sp, r["prompt"])
        cids = tok.encode(r["completion"], add_special_tokens=False).ids + [eos]
        sft.append({"pid": pid, "kind": r["kind"], "pids": pids,
                    "cids": cids, "text": r["completion"]})
    return sft


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    tok = get_tok()
    sp = special_ids(tok)
    L = StreamingGLM()
    eng = GLMFull(L)
    eng.act_dtype = torch.float32

    rows = [json.loads(l) for l in open("/root/tc-glm/data/regen.jsonl")]
    sft = build_sft(tok, sp, rows, EOS_ID)
    print(f"[train] SFT rows: {len(sft)} (shortest-correct per problem), "
          f"tokens prompt+completion = "
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

    # ---- pack SFT rows into row-aligned windows (exact: cu_seqlens isolation)
    def pack_rows(rows_):
        flat, starts, lens = [], [], []
        for r in rows_:
            starts.append(len(flat))
            ids = r["pids"] + r["cids"]
            lens.append(len(ids))
            flat.extend(ids)
        S = len(flat)
        mask_t = torch.zeros(S - 1, dtype=torch.bool, device=eng.dev)
        for r, s in zip(rows_, starts):
            ln = len(r["pids"]) + len(r["cids"])
            mask_t[s + len(r["pids"]) - 1: s + ln - 1] = True
        seq_of = torch.repeat_interleave(
            torch.arange(len(rows_), device=eng.dev),
            torch.tensor(lens, device=eng.dev))
        pos_of = torch.arange(S, device=eng.dev) - \
            torch.tensor(starts, device=eng.dev, dtype=torch.long)[seq_of]
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()),
                          dtype=torch.int32, device=eng.dev)
        return {"ids": torch.tensor(flat, dtype=torch.long, device=eng.dev),
                "mask": mask_t, "seq_of": seq_of, "pos_of": pos_of,
                "starts": starts, "lens": lens, "S": S, "cu": cu,
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
                                     p["starts"], p["lens"], p["cu"])
                del Lw
        bounds.append(st)
        st_leaf = bounds[-1].detach().requires_grad_(True)
        loss, logits = top_loss_from(st_leaf, p)
        return bounds, st_leaf, loss, logits

    # ---- step-0 bit-identity (B=0 loss == base loss on window 0)
    bounds0, st_leaf0, loss0, _ = forward_collect(packs[0])
    loss0.backward()
    eng.lora = None
    with torch.no_grad():
        b_base, _, loss_base, _ = forward_collect(packs[0])
    eng.lora = adapters
    ident = (loss0.item() == loss_base.item())
    print(f"[train] step-0 identity (B=0 loss == base loss): {ident} "
          f"({loss0.item():.6f} vs {loss_base.item():.6f})", flush=True)
    assert ident, "adapter path changed the base forward"
    del bounds0, b_base

    report = {"r": R, "alpha": ALPHA, "scale": SCALE, "lr": LR,
              "epochs": EPOCHS, "seed": SEED, "n_targets": len(adapters),
              "adapter_params": n_params,
              "frozen_target_site_params": frozen,
              "n_sft_rows": len(sft), "sft_tokens": S,
              "n_windows": len(packs), "window_s": WINDOW_S,
              "ce_targets": n_ce_total,
              "step0_identity": ident, "epoch_loss": [], "epoch_wall_s": [],
              "grad_norm": [],
              "method": "layer-wise remat + row-aligned cu_seqlens windows; "
                        "fla chunk_kda autograd recurrence"}
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
            scaled = loss * (p["n_ce"] / n_ce_total)
            scaled.backward()
            g = st_leaf.grad.detach()
            for i in reversed(range(eng.n_layers)):
                st_in = bounds[i].detach().requires_grad_(True)
                Lw = eng._load_layer(i)
                st_out = eng.block_batch(i, Lw, st_in, p["ids"], p["seq_of"],
                                         p["pos_of"], p["starts"], p["lens"],
                                         p["cu"])
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
    report["training_tokens_per_s"] = round(EPOCHS * S / wall, 1)
    save_cap(final=True)
    print(f"[train] CAP saved: {OUT}/lora.safetensors; "
          f"loss {report['epoch_loss'][0]} -> {report['epoch_loss'][-1]} "
          f"over {EPOCHS} epochs, wall {wall/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
