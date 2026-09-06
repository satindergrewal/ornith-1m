#!/usr/bin/env python3
"""dsv4_capture.py - self-capture producer for the DSV4 k35 encode.

No DSV4 teacher exists, so the calibration comes from the CAPPED model
itself: the dsv4_full engine (cap LoRA folded via attach_lora) runs over
a document corpus, and an instrumented ffn records, per routed layer, the
router input + routing decision for every token:

    hidden.bf16.bin        rows x 4096 x 2   (x entering ffn, bf16)
    topk_ids.u16le.bin     rows x 6 x 2      (sel, u16 LE)
    topk_weights.f32le.bin rows x 6 x 4      (w after renorm x scale 1.5)

Disk at 250k tokens: ~2GB/layer x 43 layers = ~88GB on the volume.

Integrity model:
  * The instrumented ffn mirrors dsv4_full.DSV4Full.ffn verbatim with record
    lines added. Window 0, check_layer 0 runs through BOTH the original and
    the instrumented path; outputs must be bitwise equal or capture refuses
    to start - the instrument may not change the arm it measures.
  * Roles are document-disjoint: fit / conditional-fit / selection /
    confirmation. selection+confirmation rows never feed encode fitting.
  * The manifest seals every payload with sha256; geometry comes from
    dsv4_geometry's discovered constants, never literals here.
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

SCHEMA = "quant-pipeline.dsv4-capture.v1"
ROLES = ("fit", "conditional-fit", "selection", "confirmation")
MAX_WINDOW_TOKENS = 4096
ROLE_CYCLE = ["fit", "fit", "fit", "fit", "conditional-fit",
              "selection", "confirmation"]


def canonical_sha(body, seal_field):
    core = {k: v for k, v in body.items() if k != seal_field}
    blob = json.dumps(core, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class FlatCaptureStore:
    """Append-only per-layer payload writers under one root."""

    def __init__(self, root, layers):
        self.root = root
        self.layers = list(layers)
        os.makedirs(root, exist_ok=True)
        self.parts = {}
        for L in self.layers:
            d = os.path.join(root, f"layer-{L}")
            os.makedirs(d, exist_ok=True)
            self.parts[L] = {
                "hidden": open(os.path.join(d, "hidden.bf16.bin"), "ab"),
                "ids": open(os.path.join(d, "topk_ids.u16le.bin"), "ab"),
                "w": open(os.path.join(d, "topk_weights.f32le.bin"), "ab"),
            }

    def append(self, layer, x, sel, w):
        p = self.parts[layer]
        p["hidden"].write(x.detach().to(torch.bfloat16).cpu()
                          .numpy().tobytes())
        p["ids"].write(sel.detach().cpu().numpy().astype("<u2").tobytes())
        p["w"].write(w.detach().cpu().numpy().astype("<f4").tobytes())

    def finalize(self, geometry, roles, windows, rows_per_layer, producer):
        files = {}
        for L in self.layers:
            d = f"layer-{L}"
            files[str(L)] = {}
            for key, fname, fh in (
                ("hidden_bf16", "hidden.bf16.bin", self.parts[L]["hidden"]),
                ("topk_ids_u16le", "topk_ids.u16le.bin", self.parts[L]["ids"]),
                ("topk_weights_f32le", "topk_weights.f32le.bin",
                 self.parts[L]["w"]),
            ):
                fh.close()
                path = os.path.join(self.root, d, fname)
                files[str(L)][key] = {
                    "path": f"{d}/{fname}",
                    "bytes": os.path.getsize(path),
                    "sha256": hashlib.sha256(
                        open(path, "rb").read()).hexdigest(),
                }
        body = {
            "schema": SCHEMA,
            "producer": producer,
            "geometry": geometry,
            "layers": self.layers,
            "roles": list(roles),
            "windows": windows,
            "rows_per_layer": rows_per_layer,
            "files": files,
        }
        body["capture_sha256"] = canonical_sha(body, "capture_sha256")
        json.dump(body, open(os.path.join(
            self.root, "capture-manifest.json"), "w"), indent=1)
        return body


def instrument(eng, store, check_layer=0):
    """Replace eng.ffn with a recording twin. The twin computes routing
    exactly as the engine does, records (x, sel, w), and delegates the
    expert compute to _ffn_body (engine tail). On the first check_layer
    call the original ffn also runs and both outputs must be bitwise
    equal, else capture aborts."""
    original = eng.ffn.__func__ if hasattr(eng.ffn, "__func__") else None
    if original is None:
        raise SystemExit("[capture] cannot unwrap engine ffn")
    state = {"checked": False}

    def capture_ffn(i, L, x, ids):
        xf = x.float()
        logits = xf @ L["ffn.gate.weight"].float().T
        scores = F.softplus(logits).sqrt()
        if i < eng.hash_layers:
            sel = L["ffn.gate.tid2eid"][ids]
        else:
            sel = (scores + L["ffn.gate.bias"]).topk(eng.exp_k, -1).indices
        w = scores.gather(1, sel)
        w = w / (w.sum(-1, keepdim=True)) * eng.route_scale
        if not state["checked"] and i == check_layer:
            y_ref = original(eng, i, L, x, ids)
            y_new = _ffn_body(eng, i, x, L, sel, w)
            if not bool(torch.equal(y_ref, y_new)):
                raise SystemExit(
                    "[capture] INSTRUMENT MISMATCH: twin ffn != engine ffn")
            state["checked"] = True
            store.append(i, x, sel, w)
            return y_ref
        store.append(i, x, sel, w)
        return _ffn_body(eng, i, x, L, sel, w)

    eng.ffn = capture_ffn


def _ffn_body(eng, i, x, L, sel, w):
    """Engine ffn compute tail given precomputed (sel, w): shared expert
    + routed experts, route-first dequant. Uses the engine's own swiglu/
    load_expert so LoRA folding and numerics are the engine's, not a copy."""
    y = eng.swiglu(x, L["ffn.shared_experts.w1.weight"],
                   L["ffn.shared_experts.w2.weight"],
                   L["ffn.shared_experts.w3.weight"],
                   li=i, prefix="ffn.shared_experts")
    for e in sel.flatten().unique():
        w_e = (w * (sel == e)).sum(-1)
        rows = w_e.nonzero().flatten()
        if rows.numel() == 0:
            continue
        w1, w2, w3 = eng.L.load_expert(i, int(e))
        y[rows] += w_e[rows, None] * eng.swiglu(x[rows], w1, w2, w3)
        del w1, w2, w3
    return y.to(eng.act_dtype)


def load_corpus(path):
    docs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        docs.append((str(d["document_id"]), str(d["text"])))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="/model/tokenizer.json")
    ap.add_argument("--lora", default="/wd/cap/lora.safetensors")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--target-tokens", type=int, default=250_000)
    ap.add_argument("--smoke", action="store_true",
                    help="3 windows, then per-expert coverage report")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    from dsv4_full import DSV4Full
    from full_loader import StreamingDSV4
    from tc_lora_train import SCALE
    from dsv4_geometry import Geometry, LORA_SCALE

    g = Geometry("/model", "/wd/tensor_meta.json", args.lora)
    assert LORA_SCALE == SCALE, "scale drift between geometry and trainer"

    tok = Tokenizer.from_file(args.tokenizer)
    docs = load_corpus(args.corpus)

    windows = []
    for did, text in docs:
        ids = tok.encode(text).ids[:args.target_tokens * 2]
        for s in range(0, len(ids), MAX_WINDOW_TOKENS):
            piece = ids[s:s + MAX_WINDOW_TOKENS]
            if len(piece) >= 256:  # skip tiny tails
                windows.append((did, piece))
    if args.smoke:
        windows = windows[:3]
    elif args.max_windows:
        windows = windows[:args.max_windows]
    doc_role = {}
    total = sum(len(w[1]) for w in windows)
    print(f"[cap] docs {len(docs)} windows {len(windows)} tokens {total}",
          flush=True)

    eng = DSV4Full(StreamingDSV4())
    n = eng.attach_lora(args.lora, SCALE, fold=True)
    print(f"[cap] folded {n} adapters", flush=True)

    store = FlatCaptureStore(args.out, range(g.n_layers))
    instrument(eng, store)

    journal, cursor, t0 = [], 0, time.time()
    for wi, (did, ids) in enumerate(windows):
        if did not in doc_role:
            doc_role[did] = ROLE_CYCLE[len(doc_role) % len(ROLE_CYCLE)]
        eng.forward(torch.tensor(ids, dtype=torch.long, device=eng.dev))
        journal.append({"window_index": wi, "rows": len(ids),
                        "role": doc_role[did], "document_id": did})
        cursor += len(ids)
        if (wi + 1) % 5 == 0 or wi == len(windows) - 1:
            el = time.time() - t0
            print(f"[cap] window {wi+1}/{len(windows)} rows {cursor} "
                  f"({el:.0f}s, {cursor/max(el,1):.0f} tok/s)", flush=True)

    manifest = store.finalize(
        geometry={"hidden_size": g.hidden, "experts": g.n_experts,
                  "top_k": g.top_k, "hash_layers": list(g.hash_layers)},
        roles=ROLES, windows=journal, rows_per_layer=cursor,
        producer="dsv4_full + attach_lora fold; corpus " +
                 os.path.basename(args.corpus))
    print(f"[cap] DONE rows/layer {cursor} -> {args.out}", flush=True)

    if args.smoke:
        for probe_layer in (0, 20, 42):
            arr = np.memmap(os.path.join(
                args.out, f"layer-{probe_layer}", "topk_ids.u16le.bin"),
                dtype="<u2", mode="r").reshape(-1, g.top_k)
            cov = np.bincount(arr.flatten(), minlength=g.n_experts)
            zero = int((cov == 0).sum())
            print(f"[cap] smoke L{probe_layer}: min/expert {cov.min()} "
                  f"max {cov.max()} zero-coverage {zero}", flush=True)


if __name__ == "__main__":
    main()
