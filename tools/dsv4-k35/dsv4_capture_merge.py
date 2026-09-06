#!/usr/bin/env python3
"""dsv4_capture_merge.py - merge data-parallel capture shards into one store.

Each shard pod ran dsv4_capture over a DISJOINT document slice into its own
root. This driver concatenates the per-layer payload bins in shard order,
re-indexes the window journal, re-hashes every artifact, and seals ONE
capture-manifest.json. Refuses mismatched geometry/schemas/layers.
"""

import argparse
import hashlib
import json
import os
import shutil

from dsv4_capture import SCHEMA, canonical_sha


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True,
                    help="capture roots in window order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifests = []
    for root in args.shards:
        m = json.load(open(os.path.join(root, "capture-manifest.json")))
        assert m["schema"] == SCHEMA, f"schema drift in {root}"
        manifests.append(m)
    base = manifests[0]
    for m in manifests[1:]:
        assert m["geometry"] == base["geometry"], "geometry drift"
        assert m["layers"] == base["layers"], "layer set drift"

    os.makedirs(args.out, exist_ok=True)
    layers = base["layers"]
    files = {}
    for L in layers:
        d = f"layer-{L}"
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
        files[str(L)] = {}
        for key, fname in (
            ("hidden_bf16", "hidden.bf16.bin"),
            ("topk_ids_u16le", "topk_ids.u16le.bin"),
            ("topk_weights_f32le", "topk_weights.f32le.bin"),
        ):
            dst = os.path.join(args.out, d, fname)
            with open(dst, "wb") as out:
                for root, m in zip(args.shards, manifests):
                    src = os.path.join(root, m["files"][str(L)][key]["path"])
                    with open(src, "rb") as f:
                        shutil.copyfileobj(f, out, 1 << 22)
            files[str(L)][key] = {
                "path": f"{d}/{fname}",
                "bytes": os.path.getsize(dst),
                "sha256": sha_file(dst),
            }

    windows, cursor, widx = [], 0, 0
    for m in manifests:
        for w in m["windows"]:
            windows.append({"window_index": widx, "rows": w["rows"],
                            "role": w["role"], "document_id": w["document_id"]})
            cursor += w["rows"]
            widx += 1
    roles = sorted({w["role"] for w in windows})
    body = {
        "schema": SCHEMA,
        "producer": "merged data-parallel shards: " + ", ".join(args.shards),
        "geometry": base["geometry"],
        "layers": layers,
        "roles": roles,
        "windows": windows,
        "rows_per_layer": cursor,
        "files": files,
    }
    body["capture_sha256"] = canonical_sha(body, "capture_sha256")
    json.dump(body, open(os.path.join(
        args.out, "capture-manifest.json"), "w"), indent=1)
    print(f"[merge] {len(args.shards)} shards -> {args.out}: "
          f"{cursor} rows/layer, {len(windows)} windows, roles {roles}")


if __name__ == "__main__":
    main()
