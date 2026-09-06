#!/usr/bin/env python3
"""dsv4_build_inventory.py - sealed release inventory over the PACKED master.

Port of the GLM-5.3 campaign's inventory builder (k35_build_inventory.py in
the k35-dsv4-study port source; walker mirror cited per function) for the
DeepSeek V4 Flash packed master. DSV4 deltas from the port source:

  - the master is the PACKED FP8/MXFP4 artifact at the model root, NOT a
    folded BF16 tree: the tensor index is /wd/tensor_meta.json
    (name -> [dtype, shape, shard]), not model.safetensors.index.json
  - source_bytes is the exact on-disk packed byte range of each tensor
    (header data_offsets), never shape x dtype-itemsize math; the packed
    layout is the ground truth this inventory describes
  - the dtype domain is the packed-source domain
    (F8_E4M3, F8_E8M0, I8, BF16, F32, I64; dsv4_capped_source dispatch)
  - scope classification is master-grammar: routed =
    layers.{i}.ffn.experts.{e}.w{1,2,3}.weight over the discovered layer
    surface; mtp_routed = mtp.* experts; native = everything else
    (attn, shared experts, routers, aligner/vision, norms, scales)
  - the identity binding is the capped-source identity (lora_sha256,
    lora_scale, lora_sites from dsv4_capped_source.CappedSource.identity),
    because the encode reads this master THROUGH CappedSource;
    model_revision = sha256 over canonical JSON of
    {shard_sha256, lora_sha256, lora_scale} (no git revision exists for a
    packed working artifact)
  - the checkpoint field (the packed master root) is written at BUILD
    time, deleting the patch step the port source needed
    (k35_patch_inventory.py exists only for that reason)
  - census asserts are re-derived from dsv4_common.G (live-discovered):
    43 x 256 x 3 = 33,024 main routed tensors and 3 x 256 x 3 = 2,304 mtp
    routed tensors, never pasted literals

Self-check: the sealed body is validated by the real consumer
(dsv4_uniform_k35._inventory_surfaces) before the file is written, mirroring
the port source's builder tail (k35_build_inventory.py:149-158).

Run inside the encode container (cwd /wd with the pod engine files; the
capped source is imported lazily after the shard walk). Output:
<work-root>/inventory.json.

ASCII only. No em-dashes. No network. Writes only under --work-root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dsv4_common as common
import dsv4_uniform_k35 as k35
from dsv4_common import die
from dsv4_geometry import MTP_ROUTED, ROUTED

INVENTORY_SCHEMA = k35.INVENTORY_SCHEMA   # quant-pipeline.dsv4-release-inventory.v1
WORK_ROOT_DEFAULT = os.environ.get("DSV4_WORK_ROOT", "/workspace/dsv4-work")


def shard_header_and_sha(path: Path) -> tuple[dict, str, dict[str, str]]:
    """One streaming pass: file sha256 + per-tensor payload sha256 + header.

    Byte-faithful mirror of the port source's walker
    (k35_build_inventory.py:36-69).  Tensor hashes are over the exact file
    byte range of each tensor ([8+header_len+start, 8+header_len+end) per
    the header's data_offsets), so no library semantics are involved.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        prefix = handle.read(8)
        digest.update(prefix)
        header_len = struct.unpack("<Q", prefix)[0]
        header_bytes = handle.read(header_len)
        digest.update(header_bytes)
        header = json.loads(header_bytes)
        spans = {
            name: (row["data_offsets"][0], row["data_offsets"][1])
            for name, row in header.items()
            if name != "__metadata__"
        }
        tensor_hashes = {name: hashlib.sha256() for name in spans}
        base = 8 + header_len
        position = 0
        total = path.stat().st_size - base
        chunk = 1 << 24
        while position < total:
            block = handle.read(min(chunk, total - position))
            if not block:
                raise RuntimeError(f"short read on {path}")
            digest.update(block)
            low, high = position, position + len(block)
            for name, (start, end) in spans.items():
                if end <= low or start >= high:
                    continue
                tensor_hashes[name].update(
                    block[max(0, start - low):min(len(block), end - low)])
            position = high
    return header, digest.hexdigest(), {
        name: hasher.hexdigest() for name, hasher in tensor_hashes.items()}


def classify_scope(name: str) -> str:
    """Master-grammar scope authority: regex re-derivation over the
    discovered surface (port source used a split-index convention pinned to
    its HF grammar, k35_build_inventory.py:97-105; here the regexes from
    dsv4_geometry are the single authority)."""
    match = ROUTED.fullmatch(name)
    if match is not None:
        layer = int(match.group(1))
        if layer not in common.MAIN_LAYERS:
            die(f"routed name outside the discovered layer surface: {name}")
        return "routed_expert"
    if MTP_ROUTED.fullmatch(name) is not None:
        return "mtp_routed_expert"
    return "native"


def build_inventory(*, device: str = "cpu") -> dict:
    """Walk the packed master and seal the release inventory.

    Paths come from dsv4_common (DSV4_MODEL / DSV4_META / DSV4_LORA env
    defaults) because dsv4_common.G is the live-discovered geometry
    authority: the census asserts compare this walk against the master G
    validated, so a divergent path would be a lie, not an override.
    """
    model_dir = Path(common.MODEL_DIR)
    meta_path = Path(common.META_PATH)
    lora_path = Path(common.LORA_PATH)
    if not model_dir.is_dir():
        die(f"packed master root is absent: {model_dir}")
    if not meta_path.is_file():
        die(f"tensor meta is absent: {meta_path}")

    meta_document = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = meta_document.get("meta") if isinstance(meta_document, dict) else None
    if not isinstance(meta, dict) or not meta:
        die(f"tensor meta lacks the meta map: {meta_path}")
    by_shard: dict[str, list[str]] = {}
    for name, entry in meta.items():
        if (
            not isinstance(entry, list)
            or len(entry) != 3
            or not isinstance(entry[0], str)
            or not isinstance(entry[1], list)
            or not isinstance(entry[2], str)
        ):
            die(f"meta row is malformed: {name}")
        by_shard.setdefault(entry[2], []).append(name)
    shards = sorted(by_shard)
    print(
        f"walking {len(shards)} shards, {len(meta)} tensors from {meta_path}",
        flush=True,
    )

    tensors: list[dict] = []
    shard_sha256: dict[str, str] = {}
    seen: set[str] = set()
    for number, shard in enumerate(shards):
        path = model_dir / shard
        if not path.is_file():
            die(f"shard listed in meta is absent: {path}")
        header, file_digest, payload_shas = shard_header_and_sha(path)
        shard_sha256[shard] = file_digest
        header_rows = {
            name: row
            for name, row in header.items()
            if name != "__metadata__"
        }
        if set(header_rows) != set(by_shard[shard]):
            missing = sorted(set(by_shard[shard]) - set(header_rows))
            extra = sorted(set(header_rows) - set(by_shard[shard]))
            die(
                f"meta/header placement differs in {shard}: "
                f"{len(missing)} missing (first {missing[0] if missing else None}), "
                f"{len(extra)} extra (first {extra[0] if extra else None})"
            )
        for name in sorted(header_rows):
            row = header_rows[name]
            if name in seen:
                raise RuntimeError(f"duplicate tensor across shards: {name}")
            seen.add(name)
            dtype, shape_raw = meta[name][0], meta[name][1]
            if row.get("dtype") != dtype:
                die(
                    f"dtype drift between meta and shard header for {name}: "
                    f"{dtype!r} vs {row.get('dtype')!r}"
                )
            header_shape = [int(v) for v in row["shape"]]
            meta_shape = [int(v) for v in shape_raw]
            if header_shape != meta_shape:
                die(f"shape drift between meta and shard header for {name}")
            start, end = row["data_offsets"]
            packed_bytes = int(end) - int(start)
            if packed_bytes < 0:
                die(f"negative payload span for {name}")
            tensors.append(
                {
                    "tensor_name": name,
                    "scope": classify_scope(name),
                    "dtype": dtype,
                    "shape": meta_shape,
                    "source_bytes": packed_bytes,
                    "source_payload_sha256": payload_shas[name],
                    "shard": shard,
                }
            )
        if (number + 1) % 20 == 0 or number + 1 == len(shards):
            print(f"  {number + 1}/{len(shards)} shards hashed", flush=True)

    if seen != set(meta):
        missing = set(meta) - seen
        extra = seen - set(meta)
        raise RuntimeError(
            f"meta closure differs: missing={len(missing)} extra={len(extra)}")

    # Census: every count re-derived from the discovered geometry.
    main_rows = [row for row in tensors if row["scope"] == "routed_expert"]
    mtp_rows = [
        row for row in tensors if row["scope"] == "mtp_routed_expert"]
    native_rows = [row for row in tensors if row["scope"] == "native"]
    if len(main_rows) != common.G.main_routed_tensors:
        die(
            f"main routed census differs from the discovered geometry: "
            f"{len(main_rows)} != {common.G.main_routed_tensors}"
        )
    if len(mtp_rows) != common.G.mtp_routed_tensors:
        die(
            f"mtp routed census differs from the discovered geometry: "
            f"{len(mtp_rows)} != {common.G.mtp_routed_tensors}"
        )
    discovered = sorted(
        {
            int(ROUTED.fullmatch(row["tensor_name"]).group(1))
            for row in main_rows
        }
    )
    if discovered != list(common.MAIN_LAYERS):
        die(
            "discovered layer surface differs: "
            f"{discovered[:3]}...{discovered[-3:]}"
        )
    dtype_census = Counter(row["dtype"] for row in tensors)
    if set(dtype_census) - set(k35.SOURCE_DTYPES):
        die(
            "dtype outside the packed-source domain: "
            f"{sorted(set(dtype_census) - set(k35.SOURCE_DTYPES))}"
        )
    for dtype in k35.SOURCE_DTYPES:
        if dtype_census[dtype] <= 0:
            die(
                f"packed-source dtype {dtype} is absent from the master; "
                f"census: {dict(sorted(dtype_census.items()))}"
            )

    # Identity binding: the capped-source identity, because the encode
    # reads this master through CappedSource.  Imported lazily (module
    # stays torch-free at import; the pod engine files sit at cwd /wd,
    # the same pattern as the phase-5 probe driver, dsv4_probe_driver.py:
    # 296-307).  CPU device: this build only hashes, it never dequants.
    from dsv4_capped_source import CappedSource

    source = CappedSource(
        model_dir=str(model_dir),
        meta_path=str(meta_path),
        lora_path=str(lora_path),
        device=device,
    )
    identity = dict(source.identity)
    if identity.get("lora_sites") != len(common.G.lora_sites or ()):
        die("capped-source lora site census differs from the discovered geometry")

    model_revision = common.sha256_bytes(
        common.canonical_json(
            {
                "shard_sha256": shard_sha256,
                "lora_sha256": identity["lora_sha256"],
                "lora_scale": identity["lora_scale"],
            }
        )
    )

    body = {
        "schema": INVENTORY_SCHEMA,
        "seal_mode": "full-shard-sha256",
        "checkpoint": str(model_dir),
        "model_revision": model_revision,
        "identity": {
            "lora_sha256": identity["lora_sha256"],
            "lora_scale": identity["lora_scale"],
            "lora_sites": identity["lora_sites"],
        },
        "config_sha256": common.sha256_file(model_dir / "config.json"),
        "tensor_meta": {
            "path": str(meta_path),
            "sha256": common.sha256_file(meta_path),
        },
        "geometry": {
            "model_type": common.G.cfg["model_type"],
            "architecture": common.G.cfg["architectures"],
            "main_layers": len(common.MAIN_LAYERS),
            "all_main_layers_routed": True,
            "hash_routed_layers": list(common.G.hash_layers),
            "mtp_modules": list(common.G.mtp_modules),
            "routed_experts": common.NUM_EXPERTS,
            "top_k": common.TOP_K,
            "hidden_size": common.HIDDEN_SIZE,
            "moe_intermediate_size": common.INTERMEDIATE_SIZE,
            "discovered_layers": discovered,
        },
        "dtype_census": dict(sorted(dtype_census.items())),
        "scope_census": {
            "routed_expert": len(main_rows),
            "mtp_routed_expert": len(mtp_rows),
            "native": len(native_rows),
        },
        "shard_sha256": shard_sha256,
        "tensors": sorted(tensors, key=lambda row: row["tensor_name"]),
    }
    body["inventory_sha256"] = common.sha256_bytes(common.canonical_json(body))

    # Self-check with the real consumer BEFORE writing (mirror of the port
    # source's builder tail, k35_build_inventory.py:149-158).
    checked_main, checked_mtp, checked_native = k35._inventory_surfaces(body)
    print(
        f"surfaces check OK: routed={len(checked_main)} "
        f"mtp={len(checked_mtp)} native={len(checked_native)}",
        flush=True,
    )
    return body


def main() -> int:
    parser = argparse.ArgumentParser(
        description="build the sealed dsv4 release inventory over the packed master"
    )
    parser.add_argument(
        "--work-root",
        default=WORK_ROOT_DEFAULT,
        type=Path,
        help="inventory output root (default env DSV4_WORK_ROOT or /workspace/dsv4-work)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="capped-source construction device (the build only hashes; default cpu)",
    )
    args = parser.parse_args()
    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    body = build_inventory(device=args.device)
    common.write_json(work_root / "inventory.json", body)
    print("INVENTORY_OK", body["inventory_sha256"][:16], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
