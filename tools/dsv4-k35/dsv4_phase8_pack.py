#!/usr/bin/env python3
"""dsv4_phase8_pack.py - phase 8: assemble the DSV4 Flash mixed K3/K4 EXL3
artifact from the sealed per-layer payload stores.

Port of the GLM-5.3 campaign's phase-8 pair: the driver k35_phase8_pack.py
(port source, k35-dsv4-study) and the materializer it drove
(quant_pipeline/checkpoint/glm53_mcg_materializer.py). Per-mirror citations
name the port-source lines. DSV4 deltas from the port source, the
load-bearing ones first:

  - NATIVES ARE DEQUANTED, NOT BYTE-COPIED. The GLM source was a BF16 tree
    and its materializer proved every native an exact byte copy
    (official_bf16_native_copy, glm53_mcg_materializer.py:282-298, verified
    again at :654-661). The DSV4 source is the PACKED FP8/MXFP4 master read
    through dsv4_capped_source.CappedSource: F8_E4M3 natives dequant to
    bf16 with the cap LoRA folded at the 258 sites, I8 natives dequant to
    bf16, BF16 natives pass through, F32 stays f32, I64 tid2eid stays raw
    (dsv4_capped_source.py:82-103). No native output byte ever equals the
    packed source bytes, so the GLM byte-exact closure is replaced by the
    inventory identity binding (lora_sha256/lora_scale/lora_sites), the
    inventory's master config/meta hash binding, and per-tensor dtype
    policy asserts. The dequant path itself is proven by
    dsv4_source_parity.py, not re-proven here.
  - MTP IS NATIVE. mtp.{0,1,2} experts dequant to bf16 through
    CappedSource.expert_tensor and materialize under their master names
    (dsv4_uniform_k35 mtp_scope_v1 policy; the GLM mtp adapter receipt
    path, k35_phase8_pack.py:62-67 and k35_surface.py:252-279, does not
    exist here).
  - SCALE PARTNERS ARE DROPPED. F8_E8M0 rows are consumed by dequant and
    never written; the count binds to the launch plan's
    native_copy_contract.dequant_scale_tensors_consumed_not_copied.
  - SHARDS GROUP BY THE PACKED MASTER'S OWN shard FIELD (tensor_meta
    placement, sealed per tensor in the inventory), not an HF
    model.safetensors.index.json the master does not have (GLM
    _source_index, glm53_mcg_materializer.py:155-170).
  - config.json carries torch_dtype bfloat16, the master's FP8 block
    quantization_config is stripped (it describes quant that no longer
    exists in the output), and an honest minimal dsv4 mixed marker is
    emitted instead: bits "mixed_k34_per_tensor" plus per_module integer
    bits read from the sealed choices, under the dsv4 schema string
    DSV4_QUANT_CONFIG_SCHEMA. The GLM artifact's r7_routed_experts + scope
    strings (its reader's contract, e.g. the moe_layers block) are
    deliberately NOT emitted: no dsv4 mixed-rate reader exists yet, and a
    mixed-rate EXL3 reader over the master w1/w2/w3 module keys is a
    documented PRECONDITION for serving this artifact.
  - total_size in model.safetensors.index.json is the sum of materialized
    tensor data bytes (numel * element_size, HF convention; GLM mirror
    glm53_mcg_materializer.py:834-835, 695); per-shard file sizes live in
    the shard receipts, never in the index.
  - resume carries an explicit .done marker per shard next to the sealed
    receipt (GLM resumed on receipt presence alone,
    glm53_mcg_materializer.py:407-423); a receipt without its .done marker
    is a crash in progress and the shard is re-derived.

Tensor names stay MASTER names end to end (layers.{L}.ffn.experts.{E}.
{w1,w2,w3}.weight); packed outputs are {module}.{trellis,suh,svh,mcg} with
module = the master weight name minus .weight. No gate_proj/up_proj/
down_proj rewrite ever happens at this layer (that vocabulary is the codec
boundary only, dsv4_worker.PROJECTION_ROLE).

This driver writes NO state transition: it only reads plan/state and
writes the artifact. Binding the k35_packed state is a later step;
dsv4_uniform_k35.seal_k35_packed wants packed_checkpoint_receipt_sha256 =
the materialization receipt seal and native_copy_receipt_sha256 = the
native-copy receipt seal, both produced here.

Usage (inside the encode container, cwd <work-root>):

  python3 dsv4_phase8_pack.py             # dry run: surface + pack plan seal
  python3 dsv4_phase8_pack.py --execute    # write the artifact

ASCII only. No em-dashes. CODE ONLY off-pod: importing dsv4_common
resolves Geometry from the live master (env DSV4_MODEL/DSV4_META/
DSV4_LORA); run inside the encode container.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dsv4_common as common
import dsv4_uniform_k35 as k35
import dsv4_surface
from dsv4_common import die
from dsv4_geometry import MTP_ROUTED, ROUTED

# ---------------------------------------------------------------------------
# NEW SURFACE schemas (never reuse glm53 strings; no k{bits} interpolation:
# the artifact is mixed-rate by construction).
# ---------------------------------------------------------------------------

PACK_PLAN_SCHEMA = "quant-pipeline.dsv4-k35-materialization-plan.v1"
PACK_SHARD_RECEIPT_SCHEMA = (
    "quant-pipeline.dsv4-k35-materialized-shard-receipt.v1")
PACK_RECEIPT_SCHEMA = "quant-pipeline.dsv4-k35-materialization-receipt.v1"
NATIVE_COPY_RECEIPT_SCHEMA = (
    "quant-pipeline.dsv4-k35-native-copy-receipt.v1")
DSV4_QUANT_CONFIG_SCHEMA = "quant-pipeline.dsv4-k35-quantization-config.v1"

# The EXL3 storage group suffixes (port mirror of
# glm53_mcg_materializer.py:41; identical to the sealed choice object set,
# dsv4_common.py:669-671).
PACKED_SUFFIXES = ("trellis", "suh", "svh", "mcg")

# MCG marker constants (mirror of the sealed choice decoder block,
# dsv4_common.py:638-641).
MCG_MULTIPLIER_HEX = "0xCBAC1FED"
MCG_MARKER_SIGNED_INT32 = -877912083

# Output dtype per packed-master source dtype (the CappedSource dispatch,
# dsv4_capped_source.py:82-103). F8_E8M0 is deliberately absent: scale
# partners are consumed by dequant and never materialized.
NATIVE_OUTPUT_DTYPE = {
    "F8_E4M3": "torch.bfloat16",
    "I8": "torch.bfloat16",
    "BF16": "torch.bfloat16",
    "F32": "torch.float32",
    "I64": "torch.int64",
}
SCALE_DTYPE = "F8_E8M0"

ORIGIN_NATIVE = "capped_source_native_dequant_fold"
ORIGIN_MTP = "capped_source_mtp_native_dequant"
ORIGIN_PACKED = "sealed_exl3_mcg_packed_choice"

# Same default as dsv4_build_inventory/dsv4_phase4_plan (their
# WORK_ROOT_DEFAULT); common.DEFAULT_WORK_ROOT points at the bare
# /workspace parent.
WORK_ROOT_DEFAULT = Path(
    os.environ.get("DSV4_WORK_ROOT", "/workspace/dsv4-work"))

# Artifact-facing JSON follows HF file conventions (indent 2, sorted, one
# trailing newline); campaign receipts keep dsv4_common.write_json.
_AUX_EXCLUDED = {
    "config.json",
    "quantization_config.json",
    "model.safetensors.index.json",
    # The master's tensor_meta.json describes the PACKED layout and is
    # meaningless for the bf16 output artifact; it is never copied.
    "tensor_meta.json",
}


def packed_tensor_name(source_weight_name: str, suffix: str) -> str:
    """Map a master weight name to the packed output tensor key.

    Port mirror of glm53_mcg_materializer.py:86-91 with master names:
    layers.0.ffn.experts.7.w1.weight -> layers.0.ffn.experts.7.w1.trellis.
    """
    if (
        not source_weight_name.endswith(".weight")
        or suffix not in PACKED_SUFFIXES
    ):
        die("packed tensor requires a master .weight and EXL3 suffix")
    return f"{source_weight_name[:-len('.weight')]}.{suffix}"


# ---------------------------------------------------------------------------
# Atomic artifact writes (port mirror of glm53_mcg_materializer.py:359-387
# and the atomic_write discipline; fsync file then rename then fsync dir).
# ---------------------------------------------------------------------------


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, body: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path, json.dumps(body, indent=2, sort_keys=True).encode() + b"\n")


def _tensor_record(
    *,
    name: str,
    value: Any,
    origin: str,
    source_tensor_name: str,
    source_dtype: str,
    choice_sha256: str | None,
    shape_authority: str,
) -> dict[str, Any]:
    # Port mirror of glm53_mcg_materializer.py:235-253 plus the DSV4
    # source-dtype and shape-authority fields the dequant policy needs.
    return {
        "name": name,
        "dtype": str(value.dtype),
        "shape": [int(item) for item in value.shape],
        "bytes": int(value.numel() * value.element_size()),
        "payload_sha256": common.tensor_sha256(value),
        "origin": origin,
        "source_tensor_name": source_tensor_name,
        "source_dtype": source_dtype,
        "choice_sha256": choice_sha256,
        "shape_authority": shape_authority,
    }


# ---------------------------------------------------------------------------
# Shard tensor assembly (port mirror of _load_shard_tensors,
# glm53_mcg_materializer.py:255-334, re-derived for the packed master).
# ---------------------------------------------------------------------------


def _native_value(source: Any, row: Mapping[str, Any]) -> Any:
    """One native tensor through the capped source, policy-asserted."""
    name = str(row["tensor_name"])
    dtype = str(row.get("dtype"))
    expected = NATIVE_OUTPUT_DTYPE.get(dtype)
    if expected is None:
        die(f"native row carries a non-materialized dtype: {name} {dtype}")
    if source is None:
        die(f"native row {name} needs the capped source (source is None)")
    value = source.load_native(name).detach().to("cpu").contiguous()
    if str(value.dtype) != expected:
        die(
            f"native output dtype differs from the dispatch policy: {name} "
            f"{value.dtype} != {expected}"
        )
    if dtype == "I8":
        # I8 masters are nibble-packed along K; the dequant expands the
        # packed span, so the output shape authority is the capped source
        # itself (the logical shape is asserted for experts, where the
        # geometry knows it; non-expert I8 shapes are recorded, not
        # guessed).
        return value
    if list(value.shape) != [int(v) for v in row.get("shape", ())]:
        die(
            f"native output shape differs from the master meta shape: "
            f"{name} {list(value.shape)} != {row.get('shape')}"
        )
    return value


def _mtp_value(source: Any, row: Mapping[str, Any]) -> Any:
    """One mtp.{0,1,2} expert tensor, dequanted bf16 under its own name."""
    name = str(row["tensor_name"])
    match = MTP_ROUTED.fullmatch(name)
    if match is None:
        die(f"mtp row does not match the master grammar: {name}")
    projection = match.group(3)
    expected_shape = list(common.G.logical_shapes[projection])
    if source is None:
        die(f"mtp row {name} needs the capped source (source is None)")
    value = (
        source.expert_tensor(name).detach().to("cpu").contiguous())
    if str(value.dtype) != "torch.bfloat16":
        die(f"mtp native dequant dtype differs: {name} {value.dtype}")
    if list(value.shape) != expected_shape:
        die(
            f"mtp native logical shape differs: {name} "
            f"{list(value.shape)} != {expected_shape}"
        )
    return value


def _verify_packed_choice_payload(
    choice: Mapping[str, Any], values: Mapping[str, Any]
) -> None:
    """Fail-closed payload geometry over the loaded choice objects.

    dsv4_common.verify_choice (called by the surface) re-sealed the row and
    re-hashed every object; this adds the trellis-bytes-vs-bits and marker
    checks the store only enforces at write time
    (dsv4_common.py:587-608, 592-595).
    """
    bits = int(choice["bits"])
    suh, svh = values["suh"], values["svh"]
    trellis, mcg = values["trellis"], values["mcg"]
    k, n = int(suh.numel()), int(svh.numel())
    if int(choice.get("param_count", -1)) != n * k:
        die("packed choice param_count disagrees with its vectors")
    expected_trellis_bytes = n * k * bits // 8
    actual = int(trellis.numel()) * int(trellis.element_size())
    if actual != expected_trellis_bytes:
        die(
            f"packed choice trellis bytes {actual} disagree with bits "
            f"{bits} geometry {n}x{k}"
        )
    if str(suh.dtype) != "torch.float16" or str(svh.dtype) != "torch.float16":
        die("packed choice scale vectors are not FP16")
    if (
        str(mcg.dtype) != "torch.int32"
        or int(mcg.numel()) != 1
        or int(mcg.reshape(-1)[0]) != MCG_MARKER_SIGNED_INT32
    ):
        die("packed choice is not marked with the MCG marker")


def _load_shard_tensors(
    *,
    source: Any,
    shard_rows: list[Mapping[str, Any]],
    surface: dsv4_surface.Dsv4Surface,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, int, int]:
    tensors: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    native_count = 0
    mtp_count = 0
    choice_count = 0
    for row in sorted(shard_rows, key=lambda item: str(item["tensor_name"])):
        source_name = str(row["tensor_name"])
        scope = row.get("scope")
        if scope == "native":
            if str(row.get("dtype")) == SCALE_DTYPE:
                # Consumed by dequant; never materialized.
                continue
            value = _native_value(source, row)
            if source_name in tensors:
                die(f"output tensor name collision: {source_name}")
            tensors[source_name] = value
            records.append(
                _tensor_record(
                    name=source_name,
                    value=value,
                    origin=ORIGIN_NATIVE,
                    source_tensor_name=source_name,
                    source_dtype=str(row["dtype"]),
                    choice_sha256=None,
                    shape_authority=(
                        "capped_source_dequant"
                        if str(row["dtype"]) == "I8"
                        else "master_meta_shape"
                    ),
                )
            )
            native_count += 1
            continue
        if scope == "mtp_routed_expert":
            value = _mtp_value(source, row)
            if source_name in tensors:
                die(f"output tensor name collision: {source_name}")
            tensors[source_name] = value
            records.append(
                _tensor_record(
                    name=source_name,
                    value=value,
                    origin=ORIGIN_MTP,
                    source_tensor_name=source_name,
                    source_dtype=str(row["dtype"]),
                    choice_sha256=None,
                    shape_authority="dsv4_geometry_logical_shape",
                )
            )
            mtp_count += 1
            continue
        if scope != "routed_expert":
            die(f"inventory row carries an unknown scope: {source_name}")
        match = ROUTED.fullmatch(source_name)
        if match is None:
            die(f"routed row does not match the master grammar: {source_name}")
        layer, expert, projection = (
            int(match.group(1)), int(match.group(2)), match.group(3))
        if (
            layer not in common.MAIN_LAYERS
            or not 0 <= expert < common.NUM_EXPERTS
            or projection not in common.PROJECTIONS
        ):
            die(f"routed inventory geometry differs: {source_name}")
        k35._routed_shape_gate(source_name, row.get("shape"))
        choice = surface.choice(layer, expert, projection)
        store = surface.store.store_for(layer)
        # The per-layer verifier re-seals the choice and re-hashes every
        # stored object (dsv4_common.py:654-675).
        verified = store.verify_choice(choice)
        values = {
            suffix: store.objects.load_tensor(
                verified["objects"][suffix]).contiguous()
            for suffix in PACKED_SUFFIXES
        }
        _verify_packed_choice_payload(verified, values)
        for suffix in PACKED_SUFFIXES:
            output_name = packed_tensor_name(source_name, suffix)
            if output_name in tensors:
                die(f"output tensor name collision: {output_name}")
            tensors[output_name] = values[suffix]
            records.append(
                _tensor_record(
                    name=output_name,
                    value=values[suffix],
                    origin=ORIGIN_PACKED,
                    source_tensor_name=source_name,
                    source_dtype=str(row["dtype"]),
                    choice_sha256=str(verified["choice_sha256"]),
                    shape_authority="sealed_choice_objects",
                )
            )
        choice_count += 1
    if set(tensors) != {record["name"] for record in records}:
        die("materialized shard tensor record census differs")
    return tensors, records, native_count, mtp_count, choice_count


# ---------------------------------------------------------------------------
# Write + verify one output shard (port mirror of
# glm53_mcg_materializer.py:337-387).
# ---------------------------------------------------------------------------


def _verify_output_shard(
    path: Path, records: Sequence[Mapping[str, Any]]
) -> None:
    from safetensors import safe_open

    if not path.is_file() or path.is_symlink():
        die(f"materialized shard is absent or symlinked: {path}")
    expected = {str(record["name"]): record for record in records}
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            die(f"materialized shard tensor census differs: {path.name}")
        for name, record in expected.items():
            value = handle.get_tensor(name)
            if (
                str(value.dtype) != record["dtype"]
                or list(value.shape) != record["shape"]
                or value.numel() * value.element_size() != record["bytes"]
                or common.tensor_sha256(value) != record["payload_sha256"]
            ):
                die(f"materialized tensor payload differs: {name}")


def _write_safetensors_atomic(
    path: Path, tensors: Mapping[str, Any], pack_plan_sha256: str
) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        save_file(
            dict(tensors),
            temporary,
            metadata={
                "format": "pt",
                "codec": "exl3-mcg",
                "pack_plan_sha256": pack_plan_sha256,
            },
        )
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _shard_receipt_path(output_root: Path, source_shard: str) -> Path:
    # Port mirror of glm53_mcg_materializer.py:390-391.
    return output_root / ".materialization" / "shards" / f"{source_shard}.json"


def _shard_done_path(output_root: Path, source_shard: str) -> Path:
    return output_root / ".materialization" / "shards" / f"{source_shard}.done"


def materialize_shard(
    *,
    pack_plan_sha256: str,
    source: Any,
    output_root: Path,
    source_shard: str,
    shard_rows: list[Mapping[str, Any]],
    surface: dsv4_surface.Dsv4Surface,
) -> dict[str, Any]:
    """Write or resume one source-aligned output shard.

    Resume (GLM mirror glm53_mcg_materializer.py:407-423 plus the .done
    marker): a shard is resumable only when the .done marker exists, its
    content is the receipt seal, the sealed receipt verifies and binds this
    pack plan, and the shard file still hashes to the receipt. A shard file
    WITHOUT its .done marker is a crash leftover and is always rewritten
    (its safetensors metadata may bind a previous pack plan, which no
    tensor-level check can see).
    """

    if Path(source_shard).name != source_shard:
        die(f"source shard path is unsafe: {source_shard}")
    output_path = output_root / source_shard
    receipt_path = _shard_receipt_path(output_root, source_shard)
    done_path = _shard_done_path(output_root, source_shard)
    if done_path.is_file() and receipt_path.is_file():
        receipt = common.load_json(receipt_path)
        seal = common.verify_seal(
            receipt,
            schema=PACK_SHARD_RECEIPT_SCHEMA,
            field="receipt_sha256",
            label=f"materialized shard receipt {source_shard}",
        )
        if (
            receipt.get("pack_plan_sha256") != pack_plan_sha256
            or receipt.get("source_shard") != source_shard
            or receipt.get("shard") != source_shard
            or receipt.get("complete") is not True
            or done_path.read_text(encoding="utf-8").strip() != seal
            or not output_path.is_file()
            or output_path.is_symlink()
            or output_path.stat().st_size
            != receipt.get("shard_bytes")
            or common.sha256_file(output_path) != receipt.get("shard_sha256")
        ):
            die(f"resumable shard receipt differs: {source_shard}")
        return receipt

    tensors, records, native_count, mtp_count, choice_count = (
        _load_shard_tensors(
            source=source, shard_rows=shard_rows, surface=surface))
    if output_path.exists() and not done_path.is_file():
        # A shard file without its .done marker is an untrusted crash
        # leftover: it may carry a previous pack plan's safetensors
        # metadata (which the tensor-level verify cannot see), so it is
        # always replaced, never verified in place.
        _write_safetensors_atomic(output_path, tensors, pack_plan_sha256)
        _verify_output_shard(output_path, records)
    elif output_path.exists():
        _verify_output_shard(output_path, records)
    else:
        _write_safetensors_atomic(output_path, tensors, pack_plan_sha256)
        _verify_output_shard(output_path, records)
    body = {
        "schema": PACK_SHARD_RECEIPT_SCHEMA,
        "pack_plan_sha256": pack_plan_sha256,
        "source_shard": source_shard,
        "shard": source_shard,
        "shard_bytes": output_path.stat().st_size,
        "shard_sha256": common.sha256_file(output_path),
        "native_tensor_count": native_count,
        "mtp_native_tensor_count": mtp_count,
        "routed_choice_count": choice_count,
        "output_tensor_count": len(records),
        "output_logical_bytes": sum(
            int(record["bytes"]) for record in records),
        "tensors": records,
        "complete": True,
    }
    receipt = common.seal(body, "receipt_sha256")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(receipt_path, receipt)
    _atomic_write_bytes(
        done_path, (receipt["receipt_sha256"] + "\n").encode("ascii"))
    return receipt


# ---------------------------------------------------------------------------
# Output index (port mirror of glm53_mcg_materializer.py:829-835; built
# over OUTPUT names, total_size from the written tensor data bytes).
# ---------------------------------------------------------------------------


def build_output_index(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    weight_map: dict[str, str] = {}
    total_size = 0
    for receipt in receipts:
        shard = str(receipt["shard"])
        for record in receipt["tensors"]:
            name = str(record["name"])
            if name in weight_map:
                die(f"output tensor name duplicated across shards: {name}")
            weight_map[name] = shard
            total_size += int(record["bytes"])
    return {"metadata": {"total_size": total_size}, "weight_map": weight_map}


# ---------------------------------------------------------------------------
# Quantization config + artifact config.json (DSV4 delta from
# glm53_mcg_materializer.py:455-503: the GLM uniform-bits qcfg and its
# r7_routed_experts/scope reader contract are replaced by the honest dsv4
# mixed marker with per-module integer bits; no reader exists yet).
# ---------------------------------------------------------------------------


def build_quantization_config(
    *,
    rate_census: Mapping[str, int],
    per_module_bits: Mapping[str, int],
    prior: Mapping[str, Any] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": DSV4_QUANT_CONFIG_SCHEMA,
        "quant_method": "exl3",
        "codebook": "mcg",
        "mixed": True,
        "bits": dsv4_surface.MIXED_BITS_MARKER,
        "allowed_bits": list(common.PER_TENSOR_ALLOWED_BITS),
        "rate": {
            "numerator": k35.RATE_NUMERATOR,
            "denominator": k35.RATE_DENOMINATOR,
        },
        "rate_census": dict(rate_census),
        "per_module_bits": dict(sorted(per_module_bits.items())),
        "module_key_rule": (
            "master weight name without .weight "
            "(layers.L.ffn.experts.E.w1|w2|w3)"
        ),
        "stored_suffixes": list(PACKED_SUFFIXES),
        "scope": "dsv4_main_routed_experts_only",
        "mtp_scope": "native_bf16_dequant_not_encoded",
        "non_routed_dtype_policy": "capped_source_dequant_fold_v1",
        "scale_tensor_policy": "consumed_by_dequant_dropped",
        "serving_reader_qualified": False,
        "serving_precondition": (
            "a dsv4 mixed-rate EXL3 reader over the master w1/w2/w3 module "
            "keys is a precondition for serving this artifact; none exists "
            "yet"
        ),
    }
    if prior is not None:
        body["original_quantization_config"] = copy.deepcopy(dict(prior))
    return body


def build_artifact_config(
    source_config: Mapping[str, Any],
    quantization_config: Mapping[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(dict(source_config))
    # The master's quantization_config (FP8 block quant) describes packed
    # bytes that no longer exist in the output; the dequant policy note and
    # the mixed marker replace it. The prior survives only nested inside
    # the emitted quantization_config as provenance.
    config.pop("quantization_config", None)
    config["torch_dtype"] = "bfloat16"
    config["quantization_config"] = copy.deepcopy(dict(quantization_config))
    return config


# ---------------------------------------------------------------------------
# Auxiliary file copy (port mirror of glm53_mcg_materializer.py:506-537;
# tensor_meta.json joins the exclusion set).
# ---------------------------------------------------------------------------


def _auxiliary_source_names(source_root: Path) -> set[str]:
    return {
        path.name
        for path in source_root.iterdir()
        if path.is_file()
        and not path.is_symlink()
        and path.name not in _AUX_EXCLUDED
        and not path.name.endswith(".safetensors")
        and path.suffix not in {".bin", ".ckpt", ".pt", ".pth"}
    }


def _copy_auxiliary_files(
    source_root: Path, output_root: Path
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for name in sorted(_auxiliary_source_names(source_root)):
        source = source_root / name
        destination = output_root / name
        expected = common.sha256_file(source)
        if destination.exists():
            if (
                not destination.is_file()
                or destination.is_symlink()
                or common.sha256_file(destination) != expected
            ):
                die(f"existing auxiliary file differs: {destination}")
        else:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{destination.name}.", dir=output_root)
            os.close(descriptor)
            try:
                shutil.copyfile(source, temporary)
                with open(temporary, "rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        copied.append(
            {
                "path": name,
                "bytes": destination.stat().st_size,
                "sha256": expected,
            }
        )
    return copied


# ---------------------------------------------------------------------------
# Native-copy receipt: the sealed native half of the artifact for the
# k35_packed state transition (dsv4_uniform_k35.seal_k35_packed wants a
# native_copy_receipt_sha256 distinct from the checkpoint receipt).
# ---------------------------------------------------------------------------


def build_native_copy_receipt(
    *,
    pack_plan_sha256: str,
    inventory: Mapping[str, Any],
    native_names: Sequence[str],
    mtp_names: Sequence[str],
    dropped_scale_count: int,
    native_output_bytes: int,
    mtp_output_bytes: int,
    per_shard: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    body = {
        "schema": NATIVE_COPY_RECEIPT_SCHEMA,
        "pack_plan_sha256": pack_plan_sha256,
        "inventory_sha256": str(inventory["inventory_sha256"]),
        "model_revision": str(inventory.get("model_revision", "")),
        "policy": "capped_source_dequant_fold_copy_v1",
        "mtp_policy": "native_copy_not_encoded",
        "scale_policy": "consumed_by_dequant_dropped",
        "native_fold_identity": copy.deepcopy(
            dict(inventory.get("identity", {}))),
        "native_tensor_count": len(native_names),
        "mtp_native_tensor_count": len(mtp_names),
        "dropped_scale_tensor_count": int(dropped_scale_count),
        "native_tensor_names_sha256": common.sha256_bytes(
            common.canonical_json(list(native_names))),
        "mtp_tensor_names_sha256": common.sha256_bytes(
            common.canonical_json(list(mtp_names))),
        "native_output_bytes": int(native_output_bytes),
        "mtp_output_bytes": int(mtp_output_bytes),
        "per_shard": dict(sorted(per_shard.items())),
        "byte_exact_vs_packed_source": False,
        "complete": True,
    }
    return common.seal(body, "receipt_sha256")


# ---------------------------------------------------------------------------
# Pack plan (port mirror of k35_phase8_pack.py:69-88 and the materializer's
# _validate_plan, glm53_mcg_materializer.py:173-195; there is no mtp
# adapter receipt and no mtp choice census in this campaign).
# ---------------------------------------------------------------------------


def build_pack_plan(
    *,
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    readiness: Mapping[str, Any],
    inventory: Mapping[str, Any],
    surface: dsv4_surface.Dsv4Surface,
    native_names: Sequence[str],
    mtp_names: Sequence[str],
    dropped_scale_count: int,
    packed_root: Path,
    model_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    census = surface.rate_census()
    body = {
        "schema": PACK_PLAN_SCHEMA,
        "bits": dsv4_surface.MIXED_BITS_MARKER,
        "rate": {
            "numerator": k35.RATE_NUMERATOR,
            "denominator": k35.RATE_DENOMINATOR,
        },
        "launch_plan_sha256": str(plan["launch_plan_sha256"]),
        "state_receipt_sha256": str(
            state["state_receipt_sha256"]),
        "readiness_receipt_sha256": str(
            readiness["readiness_receipt_sha256"]),
        "inventory_sha256": str(inventory["inventory_sha256"]),
        "model_revision": str(inventory.get("model_revision", "")),
        "codec_identity_sha256": str(
            readiness.get("codec_identity_sha256", "")),
        "packed_reader_abi_sha256": surface.packed_reader_abi_sha256,
        "layer_receipt_sha256": [
            str(digest) for digest in surface.layer_receipt_sha256],
        "main_choice_count": len(surface.choices),
        "total_choice_count": len(surface.choices),
        "packed_tensor_count": len(surface.choices) * len(PACKED_SUFFIXES),
        "rate_census": census,
        "native_tensor_count": len(native_names),
        "mtp_native_tensor_count": len(mtp_names),
        "dropped_scale_tensor_count": int(dropped_scale_count),
        "native_tensor_names_sha256": common.sha256_bytes(
            common.canonical_json(list(native_names))),
        "mtp_tensor_names_sha256": common.sha256_bytes(
            common.canonical_json(list(mtp_names))),
        "native_policy": "capped_source_dequant_fold_copy_v1",
        "mtp_policy": "native_copy_not_encoded",
        "packed_root": str(packed_root),
        "source_checkpoint": str(model_dir),
        "output_root": str(output_root),
    }
    return common.seal(body, "pack_plan_sha256")


def _validate_pack_plan(
    pack_plan: Mapping[str, Any],
    *,
    inventory: Mapping[str, Any],
    surface: dsv4_surface.Dsv4Surface,
    output_root: Path,
) -> str:
    plan_sha = common.verify_seal(
        pack_plan,
        schema=PACK_PLAN_SCHEMA,
        field="pack_plan_sha256",
        label="dsv4 materialization plan",
    )
    if (
        pack_plan.get("bits") != dsv4_surface.MIXED_BITS_MARKER
        or pack_plan.get("launch_plan_sha256")
        != surface.launch_plan_sha256
        or pack_plan.get("readiness_receipt_sha256")
        != surface.readiness_receipt_sha256
        or pack_plan.get("state_receipt_sha256")
        != surface.state_receipt_sha256
        or pack_plan.get("inventory_sha256")
        != inventory.get("inventory_sha256")
        or pack_plan.get("layer_receipt_sha256")
        != [str(d) for d in surface.layer_receipt_sha256]
        or pack_plan.get("main_choice_count") != len(surface.choices)
        or pack_plan.get("total_choice_count") != len(surface.choices)
        or pack_plan.get("packed_tensor_count")
        != len(surface.choices) * len(PACKED_SUFFIXES)
        or pack_plan.get("packed_reader_abi_sha256")
        != surface.packed_reader_abi_sha256
        or pack_plan.get("output_root") != str(output_root)
    ):
        die("materialization plan differs from the verified packed surface")
    return plan_sha


# ---------------------------------------------------------------------------
# Checkpoint materialization (port mirror of
# glm53_mcg_materializer.py:751-891).
# ---------------------------------------------------------------------------


def materialize_checkpoint(
    *,
    pack_plan: Mapping[str, Any],
    inventory: Mapping[str, Any],
    surface: dsv4_surface.Dsv4Surface,
    source: Any,
    packed_root: Path,
    model_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    plan_sha = _validate_pack_plan(
        pack_plan, inventory=inventory, surface=surface, output_root=output_root
    )
    model_dir = Path(model_dir).resolve()
    output_root = Path(output_root).resolve()
    packed_root = Path(packed_root).resolve()
    if output_root in {model_dir, packed_root}:
        die("materialized checkpoint must use a distinct output root")

    # Master bindings: the inventory sealed this exact master (resolved on
    # both sides; the pod may reach the same root through a symlink).
    if Path(str(inventory.get("checkpoint", ""))).resolve() != model_dir:
        die("inventory checkpoint root differs from the source model dir")
    if (
        common.sha256_file(model_dir / "config.json")
        != inventory.get("config_sha256")
    ):
        die("master config.json hash differs from the inventory seal")

    main_rows, mtp_rows, native_rows = k35._inventory_surfaces(inventory)
    scale_rows = [
        row for row in native_rows if row["dtype"] == SCALE_DTYPE]
    kept_native_rows = [
        row for row in native_rows if row["dtype"] != SCALE_DTYPE]
    native_names = [str(row["tensor_name"]) for row in kept_native_rows]
    mtp_names = sorted(str(row["tensor_name"]) for row in mtp_rows)

    # Identity binding: the capped source reads THIS master through THIS
    # cap adapter (the dequant analogue of the GLM byte-exact copy proof).
    if source is not None:
        identity = getattr(source, "identity", None)
        if not isinstance(identity, Mapping):
            die("capped source carries no identity block")
        for key in ("lora_sha256", "lora_scale", "lora_sites"):
            if identity.get(key) != inventory.get("identity", {}).get(key):
                die(
                    f"capped source identity {key} differs from the sealed "
                    "inventory identity"
                )

    output_root.mkdir(parents=True, exist_ok=True)
    by_shard: dict[str, list[Mapping[str, Any]]] = {}
    for row in inventory["tensors"]:
        by_shard.setdefault(str(row["shard"]), []).append(row)
    if set(by_shard) != set(inventory.get("shard_sha256", {})):
        die("inventory shard grouping differs from the sealed shard census")

    allowed_top_level = {
        *set(by_shard),
        *_auxiliary_source_names(model_dir),
        "config.json",
        "model.safetensors.index.json",
        "quantization_config.json",
        "materialization-receipt.json",
        "native-copy-receipt.json",
        ".materialization",
    }
    unexpected = {
        path.name for path in output_root.iterdir()} - allowed_top_level
    if unexpected:
        die(
            f"materialization output contains undeclared paths: "
            f"{sorted(unexpected)}"
        )

    skipped_scale_only: list[str] = []
    receipts: list[dict[str, Any]] = []
    for shard in sorted(by_shard):
        rows = by_shard[shard]
        if all(str(row["dtype"]) == SCALE_DTYPE for row in rows):
            skipped_scale_only.append(shard)
            continue
        receipts.append(
            materialize_shard(
                pack_plan_sha256=plan_sha,
                source=source,
                output_root=output_root,
                source_shard=shard,
                shard_rows=rows,
                surface=surface,
            )
        )

    tensor_records = [
        record for receipt in receipts for record in receipt["tensors"]
    ]
    output_names = [str(record["name"]) for record in tensor_records]
    if len(output_names) != len(set(output_names)):
        die("materialized checkpoint has duplicate tensor names")
    native_records = [
        record for record in tensor_records
        if record["origin"] == ORIGIN_NATIVE]
    mtp_records = [
        record for record in tensor_records if record["origin"] == ORIGIN_MTP]
    packed_records = [
        record for record in tensor_records
        if record["origin"] == ORIGIN_PACKED]
    routed_names = {str(row["tensor_name"]) for row in main_rows}
    packed_names = {
        packed_tensor_name(name, suffix)
        for name in routed_names
        for suffix in PACKED_SUFFIXES
    }
    if (
        {record["name"] for record in native_records} != set(native_names)
        or {record["name"] for record in mtp_records} != set(mtp_names)
        or {record["name"] for record in packed_records} != packed_names
        or len(packed_records)
        != len(surface.choices) * len(PACKED_SUFFIXES)
    ):
        die("final native/mtp/packed tensor census differs")

    index = build_output_index(receipts)
    total_size = int(index["metadata"]["total_size"])
    source_config = common.load_json(model_dir / "config.json")
    if not isinstance(source_config, dict):
        die("master config.json is not an object")
    per_module_bits = {}
    for (layer, expert, projection), choice in surface.choices.items():
        module = common.tensor_full_name(
            int(layer), int(expert), str(projection))[:-len(".weight")]
        per_module_bits[module] = int(choice["bits"])
    if len(per_module_bits) != len(surface.choices):
        die("per-module bits census differs from the choice census")
    quantization = build_quantization_config(
        rate_census=surface.rate_census(),
        per_module_bits=per_module_bits,
        prior=source_config.get("quantization_config"),
    )
    config = build_artifact_config(source_config, quantization)
    _atomic_write_json(
        output_root / "model.safetensors.index.json", index)
    _atomic_write_json(output_root / "config.json", config)
    _atomic_write_json(
        output_root / "quantization_config.json", quantization)
    auxiliaries = _copy_auxiliary_files(model_dir, output_root)

    per_shard_native = {
        receipt["shard"]: {
            "native_tensor_count": receipt["native_tensor_count"],
            "mtp_native_tensor_count": receipt["mtp_native_tensor_count"],
            "native_bytes": sum(
                int(record["bytes"]) for record in receipt["tensors"]
                if record["origin"] in (ORIGIN_NATIVE, ORIGIN_MTP)),
        }
        for receipt in receipts
    }
    native_copy = build_native_copy_receipt(
        pack_plan_sha256=plan_sha,
        inventory=inventory,
        native_names=native_names,
        mtp_names=mtp_names,
        dropped_scale_count=len(scale_rows),
        native_output_bytes=sum(
            int(record["bytes"]) for record in native_records),
        mtp_output_bytes=sum(
            int(record["bytes"]) for record in mtp_records),
        per_shard=per_shard_native,
    )
    common.write_json(output_root / "native-copy-receipt.json", native_copy)

    final = {
        "schema": PACK_RECEIPT_SCHEMA,
        "pack_plan_sha256": plan_sha,
        "launch_plan_sha256": surface.launch_plan_sha256,
        "state_receipt_sha256": surface.state_receipt_sha256,
        "readiness_receipt_sha256": surface.readiness_receipt_sha256,
        "source_inventory_sha256": str(inventory["inventory_sha256"]),
        "source_model_revision": str(inventory.get("model_revision", "")),
        "packed_root": str(packed_root),
        "source_checkpoint": str(model_dir),
        "output_root": str(output_root),
        "shards": [receipt["shard"] for receipt in receipts],
        "skipped_scale_only_shards": sorted(skipped_scale_only),
        "shard_receipt_sha256": [
            receipt["receipt_sha256"] for receipt in receipts],
        "shard_sha256": {
            receipt["shard"]: receipt["shard_sha256"]
            for receipt in receipts},
        "source_tensor_count": len(inventory["tensors"]),
        "native_tensor_count": len(native_records),
        "mtp_native_tensor_count": len(mtp_records),
        "dropped_scale_tensor_count": len(scale_rows),
        "routed_choice_count": len(surface.choices),
        "packed_tensor_count": len(packed_records),
        "output_tensor_count": len(output_names),
        "output_tensor_names_sha256": common.sha256_bytes(
            common.canonical_json(sorted(output_names))),
        "output_logical_bytes": total_size,
        "index_total_size_semantics": (
            "sum of materialized tensor data bytes "
            "(numel * element_size), HF index convention; per-shard file "
            "sizes live in the shard receipts"
        ),
        "index_sha256": common.sha256_file(
            output_root / "model.safetensors.index.json"),
        "config_sha256": common.sha256_file(output_root / "config.json"),
        "quantization_config_sha256": common.sha256_file(
            output_root / "quantization_config.json"),
        "native_copy_receipt_sha256": native_copy["receipt_sha256"],
        "native_copy_receipt_file_sha256": common.sha256_file(
            output_root / "native-copy-receipt.json"),
        "auxiliary_files": auxiliaries,
        "codec_family": "exl3-mcg",
        "mcg_multiplier_hex": MCG_MULTIPLIER_HEX,
        "bits": dsv4_surface.MIXED_BITS_MARKER,
        "rate": {
            "numerator": k35.RATE_NUMERATOR,
            "denominator": k35.RATE_DENOMINATOR,
        },
        "packed_reader_abi_sha256": surface.packed_reader_abi_sha256,
        "codec_identity_sha256": str(
            pack_plan.get("codec_identity_sha256", "")),
        "native_policy": "capped_source_dequant_fold_copy_v1",
        "native_fold_identity": copy.deepcopy(
            dict(inventory.get("identity", {}))),
        "nonrouted_byte_exact": False,
        "dequant_parity_binding": (
            "natives are derived through dsv4_capped_source.CappedSource "
            "(dsv4_source_parity.py proven) and bound by the inventory "
            "identity; they are never byte-compared to packed source bytes"
        ),
        "main_routed_complete": True,
        "mtp_native_complete": True,
        "serving_reader_qualified": False,
        "qualified_tp_sizes": [],
        "reader_audit_required_before_publication_as_serving_ready": True,
        "complete": True,
    }
    receipt = common.seal(final, "receipt_sha256")
    common.write_json(
        output_root / "materialization-receipt.json", receipt)
    return verify_packed_checkpoint(
        output_root=output_root,
        inventory=inventory,
        verify_shard_hashes=False,
    )


# ---------------------------------------------------------------------------
# Verification (port mirror of glm53_mcg_materializer.py:553-748; native
# closure checks are policy-based because natives are dequanted, never
# byte-copies of the packed source).
# ---------------------------------------------------------------------------


def verify_packed_checkpoint(
    *,
    output_root: str | Path,
    inventory: Mapping[str, Any],
    verify_shard_hashes: bool = True,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    receipt = common.load_json(output_root / "materialization-receipt.json")
    if receipt.get("example_only") is True:
        die(f"not an executable JSON receipt: {output_root}")
    digest = common.verify_seal(
        receipt,
        schema=PACK_RECEIPT_SCHEMA,
        field="receipt_sha256",
        label="dsv4 materialization receipt",
    )
    if (
        receipt.get("bits") != dsv4_surface.MIXED_BITS_MARKER
        or receipt.get("complete") is not True
        or receipt.get("codec_family") != "exl3-mcg"
        or receipt.get("mcg_multiplier_hex") != MCG_MULTIPLIER_HEX
        or receipt.get("nonrouted_byte_exact") is not False
        or receipt.get("main_routed_complete") is not True
        or receipt.get("mtp_native_complete") is not True
        or receipt.get("serving_reader_qualified") is not False
        or receipt.get("qualified_tp_sizes") != []
    ):
        die("materialized checkpoint receipt semantics differ")

    main_rows, mtp_rows, native_rows = k35._inventory_surfaces(inventory)
    rows = {
        str(row["tensor_name"]): row
        for row in list(main_rows) + list(mtp_rows) + list(native_rows)
    }
    scale_names = {
        str(row["tensor_name"]) for row in native_rows
        if row["dtype"] == SCALE_DTYPE}
    native_names = {
        str(row["tensor_name"]) for row in native_rows
        if row["dtype"] != SCALE_DTYPE}
    mtp_names = {str(row["tensor_name"]) for row in mtp_rows}
    routed_names = {str(row["tensor_name"]) for row in main_rows}
    packed_names = {
        packed_tensor_name(name, suffix)
        for name in routed_names
        for suffix in PACKED_SUFFIXES
    }

    shards = receipt.get("shards")
    shard_hashes = receipt.get("shard_sha256")
    shard_receipt_hashes = receipt.get("shard_receipt_sha256")
    if (
        not isinstance(shards, list)
        or not shards
        or len(shards) != len(set(shards))
        or not isinstance(shard_hashes, Mapping)
        or set(shard_hashes) != set(shards)
        or not isinstance(shard_receipt_hashes, list)
        or len(shard_receipt_hashes) != len(shards)
    ):
        die("materialized shard receipt census differs")
    tensor_rows: list[dict[str, Any]] = []
    observed_receipt_hashes: list[str] = []
    for shard in shards:
        shard_receipt = common.load_json(
            _shard_receipt_path(output_root, shard))
        if shard_receipt.get("example_only") is True:
            die(f"not an executable JSON receipt: {shard}")
        shard_receipt_sha = common.verify_seal(
            shard_receipt,
            schema=PACK_SHARD_RECEIPT_SCHEMA,
            field="receipt_sha256",
            label=f"materialized shard receipt {shard}",
        )
        if (
            shard_receipt.get("pack_plan_sha256")
            != receipt.get("pack_plan_sha256")
            or shard_receipt.get("shard") != shard
            or shard_receipt.get("complete") is not True
            or shard_receipt.get("shard_sha256") != shard_hashes[shard]
            or not isinstance(shard_receipt.get("tensors"), list)
            or shard_receipt.get("output_tensor_count")
            != len(shard_receipt["tensors"])
            or shard_receipt.get("output_logical_bytes")
            != sum(
                int(record["bytes"])
                for record in shard_receipt["tensors"])
        ):
            die(f"materialized shard closure differs: {shard}")
        shard_path = output_root / shard
        if (
            not shard_path.is_file()
            or shard_path.is_symlink()
            or shard_path.stat().st_size
            != shard_receipt.get("shard_bytes")
            or (
                verify_shard_hashes
                and common.sha256_file(shard_path) != shard_hashes[shard]
            )
        ):
            die(f"materialized shard file differs: {shard}")
        done_path = _shard_done_path(output_root, shard)
        if (
            not done_path.is_file()
            or done_path.read_text(encoding="utf-8").strip()
            != shard_receipt_sha
        ):
            die(f"materialized shard .done marker differs: {shard}")
        tensor_rows.extend(copy.deepcopy(shard_receipt["tensors"]))
        observed_receipt_hashes.append(shard_receipt_sha)
    if observed_receipt_hashes != shard_receipt_hashes:
        die("materialized shard receipt ordering/hash census differs")

    names = [str(record.get("name")) for record in tensor_rows]
    if (
        len(names) != len(set(names))
        or len(names) != receipt.get("output_tensor_count")
        or common.sha256_bytes(common.canonical_json(sorted(names)))
        != receipt.get("output_tensor_names_sha256")
        or sum(int(record["bytes"]) for record in tensor_rows)
        != receipt.get("output_logical_bytes")
    ):
        die("materialized output tensor census differs")
    by_name = {str(record["name"]): record for record in tensor_rows}
    if set(by_name) != native_names | mtp_names | packed_names:
        die(
            "materialized tensor names do not exactly cover natives, mtp "
            "natives, and replaced routed weights"
        )
    for name in native_names:
        record = by_name[name]
        source = rows[name]
        if (
            record.get("origin") != ORIGIN_NATIVE
            or record.get("source_tensor_name") != name
            or record.get("choice_sha256") is not None
            or record.get("dtype")
            != NATIVE_OUTPUT_DTYPE.get(str(source.get("dtype")))
            or record.get("source_dtype") != source.get("dtype")
        ):
            die(f"materialized native tensor closure differs: {name}")
    for name in mtp_names:
        record = by_name[name]
        source = rows[name]
        if (
            record.get("origin") != ORIGIN_MTP
            or record.get("source_tensor_name") != name
            or record.get("choice_sha256") is not None
            or record.get("dtype") != "torch.bfloat16"
            or record.get("source_dtype") != source.get("dtype")
        ):
            die(f"materialized mtp native closure differs: {name}")
    for name in packed_names:
        record = by_name[name]
        if (
            record.get("origin") != ORIGIN_PACKED
            or record.get("source_tensor_name") not in routed_names
            or record.get("name")
            != packed_tensor_name(
                str(record.get("source_tensor_name")),
                str(record.get("name")).rsplit(".", 1)[-1],
            )
        ):
            die(f"materialized packed tensor closure differs: {name}")
    if (
        receipt.get("native_tensor_count") != len(native_names)
        or receipt.get("mtp_native_tensor_count") != len(mtp_names)
        or receipt.get("dropped_scale_tensor_count") != len(scale_names)
        or receipt.get("routed_choice_count") != len(routed_names)
        or receipt.get("packed_tensor_count") != len(packed_names)
        or receipt.get("source_tensor_count") != len(rows)
    ):
        die("materialized native/mtp/packed accounting differs")

    index_path = output_root / "model.safetensors.index.json"
    config_path = output_root / "config.json"
    quantization_path = output_root / "quantization_config.json"
    if (
        common.sha256_file(index_path) != receipt.get("index_sha256")
        or common.sha256_file(config_path) != receipt.get("config_sha256")
        or common.sha256_file(quantization_path)
        != receipt.get("quantization_config_sha256")
    ):
        die("materialized config/index hash differs")
    index = common.load_json(index_path)
    expected_map = {
        str(record["name"]): shard
        for shard in shards
        for record in common.load_json(
            _shard_receipt_path(output_root, shard))["tensors"]
    }
    if (
        index.get("weight_map") != expected_map
        or index.get("metadata", {}).get("total_size")
        != receipt.get("output_logical_bytes")
    ):
        die("materialized safetensors index differs")
    config = common.load_json(config_path)
    quantization = common.load_json(quantization_path)
    if (
        config.get("torch_dtype") != "bfloat16"
        or "quantization_config" not in config
        or config.get("quantization_config") != quantization
        or quantization.get("schema") != DSV4_QUANT_CONFIG_SCHEMA
        or quantization.get("quant_method") != "exl3"
        or quantization.get("codebook") != "mcg"
        or quantization.get("bits") != dsv4_surface.MIXED_BITS_MARKER
        or quantization.get("serving_reader_qualified") is not False
        or len(quantization.get("per_module_bits", {}))
        != len(routed_names)
    ):
        die("materialized dsv4 quantization config semantics differ")
    native_copy = common.load_json(
        output_root / "native-copy-receipt.json")
    common.verify_seal(
        native_copy,
        schema=NATIVE_COPY_RECEIPT_SCHEMA,
        field="receipt_sha256",
        label="dsv4 native-copy receipt",
    )
    if (
        native_copy.get("pack_plan_sha256")
        != receipt.get("pack_plan_sha256")
        or native_copy.get("native_tensor_count") != len(native_names)
        or native_copy.get("mtp_native_tensor_count") != len(mtp_names)
        or native_copy.get("complete") is not True
    ):
        die("materialized native-copy receipt differs")
    if common.sha256_file(
            output_root / "native-copy-receipt.json"
            ) != receipt.get("native_copy_receipt_file_sha256"):
        die("materialized native-copy receipt file hash differs")
    for auxiliary in receipt.get("auxiliary_files", []):
        path = output_root / str(auxiliary["path"])
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != auxiliary.get("bytes")
            or common.sha256_file(path) != auxiliary.get("sha256")
        ):
            die(f"materialized auxiliary file differs: {path.name}")
    expected_files = {
        *shards,
        "config.json",
        "model.safetensors.index.json",
        "quantization_config.json",
        "materialization-receipt.json",
        "native-copy-receipt.json",
        *(str(row["path"]) for row in receipt.get("auxiliary_files", [])),
        *(
            str(
                _shard_receipt_path(output_root, shard).relative_to(
                    output_root))
            for shard in shards
        ),
        *(
            str(
                _shard_done_path(output_root, shard).relative_to(output_root))
            for shard in shards
        ),
    }
    observed_files = {
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        die("materialized checkpoint contains undeclared or missing files")
    return receipt


# ---------------------------------------------------------------------------
# Driver (port mirror of k35_phase8_pack.py:43-130; no monkeypatched
# surface loader is needed: dsv4_surface is the loader).
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="phase 8 dsv4: assemble the mixed K3/K4 EXL3 artifact"
    )
    parser.add_argument(
        "--work-root", default=str(WORK_ROOT_DEFAULT), type=Path
    )
    parser.add_argument(
        "--output-root",
        default=None,
        type=Path,
        help="artifact output root (default <work-root>/artifact-dsv4-k35-mixed)",
    )
    parser.add_argument("--model-dir", default=common.MODEL_DIR)
    parser.add_argument("--meta-path", default=common.META_PATH)
    parser.add_argument("--lora-path", default=common.LORA_PATH)
    parser.add_argument(
        "--device", default="cuda:0",
        help="capped-source dequant device",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="write the artifact (without it: dry run, no writes)",
    )
    args = parser.parse_args()
    args.work_root = Path(args.work_root).resolve()
    if args.output_root is None:
        args.output_root = args.work_root / "artifact-dsv4-k35-mixed"
    args.output_root = Path(args.output_root).resolve()
    args.model_dir = Path(args.model_dir).resolve()
    args.meta_path = Path(args.meta_path).resolve()
    args.lora_path = Path(args.lora_path).resolve()
    return args


def main() -> int:
    args = parse_args()

    import dsv4_worker

    plan = dsv4_worker.load_plan(args.work_root)
    _state_path, state = dsv4_worker.newest_state(args.work_root, plan)
    if state.get("phase") == "k35_main_encoding":
        die(
            "the state chain is still encoding; the artifact packs only "
            f"after every layer completes ({len(state.get('pending_layers', []))} "
            "pending)"
        )
    readiness = common.load_json(
        args.work_root / "gss" / "readiness-receipt.json")
    common.verify_seal(
        readiness,
        schema=common.DSV4_READINESS_SCHEMA,
        field="readiness_receipt_sha256",
        label="phase-6 readiness receipt",
    )
    inventory = common.load_json(args.work_root / "inventory.json")
    main_rows, mtp_rows, native_rows = k35._inventory_surfaces(inventory)

    print("[1/5] loading dsv4 surface (verifies every layer chain)...")
    surface = dsv4_surface.load_dsv4_surface(
        args.work_root, plan=plan, state=state, readiness=readiness)
    census = surface.rate_census()
    print(
        f"      surface: {len(surface.choices)} choices, "
        f"k3={census['k3_choice_count']} k4={census['k4_choice_count']}, "
        f"abi={surface.packed_reader_abi_sha256[:12]}"
    )

    scale_rows = [
        row for row in native_rows if row["dtype"] == SCALE_DTYPE]
    kept_native_rows = [
        row for row in native_rows if row["dtype"] != SCALE_DTYPE]
    native_names = [str(row["tensor_name"]) for row in kept_native_rows]
    mtp_names = sorted(str(row["tensor_name"]) for row in mtp_rows)

    # Bind the launch plan's sealed native/mtp contracts before sealing the
    # pack plan (the GLM plan validation analogue,
    # glm53_mcg_materializer.py:173-195, over the DSV4 contract blocks).
    launch_native = plan.get("native_copy_contract", {})
    if (
        launch_native.get("tensor_count") != len(native_names)
        or launch_native.get("dequant_scale_tensors_consumed_not_copied")
        != len(scale_rows)
        or launch_native.get("tensor_names_sha256")
        != common.sha256_bytes(common.canonical_json(native_names))
    ):
        die("native census differs from the launch plan native contract")
    launch_mtp = plan.get("mtp_scope_v1", {})
    if (
        launch_mtp.get("matrix_count") != len(mtp_names)
        or launch_mtp.get("tensor_names_sha256")
        != common.sha256_bytes(common.canonical_json(mtp_names))
    ):
        die("mtp census differs from the launch plan mtp scope contract")

    pack_plan = build_pack_plan(
        plan=plan,
        state=state,
        readiness=readiness,
        inventory=inventory,
        surface=surface,
        native_names=native_names,
        mtp_names=mtp_names,
        dropped_scale_count=len(scale_rows),
        packed_root=args.work_root / "layers",
        model_dir=args.model_dir,
        output_root=args.output_root,
    )
    print(f"[2/5] materialization plan sealed: {pack_plan['pack_plan_sha256'][:12]}")
    print(
        f"      natives={len(native_names)} mtp={len(mtp_names)} "
        f"scales dropped={len(scale_rows)} "
        f"packed={len(surface.choices) * len(PACKED_SUFFIXES)}"
    )

    if common.sha256_file(args.meta_path) != inventory.get(
            "tensor_meta", {}).get("sha256"):
        die("tensor_meta.json hash differs from the inventory seal")

    if not args.execute:
        print("[3/5] dry run only (pass --execute to write the artifact)")
        return 0

    print("[3/5] constructing capped source (dequant-on-demand)...")
    from dsv4_capped_source import CappedSource

    source = CappedSource(
        model_dir=str(args.model_dir),
        meta_path=str(args.meta_path),
        lora_path=str(args.lora_path),
        device=args.device,
    )
    print(
        f"      source identity: lora {source.identity['lora_sha256'][:12]} "
        f"sites {source.identity['lora_sites']} "
        f"scale {source.identity['lora_scale']}"
    )

    print("[4/5] materializing (per-shard, resumable via .done markers)...")
    receipt = materialize_checkpoint(
        pack_plan=pack_plan,
        inventory=inventory,
        surface=surface,
        source=source,
        packed_root=args.work_root / "layers",
        model_dir=args.model_dir,
        output_root=args.output_root,
    )
    print(
        "PACK COMPLETE:",
        json.dumps(
            {
                key: receipt.get(key)
                for key in (
                    "bits",
                    "routed_choice_count",
                    "native_tensor_count",
                    "mtp_native_tensor_count",
                    "dropped_scale_tensor_count",
                    "output_tensor_count",
                    "output_logical_bytes",
                    "complete",
                )
            }
        ),
    )
    print(
        "[5/5] next: bind the k35_packed state with "
        "dsv4_uniform_k35.seal_k35_packed(plan, state, "
        f"packed_checkpoint_receipt_sha256='{receipt['receipt_sha256']}', "
        "native_copy_receipt_sha256='<native-copy-receipt.json seal>')"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
