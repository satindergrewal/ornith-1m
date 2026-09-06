#!/usr/bin/env python3
"""Phase 6: rate-specific GSS preparation at BOTH rates (K3 and K4, no
cross-rate reuse) plus the dsv4 readiness receipt.

Port of k35_phase6_gss.py (GLM-5.3 campaign, _private/k35-dsv4-study) to
DeepSeek V4 Flash geometry.  Every mirrored block cites its GLM source
file and lines.

DIFFERENCE FROM THE GLM SOURCE (binding, deliberate): GLM had a sealed
uniform-K4 predecessor campaign, so its rate-4 half REUSED sealed K4
preparations (schema quant-pipeline.glm53-public-shapleymcg-layer-
preparation.v1) or rebuilt them through the SEALED builder
glm53_mcg_preparation.build_layer_preparation in a subprocess, and its
rate-3 half replayed that builder's numeric path at bits=3.  NO sealed
predecessor exists for DSV4: BOTH halves are built from scratch through
ONE builder path (build_rate_preparation below, the port of the GLM
rate-3 replay of glm53_mcg_preparation.py:258-447) at DSV4 geometry.
Deleted with the sealed half: --build-k4, --profile-selection, the
sealed-closure preflight, the vendored run_qwen_fast_encode dance, the
K4 subprocess code string, and the dual-schema load (sealed schema vs
k35 schema).  The recorded profile provenance is the honest string
profile_source = "dsv4-pod-self-capture-v1" (the calibration capture is
dsv4_capture.py's self-capture of the capped model; no public shapleymcg
run exists for this geometry).

WARN (all NEW SURFACE): nothing sealed validates any schema written
here.  The preparation manifests (dsv4 rate schemas), the readiness
receipt (DSV4_READINESS_SCHEMA), and the layer-allocation receipt
binding are defined by this driver and dsv4_common; the fail-closed
checks in this file are the entire audit surface until downstream
validators exist.

Numeric path (mirrors the sealed builder the GLM source replayed):
statistics -> shared per128 block scales -> streaming v31 fit -> per
matrix pinned GSS (13-point golden section at the layer's rate) -> FP16
vectors + permutations + sealed manifest.  The transform POLICY
(energy_balanced) and scale FAMILY (per128-grid) are recorded choices
carried over as constants, identical to the GLM campaign; what makes a
preparation rate-specific is the GSS search run at that bits value.
Reusing the other rate's fitted VECTORS is forbidden and never happens;
permutations are additionally checked identical across the two halves.

Tensor vocabulary: master names (layers.{L}.ffn.experts.{E}.{w1,w2,w3}
.weight) everywhere in campaign stores; the codec boundary maps
w1->gate_proj, w3->up_proj, w2->down_proj (dsv4_geometry role_map) and
that mapping happens ONLY at MatrixInput construction inside
_matrix_inputs_for_expert.

Usage (inside the encode container, cwd <work-root>):

  python3 dsv4_phase6_gss.py --work-root /workspace \
      --transform-seed-sha256 <64-hex>

ASCII only.  CODE ONLY: nothing here is executed by the author.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dsv4_common as common
from dsv4_common import die

ZERO_HASH = "0" * 64

# glm53_mcg_preparation.py:38 HADAMARD_BLOCK = 128 (block-grid discipline
# is geometry-independent; both DSV4 dims are multiples of 128).
HADAMARD_BLOCK = 128

# Recorded choices carried over from the GLM campaign verbatim
# (glm53_mcg_preparation.py:66-67 policy/family validation,
# k35_phase6_gss.py:663-664 pinned constants).
POLICY = "energy_balanced"
SCALE_FAMILY = "per128-grid"
PROFILE_SOURCE = "dsv4-pod-self-capture-v1"

# NEW SURFACE: the rate-4 preparation schema.  dsv4_common defines the
# rate-3 schema (DSV4_RATE3_GSS_SCHEMA); the rate-4 half built here gets
# the parallel string.  Never reuse a glm53 schema string.
DSV4_RATE4_GSS_SCHEMA = "quant-pipeline.dsv4-k35-rate4-gss-preparation.v1"

# NEW SURFACE: the layer-allocation receipt schema this driver binds
# (the phase-5 probe driver must produce it; no sealed validator).
DSV4_LAYER_ALLOCATION_SCHEMA = "quant-pipeline.dsv4-k35-layer-allocation.v1"

# Codec-boundary projection -> master tensor vocabulary
# (dsv4_geometry.py:103 role_map, inverted).  Mapping happens ONLY at
# codec call sites; stores carry master names.
CODEC_TO_MASTER = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}

RATE_SCHEMA_BY_BITS = {3: common.DSV4_RATE3_GSS_SCHEMA, 4: DSV4_RATE4_GSS_SCHEMA}

_PREPARATION_REQUIRED_KEYS = {
    "permutations",
    "w1_suh",
    "w1_svh",
    "w3_suh",
    "w3_svh",
    "w2_suh",
    "w2_svh",
}
_PREPARATION_SHAPES = {
    "permutations": (common.NUM_EXPERTS, common.INTERMEDIATE_SIZE),
    "w1_suh": (common.NUM_EXPERTS, common.HIDDEN_SIZE),
    "w1_svh": (common.NUM_EXPERTS, common.INTERMEDIATE_SIZE),
    "w3_suh": (common.NUM_EXPERTS, common.HIDDEN_SIZE),
    "w3_svh": (common.NUM_EXPERTS, common.INTERMEDIATE_SIZE),
    "w2_suh": (common.NUM_EXPERTS, common.INTERMEDIATE_SIZE),
    "w2_svh": (common.NUM_EXPERTS, common.HIDDEN_SIZE),
}


def verify_seal_field(
    document: Mapping[str, Any], *, field: str, label: str
) -> str:
    """Schema-agnostic seal recompute for documents this campaign owns
    but has no sealed validator for (plan, allocation, state)."""

    digest = common.require_hash(document.get(field), f"{label}.{field}")
    body = dict(document)
    del body[field]
    if common.sha256_bytes(common.canonical_json(body)) != digest:
        die(f"{label} seal differs")
    return digest


def add_driver_args(parser: argparse.ArgumentParser) -> None:
    """DSV4 driver argument fragment (GLM analogue: k35_common.py:1192-1217
    add_common_args; the DSV4 source is the capped packed master, so there
    is no --bf16-root/--verify-shards pair)."""

    parser.add_argument("--work-root", default=str(common.DEFAULT_WORK_ROOT), type=Path)
    parser.add_argument("--device", default=None, help="default: cuda:0")
    parser.add_argument(
        "--extension",
        default=None,
        help=f"path to the compiled exllamav3_ext .so (or env {common.ENV_EXTENSION})",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="r10 bundle root containing r7_encoder/ (default: discovered on sys.path)",
    )
    parser.add_argument(
        "--calibration-root", default=str(common.DEFAULT_CALIBRATION_ROOT), type=Path
    )
    parser.add_argument("--model-dir", default=common.MODEL_DIR)
    parser.add_argument("--meta-path", default=common.META_PATH)
    parser.add_argument("--lora-path", default=common.LORA_PATH)
    parser.add_argument("--chunk-rows", default=common.CHUNK_ROWS, type=int)
    parser.add_argument(
        "--no-verify-capture-hashes",
        action="store_true",
        help="skip per-layer capture payload hashing (manifest seal still verified)",
    )


def finish_driver_args(args: argparse.Namespace) -> None:
    """GLM analogue: k35_common.py:1220-1226 finish_common_args."""

    args.work_root = Path(args.work_root).resolve()
    args.calibration_root = Path(args.calibration_root).resolve()
    if args.device is None:
        args.device = "cuda:0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="phase 6 dsv4 rate-specific GSS")
    parser.add_argument(
        "--layers",
        default="0-42",
        help="layer range or comma list, e.g. 0-42 (default); the routed surface is 0..42",
    )
    add_driver_args(parser)
    parser.add_argument(
        "--k4-output-root", default=None, type=Path, help="default <work>/gss/k4"
    )
    parser.add_argument(
        "--k3-output-root", default=None, type=Path, help="default <work>/gss/k3"
    )
    parser.add_argument(
        "--transform-seed-sha256",
        default=None,
        help="64-hex sign-seed; required on the first build (nothing to inherit), "
        "then inherited from existing preparations and enforced across layers",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify both halves without building anything",
    )
    args = parser.parse_args()
    finish_driver_args(args)
    if args.k4_output_root is None:
        args.k4_output_root = args.work_root / "gss" / "k4"
    if args.k3_output_root is None:
        args.k3_output_root = args.work_root / "gss" / "k3"
    args.k4_output_root = Path(args.k4_output_root).resolve()
    args.k3_output_root = Path(args.k3_output_root).resolve()

    # Fail-closed layer surface re-derived from the dsv4_common constants
    # (GLM analogue: k35_phase6_gss.py:116-129; GLM allowed 3..44 plus the
    # MTP layer, DSV4 has no MTP in encode v1 - dsv4_common.py:58-59).
    lowest = min(common.MAIN_LAYERS)
    highest = max(common.MAIN_LAYERS)
    layers: list[int] = []
    for token in args.layers.split(","):
        token = token.strip()
        if "-" in token:
            low, high = token.split("-", 1)
            layers.extend(range(int(low), int(high) + 1))
        else:
            layers.append(int(token))
    for layer in layers:
        if layer not in common.MAIN_LAYERS:
            die(f"layer {layer} outside the routed surface {lowest}..{highest}")
    if len(set(layers)) != len(layers):
        die("duplicate layer in --layers")
    args.layer_list = sorted(set(layers))
    return args


def load_plan(work_root: Path) -> dict[str, Any]:
    """Load and structurally verify plan.json.

    GLM analogue: k35_common.py:1022-1025 load_plan, which called the
    SEALED k35.verify_launch_plan (glm53_uniform_k35.py:706-754).  No
    sealed DSV4 plan verifier exists: this checks the seal recompute and
    the two fields phase 6 binds (launch_plan_sha256, preflight_variant).
    The plan itself is produced by the future phase-4 driver.
    """

    plan = common.load_json(work_root / "plan.json")
    if not isinstance(plan, Mapping):
        die("plan.json is not an object")
    verify_seal_field(plan, field="launch_plan_sha256", label="launch plan")
    variant = plan.get("preflight_variant")
    if not isinstance(variant, str) or not variant.strip():
        die("plan preflight_variant is absent or empty")
    return dict(plan)


def load_layer_allocation(work_root: Path, layer: int) -> dict[str, Any]:
    """Load one sealed non-provisional layer allocation.

    GLM analogue: k35_common.py:1028-1036 load_layer_allocation + the
    sealed k35.verify_layer_allocation (glm53_uniform_k35.py:396-408).
    The DSV4 census audit is common.audit_layer_allocation (fail-closed
    over the dsv4_common constants: full 768-name census, integer bits
    in (3,4), exact 2688-unit sum, exactly 384 K4 tensors).
    """

    path = work_root / "allocations" / f"{common.probe_stem(layer)}.json"
    if not path.is_file():
        die(f"layer allocation is absent: {path} (run the phase-5 probe driver first)")
    receipt = common.load_json(path)
    if not isinstance(receipt, Mapping):
        die(f"layer allocation is not an object: {path}")
    if receipt.get("schema") != DSV4_LAYER_ALLOCATION_SCHEMA:
        die(
            f"layer allocation schema differs: expected {DSV4_LAYER_ALLOCATION_SCHEMA},"
            f" got {receipt.get('schema')!r}"
        )
    verify_seal_field(receipt, field="allocation_sha256", label=f"L{layer} allocation")
    if receipt.get("layer") != layer:
        die(f"allocation receipt layer binding differs: {path}")
    allocation = receipt.get("allocation")
    if not isinstance(allocation, Mapping):
        die(f"allocation receipt body is malformed: {path}")
    common.audit_layer_allocation(layer, allocation)
    if receipt.get("bit_units") != common.TARGET_BIT_UNITS_PER_LAYER:
        die(f"allocation receipt bit-unit census differs: {path}")
    if receipt.get("provisional") is not False:
        die(f"layer {layer} allocation is still provisional: {path}")
    return dict(receipt)


def load_preparation(
    root: Path, layer: int, *, expected_bits: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and verify one rate-specific preparation shard.

    GLM analogue: k35_common.py:273-365 load_preparation (which
    validated the sealed GLM schema or the GLM rate-3 schema).  The
    dual-schema dance is deleted: both DSV4 halves are built by this
    driver, so both are verified with full semantic binding against the
    dsv4_common constants.
    """

    from safetensors import safe_open

    if expected_bits not in RATE_SCHEMA_BY_BITS:
        die(f"preparation rate {expected_bits} outside {sorted(RATE_SCHEMA_BY_BITS)}")
    expected_schema = RATE_SCHEMA_BY_BITS[expected_bits]
    directory = Path(root) / common.layer_dir_name(layer)
    manifest_path = directory / "preparation.json"
    if not manifest_path.is_file():
        die(f"preparation manifest is absent: {manifest_path}")
    manifest = common.load_json(manifest_path)
    common.verify_seal(
        manifest,
        schema=expected_schema,
        field="preparation_sha256",
        label=f"rate-{expected_bits} GSS layer {layer}",
    )
    if (
        manifest.get("layer") != layer
        or manifest.get("complete") is not True
        or manifest.get("bits") != expected_bits
        or manifest.get("codec_family") != "exl3-mcg"
        or manifest.get("policy") != POLICY
        or manifest.get("scale_family") != SCALE_FAMILY
        or manifest.get("profile_source") != PROFILE_SOURCE
        or manifest.get("rate_specific_gss") is not True
        or manifest.get("gss_receipt_count") != common.TENSORS_PER_LAYER
    ):
        die(f"rate-{expected_bits} preparation binding differs: {manifest_path}")
    shard = directory / str(manifest["shard"])
    if not shard.is_file() or shard.is_symlink():
        die(f"preparation shard is absent or a symlink: {shard}")
    if common.sha256_file(shard) != manifest.get("shard_sha256"):
        die(f"preparation shard hash differs: {shard}")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != _PREPARATION_REQUIRED_KEYS:
            die(f"preparation tensor census differs: {sorted(keys)}")
        tensors = {name: handle.get_tensor(name).contiguous() for name in sorted(keys)}
    import torch

    for name, shape in _PREPARATION_SHAPES.items():
        if tuple(tensors[name].shape) != shape:
            die(
                f"preparation tensor {name} shape "
                f"{tuple(tensors[name].shape)} != {shape}"
            )
    if tensors["permutations"].dtype != torch.int64 or any(
        tensors[name].dtype != torch.float16
        for name in _PREPARATION_REQUIRED_KEYS - {"permutations"}
    ):
        die("preparation dtypes differ (need int64 permutations, fp16 vectors)")
    return manifest, tensors


def sweep_abandoned_staging(output: Path, layer: int) -> None:
    """Remove staging directories whose owner process is gone.

    Mirror of k35_phase6_gss.py:274-303 sweep_abandoned_staging (which
    mirrored the sealed builder's staging discipline,
    glm53_mcg_preparation.py:293-295).  A live owner means another
    phase-6 process is building this layer right now, which is not a
    supported mode: die loudly.
    """

    import shutil

    for candidate in sorted(
        output.glob(f".{common.layer_dir_name(layer)}.staging-*")
    ):
        token = candidate.name.rsplit("-", 1)[-1]
        if not token.isdigit() or int(token) <= 0:
            die(f"foreign staging entry in the rate output root: {candidate}")
        owner = int(token)
        try:
            os.kill(owner, 0)
        except ProcessLookupError:
            shutil.rmtree(candidate)
            continue
        except OSError:
            die(
                f"staging directory {candidate.name} has a live or inaccessible "
                f"owner (pid {owner}); concurrent phase-6 builds are not supported"
            )
        die(
            f"staging directory {candidate.name} has a live owner (pid {owner}); "
            "concurrent phase-6 builds are not supported"
        )


def producer_closure(source_root: Path) -> list[dict[str, str]]:
    """Hash the producer sources that actually build a preparation.

    GLM analogue: k35_phase6_gss.py:162-244 (the derived closure over
    the released tree plus the unsatisfiable-sealed-surface WARN).  That
    dance existed only because the SEALED builder's closure demanded a
    file absent from the released deliverable; no sealed surface exists
    here, so the closure is simply the live producer surface: the
    r7_encoder numeric bundle, the imported streaming/codec modules, and
    the DSV4 driver files.  Computed BEFORE the GPU pass so a broken
    surface fails in seconds, not after the expert sweep.
    """

    bundle = sorted((source_root / "r7_encoder").rglob("*.py"))
    if not bundle:
        die(f"the r7_encoder numeric closure is absent under {source_root}")
    records: list[dict[str, str]] = [
        {
            "path": path.relative_to(source_root).as_posix(),
            "sha256": common.sha256_file(path),
        }
        for path in bundle
    ]

    import quant_pipeline.codecs.exl3_mcg as exl3_mcg_module
    import quant_pipeline.normalization.streaming_v31 as streaming_module

    for name, module in (
        ("src/quant_pipeline/normalization/streaming_v31.py", streaming_module),
        ("src/quant_pipeline/codecs/exl3_mcg.py", exl3_mcg_module),
    ):
        path = Path(module.__file__).resolve() if module.__file__ else None
        if path is None or not path.is_file():
            die(f"imported producer source is absent: {name}")
        records.append(
            {"path": name, "sha256": common.sha256_file(path),
             "resolved_from": str(path)}
        )

    driver_dir = Path(__file__).resolve().parent
    for filename in (
        "dsv4_phase6_gss.py",
        "dsv4_common.py",
        "dsv4_capped_source.py",
        "dsv4_geometry.py",
    ):
        path = driver_dir / filename
        if not path.is_file():
            die(f"DSV4 producer source is absent: {path}")
        records.append(
            {"path": f"tools/dsv4-k35/{filename}", "sha256": common.sha256_file(path)}
        )
    return records


# ---------------------------------------------------------------------------
# Numeric path: DSV4 ports of the sealed builder's private helpers
# (glm53_mcg_preparation.py:101-231, replayed by the GLM source at
# k35_phase6_gss.py:388-635).  Rebound from the GLM closure's constants
# (experts 288, hidden 4096, intermediate 2048) to the dsv4_common
# constants (256 / 4096 / 2048): the expert census is the only
# geometric change; the arithmetic is unchanged.
# ---------------------------------------------------------------------------


def _sign(length: int, seed: str, *domain: Any):
    """Mirror of glm53_mcg_preparation.py:101-106 _sign."""

    import torch

    value = int(common.sha256_bytes(common.canonical_json([seed, *domain]))[:16], 16)
    generator = torch.Generator(device="cpu").manual_seed(value)
    return (
        torch.randint(0, 2, (length,), generator=generator, dtype=torch.int8).float()
        * 2.0
        - 1.0
    ).contiguous()


def _block_values(value, block: int = HADAMARD_BLOCK) -> tuple[float, ...]:
    """Mirror of glm53_mcg_preparation.py:109-113 _block_values."""

    import numpy as np

    raw = np.asarray(value, dtype=np.float64).reshape(-1)
    if raw.size % block:
        raise ValueError("profile statistic is not block aligned")
    return tuple(
        float(max(raw[index : index + block].mean(), 1e-30))
        for index in range(0, raw.size, block)
    )


def _hidden_indexed(capture: Any, rows, *, device: str, chunk_rows: int):
    """Yield (begin, stop, hidden-fp32-chunk) over row indices.

    Mirror of glm53_mcg_preparation.py:116-122 _hidden_chunks over the
    DSV4 capture ABI (dsv4_common.Dsv4CaptureView.hidden_u16).
    """

    import numpy as np
    import torch

    for begin in range(0, rows.size, chunk_rows):
        stop = min(rows.size, begin + chunk_rows)
        words = np.array(
            capture.hidden_u16[rows[begin:stop]], dtype=np.uint16, copy=True
        )
        yield begin, stop, (
            torch.from_numpy(words)
            .view(torch.bfloat16)
            .to(device=device, dtype=torch.float32)
            .contiguous()
        )


def load_triplet(source: Any, layer: int, expert: int, device: str) -> dict[str, Any]:
    """One expert's BF16 triplet in the codec vocabulary.

    CappedSource.load_expert returns (w1, w2, w3) in engine order
    (gate, down, up; dsv4_capped_source.py:65-75).  The codec boundary
    maps w1->gate_proj, w3->up_proj, w2->down_proj
    (dsv4_geometry.py:103 role_map); this adapter is that call site.
    """

    gate, down, up = source.load_expert(layer, expert)
    return {
        "gate_proj": gate.to(device),
        "up_proj": up.to(device),
        "down_proj": down.to(device),
    }


def _p2_profile_statistics(
    *, capture: Any, source: Any, layer: int, device: str, chunk_rows: int
) -> dict[str, Any]:
    """Decision statistics at DSV4 geometry.

    Mirror of glm53_mcg_preparation.py:125-181 _p2_profile_statistics:
    per-expert routed p2 mass, p2-weighted gate/up input diagonal,
    p2-weighted down input diagonal (middle = silu(h@w1.T)*(h@w3.T) at
    FULL precision - the decision statistic, not the candidate-
    conditioned encode Hessian), and the down output energy; then the
    mass-weighted shared diagonals for the shared block scales.
    """

    import hashlib
    import math

    import numpy as np
    import torch
    import torch.nn.functional as functional

    gate_diagonal = torch.empty(
        (common.NUM_EXPERTS, common.HIDDEN_SIZE), dtype=torch.float32)
    down_diagonal = torch.empty(
        (common.NUM_EXPERTS, common.INTERMEDIATE_SIZE), dtype=torch.float32)
    masses = torch.empty(common.NUM_EXPERTS, dtype=torch.float64)
    down_output_energy = torch.empty(
        (common.NUM_EXPERTS, common.HIDDEN_SIZE), dtype=torch.float32)
    row_evidence: list[dict[str, Any]] = []
    for expert in range(common.NUM_EXPERTS):
        routed = capture.routed_rows(expert, "fit")
        if routed.rows <= 0:
            # dead-in-distribution expert (receipt vocabulary matches
            # dsv4_common's dead-expert-identity-ridge-v1): mass 0 so the
            # mass-weighted shared sums exclude it entirely; neutral ones
            # diagonals so the expert's OWN GSS vector input is unguided -
            # the same treatment the phase-5 probe gives dead experts.
            # Deleting the die alone would 0/0 the per-expert diagonals.
            triplet = load_triplet(source, layer, expert, device)
            gate_diagonal[expert].fill_(1.0)
            down_diagonal[expert].fill_(1.0)
            masses[expert] = 0.0
            down_output_energy[expert].copy_(
                triplet["down_proj"].float().pow(2).mean(dim=1).cpu())
            row_evidence.append(
                {
                    "expert": expert,
                    "rows": 0,
                    "documents": 0,
                    "weight_sum": 0.0,
                    "construction": "dead_in_distribution",
                    "row_indices_sha256": hashlib.sha256(b"").hexdigest(),
                    "route_weights_sha256": hashlib.sha256(b"").hexdigest(),
                }
            )
            del triplet
            continue
        weights = np.asarray(routed.applied_weights, dtype=np.float64)
        p2_mass = float(np.square(weights).sum())
        if not math.isfinite(p2_mass) or p2_mass <= 0:
            die(f"L{layer} E{expert}: degenerate routed p2 mass")
        triplet = load_triplet(source, layer, expert, device)
        gate_weight = triplet["gate_proj"].float()
        up_weight = triplet["up_proj"].float()
        gate_sum = torch.zeros(common.HIDDEN_SIZE, dtype=torch.float64, device=device)
        down_sum = torch.zeros(
            common.INTERMEDIATE_SIZE, dtype=torch.float64, device=device)
        for begin, stop, hidden in _hidden_indexed(
            capture, routed.row_indices, device=device, chunk_rows=chunk_rows
        ):
            p2 = torch.from_numpy(
                np.square(routed.applied_weights[begin:stop], dtype=np.float32)
            ).to(device=device, dtype=torch.float64)
            gate_sum.add_((hidden.double().square() * p2.unsqueeze(1)).sum(dim=0))
            middle = functional.silu(hidden @ gate_weight.T) * (hidden @ up_weight.T)
            down_sum.add_((middle.double().square() * p2.unsqueeze(1)).sum(dim=0))
        gate_diagonal[expert].copy_((gate_sum / p2_mass).float().cpu())
        down_diagonal[expert].copy_((down_sum / p2_mass).float().cpu())
        masses[expert] = p2_mass
        down_output_energy[expert].copy_(
            triplet["down_proj"].float().pow(2).mean(dim=1).cpu())
        row_evidence.append(
            {
                "expert": expert,
                "rows": routed.rows,
                "documents": int(np.unique(routed.document_epochs).size),
                "weight_sum": p2_mass,
                "row_indices_sha256": hashlib.sha256(
                    routed.row_indices.tobytes()).hexdigest(),
                "route_weights_sha256": hashlib.sha256(
                    np.asarray(routed.applied_weights, dtype="<f4").tobytes()
                ).hexdigest(),
            }
        )
        del triplet, gate_weight, up_weight, gate_sum, down_sum
    if masses.sum() <= 0:
        die(f"L{layer}: every expert is dead-in-distribution; the shared "
            "diagonals would be 0/0")
    shared_gate = (
        gate_diagonal.double() * masses.unsqueeze(1)).sum(dim=0) / masses.sum()
    shared_down_output = (
        down_output_energy.double() * masses.unsqueeze(1)).sum(dim=0) / masses.sum()
    return {
        "gate_diagonal": gate_diagonal,
        "down_diagonal": down_diagonal,
        "masses": masses,
        "down_output_energy": down_output_energy,
        "shared_gate_diagonal": shared_gate.float(),
        "shared_down_output_energy": shared_down_output.float(),
        "row_evidence": row_evidence,
    }


def _matrix_inputs_for_expert(
    *,
    source: Any,
    layer: int,
    expert: int,
    device: str,
    policy: str,
    family: str,
    seed: str,
    statistics: Mapping[str, Any],
    shared_gate_scales,
    shared_down_scales,
    bits: int,
):
    """Fit matrices for one expert at one rate.

    Mirror of glm53_mcg_preparation.py:184-231 _matrix_inputs_for_expert
    at DSV4 geometry.  The projection vocabulary here is the codec
    boundary (gate_proj/up_proj/down_proj); stored artifacts translate
    to master names through CODEC_TO_MASTER at the storage site only.
    """

    from quant_pipeline.normalization.absolute_v31 import MatrixInput
    from quant_pipeline.normalization.prior_search import (
        permute_expert_hf,
        policy_permutations,
        scale_family_candidates,
    )

    triplet = load_triplet(source, layer, expert, device)
    diagonal = statistics["down_diagonal"][expert].tolist()
    permutation = policy_permutations(diagonal, block=HADAMARD_BLOCK)[policy]
    gate, up, down = permute_expert_hf(
        triplet["gate_proj"], triplet["up_proj"], triplet["down_proj"], permutation
    )
    shared_gate_sign = _sign(
        common.HIDDEN_SIZE, seed, layer, policy, family, "gate-up-suh").to(device)
    shared_down_sign = _sign(
        common.HIDDEN_SIZE, seed, layer, policy, family, "down-svh").to(device)
    mass = float(statistics["masses"][expert].item())
    if mass <= 0.0:
        # dead-in-distribution expert (statistics branch wrote ones diagonals
        # and mass 0): every v31 surface validates mass > 0 (absolute_v31
        # batch prepare, streaming_v31 spec, artifact_v31 record verify), so
        # the dead expert flows with a deterministic positive sentinel =
        # (layer's minimum live p2 mass) * 1e-6.  Its fit-selection weight is
        # then ~1e-9 of the layer denominator: excluded for all practical
        # purposes while write and verify stay internally consistent.  The
        # encode side gives dead experts floor bits from the sealed probe
        # allocation; the worker never reads artifact mass.
        live = statistics["masses"][statistics["masses"] > 0]
        mass = float(live.min().item()) * 1e-6
    permuted_down_diag = statistics["down_diagonal"][expert][list(permutation)].tolist()
    rows = []
    for projection, weight, hdiag, suh, svh in (
        ("gate_proj", gate, statistics["gate_diagonal"][expert].tolist(),
         shared_gate_sign,
         _sign(common.INTERMEDIATE_SIZE, seed, layer, expert, policy, family,
               "gate-svh").to(device)),
        ("up_proj", up, statistics["gate_diagonal"][expert].tolist(),
         shared_gate_sign,
         _sign(common.INTERMEDIATE_SIZE, seed, layer, expert, policy, family,
               "up-svh").to(device)),
        ("down_proj", down, permuted_down_diag,
         _sign(common.INTERMEDIATE_SIZE, seed, layer, expert, policy, family,
               "down-suh").to(device),
         shared_down_sign),
    ):
        # k scales: shared for gate/up inputs; the expert's own permuted
        # down diagonal for the down input.  n scales: the matrix's own
        # row energies for gate/up outputs; shared for the down output.
        # (The gate/up hdiag slot is carried for structural parity with
        # the sealed builder and is unused - glm53_mcg_preparation.py:214-217.)
        k_scales = (
            shared_gate_scales
            if projection != "down_proj"
            else scale_family_candidates(_block_values(hdiag))[family]
        )
        n_scales = (
            shared_down_scales
            if projection == "down_proj"
            else scale_family_candidates(
                _block_values(
                    weight.float().pow(2).mean(dim=1).detach().cpu().numpy())
            )[family]
        )
        rows.append(
            MatrixInput(
                key=f"E{expert}.{projection}",
                projection=projection,
                bits=bits,
                weight_kn=weight.T.contiguous(),
                suh_sign=suh,
                svh_sign=svh,
                k_block_scales=k_scales,
                n_block_scales=n_scales,
                mass=mass,
            )
        )
    return tuple(rows), tuple(permutation)


def build_rate_preparation(
    args: argparse.Namespace,
    codec,
    capture,
    source,
    layer: int,
    bits: int,
    policy: str,
    family: str,
) -> dict[str, Any]:
    """One rate half: statistics, streaming fit, per-matrix pinned GSS.

    Mirror of k35_phase6_gss.py:388-635 build_rate3_preparation (itself
    the replay of glm53_mcg_preparation.py:258-447 build_layer_
    preparation), parameterized over bits so BOTH halves go through this
    one path (the GLM source used the sealed builder for its rate-4
    half in a subprocess; that half and its subprocess are deleted).
    """

    import torch
    from safetensors.torch import save_file

    from quant_pipeline.campaign.qwen_services import CorrectedPinnedGSSProducer
    from quant_pipeline.normalization.artifact_v31 import (
        PinnedGSSRequest,
        tensor_identity_sha256,
        tensor_sha256,
    )
    from quant_pipeline.normalization.prior_search import scale_family_candidates
    from quant_pipeline.normalization.streaming_v31 import (
        FitSamplePlan,
        FitSampleSpec,
        StreamingLayerFitter,
    )

    if bits not in RATE_SCHEMA_BY_BITS:
        die(f"preparation rate {bits} outside {sorted(RATE_SCHEMA_BY_BITS)}")
    schema = RATE_SCHEMA_BY_BITS[bits]
    seed = args.transform_seed_sha256_resolved
    output = args.k3_output_root if bits == 3 else args.k4_output_root
    final_destination = output / common.layer_dir_name(layer)
    manifest_path = final_destination / "preparation.json"
    if manifest_path.exists():
        manifest, _tensors = load_preparation(output, layer, expected_bits=bits)
        return manifest
    if args.verify_only:
        die(f"verify-only: rate-{bits} preparation is absent for layer {layer}")
    if final_destination.exists():
        die(f"incomplete final rate-{bits} preparation directory exists: {final_destination}")
    output.mkdir(parents=True, exist_ok=True)
    sweep_abandoned_staging(output, layer)
    staging = output / f".{common.layer_dir_name(layer)}.staging-{os.getpid()}"
    staging.mkdir(exist_ok=False)

    # Compute the producer closure BEFORE the GPU pass (GLM rationale,
    # k35_phase6_gss.py:449-453): a broken producer surface must fail in
    # seconds, not after the expert statistics/fit/GSS sweep.
    source_root = common.resolve_source_root(args)
    closure = producer_closure(source_root)

    backend = codec._codec()
    producer = CorrectedPinnedGSSProducer(codec)
    statistics = _p2_profile_statistics(
        capture=capture, source=source, layer=layer,
        device=args.device, chunk_rows=args.chunk_rows,
    )
    shared_gate_scales = scale_family_candidates(
        _block_values(statistics["shared_gate_diagonal"].numpy())
    )[family]
    shared_down_scales = scale_family_candidates(
        _block_values(statistics["shared_down_output_energy"].numpy())
    )[family]

    specs: list[Any] = []
    permutations = torch.empty(
        (common.NUM_EXPERTS, common.INTERMEDIATE_SIZE), dtype=torch.int64)
    for expert in range(common.NUM_EXPERTS):
        matrices, permutation = _matrix_inputs_for_expert(
            source=source, layer=layer, expert=expert, device=args.device,
            policy=policy, family=family, seed=seed, statistics=statistics,
            shared_gate_scales=shared_gate_scales,
            shared_down_scales=shared_down_scales, bits=bits,
        )
        permutations[expert].copy_(torch.tensor(permutation, dtype=torch.int64))
        specs.extend(FitSampleSpec.from_input(matrix) for matrix in matrices)
        del matrices
    plan = FitSamplePlan.from_specs(specs, block=HADAMARD_BLOCK)
    fitter = StreamingLayerFitter(
        backend.core,
        plan,
        codebook_scale=float(backend.codebook_scale),
        numeric_core_sha256=codec.identity["numeric_core_sha256"],
    )
    for expert in range(common.NUM_EXPERTS):
        matrices, _permutation = _matrix_inputs_for_expert(
            source=source, layer=layer, expert=expert, device=args.device,
            policy=policy, family=family, seed=seed, statistics=statistics,
            shared_gate_scales=shared_gate_scales,
            shared_down_scales=shared_down_scales, bits=bits,
        )
        for matrix in matrices:
            fitter.add_fit_matrix(matrix)
        del matrices
    fit = fitter.finish()

    vectors = {
        f"{master}_suh": torch.empty(shape_suh, dtype=torch.float16)
        for master, shape_suh in (
            ("w1", (common.NUM_EXPERTS, common.HIDDEN_SIZE)),
            ("w3", (common.NUM_EXPERTS, common.HIDDEN_SIZE)),
            ("w2", (common.NUM_EXPERTS, common.INTERMEDIATE_SIZE)),
        )
    }
    for master, shape_svh in (
        ("w1", (common.NUM_EXPERTS, common.INTERMEDIATE_SIZE)),
        ("w3", (common.NUM_EXPERTS, common.INTERMEDIATE_SIZE)),
        ("w2", (common.NUM_EXPERTS, common.HIDDEN_SIZE)),
    ):
        vectors[f"{master}_svh"] = torch.empty(shape_svh, dtype=torch.float16)
    gss_receipts: list[dict[str, Any]] = []
    for expert in range(common.NUM_EXPERTS):
        matrices, _permutation = _matrix_inputs_for_expert(
            source=source, layer=layer, expert=expert, device=args.device,
            policy=policy, family=family, seed=seed, statistics=statistics,
            shared_gate_scales=shared_gate_scales,
            shared_down_scales=shared_down_scales, bits=bits,
        )
        for matrix in matrices:
            prepared = fit.prepare_matrix(matrix)
            target = prepared.gss_target()
            result = producer.search(
                PinnedGSSRequest(
                    matrix_key=matrix.key,
                    bits=bits,
                    target=target,
                    target_sha256=tensor_sha256(target),
                    source_weight_identity_sha256=tensor_identity_sha256(
                        matrix.weight_kn),
                    predecessor_checkpoint_hash=ZERO_HASH,
                )
            )
            finalized = prepared.finalize(
                prepared.bind_gss(result.scale), materialize_regularized=False
            )
            master = CODEC_TO_MASTER[matrix.projection]
            vectors[f"{master}_suh"][expert].copy_(
                finalized.stored_suh.detach().cpu())
            vectors[f"{master}_svh"][expert].copy_(
                finalized.stored_svh.detach().cpu())
            gss_receipts.append(
                {
                    "expert": expert,
                    "projection": master,
                    "scale": float(result.scale),
                    "receipt_sha256": result.receipt["receipt_sha256"],
                    "suh_sha256": finalized.suh_sha256,
                    "svh_sha256": finalized.svh_sha256,
                }
            )
        del matrices
    if len(gss_receipts) != common.TENSORS_PER_LAYER:
        die(
            f"GSS receipt census {len(gss_receipts)} differs from "
            f"{common.TENSORS_PER_LAYER}"
        )

    shard = staging / "preparation.safetensors"
    save_file(
        {"permutations": permutations, **vectors},
        str(shard),
        metadata={"schema": schema, "layer": str(layer), "bits": str(bits)},
    )
    decision_stats = staging / "profile-decision-statistics.safetensors"
    save_file(
        {
            "gate_p2_diagonal": statistics["gate_diagonal"],
            "source_down_p2_diagonal": statistics["down_diagonal"],
            "p2_mass": statistics["masses"],
            "source_down_output_energy": statistics["down_output_energy"],
        },
        str(decision_stats),
        metadata={
            "schema": schema,
            "purpose": "lossless-selected-transform-decision-statistics",
        },
    )
    body = {
        "schema": schema,
        "new_surface_warning": (
            "dsv4 rate-specific GSS preparation: no sealed DSV4 campaign or "
            "validator exists; this manifest is built and verified only by "
            "dsv4_phase6_gss.py, which replays the sealed GLM builder's "
            "numeric path (glm53_mcg_preparation.py:258-447) at DSV4 "
            "geometry with both rate halves built from scratch"
        ),
        "complete": True,
        "layer": layer,
        "bits": bits,
        "codec_family": "exl3-mcg",
        "policy": policy,
        "scale_family": family,
        "profile_source": PROFILE_SOURCE,
        "profile_fixed_before_encoding": True,
        "selection_rows_used": False,
        "selection_used_for_profile_choice": False,
        "selection_used_for_final_encoding": False,
        "confirmation_used_for_choice": False,
        "global_allocator_invoked": False,
        "candidate_rate_grid_invoked": False,
        "rate_specific_gss": True,
        "reuse_k4_gss_forbidden": True,
        "k3_and_k4_gss_both_required": True,
        "transform_seed_sha256": seed,
        "streaming_fit_plan_sha256": plan.content_sha256,
        "shared_gate_up_suh_sha256": fit.shared_gate_up_suh_sha256,
        "shared_down_svh_sha256": fit.shared_down_svh_sha256,
        "permutation_set_sha256": common.tensor_sha256(permutations),
        "gss_receipts_sha256": common.sha256_bytes(
            common.canonical_json(gss_receipts)),
        "gss_receipt_count": len(gss_receipts),
        "profile_fit_row_evidence_sha256": common.sha256_bytes(
            common.canonical_json(statistics["row_evidence"])
        ),
        "process_structure": {
            "driver": "tools/dsv4-k35/dsv4_phase6_gss.py",
            "normalization": "src/quant_pipeline/normalization/streaming_v31.py",
            "codec_adapter": "src/quant_pipeline/codecs/exl3_mcg.py",
            "operation_order": "dsv4-rate-specific-replay-v1",
            "numeric_closure": "r7_encoder",
        },
        "producer_source_closure": closure,
        "producer_source_closure_sha256": common.sha256_bytes(
            common.canonical_json(closure)
        ),
        "codec_identity": codec.identity,
        "shard": shard.name,
        "shard_sha256": common.sha256_file(shard),
        "decision_statistics": decision_stats.name,
        "decision_statistics_sha256": common.sha256_file(decision_stats),
        "exact_production_hessians": (
            "recomputed_from_dsv4_capture_and_packed_gate_up"
        ),
    }
    result = common.seal(body, "preparation_sha256")
    common.write_json(staging / "preparation.json", result)
    os.replace(staging, final_destination)
    return result


def resolve_campaign_choices(args: argparse.Namespace) -> tuple[str, str, str]:
    """Resolve (seed, policy, family), inherited or explicit.

    GLM analogue: k35_phase6_gss.py:642-680, which borrowed the triple
    from the sealed K4 preparation or the sealed profile-selection
    receipt.  No predecessor exists here: the triple is inherited from
    any existing preparation at the reference layer (either rate), else
    the pinned policy/family constants plus a REQUIRED explicit seed.
    They are shared across layers and rates and enforced per layer.
    """

    reference_layer = args.layer_list[0]
    inherited: tuple[str, str, str] | None = None
    for bits, root in (
        (4, args.k4_output_root),
        (3, args.k3_output_root),
    ):
        path = root / common.layer_dir_name(reference_layer) / "preparation.json"
        if not path.is_file():
            continue
        manifest, _tensors = load_preparation(root, reference_layer, expected_bits=bits)
        candidate = (
            manifest["transform_seed_sha256"],
            manifest["policy"],
            manifest["scale_family"],
        )
        if inherited is not None and candidate != inherited:
            die(
                f"reference layer {reference_layer}: existing preparations "
                "disagree on transform seed/policy/family across rates"
            )
        inherited = candidate
    if inherited is None:
        if not args.transform_seed_sha256:
            die(
                "no preparation exists yet at the reference layer; pass "
                "--transform-seed-sha256 (64-hex) before building from scratch"
            )
        seed = args.transform_seed_sha256
    else:
        seed = args.transform_seed_sha256 or inherited[0]
        if inherited[1] != POLICY or inherited[2] != SCALE_FAMILY:
            die(
                "existing preparation carries a foreign policy/family: "
                f"{inherited[1]}/{inherited[2]}"
            )
    common.require_hash(seed, "transform seed")
    return seed, POLICY, SCALE_FAMILY


def main() -> None:
    args = parse_args()
    plan = load_plan(args.work_root)
    seed, policy, family = resolve_campaign_choices(args)
    args.transform_seed_sha256_resolved = seed

    codec = None
    source = None
    per_layer = []
    k3_identity_shas: set[str] = set()
    k4_identity_shas: set[str] = set()
    for layer in args.layer_list:
        capture = common.open_capture(
            args.calibration_root, layer,
            verify_hashes=not args.no_verify_capture_hashes,
        )
        for bits, root in ((4, args.k4_output_root), (3, args.k3_output_root)):
            manifest_path = (
                root / common.layer_dir_name(layer) / "preparation.json")
            if manifest_path.exists():
                manifest, _tensors = load_preparation(
                    root, layer, expected_bits=bits)
            else:
                if source is None:
                    from dsv4_capped_source import CappedSource

                    source = CappedSource(
                        model_dir=args.model_dir,
                        meta_path=args.meta_path,
                        lora_path=args.lora_path,
                        device=args.device,
                    )
                if codec is None:
                    # One codec serves both rates; constructed before any
                    # r7_encoder import in this process
                    # (dsv4_common.build_codec enforces the order).
                    codec = common.build_codec(
                        common.resolve_source_root(args),
                        common.resolve_extension(args),
                        args.device,
                    )
                manifest = build_rate_preparation(
                    args, codec, capture, source, layer, bits, policy, family
                )
            # Semantic binding on EVERY layer's manifest at BOTH rates
            # (GLM analogue pinned these across the whole set,
            # k35_phase6_gss.py:707-720, on the K4 half only).
            if (
                manifest.get("policy") != policy
                or manifest.get("scale_family") != family
                or manifest.get("transform_seed_sha256") != seed
            ):
                die(
                    f"layer {layer} rate-{bits} preparation carries a foreign "
                    f"policy/family/seed: {manifest.get('policy')}/"
                    f"{manifest.get('scale_family')}"
                )
            identity_set = k4_identity_shas if bits == 4 else k3_identity_shas
            identity_set.add(
                common.sha256_bytes(
                    common.canonical_json(manifest["codec_identity"]))
            )

        import torch

        _m4, tensors_k4 = load_preparation(
            args.k4_output_root, layer, expected_bits=4)
        _m3, tensors_k3 = load_preparation(
            args.k3_output_root, layer, expected_bits=3)
        if not torch.equal(tensors_k4["permutations"], tensors_k3["permutations"]):
            die(f"layer {layer}: K3/K4 permutations differ across rate halves")
        k4_manifest = _m4
        k3_manifest = _m3
        per_layer.append(
            {
                "layer": layer,
                "k4_preparation_sha256": k4_manifest["preparation_sha256"],
                "k3_preparation_sha256": k3_manifest["preparation_sha256"],
                "permutation_identity": True,
            }
        )
        print(
            json.dumps(
                {
                    "layer": layer,
                    "k4_prep": k4_manifest["preparation_sha256"][:16],
                    "k3_prep": k3_manifest["preparation_sha256"][:16],
                }
            ),
            flush=True,
        )

    allocations = []
    for layer in args.layer_list:
        receipt = load_layer_allocation(args.work_root, layer)
        allocations.append(
            {
                "layer": layer,
                "allocation_sha256": receipt["allocation_sha256"],
                "provisional": receipt["provisional"],
            }
        )
    if any(row["provisional"] for row in allocations):
        die("a layer allocation is still provisional; re-run phase 5 first")

    # Derive the receipt's codec identity from the artifacts being
    # sealed, not from incidental execution state (GLM analogue,
    # k35_phase6_gss.py:789-818): every manifest's embedded codec
    # identity must agree within its rate AND across the two rates (both
    # halves are built at one venue here, unlike the GLM campaign whose
    # rate-4 half came from the original K4 venue - that foreign-venue
    # field is deleted), and a codec constructed this invocation must
    # reproduce the same identity.
    if len(k3_identity_shas) != 1:
        die(
            "rate-3 preparations do not share one codec identity: "
            f"{sorted(k3_identity_shas)}"
        )
    if len(k4_identity_shas) != 1:
        die(
            "rate-4 preparations do not share one codec identity: "
            f"{sorted(k4_identity_shas)}"
        )
    if k3_identity_shas != k4_identity_shas:
        die(
            "the two rate halves do not share one codec identity "
            f"(rate-3 {sorted(k3_identity_shas)} vs rate-4 "
            f"{sorted(k4_identity_shas)}); the halves were built at "
            "different venues"
        )
    codec_identity_sha256 = next(iter(k3_identity_shas))
    if codec is not None:
        live_identity_sha256 = common.sha256_bytes(
            common.canonical_json(codec.identity)
        )
        if live_identity_sha256 != codec_identity_sha256:
            die(
                "the live codec identity differs from the sealed preparation "
                "identities; this run's codec is not the codec the "
                "preparation set was built with"
            )
    body = {
        "schema": common.DSV4_READINESS_SCHEMA,
        "new_surface_warning": (
            "dsv4 readiness receipt: no sealed DSV4 campaign exists (no "
            "readiness schema, builder, or validator); this document is the "
            "binding the hash promises and dsv4_phase6_gss.py is its only "
            "verifier"
        ),
        "launch_plan_sha256": plan["launch_plan_sha256"],
        "preflight_variant": plan["preflight_variant"],
        "layers": [row["layer"] for row in per_layer],
        "per_layer": per_layer,
        "preparation_contract": {
            "rate_specific_gss_required": True,
            "reuse_k4_gss_forbidden": True,
            "k3_and_k4_gss_both_required": True,
            "candidate_conditioned_down_uses_decoded_gate_up_at_matching_rate": True,
            "reuse_raw_calibration_and_routes": True,
            "reuse_fixed_policy_and_permutations": True,
            "both_halves_built_from_scratch_at_dsv4_geometry": True,
            "down_conditioning_context": {
                "gate_bits": common.FLOOR_BITS,
                "up_bits": common.FLOOR_BITS,
                "semantics": "r7_pair_at_reference_rates_v1",
                "one_context_for_whole_down_curve": True,
                "shared_by": [
                    "probe_loss_curve",
                    "dp_allocation",
                    "encode_time_hessian",
                ],
                "note": (
                    "matching rate in the R7 pair_at sense "
                    "(r7_encoder/layer.py:901-925, 974-994): the whole down "
                    "curve (probe rates 3 and 4 and the encode-time Hessian) "
                    "is conditioned on ONE gate/up roundtrip decoded at the "
                    "reference rates; every conditioning receipt stamps the "
                    "rates actually used, never an approximation"
                ),
            },
        },
        "transform_seed_sha256": seed,
        "policy": policy,
        "scale_family": family,
        "profile_source": PROFILE_SOURCE,
        "codec_identity_sha256": codec_identity_sha256,
        "codec_identity_binding": (
            "codec_identity_sha256 is checked identical across every "
            "preparation manifest at BOTH rates and against the live codec "
            "when one was constructed, and the phase-7 worker dies unless "
            "it reproduces this exact identity; both halves are built at "
            "one venue, so no separate foreign-venue identity field exists"
        ),
        "sigma_reg": common.SIGMA_REG,
        "allocations": allocations,
    }
    receipt = common.seal(body, "readiness_receipt_sha256")
    common.write_json(
        args.work_root / "gss" / "readiness-receipt.json", receipt)
    print(
        json.dumps(
            {
                "readiness_receipt_sha256": receipt["readiness_receipt_sha256"],
                "layers": len(per_layer),
                "next": (
                    "dsv4_phase6b_enter.py (cwd "
                    f"{args.work_root}) with readiness_receipt_sha256="
                    f"'{receipt['readiness_receipt_sha256']}'"
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
