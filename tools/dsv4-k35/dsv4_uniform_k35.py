#!/usr/bin/env python3
"""dsv4_uniform_k35.py - DSV4 Flash 3.5-bpw plan and receipt state machine.

Port of the GLM-5.3 campaign's mixed K3/K4 state machine (glm53-quant-repo
src/quant_pipeline/campaign/glm53_uniform_k35.py, per-function citations
below) to DeepSeek V4 Flash geometry. The numeric discipline is unchanged
from the port source: per-tensor bits are integers 3 or 4 (the codec
accepts integer bits only, r7_encoder/trellis.py "accepts only 3, 4, or 5
bits"), the layer average is the rational 7/2, no sealed field carries a
float rate, every artifact is content-sealed with SHA-256 over canonical
JSON, and every verification fail-closes on schema, seal, census, and
rate. A planning receipt may not authorize process launch.

Geometry deltas (every constant re-derived from dsv4_common, which binds
dsv4_geometry's live-discovered Geometry; no GLM literal survives):
  - layers 0..42 are ALL routed (the port source had main layers 3..44
    plus a separate MTP layer); the encode domain is exactly
    dsv4_common.MAIN_LAYERS
  - 256 experts x 3 projections = 768 tensors per layer; floor 3 plus
    384 upgrades = 2688 bit-units = 7/2 bpw per layer
  - the mtp.{0,1,2} namespaces are NATIVE scope v1: no mtp_work_unit, no
    MTP qualification transitions, no MTP phase in sequential_states

Venue delta: no sealed multi-GPU encode attestation and no uniform 4-bpw
campaign exist for this model, so build_launch_plan takes neither a
sealed-venue preflight nor a baseline receipt. The only accepted venue
preflight is the DECLARED sm120 document (worker count 1..4 read from the
document content, no hard census), and the plan carries an explicit
declared-absent baseline block inside derived_from. The absolute KLD bar
stands alone; no comparison baseline is recorded or gated.

Worker contract: dsv4_worker.py imports this module as k35 and calls
verify_launch_plan, verify_state, verify_layer_allocation,
claim_next_layer, complete_layer, and _successor with the exact
signatures below, plus RATE_NUMERATOR and RATE_DENOMINATOR. The phase
string "k35_main_encoding" and the recovery evidence keys
k35_recovery_note / k35_recovery_quarantine stay byte-faithful to that
worker (verify_state tolerates extra evidence keys, which is what lets
the recovery transition ride through _successor).

State schema adoption: dsv4_phase6b_enter.py already mints state receipts
under quant-pipeline.dsv4-k35-state-receipt.v1 with the evidence key
dsv4_readiness_receipt_sha256 (dsv4_phase6b_enter.py:42-45); this module
adopts that schema, those phase names, and that evidence key. The
allocation schema and receipt body are byte-identical to
dsv4_probe_driver.seal_layer_allocation (dsv4_probe_driver.py:244-265)
and dsv4_phase6_gss.py:91, so probe-sealed allocations verify here
unchanged.

Naming: MASTER tensor names (layers.{L}.ffn.experts.{E}.{w1,w2,w3}
.weight) everywhere, from dsv4_common.tensor_full_name / _layer_tensor_names.

Orchestration-only: this module starts no process, imports no CUDA, and
invokes no allocator.

ASCII only. No em-dashes. CODE ONLY: nothing here is executed at
authoring time (dsv4_common resolves Geometry from the live master at
import; run inside the encode container).
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import dsv4_common as common
from dsv4_common import die
from dsv4_geometry import LORA_SCALE, MTP_ROUTED, ROUTED

# ---------------------------------------------------------------------------
# Schemas (NEW SURFACE dsv4 strings; never reuse port-source strings - seal
# collision across campaigns). Mirror of glm53_uniform_k35.py:42-48.
# ---------------------------------------------------------------------------

LAUNCH_PLAN_SCHEMA = "quant-pipeline.dsv4-k35-launch-plan.v1"
STATE_RECEIPT_SCHEMA = "quant-pipeline.dsv4-k35-state-receipt.v1"
CLAIM_RECEIPT_SCHEMA = "quant-pipeline.dsv4-k35-layer-claim.v1"
LAYER_ALLOCATION_SCHEMA = "quant-pipeline.dsv4-k35-layer-allocation.v1"
PACKED_KLD_SCHEMA = "quant-pipeline.dsv4-k35-packed-kld-receipt.v1"
PUBLICATION_RECEIPT_SCHEMA = "quant-pipeline.dsv4-k35-publication-receipt.v1"
INVENTORY_SCHEMA = "quant-pipeline.dsv4-release-inventory.v1"

# Declared sm120 encode venue (the ONLY venue; no sealed alternative
# exists). Mirror of glm53_uniform_k35.py:64-69 with the opt-in machinery
# deleted (there is no sealed venue to prefer) and the hard worker census
# relaxed to 1..4 read from the preflight document content.
SM120_DECLARED_PREFLIGHT_SCHEMA = "quant-pipeline.dsv4-sm120-declared-preflight.v1"
PREFLIGHT_VARIANT = "sm120-declared"
DECLARED_ATTESTED_BY = "sm120-declared-variant"
MIN_DECLARED_WORKERS = 1
MAX_DECLARED_WORKERS = 4

# Adopted from dsv4_phase6b_enter.py:42-45 (the entry snippet already
# mints states under this schema and evidence key).
READINESS_EVIDENCE_KEY = "dsv4_readiness_receipt_sha256"

# The dtype domain of the packed master (the CappedSource dispatch table,
# dsv4_capped_source.py:82-103; asserted by the release inventory).
SOURCE_DTYPES = ("F8_E4M3", "F8_E8M0", "I8", "BF16", "F32", "I64")

# ---------------------------------------------------------------------------
# Rate arithmetic (mirror of glm53_uniform_k35.py:76-90 on this geometry;
# the per-layer constants themselves come from dsv4_common).
# ---------------------------------------------------------------------------

RATE_NUMERATOR = 7
RATE_DENOMINATOR = 2
TARGET_BPW = "3.5"                      # display string only; never a sealed float

MAIN_ROUTED_MATRIX_COUNT = len(common.MAIN_LAYERS) * common.TENSORS_PER_LAYER
ROUTED_MATRIX_COUNT = MAIN_ROUTED_MATRIX_COUNT
GLOBAL_K4_TENSOR_COUNT = len(common.MAIN_LAYERS) * common.K4_TENSORS_PER_LAYER
GLOBAL_K3_TENSOR_COUNT = len(common.MAIN_LAYERS) * common.K3_TENSORS_PER_LAYER

# Exact capacity arithmetic over the encoded (main routed) surface,
# weights only, logical bytes; derivable without measurement. Every routed
# matrix of this model is [inter, hidden] or [hidden, inter]:
# hidden*inter elements either way. Mirror of glm53_uniform_k35.py:98-110.
ELEMENTS_PER_ROUTED_MATRIX = common.HIDDEN_SIZE * common.INTERMEDIATE_SIZE
ROUTED_ELEMENT_COUNT = ROUTED_MATRIX_COUNT * ELEMENTS_PER_ROUTED_MATRIX
TRELLIS_BYTES_PER_MATRIX = {
    3: ELEMENTS_PER_ROUTED_MATRIX * 3 // 8,
    4: ELEMENTS_PER_ROUTED_MATRIX * 4 // 8,
}
SU_SV_BYTES_PER_MATRIX = (common.HIDDEN_SIZE + common.INTERMEDIATE_SIZE) * 2
MCG_BYTES_PER_MATRIX = 4
EXACT_TRELLIS_PAYLOAD_BYTES = (
    ROUTED_ELEMENT_COUNT * RATE_NUMERATOR // (8 * RATE_DENOMINATOR))
EXACT_ROUTED_PAYLOAD_BYTES = (
    EXACT_TRELLIS_PAYLOAD_BYTES
    + ROUTED_MATRIX_COUNT * SU_SV_BYTES_PER_MATRIX
    + ROUTED_MATRIX_COUNT * MCG_BYTES_PER_MATRIX
)

# The absolute KLD bar carried unchanged from the campaign discipline. No
# uniform 4-bpw campaign exists for this model, so the bar stands alone:
# nothing is compared against a baseline and no baseline is recorded.
KLD_GATE_THRESHOLD = 0.06
PROFILE = "dsv4-mixed-k34-3p5bpw"

TODO_MEASURE_FIELDS = (
    # filled by: K3- and K4-specific GSS preparation receipts (phase 6;
    # cross-rate GSS reuse is forbidden, same rule as the port source)
    "rate_specific_gss_receipts",
    # filled by: five cold packed 3.5-bpw student KLD runs over the sealed
    # final token panel
    "k35_five_run_mean_kld",
    # filled by: measured per-GPU persistent (non-weight, non-KV) bytes and
    # usable HBM on the actual serving venue
    "serving_persistent_bytes_per_gpu",
)


def validate_rate_arithmetic() -> None:
    """Import-time census, every expected value re-derived from
    dsv4_common first (mirror of glm53_uniform_k35.py:154-169)."""
    assert common.TENSORS_PER_LAYER == common.NUM_EXPERTS * 3 == 768
    assert common.K4_TENSORS_PER_LAYER == common.K3_TENSORS_PER_LAYER == 384
    assert (
        common.TARGET_BIT_UNITS_PER_LAYER
        == common.TENSORS_PER_LAYER * RATE_NUMERATOR // RATE_DENOMINATOR
        == 2688
    )
    assert ROUTED_MATRIX_COUNT == 33_024
    assert GLOBAL_K4_TENSOR_COUNT == GLOBAL_K3_TENSOR_COUNT == 16_512
    assert GLOBAL_K4_TENSOR_COUNT + GLOBAL_K3_TENSOR_COUNT == ROUTED_MATRIX_COUNT
    assert ELEMENTS_PER_ROUTED_MATRIX == 8_388_608
    assert ROUTED_ELEMENT_COUNT == 277_025_390_592
    assert EXACT_TRELLIS_PAYLOAD_BYTES == 121_198_608_384
    assert EXACT_ROUTED_PAYLOAD_BYTES == 121_604_539_392


validate_rate_arithmetic()


# ---------------------------------------------------------------------------
# Sealed per-layer allocation document (required plan input; fail-closed).
# The receipt body is byte-identical to dsv4_probe_driver.py:244-265 so
# probe-sealed allocations verify unchanged; census authority is
# dsv4_common.audit_layer_allocation. Mirror of glm53_uniform_k35.py:348-411.
# ---------------------------------------------------------------------------


def build_provisional_allocation(layer: int) -> dict[str, int]:
    """Deterministic provisional split: K4 on every w2 (down) plus w1/w3
    of experts [0, upgrade_experts). Needs no measurement, is exactly
    384 K4 tensors, and is explicitly PROVISIONAL: encode requires the
    sensitivity DP allocation the phase-5 probe driver seals."""
    if layer not in common.MAIN_LAYERS:
        die(f"layer {layer} outside the routed surface")
    upgrade_experts = (
        common.K4_TENSORS_PER_LAYER - common.NUM_EXPERTS) // 2
    if not 0 <= upgrade_experts < common.NUM_EXPERTS:
        die("provisional upgrade census does not close at this geometry")
    allocation: dict[str, int] = {}
    for expert in range(common.NUM_EXPERTS):
        for projection in common.PROJECTIONS:
            upgraded = projection == "w2" or expert < upgrade_experts
            allocation[common.tensor_full_name(layer, expert, projection)] = (
                4 if upgraded else 3)
    common.audit_layer_allocation(layer, allocation)
    return allocation


def seal_layer_allocation(
    layer: int, allocation: Mapping[str, int], *, provisional: bool, basis: str
) -> dict[str, Any]:
    common.audit_layer_allocation(layer, allocation)
    if common.TARGET_BIT_UNITS_PER_LAYER * RATE_DENOMINATOR != (
            common.TENSORS_PER_LAYER * RATE_NUMERATOR):
        die("3.5-bpw rate arithmetic does not close at this census")
    body = {
        "schema": LAYER_ALLOCATION_SCHEMA,
        "layer": layer,
        "allocation": dict(sorted(allocation.items())),
        "bit_units": common.TARGET_BIT_UNITS_PER_LAYER,
        "k4_tensor_count": common.K4_TENSORS_PER_LAYER,
        "k3_tensor_count": common.K3_TENSORS_PER_LAYER,
        "rate": {
            "numerator": RATE_NUMERATOR,
            "denominator": RATE_DENOMINATOR,
        },
        "provisional": bool(provisional),
        "basis": basis,
    }
    return common.seal(body, "allocation_sha256")


def verify_layer_allocation(receipt: Mapping[str, Any], *, layer: int) -> str:
    seal = common.verify_seal(
        receipt,
        schema=LAYER_ALLOCATION_SCHEMA,
        field="allocation_sha256",
        label=f"layer {layer} 3.5-bpw allocation",
    )
    if receipt.get("layer") != layer:
        die("allocation receipt layer binding differs")
    if receipt.get("rate") != {
        "numerator": RATE_NUMERATOR,
        "denominator": RATE_DENOMINATOR,
    }:
        die("allocation receipt rate differs from 7/2")
    for key, expected in (
        ("bit_units", common.TARGET_BIT_UNITS_PER_LAYER),
        ("k4_tensor_count", common.K4_TENSORS_PER_LAYER),
        ("k3_tensor_count", common.K3_TENSORS_PER_LAYER),
    ):
        if receipt.get(key) != expected:
            die(f"allocation receipt {key} census differs")
    if not isinstance(receipt.get("provisional"), bool):
        die("allocation receipt provisional flag is malformed")
    if not isinstance(receipt.get("basis"), str) or not receipt["basis"].strip():
        die("allocation receipt basis is malformed")
    allocation = receipt.get("allocation")
    if not isinstance(allocation, Mapping):
        die("allocation receipt body is malformed")
    common.audit_layer_allocation(layer, allocation)
    return seal


# ---------------------------------------------------------------------------
# Release-inventory surfaces (fail-closed re-derivation of scope, census,
# and shapes from dsv4_common + dsv4_geometry; the builder self-checks
# through this before writing). Mirror of the port source's
# _inventory_surfaces (glm53_uniform_k4.py:69-140) with the revision
# binding relaxed from a git sha to the sha256 identity the dsv4 release
# inventory seals.
# ---------------------------------------------------------------------------


def _routed_shape_gate(name: str, shape: Any) -> None:
    match = ROUTED.fullmatch(name) or MTP_ROUTED.fullmatch(name)
    if match is None:
        die(f"routed row does not match the master grammar: {name}")
    projection = match.group(3)
    expected = (
        (common.INTERMEDIATE_SIZE, common.HIDDEN_SIZE)
        if projection in ("w1", "w3")
        else (common.HIDDEN_SIZE, common.INTERMEDIATE_SIZE)
    )
    shape_t = tuple(int(v) for v in shape)
    if shape_t == expected:
        return  # logical (BF16-form) layout
    packed = (expected[0], expected[1] // 2)
    if shape_t == packed:
        return  # packed master: I8 nibble pairs along K (element 2i=LOW)
    die(
        f"routed shape drift for {name}: {list(shape)} differs from "
        f"logical {list(expected)} or packed {list(packed)} "
        "(orientation or geometry drift)"
    )


def _routed_logical_elements(row: Mapping[str, Any]) -> int:
    """Logical element count for a routed row: packed I8 rows carry the
    nibble pairs of 2x the logical K; logical rows count directly."""
    match = ROUTED.fullmatch(row["tensor_name"]) or MTP_ROUTED.fullmatch(
        row["tensor_name"])
    if match is None:
        die(f"routed row does not match the master grammar: "
            f"{row['tensor_name']}")
    projection = match.group(3)
    expected = (
        (common.INTERMEDIATE_SIZE, common.HIDDEN_SIZE)
        if projection in ("w1", "w3")
        else (common.HIDDEN_SIZE, common.INTERMEDIATE_SIZE)
    )
    shape = [int(v) for v in row["shape"]]
    if tuple(shape) == expected:
        return shape[0] * shape[1]
    if shape == [expected[0], expected[1] // 2]:
        return shape[0] * shape[1] * 2
    die(f"routed shape is neither logical nor packed: {row['tensor_name']}")


def _inventory_surfaces(
    inventory: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    common.verify_seal(
        inventory,
        schema=INVENTORY_SCHEMA,
        field="inventory_sha256",
        label="dsv4 release inventory",
    )
    if inventory.get("seal_mode") != "full-shard-sha256":
        die("execution requires a full-shard SHA256 inventory")
    common.require_hash(inventory.get("model_revision"), "model revision")
    geometry = inventory.get("geometry")
    if not isinstance(geometry, Mapping):
        die("inventory geometry is absent")
    required_geometry = {
        "model_type": common.G.cfg["model_type"],
        "main_layers": len(common.MAIN_LAYERS),
        "all_main_layers_routed": True,
        "hash_routed_layers": list(common.G.hash_layers),
        "mtp_modules": list(common.G.mtp_modules),
        "routed_experts": common.NUM_EXPERTS,
        "top_k": common.TOP_K,
        "hidden_size": common.HIDDEN_SIZE,
        "moe_intermediate_size": common.INTERMEDIATE_SIZE,
    }
    for key, expected in required_geometry.items():
        if geometry.get(key) != expected:
            die(
                f"released geometry {key} differs: "
                f"{geometry.get(key)!r} != {expected!r}"
            )
    if geometry.get("discovered_layers") != list(common.MAIN_LAYERS):
        die("inventory does not discover the complete routed layer surface")
    identity = inventory.get("identity")
    if not isinstance(identity, Mapping):
        die("inventory identity binding is absent")
    common.require_hash(identity.get("lora_sha256"), "inventory lora identity")
    if identity.get("lora_sites") != len(common.G.lora_sites or ()):
        die("inventory lora site census differs from the discovered geometry")
    if identity.get("lora_scale") != LORA_SCALE:
        die("inventory lora scale differs from the discovered fold semantics")

    tensors = inventory.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        die("inventory tensor surface is absent")
    names: set[str] = set()
    main: list[dict[str, Any]] = []
    mtp: list[dict[str, Any]] = []
    native: list[dict[str, Any]] = []
    dtypes: set[str] = set()
    for raw in tensors:
        if not isinstance(raw, Mapping):
            die("inventory contains a malformed tensor row")
        row = dict(raw)
        name = row.get("tensor_name")
        if not isinstance(name, str) or not name or name in names:
            die("inventory tensor names are absent or duplicated")
        names.add(name)
        shape = row.get("shape")
        source_bytes = row.get("source_bytes")
        if (
            isinstance(source_bytes, bool)
            or not isinstance(source_bytes, int)
            or source_bytes < 0
            or not isinstance(shape, list)
            or not shape
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                   for v in shape)
        ):
            die(f"inventory row is malformed: {name}")
        common.require_hash(
            row.get("source_payload_sha256"), f"payload hash {name}")
        dtype = row.get("dtype")
        if dtype not in SOURCE_DTYPES:
            die(f"inventory dtype outside the packed-source domain: {name}")
        dtypes.add(dtype)
        scope = row.get("scope")
        if scope == "routed_expert":
            match = ROUTED.fullmatch(name)
            if match is None or int(match.group(1)) not in common.MAIN_LAYERS:
                die(f"routed scope does not re-derive: {name}")
            _routed_shape_gate(name, shape)
            main.append(row)
        elif scope == "mtp_routed_expert":
            if MTP_ROUTED.fullmatch(name) is None:
                die(f"mtp scope does not re-derive: {name}")
            _routed_shape_gate(name, shape)
            mtp.append(row)
        elif scope == "native":
            if ROUTED.fullmatch(name) is not None or MTP_ROUTED.fullmatch(
                    name) is not None:
                die(f"native scope does not re-derive: {name}")
            native.append(row)
        else:
            die(f"inventory row carries an unknown scope: {name}")
    if len(main) != common.G.main_routed_tensors:
        die(f"routed census differs: {len(main)}")
    if len(mtp) != common.G.mtp_routed_tensors:
        die(f"mtp routed census differs: {len(mtp)}")
    per_layer: dict[int, int] = {}
    for row in main:
        layer = int(ROUTED.fullmatch(row["tensor_name"]).group(1))
        per_layer[layer] = per_layer.get(layer, 0) + 1
    if sorted(per_layer) != list(common.MAIN_LAYERS) or any(
            count != common.TENSORS_PER_LAYER
            for count in per_layer.values()):
        die("per-layer routed census differs from the discovered geometry")
    if dtypes != set(SOURCE_DTYPES):
        die(f"inventory dtype surface incomplete: {sorted(dtypes)}")
    if names != set(common.G.meta):
        die("inventory tensor closure differs from the discovered meta")
    return main, mtp, native


# ---------------------------------------------------------------------------
# Declared sm120 encode venue (mirror of glm53_uniform_k35.py:233-303 with
# the worker count read from the document, 1..4, and no hard census).
# ---------------------------------------------------------------------------


def _declared_workers(
    preflight: Mapping[str, Any], inventory_sha: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    preflight_sha = common.verify_seal(
        preflight,
        schema=SM120_DECLARED_PREFLIGHT_SCHEMA,
        field="preflight_sha256",
        label="declared sm120 preflight",
    )
    workers_count = preflight.get("workers")
    if (
        isinstance(workers_count, bool)
        or not isinstance(workers_count, int)
        or not MIN_DECLARED_WORKERS <= workers_count <= MAX_DECLARED_WORKERS
    ):
        die(
            "declared sm120 layer-streaming preflight must declare "
            f"{MIN_DECLARED_WORKERS}..{MAX_DECLARED_WORKERS} workers"
        )
    if (
        preflight.get("ready") is not True
        or preflight.get("mode") != "layer-streaming"
        or preflight.get("checkpoint_seal_mode") != "full-shard-sha256"
        or preflight.get("checkpoint_inventory_sha256") != inventory_sha
    ):
        die("declared sm120 layer-streaming preflight is not execution-ready")
    declaration = preflight.get("declaration")
    if (
        not isinstance(declaration, Mapping)
        or declaration.get("attested_by") != DECLARED_ATTESTED_BY
    ):
        die("declared sm120 preflight lacks the declaration block")
    rationale = declaration.get("rationale")
    if (
        not isinstance(rationale, str)
        or not rationale.strip()
        or "\n" in rationale
        or len(rationale) > 200
    ):
        die("declared sm120 rationale must be one line of at most 200 characters")
    common.require_hash(
        declaration.get("runtime_receipt_sha256"),
        "declared preflight runtime receipt link",
    )
    gpus = preflight.get("gpus")
    if not isinstance(gpus, list) or len(gpus) != workers_count:
        die("declared sm120 preflight gpus census differs from its workers count")
    workers: list[dict[str, Any]] = []
    indices: set[int] = set()
    for slot, raw in enumerate(gpus):
        if not isinstance(raw, Mapping):
            die("declared sm120 preflight GPU row is malformed")
        index = raw.get("index")
        name = raw.get("name")
        capability = raw.get("compute_capability")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index in indices
            or not isinstance(name, str)
            or "RTX PRO 6000" not in name
            or not isinstance(capability, str)
            or not capability.startswith("12.")
        ):
            die(
                "declared sm120 workers must be distinct RTX PRO 6000 "
                "(compute capability 12.x) devices"
            )
        indices.add(index)
        workers.append(
            {
                "worker_id": f"sm120-{slot}",
                "physical_gpu": index,
                "cuda_visible_devices": str(index),
                "codec_device": "cuda:0",
                "name": name,
                "compute_capability": capability,
                "preflight_sha256": preflight_sha,
                "preflight_variant": PREFLIGHT_VARIANT,
            }
        )
    block = {
        "attested_by": declaration["attested_by"],
        "rationale": declaration["rationale"],
        "runtime_receipt_sha256": declaration["runtime_receipt_sha256"],
    }
    return workers, block


# ---------------------------------------------------------------------------
# Launch plan (mirror of glm53_uniform_k35.py:445-703; no baseline receipt
# argument, no mtp_work_unit, capacity re-derived for this geometry and
# the packed source domain).
# ---------------------------------------------------------------------------


def build_launch_plan(
    inventory: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    layer_allocations: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the immutable, no-launch 3.5-bpw work contract from sealed
    evidence. The venue is the declared sm120 preflight (the only venue);
    no uniform 4-bpw baseline receipt is demanded or recorded."""
    main_rows, mtp_rows, native_rows = _inventory_surfaces(inventory)
    inventory_sha = str(inventory["inventory_sha256"])
    workers, declaration_block = _declared_workers(preflight, inventory_sha)

    by_layer: dict[int, list[dict[str, Any]]] = {
        layer: [] for layer in common.MAIN_LAYERS}
    tensor_contract: list[dict[str, Any]] = []
    for row in main_rows:
        layer = int(ROUTED.fullmatch(row["tensor_name"]).group(1))
        by_layer[layer].append(row)
        tensor_contract.append(
            {
                "tensor_name": row["tensor_name"],
                "source_payload_sha256": row.get("source_payload_sha256"),
                "bits": "per_layer_allocation",
                "disposition": "mixed_k34_3p5bpw_direct_encode",
                "execution_track": "main_dynamic_layer_scheduler",
            }
        )

    units: list[dict[str, Any]] = []
    for layer in common.MAIN_LAYERS:
        rows = by_layer[layer]
        allocation_receipt = layer_allocations.get(layer)
        if not isinstance(allocation_receipt, Mapping):
            die(f"layer {layer} has no sealed 3.5-bpw allocation document")
        allocation_sha = verify_layer_allocation(allocation_receipt, layer=layer)
        names = sorted(row["tensor_name"] for row in rows)
        if len(rows) != common.TENSORS_PER_LAYER:
            die("validated routed layer lost matrices")
        units.append(
            {
                "layer": layer,
                "expert_count": common.NUM_EXPERTS,
                "matrix_count": len(rows),
                "source_bytes": sum(int(row["source_bytes"]) for row in rows),
                "source_elements": sum(
                    _routed_logical_elements(row) for row in rows),
                "tensor_names_sha256": common.sha256_bytes(
                    common.canonical_json(names)),
                "rate": {
                    "numerator": RATE_NUMERATOR,
                    "denominator": RATE_DENOMINATOR,
                },
                "target_bpw": TARGET_BPW,
                "per_tensor_allowed_bits": list(common.PER_TENSOR_ALLOWED_BITS),
                "bit_units": common.TARGET_BIT_UNITS_PER_LAYER,
                "k4_tensor_count": common.K4_TENSORS_PER_LAYER,
                "k3_tensor_count": common.K3_TENSORS_PER_LAYER,
                "allocation_sha256": allocation_sha,
                "allocation_provisional": bool(
                    allocation_receipt.get("provisional")),
                "global_allocator": False,
                "candidate_rate_grid": False,
            }
        )
    if any(unit["allocation_provisional"] for unit in units):
        # Planning with the deterministic split is allowed; ENCODE is not.
        # The phase-5 probe driver's sensitivity DP allocation must replace
        # every provisional layer (claim_next_layer fail-closes on
        # provisional units) and the plan must then be rebuilt.
        pass

    scale_rows = [r for r in native_rows if r["dtype"] == "F8_E8M0"]
    native_copy_rows = [r for r in native_rows if r["dtype"] != "F8_E8M0"]
    mtp_names = sorted(row["tensor_name"] for row in mtp_rows)
    native_names = [row["tensor_name"] for row in native_copy_rows]

    queue = [
        unit["layer"]
        for unit in sorted(
            units, key=lambda unit: (-int(unit["source_bytes"]), int(unit["layer"])))
    ]

    body = {
        "schema": LAUNCH_PLAN_SCHEMA,
        "model_revision": inventory.get("model_revision"),
        "inventory_sha256": inventory_sha,
        "preflight_sha256": preflight["preflight_sha256"],
        "preflight_variant": PREFLIGHT_VARIANT,
        "launch_authorized": False,
        "boundary": (
            "sealed planning and receipt transitions only; this document "
            "starts no process"
        ),
        "profile": PROFILE,
        "runtime_target": {
            "encode_workers": len(workers),
            "preflight_variant": PREFLIGHT_VARIANT,
            "serving_layout": (
                "not bound by this encode plan; decided at the pack phase"
            ),
        },
        "derived_from": {
            "state_machine_port": (
                "tools/dsv4-k35/dsv4_uniform_k35.py (ported state machine; "
                "per-function provenance citations in that source)"
            ),
            "four_bpw_baseline": {
                "status": "absent_by_declaration",
                "reason": (
                    "no uniform 4-bpw campaign exists for this model; no "
                    "sealed baseline receipt can be demanded"
                ),
                "baseline_comparison_recorded_not_gating": (
                    "not_applicable_no_baseline"
                ),
            },
            "absolute_kld_bar_only": True,
        },
        "geometry": {
            "main_layers": list(common.MAIN_LAYERS),
            "all_main_layers_routed": True,
            "hash_routed_layers": list(common.G.hash_layers),
            "mtp_modules": list(common.G.mtp_modules),
            "mtp_scope_v1_native": True,
            "routed_experts": common.NUM_EXPERTS,
            "projections": list(common.PROJECTIONS),
        },
        "rate_contract": {
            "allocation": "per_layer_exact_bit_unit_budget",
            "rate": {
                "numerator": RATE_NUMERATOR,
                "denominator": RATE_DENOMINATOR,
            },
            "target_bpw": TARGET_BPW,
            "per_tensor_allowed_bits": list(common.PER_TENSOR_ALLOWED_BITS),
            "global_allocator_invoked": False,
            "candidate_rate_grid_invoked": False,
            "K3": GLOBAL_K3_TENSOR_COUNT,
            "K4": GLOBAL_K4_TENSOR_COUNT,
            "main_routed_matrix_count": MAIN_ROUTED_MATRIX_COUNT,
            "bit_units_per_layer": common.TARGET_BIT_UNITS_PER_LAYER,
            "sensitivity_dp_allocation_required_for_encode": True,
            "provisional_allocation_valid_for_planning_only": True,
        },
        "preparation_contract": {
            "reuse_raw_calibration_and_routes": True,
            "reuse_fixed_policy_and_permutations": True,
            "rate_specific_gss_required": True,
            "reuse_k4_gss_forbidden": True,
            "k3_and_k4_gss_both_required": True,
            "down_conditioning_reference_bits": common.FLOOR_BITS,
            "down_conditioning_semantics": (
                "r7_pair_at_reference_rates_v1: both candidate rates and "
                "the encode-time down Hessian condition on w1/w3 decoded "
                "at the reference bits (dsv4_common.FLOOR_BITS), the "
                "identical context the phase-5 probe measured and the DP "
                "allocation ranked under"
            ),
        },
        "capacity_contract": {
            "routed_matrix_count": ROUTED_MATRIX_COUNT,
            "routed_element_count": ROUTED_ELEMENT_COUNT,
            "elements_per_routed_matrix": ELEMENTS_PER_ROUTED_MATRIX,
            "trellis_payload_bytes": EXACT_TRELLIS_PAYLOAD_BYTES,
            "suh_svh_bytes": ROUTED_MATRIX_COUNT * SU_SV_BYTES_PER_MATRIX,
            "mcg_bytes": ROUTED_MATRIX_COUNT * MCG_BYTES_PER_MATRIX,
            "routed_payload_bytes": EXACT_ROUTED_PAYLOAD_BYTES,
            "native_packed_source_bytes": sum(
                int(row["source_bytes"]) for row in native_rows),
            "mtp_routed_packed_source_bytes": sum(
                int(row["source_bytes"]) for row in mtp_rows),
            "native_output_layout_binding": (
                "the packed output layout for native and mtp tensors binds "
                "at the pack phase; this plan records packed-source "
                "accounting only"
            ),
            "kv_pool_note": (
                "kv_available = floor(hbm * utilization) - weight - "
                "persistents - safety; TODO(measure): persistents and "
                "utilization on the serving venue"
            ),
        },
        "native_copy_contract": {
            "policy": "capped_source_dequant_fold_copy_v1",
            "includes_all_nonrouted_nonscale": True,
            "includes_mtp_experts": False,
            "dequant_scale_tensors_consumed_not_copied": len(scale_rows),
            "tensor_count": len(native_copy_rows),
            "source_bytes": sum(
                int(row["source_bytes"]) for row in native_copy_rows),
            "tensor_names_sha256": common.sha256_bytes(
                common.canonical_json(native_names)),
        },
        "mtp_scope_v1": {
            "policy": "native_copy_not_encoded",
            "modules": list(common.G.mtp_modules),
            "matrix_count": len(mtp_rows),
            "source_bytes": sum(int(row["source_bytes"]) for row in mtp_rows),
            "tensor_names_sha256": common.sha256_bytes(
                common.canonical_json(mtp_names)),
            "note": (
                "mtp expert tensors are native scope v1: byte-derived "
                "through the capped source like other natives, never "
                "encoded, no adapter receipt, no MTP phase"
            ),
        },
        "routed_tensor_contract": tensor_contract,
        "work_units": units,
        "scheduler": {
            "policy": "dynamic_next_unclaimed_whole_layer",
            "static_layer_partition_forbidden": True,
            "one_active_layer_per_worker": True,
            "workers": workers,
            "initial_queue": queue,
        },
        "sequential_states": [
            "planned",
            "k35_main_encoding",
            "k35_main_encoded",
            "k35_packed",
            "k35_kld_qualified",
            "publication_authorized",
        ],
        "kld_gate": {
            "metric": "mean_of_five_run_mean_tokenwise_kld",
            "threshold_lt": KLD_GATE_THRESHOLD,
            "direction": "teacher_to_student",
            "five_cold_runs_required": True,
            "absolute_bar_only_no_baseline_comparison": True,
            "requires_reader_audit": True,
        },
        "publication_gate": {
            "required_predecessor_state": "k35_kld_qualified",
            "required_receipt_schema": PACKED_KLD_SCHEMA,
            "requires_same_packed_3p5bpw_checkpoint": True,
            "requires_reader_audit": True,
            "hf_publication_receipt_required": True,
        },
        "todo_measure_fields": list(TODO_MEASURE_FIELDS),
        "preflight_declaration": dict(declaration_block),
        "venue_attestation": "declared_variant_not_a_sealed_attestation",
    }
    if len(tensor_contract) != ROUTED_MATRIX_COUNT:
        die("3.5-bpw routed census drift")
    if sum(unit["source_elements"] for unit in units) != ROUTED_ELEMENT_COUNT:
        die("routed element census disagrees with capacity constants")
    if not MIN_DECLARED_WORKERS <= len(workers) <= MAX_DECLARED_WORKERS:
        die("encode worker census differs from the declared venue")
    return common.seal(body, "launch_plan_sha256")


def verify_launch_plan(plan: Mapping[str, Any]) -> str:
    seal = common.verify_seal(
        plan,
        schema=LAUNCH_PLAN_SCHEMA,
        field="launch_plan_sha256",
        label="mixed K3/K4 3.5-bpw launch plan",
    )
    rate = plan.get("rate_contract")
    if not isinstance(rate, Mapping) or (
        rate.get("rate") != {
            "numerator": RATE_NUMERATOR,
            "denominator": RATE_DENOMINATOR,
        }
        or rate.get("K3") != GLOBAL_K3_TENSOR_COUNT
        or rate.get("K4") != GLOBAL_K4_TENSOR_COUNT
        or rate.get("bit_units_per_layer") != common.TARGET_BIT_UNITS_PER_LAYER
        or rate.get("per_tensor_allowed_bits")
        != list(common.PER_TENSOR_ALLOWED_BITS)
    ):
        die("3.5-bpw launch-plan rate census differs")
    capacity = plan.get("capacity_contract")
    if not isinstance(capacity, Mapping) or (
        capacity.get("trellis_payload_bytes") != EXACT_TRELLIS_PAYLOAD_BYTES
        or capacity.get("routed_payload_bytes") != EXACT_ROUTED_PAYLOAD_BYTES
    ):
        die("3.5-bpw launch-plan capacity arithmetic differs")
    if plan.get("preflight_variant") != PREFLIGHT_VARIANT:
        die("launch plan preflight variant is not the declared sm120 venue")
    declaration = plan.get("preflight_declaration")
    if not isinstance(declaration, Mapping) or (
        declaration.get("attested_by") != DECLARED_ATTESTED_BY
        or not isinstance(declaration.get("rationale"), str)
        or not declaration.get("rationale").strip()
        or "\n" in declaration.get("rationale")
    ):
        die("declared sm120 preflight block in the plan is malformed")
    common.require_hash(
        declaration.get("runtime_receipt_sha256"),
        "declared preflight runtime receipt link",
    )
    if plan.get("venue_attestation") != "declared_variant_not_a_sealed_attestation":
        die("declared-venue plan lacks its explicit attestation disclaimer")
    baseline = plan.get("derived_from", {}).get("four_bpw_baseline")
    if not isinstance(baseline, Mapping) or baseline.get(
            "status") != "absent_by_declaration":
        die("launch plan lacks the declared-absent baseline block")
    if plan.get("launch_authorized") is not False:
        die("a planning receipt may not authorize process launch")
    return seal


# ---------------------------------------------------------------------------
# State machine (mirror of glm53_uniform_k35.py:756-1182 minus every MTP
# transition; schema and readiness evidence key adopted from
# dsv4_phase6b_enter.py:42-45).
# ---------------------------------------------------------------------------


def initial_state(plan: Mapping[str, Any]) -> dict[str, Any]:
    plan_sha = verify_launch_plan(plan)
    body = {
        "schema": STATE_RECEIPT_SCHEMA,
        "launch_plan_sha256": plan_sha,
        "sequence": 0,
        "previous_state_receipt_sha256": None,
        "phase": "planned",
        "pending_layers": list(plan["scheduler"]["initial_queue"]),
        "active_claims": {},
        "completed_layers": {},
        "evidence": {},
        "publication_authorized": False,
    }
    return common.seal(body, "state_receipt_sha256")


def verify_state(plan: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    plan_sha = verify_launch_plan(plan)
    state_sha = common.verify_seal(
        state,
        schema=STATE_RECEIPT_SCHEMA,
        field="state_receipt_sha256",
        label="3.5-bpw state receipt",
    )
    if state.get("launch_plan_sha256") != plan_sha:
        die("state receipt targets a different launch plan")
    pending = state.get("pending_layers")
    active = state.get("active_claims")
    completed = state.get("completed_layers")
    if (
        not isinstance(pending, list)
        or not isinstance(active, Mapping)
        or not isinstance(completed, Mapping)
    ):
        die("state scheduler domains are malformed")
    sequence = state.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        die("state sequence is malformed")
    predecessor = state.get("previous_state_receipt_sha256")
    if sequence == 0 and predecessor is not None:
        die("initial state may not name a predecessor")
    if sequence > 0:
        common.require_hash(predecessor, "state predecessor receipt")
    domain = set(common.MAIN_LAYERS)
    pending_set = set(pending)
    active_layers = {
        row.get("layer") for row in active.values()
        if isinstance(row, Mapping)}
    try:
        complete_set = {int(layer) for layer in completed}
    except (TypeError, ValueError):
        die("completed layer keys are malformed")
    if (
        len(pending) != len(pending_set)
        or len(active_layers) != len(active)
        or pending_set & active_layers
        or pending_set & complete_set
        or active_layers & complete_set
        or pending_set | active_layers | complete_set != domain
    ):
        die("state layer partition does not close exactly")
    worker_ids = {row["worker_id"] for row in plan["scheduler"]["workers"]}
    if set(active) - worker_ids:
        die("state has a claim for an unknown plan worker")
    for worker_id, claim in active.items():
        if not isinstance(claim, Mapping):
            die("state contains a malformed active claim")
        common.verify_seal(
            claim,
            schema=CLAIM_RECEIPT_SCHEMA,
            field="claim_receipt_sha256",
            label="active layer claim",
        )
        if (
            claim.get("launch_plan_sha256") != plan_sha
            or claim.get("worker_id") != worker_id
            or claim.get("target_bpw") != TARGET_BPW
            or claim.get("bit_units") != common.TARGET_BIT_UNITS_PER_LAYER
            or claim.get("preflight_variant") != plan.get("preflight_variant")
        ):
            die("active layer claim binding differs")
    for layer, completion in completed.items():
        if not isinstance(completion, Mapping) or completion.get(
                "worker_id") not in worker_ids:
            die(f"completed layer {layer} receipt is malformed")
        common.require_hash(
            completion.get("claim_receipt_sha256"),
            f"completed layer {layer} claim")
        common.require_hash(
            completion.get("layer_receipt_sha256"),
            f"completed layer {layer} artifact")
    phase = state.get("phase")
    if phase not in plan["sequential_states"]:
        die("state phase is outside the launch plan")
    if phase != "publication_authorized" and state.get(
            "publication_authorized") is not False:
        die("publication is authorized before the 3.5-bpw KLD gate")
    if phase == "publication_authorized" and state.get(
            "publication_authorized") is not True:
        die("publication authorization state is internally inconsistent")
    evidence = state.get("evidence")
    if not isinstance(evidence, Mapping):
        die("state evidence is malformed")
    if phase == "planned" and (
        list(pending) != list(plan["scheduler"]["initial_queue"])
        or active
        or completed
        or evidence
        or sequence != 0
    ):
        die("planned state contains premature execution evidence")
    if phase != "planned":
        common.require_hash(
            evidence.get(READINESS_EVIDENCE_KEY), "3.5-bpw readiness receipt")
    closed_main_phases = {
        "k35_main_encoded",
        "k35_packed",
        "k35_kld_qualified",
        "publication_authorized",
    }
    if phase in closed_main_phases and (
        pending or active or complete_set != set(common.MAIN_LAYERS)
    ):
        die("post-main state does not close all main 3.5-bpw layers")
    if phase in closed_main_phases:
        common.require_hash(
            evidence.get("main_routed_receipt_sha256"),
            "main routed 3.5-bpw receipt")
    if phase in {"k35_packed", "k35_kld_qualified", "publication_authorized"}:
        common.require_hash(
            evidence.get("packed_checkpoint_receipt_sha256"),
            "packed 3.5-bpw checkpoint receipt")
        common.require_hash(
            evidence.get("native_copy_receipt_sha256"),
            "native non-routed copy receipt")
    if phase in {"k35_kld_qualified", "publication_authorized"}:
        common.require_hash(
            evidence.get("k35_packed_kld_receipt_sha256"),
            "packed 3.5-bpw KLD receipt")
    return state_sha


def _successor(
    plan: Mapping[str, Any], state: Mapping[str, Any], **updates: Any
) -> dict[str, Any]:
    previous = verify_state(plan, state)
    body = copy.deepcopy(dict(state))
    del body["state_receipt_sha256"]
    body.update(copy.deepcopy(updates))
    body["sequence"] = int(state["sequence"]) + 1
    body["previous_state_receipt_sha256"] = previous
    return common.seal(body, "state_receipt_sha256")


def enter_k35_encoding(
    plan: Mapping[str, Any], state: Mapping[str, Any],
    *, readiness_receipt_sha256: str,
) -> dict[str, Any]:
    """Machine-native twin of dsv4_phase6b_enter.enter_k35_encoding (the
    operational entry path is that snippet; both mint the same successor
    under the adopted schema and evidence key)."""
    verify_state(plan, state)
    if state.get("phase") != "planned":
        die("3.5-bpw encoding may start only from planned")
    evidence = dict(state.get("evidence", {}))
    evidence[READINESS_EVIDENCE_KEY] = common.require_hash(
        readiness_receipt_sha256, "3.5-bpw readiness receipt")
    return _successor(plan, state, phase="k35_main_encoding", evidence=evidence)


def claim_next_layer(
    plan: Mapping[str, Any], state: Mapping[str, Any], *, worker_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_state(plan, state)
    if state.get("phase") != "k35_main_encoding":
        die("layer claims are allowed only during the k35_main_encoding phase")
    workers = {row["worker_id"] for row in plan["scheduler"]["workers"]}
    if worker_id not in workers:
        die("unknown encode worker (not in the plan scheduler)")
    active = copy.deepcopy(dict(state["active_claims"]))
    if worker_id in active:
        die("worker already owns an active layer")
    pending = list(state["pending_layers"])
    if not pending:
        die("no unclaimed 3.5-bpw layers remain")
    layer = int(pending.pop(0))
    unit = next(row for row in plan["work_units"] if row["layer"] == layer)
    if unit.get("allocation_provisional"):
        # Fail-closed: the deterministic split plans the budget; only the
        # measured sensitivity DP allocation may be encoded.
        die(
            f"layer {layer} still carries a provisional allocation; encode "
            "requires the phase-5 sensitivity DP allocation and a plan "
            "rebuild"
        )
    claim = common.seal(
        {
            "schema": CLAIM_RECEIPT_SCHEMA,
            "launch_plan_sha256": plan["launch_plan_sha256"],
            "parent_state_receipt_sha256": state["state_receipt_sha256"],
            "worker_id": worker_id,
            "layer": layer,
            "tensor_names_sha256": unit["tensor_names_sha256"],
            "allocation_sha256": unit["allocation_sha256"],
            "target_bpw": TARGET_BPW,
            "rate": {
                "numerator": RATE_NUMERATOR,
                "denominator": RATE_DENOMINATOR,
            },
            "per_tensor_allowed_bits": list(common.PER_TENSOR_ALLOWED_BITS),
            "bit_units": common.TARGET_BIT_UNITS_PER_LAYER,
            "preflight_variant": plan["preflight_variant"],
        },
        "claim_receipt_sha256",
    )
    active[worker_id] = claim
    successor = _successor(
        plan, state, pending_layers=pending, active_claims=active)
    return successor, claim


def complete_layer(
    plan: Mapping[str, Any], state: Mapping[str, Any],
    *,
    worker_id: str,
    layer: int,
    layer_receipt_sha256: str,
) -> dict[str, Any]:
    verify_state(plan, state)
    if state.get("phase") != "k35_main_encoding":
        die("layers may complete only during the k35_main_encoding phase")
    active = copy.deepcopy(dict(state["active_claims"]))
    claim = active.get(worker_id)
    if not isinstance(claim, Mapping) or claim.get("layer") != layer:
        die("layer completion does not match the worker claim")
    common.verify_seal(
        claim,
        schema=CLAIM_RECEIPT_SCHEMA,
        field="claim_receipt_sha256",
        label="layer claim",
    )
    completed = copy.deepcopy(dict(state["completed_layers"]))
    completed[str(layer)] = {
        "worker_id": worker_id,
        "claim_receipt_sha256": claim["claim_receipt_sha256"],
        "layer_receipt_sha256": common.require_hash(
            layer_receipt_sha256, "3.5-bpw layer receipt"),
    }
    del active[worker_id]
    return _successor(
        plan, state, active_claims=active, completed_layers=completed)


def seal_main_k35(
    plan: Mapping[str, Any], state: Mapping[str, Any],
    *, main_routed_receipt_sha256: str,
) -> dict[str, Any]:
    verify_state(plan, state)
    if state.get("phase") != "k35_main_encoding":
        die("main 3.5-bpw may seal only after main encoding")
    if (
        state["pending_layers"]
        or state["active_claims"]
        or len(state["completed_layers"]) != len(common.MAIN_LAYERS)
    ):
        die("main 3.5-bpw sealing is blocked until every main routed layer completes")
    evidence = dict(state.get("evidence", {}))
    evidence["main_routed_receipt_sha256"] = common.require_hash(
        main_routed_receipt_sha256, "main routed 3.5-bpw receipt")
    return _successor(plan, state, phase="k35_main_encoded", evidence=evidence)


def seal_k35_packed(
    plan: Mapping[str, Any], state: Mapping[str, Any],
    *,
    packed_checkpoint_receipt_sha256: str,
    native_copy_receipt_sha256: str,
) -> dict[str, Any]:
    verify_state(plan, state)
    if state.get("phase") != "k35_main_encoded":
        die("final 3.5-bpw packing requires all main layers sealed")
    evidence = dict(state.get("evidence", {}))
    common.require_hash(
        evidence.get("main_routed_receipt_sha256"),
        "main routed 3.5-bpw receipt")
    evidence.update(
        {
            "packed_checkpoint_receipt_sha256": common.require_hash(
                packed_checkpoint_receipt_sha256,
                "packed 3.5-bpw checkpoint receipt"),
            "native_copy_receipt_sha256": common.require_hash(
                native_copy_receipt_sha256, "native non-routed copy receipt"),
        }
    )
    return _successor(plan, state, phase="k35_packed", evidence=evidence)


def verify_k35_kld_receipt(
    receipt: Mapping[str, Any],
    *,
    packed_checkpoint_receipt_sha256: str,
) -> str:
    """The 3.5-bpw KLD gate is the ABSOLUTE bar: the mean of five cold-run
    mean tokenwise KL divergences over the sealed final token panel must
    be < 0.06. No baseline comparison exists or is gated."""
    seal = common.verify_seal(
        receipt,
        schema=PACKED_KLD_SCHEMA,
        field="receipt_sha256",
        label="packed 3.5-bpw KLD receipt",
    )
    required = {
        "profile": PROFILE,
        "target_bpw": TARGET_BPW,
        "qualified": True,
        "kld_direction": "teacher_to_student",
        "quality_gate_passed": True,
        "cold_execution_count": 5,
        "checkpoint_receipt_sha256": packed_checkpoint_receipt_sha256,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            die(f"packed 3.5-bpw KLD qualification {key} differs")
    gate = receipt.get("quality_gate")
    if not isinstance(gate, Mapping) or (
        gate.get("metric") != "mean_of_five_run_mean_tokenwise_kld"
        or gate.get("threshold_lt") != KLD_GATE_THRESHOLD
    ):
        die("packed 3.5-bpw KLD gate definition differs")
    measured = receipt.get("five_run_mean_kld")
    if (
        isinstance(measured, bool)
        or not isinstance(measured, (int, float))
        or not measured < KLD_GATE_THRESHOLD
    ):
        die("measured five-run mean KLD does not clear the absolute 0.06 bar")
    common.require_hash(
        receipt.get("token_panel_receipt_sha256"),
        "3.5-bpw KLD token-panel receipt")
    common.require_hash(
        receipt.get("reader_audit_receipt_sha256"),
        "3.5-bpw KLD reader-audit receipt")
    evidence = receipt.get("evidence_artifacts")
    mandatory_roles = {"teacher_logits", "final_student_logits", "tokenwise_kl"}
    if not isinstance(evidence, Mapping) or not mandatory_roles <= set(evidence):
        die("packed 3.5-bpw KLD receipt lacks preserved logits/tokenwise-KL evidence")
    for role in mandatory_roles:
        common.require_hash(evidence[role], f"3.5-bpw KLD evidence {role}")
    return seal


def qualify_k35_kld(
    plan: Mapping[str, Any], state: Mapping[str, Any],
    *, kld_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    verify_state(plan, state)
    if state.get("phase") != "k35_packed":
        die("3.5-bpw KLD may qualify only the packed state")
    packed = state.get("evidence", {}).get("packed_checkpoint_receipt_sha256")
    kld_sha = verify_k35_kld_receipt(
        kld_receipt, packed_checkpoint_receipt_sha256=packed)
    evidence = dict(state.get("evidence", {}))
    evidence["k35_packed_kld_receipt_sha256"] = kld_sha
    return _successor(plan, state, phase="k35_kld_qualified", evidence=evidence)


def verify_publication_receipt(
    receipt: Mapping[str, Any], *, kld_receipt_sha256: str
) -> str:
    seal = common.verify_seal(
        receipt,
        schema=PUBLICATION_RECEIPT_SCHEMA,
        field="receipt_sha256",
        label="3.5-bpw HF publication receipt",
    )
    required = {
        "profile": PROFILE,
        "target_bpw": TARGET_BPW,
        "kld_receipt_sha256": kld_receipt_sha256,
        "reader_audit_qualified": True,
        "native_copy_exact": True,
        "published": True,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            die(f"3.5-bpw publication receipt {key} differs")
    common.require_hash(receipt.get("hf_upload_receipt_sha256"), "HF upload receipt")
    return seal


def authorize_publication(
    plan: Mapping[str, Any], state: Mapping[str, Any],
    *, publication_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    verify_state(plan, state)
    if state.get("phase") != "k35_kld_qualified":
        die("publication is prohibited until packed 3.5-bpw KLD qualifies")
    kld_sha = common.require_hash(
        state.get("evidence", {}).get("k35_packed_kld_receipt_sha256"),
        "packed 3.5-bpw KLD receipt",
    )
    publication_sha = verify_publication_receipt(
        publication_receipt, kld_receipt_sha256=kld_sha)
    evidence = dict(state.get("evidence", {}))
    evidence["publication_receipt_sha256"] = publication_sha
    return _successor(
        plan, state, phase="publication_authorized",
        publication_authorized=True, evidence=evidence)
