#!/usr/bin/env python3
"""dsv4_surface.py - DSV4 Flash mixed-rate packed surface: loads the sealed
per-layer payload stores into the shape the phase-8 pack driver consumes.

Port of the GLM-5.3 campaign's k35_surface.py (k35-dsv4-study port source)
to the DSV4 Flash geometry, bound to dsv4_common/dsv4_uniform_k35/
dsv4_worker instead of k35_common and the glm53 campaign modules.
Per-mirror citations name the port-source lines.

Integrity rules (port mirror of k35_surface.py:11-18, re-derived here):
- Every layer receipt must be sealed, complete, mixed-rate, and bound to
  the plan work unit, the phase-6 readiness allocation, and the state
  chain's completed_layers entry for that layer.
- The layer verification is a CHAIN (stronger than the port source, which
  read choice files by digest only, k35_surface.py:223-232): layer receipt
  seal -> every expert receipt under experts/layer-NN/ -> every choice
  re-verified against its layer payload store (Dsv4PackedPayloadStore
  .verify_choice re-hashes every stored object). Census per layer is
  NUM_EXPERTS * len(PROJECTIONS) exactly; rates are read from the choices
  themselves and must land in PER_TENSOR_ALLOWED_BITS; no global rate is
  invented.
- The choice predecessor chain inside each expert must root at the layer
  claim and cover all three projections exactly once (the encode order is
  w1 -> w3 -> w2, dsv4_worker.py:1014-1065, 1112-1186; this check is
  order-agnostic but proves the chain is unbroken and acyclic).
- No MTP surface exists: mtp.{0,1,2} is native scope v1 (never encoded),
  so the port source's MTP adapter receipt builder (k35_surface.py:252-279)
  has NO counterpart here and no mtp layer receipt is read.

Geometry: layers 0..42 all routed (common.MAIN_LAYERS), 256 experts x 3
projections, stores at layers/L{NN}/payload-store with choices/*.json and
content-addressed objects/{xx}/{sha256}.bin (dsv4_common
.Dsv4PackedPayloadStore).

ASCII only. No em-dashes. CODE ONLY off-pod: importing dsv4_common
resolves Geometry from the live master (env DSV4_MODEL/DSV4_META/
DSV4_LORA); run inside the encode container.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import dsv4_common as common
import dsv4_uniform_k35 as k35
from dsv4_common import die

# The mixed-rate marker every layer receipt and the packed artifact carry
# (port mirror of k35_surface.py:43; dsv4_worker.build_layer_receipt seals
# the same string, dsv4_worker.py:711).
MIXED_BITS_MARKER = "mixed_k34_per_tensor"

# Phases whose state chain has every main layer complete
# (dsv4_uniform_k35.verify_state closed_main_phases, dsv4_uniform_k35.py:
# 959-964). The surface may load from any of them.
CLOSED_MAIN_PHASES = (
    "k35_main_encoded",
    "k35_packed",
    "k35_kld_qualified",
    "publication_authorized",
)

_LAYER_DIR = re.compile(r"^L(\d{2})$")

# MCG marker constants mirrored from the sealed choice decoder block
# (dsv4_common.Dsv4PackedPayloadStore.put_choice, dsv4_common.py:638-641).
MCG_MULTIPLIER_HEX = "0xCBAC1FED"
MCG_MARKER_SIGNED_INT32 = -877912083


def expected_choice_id(layer: int, expert: int, projection: str,
                       bits: int) -> str:
    """The one choice_id format this campaign seals (grep authority:
    dsv4_worker.py:1047 and dsv4_worker.py:1160)."""
    return f"L{layer:02d}.E{expert:03d}.{projection}.K{bits}"


# ---------------------------------------------------------------------------
# Layer receipt seal + census (port mirror of k35_surface.py:60-77 with the
# DSV4 receipt body from dsv4_worker.build_layer_receipt,
# dsv4_worker.py:699-723).
# ---------------------------------------------------------------------------


def _verify_layer_receipt(receipt: Mapping[str, Any], *, layer: int) -> str:
    seal = common.verify_seal(
        receipt,
        schema=common.DSV4_LAYER_RECEIPT_SCHEMA,
        field="receipt_sha256",
        label=f"dsv4 layer L{layer} receipt",
    )
    if (
        receipt.get("layer") != layer
        or receipt.get("complete") is not True
        or receipt.get("bits") != MIXED_BITS_MARKER
    ):
        die(f"L{layer} receipt is not a complete mixed-rate seal")
    for key, expected in (
        ("experts", common.NUM_EXPERTS),
        ("matrix_count", common.TENSORS_PER_LAYER),
        ("bit_units", common.TARGET_BIT_UNITS_PER_LAYER),
        ("k4_tensor_count", common.K4_TENSORS_PER_LAYER),
        ("k3_tensor_count", common.K3_TENSORS_PER_LAYER),
    ):
        if receipt.get(key) != expected:
            die(
                f"L{layer} receipt census field {key} differs: "
                f"{receipt.get(key)!r} != {expected}"
            )
    if receipt.get("rate") != {
        "numerator": k35.RATE_NUMERATOR,
        "denominator": k35.RATE_DENOMINATOR,
    }:
        die(f"L{layer} receipt rate differs from 7/2")
    if len(receipt.get("expert_receipt_sha256", ())) != common.NUM_EXPERTS:
        die(f"L{layer} receipt expert receipt census differs")
    if len(receipt.get("choice_sha256", ())) != common.TENSORS_PER_LAYER:
        die(f"L{layer} receipt choice census differs")
    common.require_hash(
        receipt.get("claim_receipt_sha256"), f"L{layer} claim receipt")
    common.require_hash(
        receipt.get("allocation_sha256"), f"L{layer} allocation receipt")
    return seal


# ---------------------------------------------------------------------------
# Multi-layer store view (port mirror of k35_surface.py:124-146). Object
# loads stay PER LAYER: a choice's object refs are relative to its own
# layer store, and ExactCodecPayloadStore.load_tensor re-hashes every
# content-addressed object it loads, so no cross-layer index is needed
# (delta from the port source's global K35ObjectIndex, k35_surface.py:80-121).
# ---------------------------------------------------------------------------


class Dsv4MultiLayerStore:
    """The store face the pack driver expects, over per-layer stores."""

    def __init__(self, layers_root: str | Path):
        self.layers_root = Path(layers_root)
        if not self.layers_root.is_dir():
            die(f"dsv4 layers root is absent: {self.layers_root}")
        self.by_layer: dict[int, Any] = {}
        for directory in sorted(self.layers_root.iterdir()):
            match = _LAYER_DIR.match(directory.name)
            if match and (directory / "payload-store").is_dir():
                self.by_layer[int(match.group(1))] = (
                    common.Dsv4PackedPayloadStore(
                        directory / "payload-store"))
        if not self.by_layer:
            die("no dsv4 per-layer payload stores found")

    def store_for(self, layer: int):
        store = self.by_layer.get(int(layer))
        if store is None:
            die(f"choice references absent layer store: L{layer}")
        return store

    def verify_choice(self, choice: Mapping[str, Any]) -> dict[str, Any]:
        # Route through the per-layer verifier (seal + honest bits +
        # dtype + object re-hash).
        return self.store_for(int(choice["layer"])).verify_choice(choice)

    def load_object(self, layer: int, ref: Mapping[str, Any]):
        """Load one stored object of a layer's choice (re-hashed on load)."""
        return self.store_for(layer).objects.load_tensor(ref)


# ---------------------------------------------------------------------------
# Layer chain verification: receipt -> expert receipts -> choices.
# ---------------------------------------------------------------------------


def _verify_expert_choice_chain(
    store: Any,
    *,
    layer: int,
    expert: int,
    expert_receipt: Mapping[str, Any],
    claim_sha: str,
    allocation_sha: str,
) -> dict[str, dict[str, Any]]:
    """Verify one expert's three choices against the payload store.

    Returns {projection: verified choice row}. The expert receipt embeds
    the full choice bodies the worker sealed (dsv4_worker.py:619-622);
    each must be byte-equal to the store's own choice file, and the store
    verifier re-hashes every referenced object (dsv4_common.py:654-675).
    """

    from dsv4_worker import verify_expert_receipt  # single authority

    verify_expert_receipt(expert_receipt)
    if (
        expert_receipt.get("layer") != layer
        or expert_receipt.get("expert") != expert
        or expert_receipt.get("claim_receipt_sha256") != claim_sha
        or expert_receipt.get("allocation_sha256") != allocation_sha
    ):
        die(
            f"L{layer} E{expert} receipt binding differs (layer/expert/"
            "claim/allocation)"
        )
    bits_by_projection = expert_receipt.get("bits", {})
    choices: dict[str, dict[str, Any]] = {}
    shas = {
        projection: str(expert_receipt["choices"][projection]
                        .get("choice_sha256"))
        for projection in common.PROJECTIONS
    }
    predecessors = {
        projection: str(expert_receipt["choices"][projection]
                        .get("predecessor_state_hash"))
        for projection in common.PROJECTIONS
    }
    # Predecessor chain: rooted at the claim, covers each projection exactly
    # once, no branch, no cycle (order-agnostic walk).
    owners: dict[str, list[str]] = {}
    for projection, pred in predecessors.items():
        owners.setdefault(pred, []).append(projection)
    current = claim_sha
    visited: list[str] = []
    while current in owners:
        holders = owners[current]
        if len(holders) != 1:
            die(
                f"L{layer} E{expert} choice predecessor chain branches at "
                f"{current[:12]}"
            )
        projection = holders[0]
        if projection in visited:
            die(f"L{layer} E{expert} choice predecessor chain cycles")
        visited.append(projection)
        current = shas[projection]
    if sorted(visited) != sorted(common.PROJECTIONS):
        die(
            f"L{layer} E{expert} choice predecessor chain does not cover the "
            f"triplet from the claim (covered {sorted(visited)})"
        )
    for projection in common.PROJECTIONS:
        embedded = expert_receipt["choices"][projection]
        digest = shas[projection]
        common.require_hash(digest, f"L{layer} E{expert} {projection} choice")
        path = store.choices / f"{digest}.json"
        if not path.is_file():
            die(f"L{layer} E{expert} {projection} choice file absent: {path}")
        disk = common.load_json(path)
        if disk != dict(embedded):
            die(
                f"L{layer} E{expert} {projection} embedded choice differs "
                f"from the store file (receipt vs {path.name})"
            )
        verified = store.verify_choice(disk)
        bits = bits_by_projection.get(projection)
        if (
            verified.get("layer") != layer
            or verified.get("expert") != expert
            or verified.get("projection") != projection
            or verified.get("bits") != bits
            or bits not in common.PER_TENSOR_ALLOWED_BITS
        ):
            die(
                f"L{layer} E{expert} {projection} choice binding differs "
                "(layer/expert/projection/bits)"
            )
        if verified.get("choice_id") != expected_choice_id(
                layer, expert, projection, int(bits)):
            die(f"L{layer} E{expert} {projection} choice_id format differs")
        decoder = verified.get("decoder", {})
        if (
            decoder.get("codec_family") != "exl3-mcg"
            or decoder.get("mcg_multiplier_hex") != MCG_MULTIPLIER_HEX
            or decoder.get("mcg_marker_signed_int32") != MCG_MARKER_SIGNED_INT32
            or decoder.get("reader_abi_sha256") in (None, "")
        ):
            die(f"L{layer} E{expert} {projection} decoder block differs")
        provenance = verified.get("provenance", {})
        if (
            provenance.get("claim_receipt_sha256") != claim_sha
            or provenance.get("allocation_sha256") != allocation_sha
        ):
            die(
                f"L{layer} E{expert} {projection} choice provenance does not "
                "bind the layer claim/allocation"
            )
        choices[projection] = verified
    return choices


def verify_layer_chain(
    layers_root: str | Path,
    layer: int,
    *,
    receipt: Mapping[str, Any] | None = None,
    expected_allocation_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify one layer's full receipt chain and return its pieces.

    Returns {"layer_receipt", "receipt_sha256", "expert_receipts",
    "choices": {(expert, projection): verified choice}}.
    """

    if layer not in common.MAIN_LAYERS:
        die(f"layer {layer} outside the routed surface 0..42")
    root = Path(layers_root) / common.probe_stem(layer)
    if receipt is None:
        receipt = common.load_json(root / "layer-receipt.json")
    seal = _verify_layer_receipt(receipt, layer=layer)
    claim_sha = str(receipt["claim_receipt_sha256"])
    allocation_sha = str(receipt["allocation_sha256"])
    if (
        expected_allocation_sha256 is not None
        and allocation_sha != expected_allocation_sha256
    ):
        die(
            f"L{layer} receipt allocation {allocation_sha[:12]} differs from "
            f"the expected {expected_allocation_sha256[:12]}"
        )
    store = common.Dsv4PackedPayloadStore(root / "payload-store")
    choices: dict[tuple, dict[str, Any]] = {}
    expert_receipts: list[dict[str, Any]] = []
    flat_choice_shas: list[str] = []
    k4_tensors = 0
    bit_units = 0
    for expert in range(common.NUM_EXPERTS):
        path = (
            root / "experts" / common.layer_dir_name(layer)
            / f"expert-{expert:03d}.json"
        )
        if not path.is_file():
            die(f"L{layer} expert receipt absent: {path}")
        expert_receipt = common.load_json(path)
        if (
            expert_receipt.get("receipt_sha256")
            != receipt["expert_receipt_sha256"][expert]
        ):
            die(
                f"L{layer} expert receipt {expert} does not match the layer "
                "receipt chain"
            )
        expert_choices = _verify_expert_choice_chain(
            store,
            layer=layer,
            expert=expert,
            expert_receipt=expert_receipt,
            claim_sha=claim_sha,
            allocation_sha=allocation_sha,
        )
        for projection in common.PROJECTIONS:
            choice = expert_choices[projection]
            key = (layer, int(expert), projection)
            if key in choices:
                die(f"duplicate choice key: {key}")
            choices[key] = choice
            flat_choice_shas.append(str(choice["choice_sha256"]))
            bits = int(expert_receipt["bits"][projection])
            k4_tensors += 1 if bits == 4 else 0
            bit_units += bits
        expert_receipts.append(expert_receipt)
    # The layer receipt's ordered census must be exactly what the chain
    # produced (dsv4_worker.build_layer_receipt append order: expert-major,
    # PROJECTIONS order).
    if list(receipt["choice_sha256"]) != flat_choice_shas:
        die(f"L{layer} receipt choice_sha256 order/census differs from chain")
    if (
        k4_tensors != common.K4_TENSORS_PER_LAYER
        or bit_units != common.TARGET_BIT_UNITS_PER_LAYER
    ):
        die(
            f"L{layer} chain rate census differs: k4 {k4_tensors} "
            f"(need {common.K4_TENSORS_PER_LAYER}), bit units {bit_units} "
            f"(need {common.TARGET_BIT_UNITS_PER_LAYER})"
        )
    return {
        "layer_receipt": dict(receipt),
        "receipt_sha256": seal,
        "expert_receipts": expert_receipts,
        "choices": choices,
    }


# ---------------------------------------------------------------------------
# Surface (port mirror of k35_surface.py:149-191; bits is the STRING marker,
# never an int surface rate).
# ---------------------------------------------------------------------------


class Dsv4Surface:
    """The shape the phase-8 pack driver consumes."""

    def __init__(
        self,
        *,
        root: Path,
        choices: Mapping[tuple, Mapping[str, Any]],
        layer_receipt_sha256: tuple,
        layer_receipts: tuple,
        store: Dsv4MultiLayerStore,
        launch_plan_sha256: str = "",
        readiness_receipt_sha256: str = "",
        state_receipt_sha256: str = "",
        packed_reader_abi_sha256: str = "",
    ):
        self.root = root
        self.choices = dict(choices)
        self.layer_receipt_sha256 = tuple(layer_receipt_sha256)
        self.layer_receipts = tuple(layer_receipts)
        self.bits = MIXED_BITS_MARKER
        self.store = store
        self.launch_plan_sha256 = launch_plan_sha256
        self.readiness_receipt_sha256 = readiness_receipt_sha256
        self.state_receipt_sha256 = state_receipt_sha256
        self.packed_reader_abi_sha256 = packed_reader_abi_sha256

    def choice(self, layer: int, expert: int, projection: str):
        key = (int(layer), int(expert), str(projection))
        if key not in self.choices:
            die(f"dsv4 surface lacks choice L{layer} E{expert} {projection}")
        return self.choices[key]

    def rate_census(self) -> dict[str, int]:
        """Census from the choice bits themselves (port mirror of
        k35_surface.py:184-191); the only accepted set is
        PER_TENSOR_ALLOWED_BITS."""
        counts = {3: 0, 4: 0}
        for choice in self.choices.values():
            bits = int(choice["bits"])
            if bits not in common.PER_TENSOR_ALLOWED_BITS:
                die(f"choice rate outside allowed set: {bits}")
            counts[bits] += 1
        return {
            "k3_choice_count": counts[3],
            "k4_choice_count": counts[4],
        }


# ---------------------------------------------------------------------------
# Loader with plan/state/readiness binding (port mirror of
# k35_surface.py:194-249; the GLM readiness-allocation binding at
# k35_surface.py:200-204, 217-218 becomes a three-way binding: plan work
# unit + readiness allocation + state completed_layers entry).
# ---------------------------------------------------------------------------


def load_dsv4_surface(
    work_root: str | Path,
    *,
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> Dsv4Surface:
    work_root = Path(work_root).resolve()
    plan_sha = k35.verify_launch_plan(plan)
    state_sha = k35.verify_state(plan, state)
    phase = state.get("phase")
    if phase not in CLOSED_MAIN_PHASES:
        die(
            f"state phase {phase!r} has not closed the main encode; the "
            "surface loads only from "
            f"{'/'.join(CLOSED_MAIN_PHASES)}"
        )
    readiness_sha = common.verify_seal(
        readiness,
        schema=common.DSV4_READINESS_SCHEMA,
        field="readiness_receipt_sha256",
        label="dsv4 phase-6 readiness receipt",
    )
    if readiness.get("launch_plan_sha256") != plan_sha:
        die("readiness receipt binds a different launch plan")
    if readiness.get("layers") != list(common.MAIN_LAYERS):
        die("readiness receipt layer surface differs from 0..42")
    allocation_rows = readiness.get("allocations")
    if not isinstance(allocation_rows, list):
        die("readiness receipt allocation list is absent")
    readiness_alloc: dict[int, str] = {}
    for row in allocation_rows:
        if (
            not isinstance(row, Mapping)
            or row.get("provisional") is not False
        ):
            die("readiness allocation row is absent or still provisional")
        readiness_alloc[int(row["layer"])] = str(row["allocation_sha256"])
    if sorted(readiness_alloc) != list(common.MAIN_LAYERS):
        die("readiness allocation surface differs from 0..42")

    unit_by_layer = {
        int(unit["layer"]): unit for unit in plan.get("work_units", ())}
    if sorted(unit_by_layer) != list(common.MAIN_LAYERS):
        die("launch plan work unit surface differs from 0..42")
    completed = state.get("completed_layers", {})
    if {int(layer) for layer in completed} != set(common.MAIN_LAYERS):
        die("state completed layers do not close the routed surface")

    layers_root = work_root / "layers"
    store = Dsv4MultiLayerStore(layers_root)
    if set(store.by_layer) != set(common.MAIN_LAYERS):
        die(
            "per-layer payload store surface differs from 0..42: "
            f"{sorted(store.by_layer)}"
        )
    choices: dict[tuple, Mapping[str, Any]] = {}
    layer_seals: list[str] = []
    layer_receipts: list[dict[str, Any]] = []
    for layer in common.MAIN_LAYERS:
        unit = unit_by_layer[layer]
        chain = verify_layer_chain(
            layers_root,
            layer,
            expected_allocation_sha256=str(unit["allocation_sha256"]),
        )
        receipt = chain["layer_receipt"]
        if str(receipt["allocation_sha256"]) != readiness_alloc[layer]:
            die(
                f"L{layer} receipt is not bound to the readiness allocation"
            )
        completion = completed.get(str(layer))
        if not isinstance(completion, Mapping):
            die(f"L{layer} has no completed_layers entry in the state chain")
        if (
            completion.get("layer_receipt_sha256")
            != chain["receipt_sha256"]
            or completion.get("claim_receipt_sha256")
            != receipt["claim_receipt_sha256"]
            or completion.get("worker_id") != receipt.get("worker_id")
        ):
            die(
                f"L{layer} state completed_layers entry does not bind the "
                "on-disk layer receipt (seal/claim/worker)"
            )
        layer_seals.append(chain["receipt_sha256"])
        layer_receipts.append(receipt)
        choices.update(chain["choices"])
    expected = len(common.MAIN_LAYERS) * common.TENSORS_PER_LAYER
    if len(choices) != expected:
        die(f"dsv4 choice census differs: {len(choices)} != {expected}")
    census = Dsv4Surface(
        root=layers_root,
        choices=choices,
        layer_receipt_sha256=tuple(layer_seals),
        layer_receipts=tuple(layer_receipts),
        store=store,
        launch_plan_sha256=plan_sha,
        readiness_receipt_sha256=readiness_sha,
        state_receipt_sha256=state_sha,
    ).rate_census()
    if census["k4_choice_count"] != len(common.MAIN_LAYERS) * (
            common.K4_TENSORS_PER_LAYER):
        die(
            "global K4 census differs: "
            f"{census['k4_choice_count']} != "
            f"{len(common.MAIN_LAYERS) * common.K4_TENSORS_PER_LAYER}"
        )
    if census["k3_choice_count"] != len(common.MAIN_LAYERS) * (
            common.K3_TENSORS_PER_LAYER):
        die(
            "global K3 census differs: "
            f"{census['k3_choice_count']} != "
            f"{len(common.MAIN_LAYERS) * common.K3_TENSORS_PER_LAYER}"
        )
    abis = {
        str(choice["decoder"]["reader_abi_sha256"])
        for choice in choices.values()
    }
    if len(abis) != 1:
        die("dsv4 choices do not share one sealed MCG reader ABI")
    return Dsv4Surface(
        root=layers_root,
        choices=choices,
        layer_receipt_sha256=tuple(layer_seals),
        layer_receipts=tuple(layer_receipts),
        store=store,
        launch_plan_sha256=plan_sha,
        readiness_receipt_sha256=readiness_sha,
        state_receipt_sha256=state_sha,
        packed_reader_abi_sha256=abis.pop(),
    )
