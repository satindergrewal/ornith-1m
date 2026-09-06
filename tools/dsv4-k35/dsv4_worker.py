#!/usr/bin/env python3
"""Phase 7 dsv4 encode worker: one process per GPU, dynamic whole-layer
claims through the dsv4 mixed K3/K4 state machine.

Port of the k35 study's phase-7 worker (k35_worker.py) to the DSV4 Flash
geometry.  Scope: MAIN_LAYERS 0..42 only, all routed, 256 experts x 3
projections = 768 tensors per layer; mtp.{0,1,2} is NATIVE scope v1 and is
never encoded here (no MTP branch exists in this file).

Loop per invocation (runbook phase 7, port mirror of k35_worker.py:7-15):

  while layers remain:
    load newest state/state-NNNN.json (under the state flock)
    (successor, claim) = k35.claim_next_layer(plan, state, worker_id=...)
    WRITE THE SUCCESSOR STATE FILE BEFORE ENCODING
    encode all 768 tensors at their allocated integer rates
    write layers/L{NN}/ artifacts + layer receipt
    state = k35.complete_layer(...)
    write the next state file
    repeat

Drain pause: before each claim the worker checks <work-root>/PAUSE; if the
file exists it takes no new claim and exits 0 (a layer already in flight
always completes first).  This is the preferred pause; the alternative is
`docker pause` freezing the processes mid-layer.

Claim recovery: if a worker process dies mid-layer, its active claim stays
in the newest state.  Restarting the SAME worker id resumes that layer
(per-expert receipts are skipped when they already verify, with every
pre-existing choice re-hashed against the payload store and every Hessian
artifact re-hashed on disk).  If the worker is dead for good, run this
driver once with --recover-worker <id>: it drops the claim, QUARANTINES the
layer's partial per-claim artifacts (expert receipts, payload-store, stale
layer receipt; Hessians are kept), and re-queues the layer at the front of
pending in a recovery successor.  A fresh claim mints a new
claim_receipt_sha256, and every expert receipt and choice embeds the dead
claim's hash, so deletion under the state lock is the only seal-safe
recovery; the removal is recorded in the successor's evidence block.

The worker re-loads plan.json every loop iteration and exits cleanly when
the launch plan changes on disk (a re-plan), so a live old-plan worker
stops appending old-plan successors instead of deepening the wedge; state
selection itself only ever considers states bound to the current plan
(newest_state below).

WARN: the plan/state/claim/allocation transitions and their schema strings
live in the sibling module dsv4_uniform_k35 (the campaign state-machine
authority for this venue).  This worker imports that module and defines NO
schema of its own beyond the two encode-path strings noted below; run this
file only after that sibling exists in this directory.

Usage (inside the encode container, cwd <work>):

  CUDA_VISIBLE_DEVICES=0 python3 dsv4_worker.py --worker sm120-0 --work-root /workspace
  CUDA_VISIBLE_DEVICES=1 python3 dsv4_worker.py --worker sm120-1 --work-root /workspace

Worker ids come from the plan's scheduler (the preflight enumeration
assigns f"sm120-{slot}" for slot 0..N-1 over the declared GPU list; this
worker accepts any plan worker id and asserts no venue census).

Weights source: dsv4_capped_source.CappedSource, the sealed dequant-on-
demand reader over the packed master (per-expert load_expert has no LoRA
sites; numerically identical to the BF16 fold by construction, proven by
dsv4_source_parity.py).

Encode notes (ported mismatches, details in the port runbook):
  - the sealed prepared backend accepts only uniform bits in (4, 6), so
    the worker drives the same numeric path per expert through
    Exl3MCGCodec.encode_candidates at the allocated rate, with
    rate-specific preparation vectors (gss/k3, gss/k4);
  - the sealed PackedMCGPayloadStore hardcodes bits=4 per choice, so layer
    artifacts use dsv4_common.Dsv4PackedPayloadStore (same layout, honest
    per-choice bits, trellis-bytes-vs-bits check included);
  - gate/up encode is per tensor (the backend's grouped lockstep batching
    is a uniform-rate surface);
  - the down Hessian is conditioned on gate/up decoded at the reference
    rate dsv4_common.FLOOR_BITS (R7 pair_at semantics), the SAME context
    the probe measured and the DP allocation ranked under; mixed-rate
    triplets add reference-rate conditioning encodes (deployed choices
    keep the allocated rates), and every receipt stamps the rates used.

Name vocabulary: MASTER names (layers.{L}.ffn.experts.{E}.{w1,w2,w3}
.weight) everywhere in campaign code, stores, receipts, and allocations
(single naming authority: dsv4_common.tensor_full_name).  The codec
boundary speaks gate/up/down: w1 -> gate_proj, w3 -> up_proj, w2 ->
down_proj (engine swiglu: gate = x @ w1.T silu'd, up = x @ w3.T,
down = h @ w2.T).  The mapping is applied ONLY at codec call sites and at
the GSS preparation-vector boundary; every such site carries a comment.

ASCII only.  CODE ONLY: nothing here is executed by the author.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dsv4_common as common
import dsv4_uniform_k35 as k35  # sibling state-machine authority (see WARN)
from dsv4_common import die

# Master projection -> codec role.  THE mapping site for the whole worker;
# the engine swiglu defines it (gate = x @ w1.T silu'd, up = x @ w3.T,
# down = h @ w2.T), confirmed by CappedSource.load_expert which returns
# (w1, w2, w3) in engine order (gate, down, up).
PROJECTION_ROLE = {
    "w1": "gate_proj",
    "w3": "up_proj",
    "w2": "down_proj",
}

# Vector sharing topology per MASTER projection (the GSS process structure;
# gate and up share a layer-wide suh basis over the hidden input, down
# shares a layer-wide svh basis over the intermediate input).  Port mirror
# of the port source's VECTOR_TOPOLOGY, re-keyed to master names.
VECTOR_TOPOLOGY = {
    "w1": {"suh": "layer_shared", "svh": "expert_private"},
    "w3": {"suh": "layer_shared", "svh": "expert_private"},
    "w2": {"suh": "expert_private", "svh": "layer_shared"},
}

# NEW SURFACE encode-path schemas minted here (no dsv4_common counterpart;
# the phase-6 GSS driver and the resume path must agree):
#   rate-4 GSS preparation manifest (pattern sibling of
#   dsv4_common.DSV4_RATE3_GSS_SCHEMA) and the routed Hessian pair file.
RATE4_GSS_SCHEMA = "quant-pipeline.dsv4-k35-rate4-gss-preparation.v1"
HESSIAN_PAIR_SCHEMA = "quant-pipeline.dsv4-routed-p2-hessian-pair.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="phase 7 dsv4 mixed K3/K4 encode worker"
    )
    parser.add_argument("--worker", default=None, help="plan worker id, e.g. sm120-0")
    parser.add_argument(
        "--work-root", default=str(common.DEFAULT_WORK_ROOT), type=Path
    )
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
        "--calibration-root",
        default=str(common.DEFAULT_CALIBRATION_ROOT),
        type=Path,
    )
    parser.add_argument("--model-dir", default=common.MODEL_DIR, help="packed master root")
    parser.add_argument("--meta-path", default=common.META_PATH, help="tensor_meta.json path")
    parser.add_argument("--lora-path", default=common.LORA_PATH, help="cap LoRA safetensors")
    parser.add_argument("--chunk-rows", default=common.CHUNK_ROWS, type=int)
    parser.add_argument(
        "--no-verify-capture-hashes",
        action="store_true",
        help="skip per-layer capture payload hashing (manifest seal still verified)",
    )
    parser.add_argument(
        "--k4-preparation-root", default=None, type=Path, help="default <work>/gss/k4"
    )
    parser.add_argument(
        "--k3-preparation-root", default=None, type=Path, help="default <work>/gss/k3"
    )
    parser.add_argument(
        "--reader-abi-sha256",
        default=None,
        help="sealed reader ABI hash recorded in every packed choice",
    )
    parser.add_argument(
        "--recover-worker",
        default=None,
        metavar="ID",
        help="maintenance: drop a stale active claim and re-queue its layer",
    )
    args = parser.parse_args()
    args.work_root = Path(args.work_root).resolve()
    args.calibration_root = Path(args.calibration_root).resolve()
    if args.device is None:
        args.device = "cuda:0"
    if args.k4_preparation_root is None:
        args.k4_preparation_root = args.work_root / "gss" / "k4"
    if args.k3_preparation_root is None:
        args.k3_preparation_root = args.work_root / "gss" / "k3"
    args.k4_preparation_root = Path(args.k4_preparation_root).resolve()
    args.k3_preparation_root = Path(args.k3_preparation_root).resolve()
    if args.recover_worker is None and args.worker is None:
        die("pass --worker ID (or --recover-worker ID for maintenance)")
    return args


# ---------------------------------------------------------------------------
# State-chain helpers (port mirror of the port source's common layer:
# newest_state, StateLock, plan/allocation loaders.  The transitions
# themselves are the sibling state machine's, imported as k35 above.)
# ---------------------------------------------------------------------------


def state_dir(work_root: Path) -> Path:
    return Path(work_root) / "state"


def state_path(work_root: Path, sequence: int) -> Path:
    return state_dir(work_root) / f"state-{int(sequence):04d}.json"


def newest_state(
    work_root: Path, plan: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Newest state receipt bound to THIS launch plan.

    State files whose launch_plan_sha256 differs from the passed plan are
    ignored: the state machine rejects them, so a superseded-plan file at
    the max sequence would wedge every worker and every --recover-worker
    start.  Superseded states belong in state/history/; files left in
    place are filtered out here instead of poisoning the selection.
    """

    directory = state_dir(work_root)
    plan_sha = plan["launch_plan_sha256"]
    paths = sorted(directory.glob("state-*.json"))
    if not paths:
        die(f"no state receipts under {directory}; run the phase-4 plan bootstrap first")
    parsed: list[tuple[int, Path]] = []
    states: dict[int, dict[str, Any]] = {}
    for path in paths:
        match = re.fullmatch(r"state-(\d+)\.json", path.name)
        if match is None:
            die(f"foreign file in the state directory: {path}")
        state = common.load_json(path)
        if not isinstance(state, Mapping):
            die(f"state file is not an object: {path}")
        if state.get("launch_plan_sha256") != plan_sha:
            continue
        sequence = int(match.group(1))
        if state.get("sequence") != sequence:
            die(f"state file sequence stamp differs: {path}")
        parsed.append((sequence, path))
        states[sequence] = dict(state)
    if not parsed:
        die(
            f"no state bound to the current plan {plan_sha[:16]} under "
            f"{directory}; run the re-plan bootstrap (rebuild plan.json, "
            "rewrite state-0000.json, and move superseded state files into "
            "state/history/)"
        )
    parsed.sort()
    top = [item for item in parsed if item[0] == parsed[-1][0]]
    if len(top) != 1:
        die(f"multiple newest state receipts: {[str(item[1]) for item in top]}")
    sequence, path = top[0]
    return path, states[sequence]


class StateLock:
    """Cross-process lock serializing claim/complete transitions.

    The state machine is the lock (runbook phase 7): claim_next_layer
    refuses a worker that already owns a layer, and every successor is
    sealed.  With file-based state, the read-claim-write critical section
    must also be serialized between the worker processes, hence this
    flock.
    """

    def __init__(self, work_root: Path) -> None:
        directory = state_dir(work_root)
        directory.mkdir(parents=True, exist_ok=True)
        self.handle = (directory / ".lock").open("a")

    def __enter__(self) -> "StateLock":
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *_exc: Any) -> None:
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def verify_plan_worker(plan: Mapping[str, Any], worker_id: str) -> dict[str, Any]:
    rows = plan.get("scheduler", {}).get("workers", [])
    for row in rows:
        if row.get("worker_id") == worker_id:
            return dict(row)
    die(
        f"worker id {worker_id!r} is not in the plan; accepted ids for this "
        f"plan: {[row.get('worker_id') for row in rows]}.  The declared "
        'preflight assigns worker ids f"sm120-{{slot}}" from its gpus '
        "enumeration (slot position, not the physical index)."
    )
    raise AssertionError("unreachable")


def load_plan(work_root: Path) -> dict[str, Any]:
    plan = common.load_json(work_root / "plan.json")
    k35.verify_launch_plan(plan)
    return plan


def load_layer_allocation(work_root: Path, layer: int) -> dict[str, Any]:
    path = work_root / "allocations" / f"{common.probe_stem(layer)}.json"
    if not path.is_file():
        die(f"sealed layer allocation is absent: {path} (run the phase-5 probe driver)")
    receipt = common.load_json(path)
    k35.verify_layer_allocation(receipt, layer=layer)
    if receipt.get("provisional") is not False:
        die(f"layer {layer} allocation is still provisional: {path}")
    return receipt


# ---------------------------------------------------------------------------
# Recovery transition (the ONE transition the public state machine does not
# offer; sealed through the module-private k35._successor under an explicit
# flag and recorded in the successor's evidence block; never run it while
# the target worker might still be alive).
# ---------------------------------------------------------------------------


def quarantine_partial_layer(work_root: Path, layer: int) -> dict[str, Any]:
    """Remove the partial per-claim artifacts of a recovered (crashed) layer.

    Every expert receipt, every packed choice, and the choice predecessor
    chain embed the dead claim's claim_receipt_sha256; the re-claim
    necessarily mints a new claim hash (the claim seals its
    parent_state_receipt_sha256, which the recovery successor moved), so the
    resume gate and build_layer_receipt would reject every surviving receipt
    forever and the layer could never complete.  Deletion is the only
    seal-safe recovery.  The payload store is per-layer, so removing the
    whole store removes exactly this layer's entries; Hessians are
    claim-independent and save_hessians tolerates byte-identical existing
    artifacts, so they are kept.
    """

    import shutil

    stem = common.probe_stem(layer)
    layer_root = work_root / "layers" / stem
    removed: list[str] = []
    expert_dir = layer_root / "experts" / common.layer_dir_name(layer)
    receipts_removed = 0
    if expert_dir.exists():
        receipts_removed = len(list(expert_dir.glob("expert-*.json")))
        shutil.rmtree(expert_dir)
        removed.append(f"layers/{stem}/experts/{common.layer_dir_name(layer)}")
    store_dir = layer_root / "payload-store"
    if store_dir.exists():
        shutil.rmtree(store_dir)
        removed.append(f"layers/{stem}/payload-store")
    layer_receipt_path = layer_root / "layer-receipt.json"
    if layer_receipt_path.exists():
        layer_receipt_path.unlink()
        removed.append(f"layers/{stem}/layer-receipt.json")
    return {
        "layer": int(layer),
        "removed": removed,
        "expert_receipts_removed": receipts_removed,
        "kept": ["hessians (claim-independent; byte-identical artifacts tolerated)"],
        "reason": (
            "receipts, choices, and the choice predecessor chain embed the dead "
            "claim's claim_receipt_sha256; the re-claim mints a new hash, so "
            "deletion is the only seal-safe recovery"
        ),
    }


def recover_worker(args: argparse.Namespace) -> None:
    plan = load_plan(args.work_root)
    with StateLock(args.work_root):
        _path, state = newest_state(args.work_root, plan)
        k35.verify_state(plan, state)
        active = dict(state["active_claims"])
        target = active.get(args.recover_worker)
        if target is None:
            die(
                f"worker {args.recover_worker} holds no active claim in the newest "
                "state; nothing to recover"
            )
        layer = int(target["layer"])
        pending = [layer] + [int(x) for x in state["pending_layers"]]
        del active[args.recover_worker]
        quarantine = quarantine_partial_layer(args.work_root, layer)
        evidence = dict(state.get("evidence", {}))
        evidence["k35_recovery_note"] = (
            f"stale claim of worker {args.recover_worker} on layer "
            f"{layer} dropped and re-queued by dsv4_worker --recover-worker; "
            "the layer's partial per-claim artifacts (expert receipts, "
            "payload-store, stale layer receipt) were quarantined under the "
            "state lock because they all embed the dead claim hash; "
            "transition via k35._successor because the public state machine "
            "offers no recovery transition"
        )
        evidence["k35_recovery_quarantine"] = quarantine
        successor = k35._successor(
            plan,
            state,
            pending_layers=pending,
            active_claims=active,
            evidence=evidence,
        )
        common.write_json(
            state_path(args.work_root, successor["sequence"]), successor
        )
    print(
        json.dumps(
            {
                "recovered_worker": args.recover_worker,
                "requeued_layer": layer,
                "quarantined": quarantine["removed"],
                "state": successor["state_receipt_sha256"],
            }
        )
    )


# ---------------------------------------------------------------------------
# Preparation shards (vectors + permutations)
# ---------------------------------------------------------------------------

# Shard keys are the CODEC role vocabulary (gate/up/down), not master
# projections; load_preparation is the GSS boundary and keeps the sealed
# layout, the caller applies PROJECTION_ROLE at the call site.
# Preparation shards carry MASTER projection names (the phase-6 producer
# keys tensors w1/w3/w2; master vocabulary is the campaign rule). The
# codec boundary maps via PROJECTION_ROLE at the call site.
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


def load_preparation(root: Path, layer: int, *, expected_bits: int):
    """Load and verify one dsv4 rate-specific GSS preparation shard.

    Both rate halves are campaign-built NEW SURFACE manifests under the
    rate-pattern schema (rate 3 from dsv4_common.DSV4_RATE3_GSS_SCHEMA,
    rate 4 from RATE4_GSS_SCHEMA above): a sealed preparation.json
    (preparation_sha256 over the canonical body) binding layer, complete,
    bits, shard, and shard_sha256, plus the safetensors shard with the
    codec-role tensor census below.  Returns (manifest, tensors) with
    tensors on CPU.
    """

    from safetensors import safe_open

    schema = (
        common.DSV4_RATE3_GSS_SCHEMA
        if expected_bits == 3
        else RATE4_GSS_SCHEMA
        if expected_bits == 4
        else None
    )
    if schema is None:
        die(f"no dsv4 GSS preparation schema for bits {expected_bits}")
    directory = Path(root) / common.layer_dir_name(layer)
    manifest_path = directory / "preparation.json"
    if not manifest_path.is_file():
        die(f"preparation manifest is absent: {manifest_path}")
    manifest = common.load_json(manifest_path)
    common.verify_seal(
        manifest,
        schema=schema,
        field="preparation_sha256",
        label=f"rate-{expected_bits} GSS layer {layer}",
    )
    if manifest.get("layer") != layer or manifest.get("complete") is not True:
        die(f"GSS manifest layer binding differs: {manifest_path}")
    bits = int(manifest["bits"])
    if bits != expected_bits:
        die(
            f"preparation bits {bits} differs from expected {expected_bits}: "
            f"{manifest_path}"
        )
    shard = directory / str(manifest["shard"])
    if not shard.is_file() or shard.is_symlink():
        die(f"preparation shard is absent or a symlink: {shard}")
    if common.sha256_file(shard) != manifest.get("shard_sha256"):
        die(f"preparation shard hash differs: {shard}")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != _PREPARATION_REQUIRED_KEYS:
            die(f"preparation tensor census differs: {sorted(keys)}")
        tensors = {
            name: handle.get_tensor(name).contiguous() for name in sorted(keys)
        }
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


def preparation_vectors(
    tensors: Mapping[str, Any], projection: str, expert: int, device: str
):
    """Move one expert's suh/svh rows to the encode device.

    ``projection`` is the MASTER projection (w1/w3/w2); the preparation
    shard carries master-vocabulary keys, matching the phase-6 producer.
    """

    if projection not in common.PROJECTIONS:
        die(f"preparation_vectors: unknown projection {projection}")
    return (
        tensors[f"{projection}_suh"][expert].to(device),
        tensors[f"{projection}_svh"][expert].to(device),
    )


# ---------------------------------------------------------------------------
# One-rate encode helper (codec boundary; role vocabulary)
# ---------------------------------------------------------------------------


def encode_one_rate(
    codec,
    *,
    layer: int,
    expert: int,
    role: str,
    weight_hf: Any,
    covariance: Any,
    bits: int,
    suh: Any,
    svh: Any,
    provenance: dict[str, Any] | None = None,
):
    """encode_candidates narrowed to one integer rate (worker path)."""
    if int(bits) not in common.PER_TENSOR_ALLOWED_BITS:
        die(f"encode rate {bits} outside {common.PER_TENSOR_ALLOWED_BITS}")
    return codec.encode_candidates(
        unit_id=f"L{layer}.E{expert}.{role}",
        weight_hf=weight_hf,
        covariance=covariance,
        bits=(int(bits),),
        input_vector=suh,
        output_vector=svh,
        provenance=provenance,
    )[int(bits)]


# ---------------------------------------------------------------------------
# NEW SURFACE receipts (dsv4 schemas from dsv4_common; no sealed validator
# exists; the port runbook registers both schemas)
# ---------------------------------------------------------------------------


def build_expert_receipt(
    *,
    layer: int,
    expert: int,
    bits_by_projection: Mapping[str, int],
    choices: Mapping[str, Mapping[str, Any]],
    claim_receipt_sha256: str,
    allocation_sha256: str,
    capture_binding: Mapping[str, Any],
    hessian_artifact: Mapping[str, Any],
    down_conditioning: Mapping[str, Any],
    codec_identity_sha256: str,
) -> dict[str, Any]:
    if sorted(bits_by_projection) != sorted(common.PROJECTIONS):
        die("expert receipt does not close its triplet")
    body = {
        "schema": common.DSV4_EXPERT_RECEIPT_SCHEMA,
        "claim_receipt_sha256": common.require_hash(
            claim_receipt_sha256, "claim receipt"
        ),
        "allocation_sha256": common.require_hash(
            allocation_sha256, "allocation receipt"
        ),
        "layer": int(layer),
        "expert": int(expert),
        "projections": list(common.PROJECTIONS),
        "bits": dict(sorted(bits_by_projection.items())),
        "bit_units": int(sum(bits_by_projection.values())),
        "rate": {
            "numerator": k35.RATE_NUMERATOR,
            "denominator": k35.RATE_DENOMINATOR,
        },
        "capture_binding": copy.deepcopy(dict(capture_binding)),
        "codec_identity_sha256": common.require_hash(
            codec_identity_sha256, "codec identity"
        ),
        "sigma_reg": common.SIGMA_REG,
        "down_conditioning": copy.deepcopy(dict(down_conditioning)),
        "hessian_artifact": copy.deepcopy(dict(hessian_artifact)),
        "choices": copy.deepcopy(
            {p: dict(choices[p]) for p in common.PROJECTIONS}
        ),
    }
    return common.seal(body, "receipt_sha256")


def verify_expert_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    common.verify_seal(
        receipt,
        schema=common.DSV4_EXPERT_RECEIPT_SCHEMA,
        field="receipt_sha256",
        label=(
            f"dsv4 expert receipt L{receipt.get('layer')}E{receipt.get('expert')}"
        ),
    )
    if sorted(receipt.get("choices", {})) != sorted(common.PROJECTIONS):
        die("dsv4 expert receipt choice census differs")
    bits = receipt.get("bits", {})
    if sorted(bits) != sorted(common.PROJECTIONS) or any(
        bits.get(p) not in common.PER_TENSOR_ALLOWED_BITS
        for p in common.PROJECTIONS
    ):
        die("dsv4 expert receipt bits binding differs")
    conditioning = receipt.get("down_conditioning")
    if not isinstance(conditioning, Mapping):
        die("dsv4 expert receipt lacks its down-conditioning block")
    evidence = conditioning.get("evidence")
    if not isinstance(evidence, Mapping):
        die("dsv4 expert receipt down-conditioning evidence is absent")
    if (
        evidence.get("conditioning_gate_bits") != conditioning.get("gate_rate")
        or evidence.get("conditioning_up_bits") != conditioning.get("up_rate")
    ):
        die(
            "dsv4 expert receipt down-conditioning stamp disagrees with its "
            "recorded gate/up conditioning rates"
        )
    return dict(receipt)


def build_layer_receipt(
    *,
    layer: int,
    worker_id: str,
    claim_receipt_sha256: str,
    allocation_sha256: str,
    expert_receipts,
) -> dict[str, Any]:
    if len(expert_receipts) != common.NUM_EXPERTS:
        die(f"layer receipt census differs: {len(expert_receipts)} experts")
    expert_shas: list[str] = []
    choice_shas: list[str] = []
    k4_tensors = 0
    bit_units = 0
    for receipt in expert_receipts:
        verify_expert_receipt(receipt)
        if receipt["layer"] != layer:
            die("layer receipt expert targets a foreign layer")
        if receipt["claim_receipt_sha256"] != claim_receipt_sha256:
            die("layer receipt expert binds a foreign claim")
        if receipt["allocation_sha256"] != allocation_sha256:
            die("layer receipt expert binds a foreign allocation")
        expert_shas.append(receipt["receipt_sha256"])
        for projection in common.PROJECTIONS:
            choice = receipt["choices"][projection]
            common.require_hash(choice.get("choice_sha256"), "choice seal")
            bits = receipt["bits"][projection]
            k4_tensors += 1 if bits == 4 else 0
            bit_units += int(bits)
            choice_shas.append(choice["choice_sha256"])
    if (
        k4_tensors != common.K4_TENSORS_PER_LAYER
        or bit_units != common.TARGET_BIT_UNITS_PER_LAYER
    ):
        die(
            f"layer receipt rate census differs: k4 {k4_tensors} "
            f"(need {common.K4_TENSORS_PER_LAYER}), bit units {bit_units} "
            f"(need {common.TARGET_BIT_UNITS_PER_LAYER})"
        )
    body = {
        "schema": common.DSV4_LAYER_RECEIPT_SCHEMA,
        "layer": int(layer),
        "worker_id": str(worker_id),
        "claim_receipt_sha256": common.require_hash(
            claim_receipt_sha256, "claim receipt"
        ),
        "allocation_sha256": common.require_hash(
            allocation_sha256, "allocation receipt"
        ),
        "experts": common.NUM_EXPERTS,
        "matrix_count": common.TENSORS_PER_LAYER,
        "bits": "mixed_k34_per_tensor",
        "bit_units": common.TARGET_BIT_UNITS_PER_LAYER,
        "k4_tensor_count": common.K4_TENSORS_PER_LAYER,
        "k3_tensor_count": common.K3_TENSORS_PER_LAYER,
        "rate": {
            "numerator": k35.RATE_NUMERATOR,
            "denominator": k35.RATE_DENOMINATOR,
        },
        "expert_receipt_sha256": expert_shas,
        "choice_sha256": choice_shas,
        "complete": True,
    }
    return common.seal(body, "receipt_sha256")


# ---------------------------------------------------------------------------
# Capture binding (the dsv4 capture view has no binding() accessor; the
# expert receipt embeds this block derived from the sealed manifest)
# ---------------------------------------------------------------------------


def capture_binding(capture: Any) -> dict[str, Any]:
    manifest = common.load_json(Path(capture.root) / "capture-manifest.json")
    digest = common.verify_seal(
        manifest,
        schema=common.DSV4_CAPTURE_SCHEMA,
        field="capture_sha256",
        label="dsv4 capture manifest",
    )
    rows = manifest.get("rows_per_layer")
    roles = manifest.get("roles")
    if not isinstance(rows, int) or rows <= 0:
        die("dsv4 capture manifest rows census is malformed")
    if not isinstance(roles, list) or not roles:
        die("dsv4 capture manifest role census is malformed")
    return {
        "schema": common.DSV4_CAPTURE_SCHEMA,
        "capture_sha256": digest,
        "layer": int(capture.layer),
        "rows_per_layer": int(rows),
        "roles": sorted(str(role) for role in roles),
        "calibration_root": str(capture.root),
    }


# ---------------------------------------------------------------------------
# Hessian artifacts
# ---------------------------------------------------------------------------


def save_hessians(layer_root: Path, layer: int, expert: int, gate_up, down,
                  evidence) -> dict:
    """Routed p2 Hessian pair on disk: FP16 matrices plus exact FP32
    recomputation hashes in the metadata block (port mirror of the port
    source's save_hessians; shapes re-derived from dsv4_common)."""

    import os

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    directory = layer_root / "hessians" / common.layer_dir_name(layer)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"expert-{expert:03d}.safetensors"
    metadata = {
        "schema": HESSIAN_PAIR_SCHEMA,
        "layer": str(layer),
        "expert": str(expert),
        "stored_dtype": "float16",
        "gate_up_exact_fp32_sha256": str(evidence["gate_up"]["matrix_sha256"]),
        "down_exact_fp32_sha256": str(evidence["down"]["matrix_sha256"]),
        "exact_recomputation": "sealed_capture_routes_plus_decoded_gate_up",
    }
    if path.exists():
        if path.is_symlink():
            die(f"Hessian artifact is a symlink: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            if (
                set(handle.keys()) != {"gate_up_hessian", "down_hessian"}
                or (handle.metadata() or {}) != metadata
                or tuple(handle.get_slice("gate_up_hessian").get_shape())
                != (common.HIDDEN_SIZE, common.HIDDEN_SIZE)
                or tuple(handle.get_slice("down_hessian").get_shape())
                != (common.INTERMEDIATE_SIZE, common.INTERMEDIATE_SIZE)
            ):
                die(f"existing Hessian artifact differs: {path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        save_file(
            {
                "gate_up_hessian": torch.as_tensor(gate_up)
                .detach()
                .to(device="cpu", dtype=torch.float16)
                .contiguous(),
                "down_hessian": torch.as_tensor(down)
                .detach()
                .to(device="cpu", dtype=torch.float16)
                .contiguous(),
            },
            str(temporary),
            metadata=metadata,
        )
        os.replace(temporary, path)
    return {
        "schema": metadata["schema"],
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
        "stored_dtype": "float16",
        "gate_up_shape": [common.HIDDEN_SIZE, common.HIDDEN_SIZE],
        "down_shape": [common.INTERMEDIATE_SIZE, common.INTERMEDIATE_SIZE],
        "gate_up_exact_fp32_sha256": metadata["gate_up_exact_fp32_sha256"],
        "down_exact_fp32_sha256": metadata["down_exact_fp32_sha256"],
        "exact_recomputation_inputs": (
            "sealed_raw_capture_routes_plus_decoded_gate_up_plus_numeric_core"
        ),
    }


# ---------------------------------------------------------------------------
# Layer encode
# ---------------------------------------------------------------------------


def encode_layer(
    args: argparse.Namespace,
    codec,
    source,
    capture,
    plan: Mapping[str, Any],
    claim: Mapping[str, Any],
    readiness_preparations: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    import torch

    from quant_pipeline.normalization.prior_search import permute_expert_hf

    layer = int(claim["layer"])
    stem = common.probe_stem(layer)
    layer_root = args.work_root / "layers" / stem
    expert_root = layer_root / "experts"
    store = common.Dsv4PackedPayloadStore(layer_root / "payload-store")

    unit = next(row for row in plan["work_units"] if row["layer"] == layer)
    allocation = load_layer_allocation(args.work_root, layer)
    if allocation["allocation_sha256"] != claim["allocation_sha256"]:
        die(
            f"layer {layer} allocation seal differs from the claim: claim "
            f"{claim['allocation_sha256']} disk {allocation['allocation_sha256']}"
        )
    if allocation["allocation_sha256"] != unit["allocation_sha256"]:
        die(
            f"layer {layer} allocation seal differs from the plan unit; the plan "
            "was not rebuilt after the probe re-allocation (re-run the plan "
            "phase after phase 5)"
        )
    bits_map: dict[str, int] = {
        name: int(bits) for name, bits in allocation["allocation"].items()
    }

    _manifest_k4, tensors_k4 = load_preparation(
        args.k4_preparation_root, layer, expected_bits=4
    )
    _manifest_k3, tensors_k3 = load_preparation(
        args.k3_preparation_root, layer, expected_bits=3
    )
    readiness_row = readiness_preparations.get(layer)
    if not isinstance(readiness_row, Mapping):
        die(
            f"layer {layer} is absent from the phase-6 readiness receipt "
            "per_layer list; re-run phase 6"
        )
    if _manifest_k4["preparation_sha256"] != readiness_row.get(
        "k4_preparation_sha256"
    ):
        die(
            f"layer {layer} K4 preparation seal differs from the phase-6 "
            "readiness receipt; the preparation tree is not the sealed one"
        )
    if _manifest_k3["preparation_sha256"] != readiness_row.get(
        "k3_preparation_sha256"
    ):
        die(
            f"layer {layer} K3 preparation seal differs from the phase-6 "
            "readiness receipt; the preparation tree is not the sealed one"
        )
    if not torch.equal(tensors_k4["permutations"], tensors_k3["permutations"]):
        die(
            f"layer {layer}: K3 and K4 preparation permutations differ; the "
            "permuted basis must be shared across the triplet"
        )

    codec_identity_sha256 = common.require_hash(
        common.sha256_bytes(common.canonical_json(codec.identity)),
        "codec identity",
    )
    reader_abi = args.reader_abi_sha256
    if reader_abi is None:
        die(
            "--reader-abi-sha256 is required: every packed choice binds the sealed "
            "reader ABI (port mirror of the encode-work-unit reader binding)"
        )
    common.require_hash(reader_abi, "reader ABI")

    claim_sha = claim["claim_receipt_sha256"]
    allocation_sha = allocation["allocation_sha256"]
    binding = capture_binding(capture)
    mcg_marker = torch.tensor([-877912083], dtype=torch.int32)
    started = time.monotonic()

    for expert in range(common.NUM_EXPERTS):
        receipt_path = (
            expert_root / common.layer_dir_name(layer) / f"expert-{expert:03d}.json"
        )
        bits_by_projection = {
            projection: bits_map[
                common.tensor_full_name(layer, expert, projection)
            ]
            for projection in common.PROJECTIONS
        }
        if receipt_path.exists():
            receipt = verify_expert_receipt(common.load_json(receipt_path))
            if (
                receipt["claim_receipt_sha256"] != claim_sha
                or receipt["allocation_sha256"] != allocation_sha
                or receipt["bits"] != bits_by_projection
            ):
                die(
                    f"existing expert receipt binds a foreign claim, allocation, "
                    f"or rate: {receipt_path}"
                )
            # Mirror the resume verifier: re-verify every pre-existing
            # choice against the store.  Dsv4PackedPayloadStore.verify_choice
            # loads every object through ExactCodecPayloadStore.load_tensor,
            # which re-hashes each content-addressed object file and fails
            # loudly on truncation, corruption, or deletion; without this
            # the damage surfaces only at materialization.
            for projection in common.PROJECTIONS:
                choice = receipt["choices"][projection]
                verified = store.verify_choice(choice)
                if (
                    verified.get("layer") != layer
                    or verified.get("expert") != expert
                    or verified.get("projection") != projection
                    or verified.get("bits") != receipt["bits"][projection]
                ):
                    die(
                        f"existing expert receipt choice binding differs: "
                        f"{receipt_path}"
                    )
            # Re-hash the Hessian artifact on disk like the resume
            # verifier; save_hessians only guards the fresh path.
            hessian_record = receipt.get("hessian_artifact")
            if not isinstance(hessian_record, Mapping):
                die(
                    f"existing expert receipt lacks its Hessian artifact: "
                    f"{receipt_path}"
                )
            hessian_path = Path(str(hessian_record.get("path", ""))).resolve()
            try:
                hessian_path.relative_to(layer_root.resolve())
            except ValueError:
                die(
                    f"existing expert receipt Hessian artifact escapes the layer "
                    f"root: {receipt_path}"
                )
            if (
                not hessian_path.is_file()
                or hessian_path.is_symlink()
                or hessian_path.stat().st_size != hessian_record.get("bytes")
                or common.sha256_file(hessian_path)
                != hessian_record.get("sha256")
            ):
                die(
                    f"existing expert receipt Hessian artifact differs on disk: "
                    f"{receipt_path}"
                )
            continue

        # Codec-role mapping site: load_expert returns (w1, w2, w3) in
        # engine order (gate, down, up); w1 is the gate path, w3 the up
        # path, w2 the down path.
        w1, w2, w3 = source.load_expert(layer, expert)
        w1 = w1.to(args.device)
        w2 = w2.to(args.device)
        w3 = w3.to(args.device)
        permutation = tensors_k4["permutations"][expert].tolist()
        gate_weight, up_weight, down_weight = permute_expert_hf(
            w1, w3, w2, permutation
        )

        gate_cov, gate_up_evidence = common.gate_covariance(
            codec, capture, expert, args.device, args.chunk_rows
        )
        gate_up_evidence["matrix_sha256"] = common.tensor_sha256(gate_cov)

        encoded = {}
        choices: dict[str, dict[str, Any]] = {}
        used_vectors: dict[str, tuple[Any, Any]] = {}
        predecessor = claim_sha
        # Gate and up first (w1 and w3): both consume the hidden-input
        # gate covariance; roles gate_proj/up_proj at the codec boundary.
        for projection, weight in (
            ("w1", gate_weight),
            ("w3", up_weight),
        ):
            bits = bits_by_projection[projection]
            vectors = tensors_k4 if bits == 4 else tensors_k3
            suh, svh = preparation_vectors(
                vectors, projection, expert, args.device
            )
            candidate = encode_one_rate(
                codec,
                layer=layer,
                expert=expert,
                role=PROJECTION_ROLE[projection],
                weight_hf=weight,
                covariance=gate_cov,
                bits=bits,
                suh=suh,
                svh=svh,
                provenance={
                    "claim_receipt_sha256": claim_sha,
                    "allocation_sha256": allocation_sha,
                    "public_shapleymcg_mixed_rate": True,
                    "global_allocator": False,
                },
            )
            encoded[projection] = candidate
            used_vectors[projection] = (suh, svh)
            choices[projection] = store.put_choice(
                layer=layer,
                expert=expert,
                projection=projection,
                bits=bits,
                choice_id=f"L{layer:02d}.E{expert:03d}.{projection}.K{bits}",
                trellis=candidate.packed,
                suh=suh,
                svh=svh,
                mcg=mcg_marker,
                reconstruction=candidate.reconstructed.half().contiguous(),
                vector_topology=VECTOR_TOPOLOGY[projection],
                reader_abi_sha256=reader_abi,
                provenance={
                    "claim_receipt_sha256": claim_sha,
                    "allocation_sha256": allocation_sha,
                    "bits": bits,
                    "packed_sha256": candidate.packed_sha256,
                    "reconstruction_sha256": candidate.reconstruction_sha256,
                    "vector_rate": bits,
                },
                predecessor_state_hash=predecessor,
            )
            predecessor = choices[projection]["choice_sha256"]

        # Corrected operation order: a fresh factor domain for the
        # candidate-conditioned down covariance after exact gate/up decode.
        codec._codec().clear_caches()

        # ONE conditioning context for the whole down curve, identical to
        # the probe and the DP allocation (R7 pair_at semantics): the down
        # Hessian is conditioned on gate/up reconstructions decoded at the
        # reference rate dsv4_common.FLOOR_BITS regardless of the allocated
        # gate/up rates.  When an allocated rate differs, the mismatched
        # tensor is additionally encoded at the reference rate for the
        # conditioning only; the deployed choice still carries the
        # allocated rate, and the receipt records both.
        reference_bits = common.FLOOR_BITS
        conditioning: dict[str, Any] = {}
        for projection, weight in (
            ("w1", gate_weight),
            ("w3", up_weight),
        ):
            if bits_by_projection[projection] == reference_bits:
                conditioning[projection] = encoded[projection]
            else:
                vectors = tensors_k4 if reference_bits == 4 else tensors_k3
                suh, svh = preparation_vectors(
                    vectors, projection, expert, args.device
                )
                conditioning[projection] = encode_one_rate(
                    codec,
                    layer=layer,
                    expert=expert,
                    role=PROJECTION_ROLE[projection],
                    weight_hf=weight,
                    covariance=gate_cov,
                    bits=reference_bits,
                    suh=suh,
                    svh=svh,
                    provenance={
                        "claim_receipt_sha256": claim_sha,
                        "allocation_sha256": allocation_sha,
                        "conditioning_only": True,
                        "conditioning_reference_bits": reference_bits,
                        "public_shapleymcg_mixed_rate": True,
                        "global_allocator": False,
                    },
                )

        down_bits = bits_by_projection["w2"]
        down_cov, down_evidence = common.down_covariance(
            codec,
            capture,
            expert,
            # w1 reconstruction is the gate conditioning, w3 the up
            # conditioning (codec-role mapping site).
            conditioning["w1"].reconstructed.t().contiguous(),
            conditioning["w3"].reconstructed.t().contiguous(),
            gate_bits=reference_bits,
            up_bits=reference_bits,
            device=args.device,
            chunk_rows=args.chunk_rows,
        )
        down_evidence["matrix_sha256"] = common.tensor_sha256(down_cov)
        down_evidence["gate_reconstruction_sha256"] = (
            conditioning["w1"].reconstruction_sha256
        )
        down_evidence["up_reconstruction_sha256"] = (
            conditioning["w3"].reconstruction_sha256
        )

        vectors_down = tensors_k4 if down_bits == 4 else tensors_k3
        suh_down, svh_down = preparation_vectors(
            vectors_down, "w2", expert, args.device
        )
        down_candidate = encode_one_rate(
            codec,
            layer=layer,
            expert=expert,
            role=PROJECTION_ROLE["w2"],
            weight_hf=down_weight,
            covariance=down_cov,
            bits=down_bits,
            suh=suh_down,
            svh=svh_down,
            provenance={
                "claim_receipt_sha256": claim_sha,
                "allocation_sha256": allocation_sha,
                "public_shapleymcg_mixed_rate": True,
                "global_allocator": False,
            },
        )
        choices["w2"] = store.put_choice(
            layer=layer,
            expert=expert,
            projection="w2",
            bits=down_bits,
            choice_id=f"L{layer:02d}.E{expert:03d}.w2.K{down_bits}",
            trellis=down_candidate.packed,
            suh=suh_down,
            svh=svh_down,
            mcg=mcg_marker,
            reconstruction=down_candidate.reconstructed.half().contiguous(),
            vector_topology=VECTOR_TOPOLOGY["w2"],
            reader_abi_sha256=reader_abi,
            provenance={
                "claim_receipt_sha256": claim_sha,
                "allocation_sha256": allocation_sha,
                "bits": down_bits,
                "packed_sha256": down_candidate.packed_sha256,
                "reconstruction_sha256": down_candidate.reconstruction_sha256,
                "vector_rate": down_bits,
                "down_conditioning": {
                    "gate_bits": reference_bits,
                    "up_bits": reference_bits,
                    "semantics": "r7_pair_at_reference_rates_v1",
                },
                "gate_up_roundtrip_sha256": {
                    "gate": conditioning["w1"].reconstruction_sha256,
                    "up": conditioning["w3"].reconstruction_sha256,
                },
            },
            predecessor_state_hash=predecessor,
        )

        hessian_artifact = save_hessians(
            layer_root,
            layer,
            expert,
            gate_cov,
            down_cov,
            {"gate_up": gate_up_evidence, "down": down_evidence},
        )
        receipt = build_expert_receipt(
            layer=layer,
            expert=expert,
            bits_by_projection=bits_by_projection,
            choices=choices,
            claim_receipt_sha256=claim_sha,
            allocation_sha256=allocation_sha,
            capture_binding=binding,
            hessian_artifact=hessian_artifact,
            down_conditioning={
                "gate_rate": reference_bits,
                "up_rate": reference_bits,
                "down_rate": down_bits,
                "deployed_gate_rate": bits_by_projection["w1"],
                "deployed_up_rate": bits_by_projection["w3"],
                "semantics": "r7_pair_at_reference_rates_v1",
                "note": (
                    "gate_rate/up_rate are the rates whose decoded "
                    "reconstructions conditioned the Hessian (the R7 pair_at "
                    "reference context shared with the probe and the DP "
                    "allocation); deployed rates are the shipped encodes "
                    "(gate is w1, up is w3, down is w2)"
                ),
                "evidence": down_evidence,
            },
            codec_identity_sha256=codec_identity_sha256,
        )
        common.write_json(receipt_path, receipt)
        verify_expert_receipt(receipt)
        for projection in common.PROJECTIONS:
            store.verify_choice(choices[projection])
        del (
            w1,
            w2,
            w3,
            gate_weight,
            up_weight,
            down_weight,
            gate_cov,
            down_cov,
            encoded,
            conditioning,
        )
        if expert % 8 == 7:
            gc.collect()
            torch.cuda.empty_cache()

    expert_receipts = [
        verify_expert_receipt(
            common.load_json(
                expert_root / common.layer_dir_name(layer)
                / f"expert-{expert:03d}.json"
            )
        )
        for expert in range(common.NUM_EXPERTS)
    ]
    layer_receipt = build_layer_receipt(
        layer=layer,
        worker_id=args.worker,
        claim_receipt_sha256=claim_sha,
        allocation_sha256=allocation_sha,
        expert_receipts=expert_receipts,
    )
    common.write_json(layer_root / "layer-receipt.json", layer_receipt)
    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "layer": layer,
                "worker": args.worker,
                "elapsed_seconds": round(elapsed, 1),
                "layer_receipt_sha256": layer_receipt["receipt_sha256"],
            }
        ),
        flush=True,
    )
    return layer_receipt


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.recover_worker is not None:
        recover_worker(args)
        return

    plan = load_plan(args.work_root)
    worker_row = verify_plan_worker(plan, args.worker)
    print(
        json.dumps(
            {
                "worker": args.worker,
                "preflight_variant": plan["preflight_variant"],
                "plan_cuda_visible_devices": worker_row.get("cuda_visible_devices"),
                "device": args.device,
                "note": (
                    "docker --gpus renumbers devices; verify CUDA_VISIBLE_DEVICES "
                    "against the enumeration visible inside this container"
                ),
            }
        ),
        flush=True,
    )

    source_root = common.resolve_source_root(args)
    codec = common.build_codec(
        source_root, common.resolve_extension(args), args.device
    )
    codec_identity_sha256 = common.require_hash(
        common.sha256_bytes(common.canonical_json(codec.identity)),
        "codec identity",
    )

    # Bind this worker to the phase-6 readiness receipt BEFORE any expert
    # receipt is sealed: the receipt's codec_identity_sha256 is derived from
    # the sealed rate-3 preparation manifests (checked identical across
    # layers and against the live codec by phase 6), so equality here means
    # the encode codec is the codec the preparations were built with, not an
    # unbound attestation.  The per_layer preparation seals are enforced per
    # layer inside encode_layer.
    readiness_path = args.work_root / "gss" / "readiness-receipt.json"
    if not readiness_path.is_file():
        die(f"the phase-6 readiness receipt is absent: {readiness_path}")
    readiness = common.load_json(readiness_path)
    common.verify_seal(
        readiness,
        schema=common.DSV4_READINESS_SCHEMA,
        field="readiness_receipt_sha256",
        label="phase-6 readiness receipt",
    )
    if readiness.get("launch_plan_sha256") != plan["launch_plan_sha256"]:
        die(
            "the phase-6 readiness receipt binds a different launch plan; "
            "re-run phase 6 after any plan rebuild"
        )
    if readiness.get("codec_identity_sha256") != codec_identity_sha256:
        die(
            "the phase-6 readiness receipt codec identity differs from this "
            "worker's codec (extension/torch/environment differ); refusing to "
            "encode under a codec the preparations were not built with"
        )
    readiness_preparations: dict[int, dict[str, Any]] = {}
    for row in readiness.get("per_layer", []):
        if not isinstance(row, Mapping) or not isinstance(row.get("layer"), int):
            die("phase-6 readiness receipt per_layer list is malformed")
        readiness_preparations[int(row["layer"])] = dict(row)

    from dsv4_capped_source import CappedSource

    source = CappedSource(
        model_dir=args.model_dir,
        meta_path=args.meta_path,
        lora_path=args.lora_path,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "worker": args.worker,
                "source": "dsv4_capped_source.CappedSource",
                "source_identity": source.identity,
            }
        ),
        flush=True,
    )

    while True:
        active_claim = None
        with StateLock(args.work_root):
            current_plan = load_plan(args.work_root)
            if (
                current_plan["launch_plan_sha256"]
                != plan["launch_plan_sha256"]
            ):
                print(
                    json.dumps(
                        {
                            "worker": args.worker,
                            "plan_changed": True,
                            "note": (
                                "launch plan changed on disk (a re-plan); "
                                "exiting cleanly without new claims; restart "
                                "the worker on the new plan"
                            ),
                        }
                    ),
                    flush=True,
                )
                return
            _path, state = newest_state(args.work_root, plan)
            k35.verify_state(plan, state)
            phase = state["phase"]
            if phase == "planned":
                die(
                    "the state chain is still in phase 'planned'; run the "
                    "phase-6 readiness receipt and enter the encoding phase "
                    "first"
                )
            if phase != "k35_main_encoding":
                print(
                    json.dumps(
                        {
                            "worker": args.worker,
                            "phase": phase,
                            "note": (
                                "encoding phase is closed; nothing to claim"
                            ),
                        }
                    )
                )
                return
            mine = state["active_claims"].get(args.worker)
            if mine is not None:
                active_claim = dict(mine)
            elif (args.work_root / "PAUSE").exists():
                print(
                    json.dumps(
                        {
                            "worker": args.worker,
                            "drain_pause": True,
                            "note": "PAUSE file present; no new claim taken",
                        }
                    )
                )
                return
            elif not state["pending_layers"]:
                print(
                    json.dumps(
                        {
                            "worker": args.worker,
                            "pending": 0,
                            "note": (
                                "no unclaimed layers remain; ready for the "
                                "packing phase"
                            ),
                        }
                    )
                )
                return
            else:
                successor, claim = k35.claim_next_layer(
                    plan, state, worker_id=args.worker
                )
                common.write_json(
                    state_path(args.work_root, successor["sequence"]),
                    successor,
                )
                active_claim = dict(claim)
                print(
                    json.dumps(
                        {
                            "worker": args.worker,
                            "claimed_layer": claim["layer"],
                            "state": successor["sequence"],
                        }
                    ),
                    flush=True,
                )

        layer = int(active_claim["layer"])
        capture = common.open_capture(
            args.calibration_root,
            layer,
            verify_hashes=not args.no_verify_capture_hashes,
        )
        layer_receipt = encode_layer(
            args, codec, source, capture, plan, active_claim,
            readiness_preparations
        )
        del capture

        with StateLock(args.work_root):
            _path, newest = newest_state(args.work_root, plan)
            k35.verify_state(plan, newest)
            claim_now = newest["active_claims"].get(args.worker)
            if (
                claim_now is None
                or claim_now.get("claim_receipt_sha256")
                != active_claim["claim_receipt_sha256"]
            ):
                die(
                    f"worker {args.worker} claim on layer {layer} vanished from "
                    "the newest state before completion; inspect the chain"
                )
            successor = k35.complete_layer(
                plan,
                newest,
                worker_id=args.worker,
                layer=layer,
                layer_receipt_sha256=layer_receipt["receipt_sha256"],
            )
            common.write_json(
                state_path(args.work_root, successor["sequence"]), successor
            )
            print(
                json.dumps(
                    {
                        "worker": args.worker,
                        "completed_layer": layer,
                        "completed_total": len(successor["completed_layers"]),
                        "state": successor["sequence"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
