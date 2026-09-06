#!/usr/bin/env python3
"""dsv4_common.py - shared bindings for the DSV4 Flash mixed K3/K4 encode.

Port of k35_common.py (GLM-5.3 campaign) to DeepSeek V4 Flash geometry,
bound to dsv4_geometry's discovered constants instead of GLM literals.

Geometry deltas from GLM (all discovered, none assumed):
  layers 0..42 ALL routed (GLM 3..44); hash-routed 0..2 (tid2eid)
  256 experts x 3 = 768 tensors/layer (GLM 288x3=864)
  rate arithmetic: floor 3 + 384 upgrades = 2688 units = 3.5 bpw/layer
  r7_encoder's own constants (256/768/2688) already match this geometry,
  but this module keeps its own byte-faithful DP copy for self-contained
  audit (no dependence on r7's owner-locked constants module).
  MTP mtp.{0,1,2} is NATIVE scope v1 - no MTP capture, encode, or receipt.

Tensor-name vocabulary: MASTER names (layers.{L}.ffn.experts.{E}.w{1,2,3}
.weight) everywhere in campaign code and stores. The codec boundary maps
w1->gate_proj, w3->up_proj, w2->down_proj (engine swiglu: gate=x@w1.T
silu'd, up=x@w3.T, down=h@w2.T) - map ONLY at codec call sites.

Capture ABI: dsv4_capture.py's sealed u16 manifest
(quant-pipeline.dsv4-capture.v1) - hidden.bf16.bin rows x 4096 x 2,
topk_ids.u16le.bin rows x 6 x 2, topk_weights.f32le.bin rows x 6 x 4.

ASCII only. No em-dashes. No network. Writes only under --work-root.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Geometry singleton - resolved from the LIVE master at import
# ---------------------------------------------------------------------------

MODEL_DIR = os.environ.get("DSV4_MODEL", "/model")
META_PATH = os.environ.get("DSV4_META", "/wd/tensor_meta.json")
LORA_PATH = os.environ.get("DSV4_LORA", "/wd/cap/lora.safetensors")

from dsv4_geometry import Geometry, LORA_SCALE  # noqa: E402

G = Geometry(MODEL_DIR, META_PATH, LORA_PATH)

HIDDEN_SIZE = G.hidden              # 4096
INTERMEDIATE_SIZE = G.inter         # 2048
NUM_EXPERTS = G.n_experts           # 256
TOP_K = G.top_k                     # 6
MAIN_LAYERS = tuple(range(G.n_layers))          # 0..42, all routed
ALL_PROBE_LAYERS = MAIN_LAYERS                    # no MTP in encode v1
PROJECTIONS = ("w1", "w2", "w3")                  # master vocabulary

# rate arithmetic (mirrors r7_encoder/constants.py:35-40 for this geometry)
TENSORS_PER_LAYER = NUM_EXPERTS * 3               # 768
FLOOR_BITS = 3
PER_TENSOR_ALLOWED_BITS = (3, 4)
K4_TENSORS_PER_LAYER = TENSORS_PER_LAYER // 2     # 384 upgrades
K3_TENSORS_PER_LAYER = TENSORS_PER_LAYER - K4_TENSORS_PER_LAYER
TARGET_BIT_UNITS_PER_LAYER = (
    TENSORS_PER_LAYER * FLOOR_BITS + K4_TENSORS_PER_LAYER)  # 2688

SIGMA_REG = 0.025  # pinned at codec construction (campaign discipline)

DEFAULT_WORK_ROOT = Path(os.environ.get("DSV4_WORK_ROOT", "/workspace"))
DEFAULT_CALIBRATION_ROOT = Path(
    os.environ.get("DSV4_CALIBRATION", "/workspace/calibration/main-full"))
ENV_EXTENSION = "K35_EXLLAMAV3_EXT"
CHUNK_ROWS = 1024

# ---------------------------------------------------------------------------
# NEW SURFACE schemas (never reuse glm53 strings - seal collision)
# ---------------------------------------------------------------------------

DSV4_PROBE_LEDGER_SCHEMA = "quant-pipeline.dsv4-k35-probe-ledger.v1"
DSV4_RATE3_GSS_SCHEMA = "quant-pipeline.dsv4-k35-rate3-gss-preparation.v1"
DSV4_READINESS_SCHEMA = "quant-pipeline.dsv4-k35-readiness-receipt.v1"
DSV4_PACKED_CHOICE_SCHEMA = "quant-pipeline.dsv4-k35-packed-choice.v1"
DSV4_EXPERT_RECEIPT_SCHEMA = "quant-pipeline.dsv4-k35-expert-receipt.v1"
DSV4_LAYER_RECEIPT_SCHEMA = "quant-pipeline.dsv4-k35-layer-receipt.v1"
DSV4_CAPTURE_SCHEMA = "quant-pipeline.dsv4-capture.v1"

_HASH = re.compile(r"[0-9a-f]{64}")
DP_SCORE_SCALE = 10**15  # mirrors r7_encoder/allocation.py:25


def die(message: str) -> None:
    raise SystemExit(f"dsv4: FAIL: {message}")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, body: Mapping[str, Any]) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(
        json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def canonical_json(body: Any) -> bytes:
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        die(f"{label} must be a lowercase 64-hex SHA-256, got {value!r}")
    return value


def seal(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(body))
    result[field] = sha256_bytes(canonical_json(result))
    return result


def verify_seal(document: Mapping[str, Any], *, schema: str, field: str,
                label: str) -> str:
    if document.get("schema") != schema:
        die(f"{label} schema differs: expected {schema}, "
            f"got {document.get('schema')!r}")
    digest = require_hash(document.get(field), f"{label}.{field}")
    body = copy.deepcopy(dict(document))
    del body[field]
    if sha256_bytes(canonical_json(body)) != digest:
        die(f"{label} seal differs")
    return digest


def probe_stem(layer: int) -> str:
    return f"L{layer:02d}"


def layer_dir_name(layer: int) -> str:
    return f"layer-{layer:02d}"


def tensor_full_name(layer: int, expert: int, projection: str) -> str:
    """Master tensor name - the single naming authority."""
    if projection not in PROJECTIONS:
        die(f"unknown projection {projection}")
    return f"layers.{layer}.ffn.experts.{expert}.{projection}.weight"


def _layer_tensor_names(layer: int) -> list[str]:
    return sorted(
        tensor_full_name(layer, expert, projection)
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    )


def resolve_extension(args) -> Path:
    raw = getattr(args, "extension", None) or os.environ.get(ENV_EXTENSION)
    if not raw:
        die("the compiled exllamav3_ext .so is required: pass --extension "
            f"PATH or set {ENV_EXTENSION}")
    path = Path(raw).resolve()
    if not path.is_file():
        die(f"extension is not a file: {path}")
    return path


def resolve_source_root(args) -> Path:
    raw = getattr(args, "repo_root", None)
    if raw:
        root = Path(raw).resolve()
        if not (root / "r7_encoder" / "r10_codec.py").is_file():
            die(f"--repo-root lacks r7_encoder/r10_codec.py: {root}")
        return root
    for entry in sys.path:
        candidate = Path(entry or ".").resolve()
        if (candidate / "r7_encoder" / "r10_codec.py").is_file():
            return candidate
    die("cannot find the r10 bundle on sys.path; pass --repo-root")


def numeric_core_path(source_root: Path) -> Path:
    path = source_root / "lineage" / "encode_tr3_v31.py"
    if not path.is_file():
        die(f"numeric core is absent: {path}")
    return path


def build_codec(source_root: Path, extension: Path, device: str):
    """Construct Exl3MCGCodec and force the sealed import NOW (the codec
    refuses to run when r7_encoder is already cached - construct before any
    r7_encoder.* import in every driver)."""
    from quant_pipeline.codecs.exl3_mcg import Exl3MCGCodec

    codec = Exl3MCGCodec(
        source_root=source_root,
        numeric_core=numeric_core_path(source_root),
        extension=extension,
        device=device,
        sigma_reg=SIGMA_REG,
    )
    codec._codec()
    return codec


def r7_hessian():
    import r7_encoder.hessian as hessian

    return hessian


# ---------------------------------------------------------------------------
# Capture view (consumer of dsv4_capture.py's sealed ABI)
# ---------------------------------------------------------------------------


class RoutedRows:
    """Row selection for one expert+role, with document epochs."""

    def __init__(self, rows, row_indices, applied_weights, document_epochs):
        self.rows = rows
        self.row_indices = row_indices
        self.applied_weights = applied_weights
        self.document_epochs = document_epochs


class Dsv4CaptureView:
    """Memory-mapped adapter from the sealed dsv4 capture ABI."""

    def __init__(self, root: str | Path, layer: int, *,
                 verify_hashes: bool = True,
                 required_roles: Sequence[str] = (
                     "fit", "conditional-fit", "selection", "confirmation")):
        if layer not in MAIN_LAYERS:
            raise ValueError(f"layer {layer} outside routed surface 0..42")
        self.root = Path(root).resolve()
        self.layer = layer
        manifest = load_json(self.root / "capture-manifest.json")
        verify_seal(manifest, schema=DSV4_CAPTURE_SCHEMA,
                    field="capture_sha256", label="capture manifest")
        geom = manifest.get("geometry", {})
        if (manifest.get("layers") != list(MAIN_LAYERS)
                or geom.get("hidden_size") != HIDDEN_SIZE
                or geom.get("experts") != NUM_EXPERTS
                or geom.get("top_k") != TOP_K):
            raise ValueError("capture geometry differs from discovered DSV4")
        roles = tuple(manifest.get("roles", ()))
        if not set(required_roles) <= set(roles):
            raise ValueError(f"capture lacks roles: {required_roles}")
        windows = manifest.get("windows")
        if not isinstance(windows, list) or not windows:
            raise ValueError("capture window journal is absent")
        self.offsets = []
        cursor = 0
        document_epoch: dict[str, int] = {}
        for index, raw in enumerate(windows):
            if (not isinstance(raw, Mapping)
                    or raw.get("window_index") != index
                    or not isinstance(raw.get("rows"), int)
                    or raw["rows"] <= 0
                    or raw.get("role") not in roles):
                raise ValueError("capture window boundary is malformed")
            doc = str(raw.get("document_id") or "")
            if not doc:
                raise ValueError("capture window document is malformed")
            # multi-doc batches carry comma-joined ids; epoch per atom
            for atom in doc.split(","):
                document_epoch.setdefault(atom, len(document_epoch))
            self.offsets.append(
                (cursor, cursor + raw["rows"], str(raw["role"]), doc))
            cursor += raw["rows"]
        if manifest.get("rows_per_layer") != cursor:
            raise ValueError("capture row census differs from windows")
        files = manifest.get("files", {}).get(str(layer), {})
        paths = {}
        for key, fname in (
            ("hidden_bf16", "hidden.bf16.bin"),
            ("topk_ids_u16le", "topk_ids.u16le.bin"),
            ("topk_weights_f32le", "topk_weights.f32le.bin"),
        ):
            record = files.get(key)
            if not isinstance(record, Mapping):
                raise ValueError(f"capture lacks {key}")
            artifact = (self.root / str(record.get("path", ""))).resolve()
            try:
                artifact.relative_to(self.root)
            except ValueError as error:
                raise ValueError("capture artifact escapes root") from error
            if (not artifact.is_file() or artifact.is_symlink()
                    or artifact.stat().st_size != record.get("bytes")
                    or (verify_hashes
                        and sha256_file(artifact) != record.get("sha256"))):
                raise ValueError(f"capture artifact differs: {key}")
            paths[key] = artifact
        expected = {
            "hidden_bf16": cursor * HIDDEN_SIZE * 2,
            "topk_ids_u16le": cursor * TOP_K * 2,
            "topk_weights_f32le": cursor * TOP_K * 4,
        }
        if any(paths[k].stat().st_size != v for k, v in expected.items()):
            raise ValueError("capture artifact sizes disagree with rows")
        self.hidden_u16 = np.memmap(
            paths["hidden_bf16"], dtype="<u2", mode="r")
        self.ids = np.memmap(
            paths["topk_ids_u16le"], dtype="<u2", mode="r").reshape(
                cursor, TOP_K)
        self.weights = np.memmap(
            paths["topk_weights_f32le"], dtype="<f4", mode="r").reshape(
                cursor, TOP_K)
        self.row_roles = np.empty(cursor, dtype=object)
        self.row_docs = np.empty(cursor, dtype=object)
        self._doc_epoch: dict[str, int] = {}
        for begin, stop, role, doc in self.offsets:
            self.row_roles[begin:stop] = role
            self.row_docs[begin:stop] = doc
            for atom in doc.split(","):
                self._doc_epoch.setdefault(atom, begin)
        if int(self.ids.max(initial=0)) >= NUM_EXPERTS:
            raise ValueError("capture ids exceed expert census")

    def routed_rows(self, expert: int, role: str) -> RoutedRows:
        if not 0 <= expert < NUM_EXPERTS:
            raise ValueError(f"expert {expert} outside 0..{NUM_EXPERTS-1}")
        hit = (self.ids == expert)
        keep = hit.any(axis=1) & (self.row_roles == role)
        rows = int(keep.sum())
        idx = np.flatnonzero(keep)
        # per-row applied weight for this expert (0 where not routed);
        # a row routes an expert at most once (no-duplicate invariant)
        w = np.where(hit[idx], self.weights[idx], 0.0).astype(np.float64)
        # document epochs: min epoch over the row's doc atoms
        epoch = np.array(
            [min(self._doc_epoch[a] for a in d.split(",") if a in
                 self._doc_epoch)
             for d in self.row_docs[idx]], dtype=np.int64)
        return RoutedRows(rows, idx.astype(np.int64), w, epoch)


def open_capture(calibration_root: Path, layer: int, *,
                 verify_hashes: bool = True):
    return Dsv4CaptureView(
        calibration_root, layer, verify_hashes=verify_hashes)


# ---------------------------------------------------------------------------
# Covariances (mirrors of glm53_prepared_backend.py:253-305, same arithmetic)
# ---------------------------------------------------------------------------


def _hidden_chunks(capture: Any, row_indices, device: str, chunk_rows: int):
    import torch

    for begin in range(0, row_indices.size, chunk_rows):
        stop = min(row_indices.size, begin + chunk_rows)
        words = np.array(
            capture.hidden_u16[row_indices[begin:stop]],
            dtype=np.uint16, copy=True)
        yield (
            torch.from_numpy(words)
            .view(torch.bfloat16)
            .to(device, dtype=torch.float32)
            .contiguous()
        )


def _routed_evidence(routed) -> dict[str, Any]:
    return {
        "rows": int(routed.rows),
        "documents": int(np.unique(routed.document_epochs).size),
        "row_indices_sha256": hashlib.sha256(
            routed.row_indices.tobytes()).hexdigest(),
        "route_weights_sha256": hashlib.sha256(
            np.asarray(routed.applied_weights, dtype="<f4").tobytes()
        ).hexdigest(),
    }


def expert_p2_mass(capture: Any, expert: int) -> float:
    """Per-expert p2 mass over fit rows (the DP gain weight)."""
    routed = capture.routed_rows(expert, "fit")
    if routed.rows <= 0:
        die(f"L{capture.layer} E{expert}: fit rows are empty; "
            "mass is undefined (extend corpus or reassign role)")
    weights = np.asarray(routed.applied_weights, dtype=np.float64)
    mass = float(np.square(weights).sum())
    if mass <= 0:
        die(f"L{capture.layer} E{expert}: degenerate p2 mass")
    return mass


def gate_covariance(codec, capture: Any, expert: int, device: str,
                    chunk_rows: int):
    """Routed p2 uncentered full covariance over 'fit' rows (w1 and w3
    input space = hidden)."""
    import torch

    hessian = r7_hessian()
    routed = capture.routed_rows(expert, "fit")
    if routed.rows <= 0:
        die(f"L{capture.layer} E{expert}: empty fit rows")
    accumulator = hessian.FullCovarianceAccumulator(
        HIDDEN_SIZE, device=device, guided=True)
    cursor = 0
    for hidden in _hidden_chunks(capture, routed.row_indices, device,
                                 chunk_rows):
        stop = cursor + int(hidden.shape[0])
        weights = np.square(routed.applied_weights[cursor:stop],
                            dtype=np.float32)
        accumulator.add(hidden, weights)
        cursor = stop
    value = accumulator.finalize(SIGMA_REG, add_damping=False)
    evidence = dict(_routed_evidence(routed))
    evidence.update({
        "construction": "routed-p2-uncentered-second-moment-v1",
        "weight_sum": float(value.weight_sum),
    })
    return value.matrix, evidence


def down_covariance(codec, capture: Any, expert: int, gate_kn, up_kn, *,
                    gate_bits: int, up_bits: int, device: str,
                    chunk_rows: int):
    """Candidate-conditioned down covariance over 'conditional-fit' rows
    (w2 input space = intermediate). Conditioning per R7 pair_at semantics:
    gate/up decoded at FLOOR_BITS for the whole curve."""
    import torch

    hessian = r7_hessian()
    routed = capture.routed_rows(expert, "conditional-fit")
    if routed.rows <= 0:
        die(f"L{capture.layer} E{expert}: empty conditional-fit rows")
    accumulator = hessian.FullCovarianceAccumulator(
        INTERMEDIATE_SIZE, device=device, guided=True)
    gate_rt = gate_kn.to(device)
    up_rt = up_kn.to(device)
    cursor = 0
    for hidden in _hidden_chunks(capture, routed.row_indices, device,
                                 chunk_rows):
        stop = cursor + int(hidden.shape[0])
        middle = hessian.down_inputs_from_roundtrip(hidden, gate_rt, up_rt)
        weights = np.square(routed.applied_weights[cursor:stop],
                            dtype=np.float32)
        accumulator.add(middle, weights)
        cursor = stop
    value = accumulator.finalize(SIGMA_REG, add_damping=False)
    evidence = dict(_routed_evidence(routed))
    evidence.update({
        "construction": (
            f"decoded-gate-k{int(gate_bits)}-up-k{int(up_bits)}-"
            "candidate-conditioned-routed-p2-uncentered-second-moment-v1"),
        "conditioning_gate_bits": int(gate_bits),
        "conditioning_up_bits": int(up_bits),
        "weight_sum": float(value.weight_sum),
    })
    return value.matrix, evidence


def tensor_sha256(value: Any) -> str:
    import torch

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(map(str, tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def raw_payload_sha256(value: Any) -> str:
    import torch

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(
        tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def covariance_proxy_loss(weight_hf: Any, reconstructed_hf: Any,
                          covariance: Any) -> float:
    """Per-rate probe loss: relative covariance quadratic, float64,
    trellis.py:383-396 formula on HF-orientation reconstructions."""
    import torch

    weight = torch.as_tensor(weight_hf)
    recon = torch.as_tensor(reconstructed_hf)
    if tuple(weight.shape) != tuple(recon.shape):
        die(f"proxy-loss orientation differs: weight {tuple(weight.shape)} "
            f"reconstruction {tuple(recon.shape)}")
    covariance_tensor = torch.as_tensor(
        covariance, dtype=torch.float64, device=weight.device)
    error = recon.double() - weight.double()
    numerator = torch.einsum("nk,kl,nl->", error, covariance_tensor, error)
    denominator = torch.einsum(
        "nk,kl,nl->", weight.double(), covariance_tensor,
        weight.double()).clamp_min(1e-30)
    value = float((numerator / denominator).item())
    if value < 0:
        die("proxy loss is negative; covariance or reconstruction malformed")
    return value


# ---------------------------------------------------------------------------
# NEW SURFACE: bits-honest packed choice store (dsv4 schemas)
# ---------------------------------------------------------------------------


class Dsv4PackedPayloadStore:
    def __init__(self, root: str | Path) -> None:
        from quant_pipeline.checkpoint.exact_payload import (
            ExactCodecPayloadStore,
        )

        self.root = Path(root)
        self.objects = ExactCodecPayloadStore(self.root)
        self.choices = self.root / "choices"
        self.choices.mkdir(parents=True, exist_ok=True)

    def put_choice(self, *, layer: int, expert: int, projection: str,
                   bits: int, choice_id: str, trellis, suh, svh, mcg,
                   reconstruction, vector_topology: Mapping[str, str],
                   reader_abi_sha256: str, provenance: Mapping[str, Any],
                   predecessor_state_hash: str) -> dict[str, Any]:
        import torch

        from quant_pipeline.checkpoint.exact_payload import (
            packed_payload_sha256,
            tensor_sha256 as q_tensor_sha256,
        )
        from quant_pipeline.checkpoint.packed_payload import (
            checkpoint_payload_sha256,
        )

        if projection not in PROJECTIONS:
            die(f"unknown projection {projection}")
        if bits not in PER_TENSOR_ALLOWED_BITS:
            die(f"choice bits {bits} outside {PER_TENSOR_ALLOWED_BITS}")
        require_hash(predecessor_state_hash, "choice predecessor state")
        require_hash(reader_abi_sha256, "reader ABI")
        values = {}
        for name, value in (("trellis", trellis), ("suh", suh),
                            ("svh", svh), ("mcg", mcg),
                            ("reconstruction", reconstruction)):
            tensor = torch.as_tensor(value).detach().contiguous().cpu()
            values[name] = (tensor.reshape(1) if tensor.ndim == 0
                            else tensor)
        if values["trellis"].dtype != torch.int16:
            die("trellis must be int16")
        if any(values[n].dtype != torch.float16
               for n in ("suh", "svh", "reconstruction")):
            die("scales and closure reconstruction must be FP16")
        if (values["mcg"].dtype != torch.int32
                or values["mcg"].numel() != 1
                or int(values["mcg"].reshape(-1)[0]) != -877912083):
            die("choice is not marked with MCG 0xCBAC1FED")
        reconstruction = values["reconstruction"]
        if (reconstruction.ndim != 2 or values["suh"].ndim != 1
                or values["svh"].ndim != 1):
            die("choice tensor ranks differ")
        n, k = map(int, reconstruction.shape)
        if values["suh"].numel() != k or values["svh"].numel() != n:
            die("vectors disagree with reconstruction geometry")
        expected_trellis_bytes = n * k * int(bits) // 8
        actual = (values["trellis"].numel()
                  * values["trellis"].element_size())
        if actual != expected_trellis_bytes:
            die(f"trellis bytes {actual} disagree with bits {bits} "
                f"geometry {n}x{k} (expected {expected_trellis_bytes})")
        stored = {n_: values[n_] for n_ in ("trellis", "suh", "svh", "mcg")}
        refs = {n_: self.objects.put_tensor(v).as_dict()
                for n_, v in stored.items()}
        body = {
            "schema": DSV4_PACKED_CHOICE_SCHEMA,
            "layer": int(layer),
            "expert": int(expert),
            "projection": projection,
            "choice_id": str(choice_id),
            "bits": int(bits),
            "predecessor_state_hash": str(predecessor_state_hash),
            "objects": refs,
            "packed_sha256": packed_payload_sha256(
                {n_: stored[n_] for n_ in ("trellis", "suh", "svh")}),
            "checkpoint_payload_sha256": checkpoint_payload_sha256(stored),
            "logical_payload_bytes": sum(int(r["bytes"]) for r in
                                         refs.values()),
            "param_count": n * k,
            "vector_topology": dict(vector_topology),
            "reconstruction_closure": {
                "schema": "quant-pipeline.exl3-mcg-fp16-closure.v1",
                "dtype": "float16",
                "shape": [n, k],
                "orientation": "huggingface_out_in",
                "payload_sha256": q_tensor_sha256(reconstruction),
                "persisted": False,
                "encoder_full_decode_closure": True,
            },
            "decoder": {
                "codec_family": "exl3-mcg",
                "mcg_multiplier_hex": "0xCBAC1FED",
                "mcg_marker_signed_int32": -877912083,
                "reader_abi_sha256": str(reader_abi_sha256),
            },
            "provenance": copy.deepcopy(dict(provenance)),
        }
        body["choice_sha256"] = sha256_bytes(canonical_json(body))
        path = self.choices / f"{body['choice_sha256']}.json"
        if path.exists():
            if load_json(path) != body:
                die("EXL3/MCG choice hash collision")
        else:
            write_json(path, body)
        return body

    def verify_choice(self, choice) -> dict[str, Any]:
        import torch

        row = (load_json(choice) if isinstance(choice, (str, Path))
               else copy.deepcopy(dict(choice)))
        expected = row.get("choice_sha256")
        unsigned = {k: v for k, v in row.items() if k != "choice_sha256"}
        if (row.get("schema") != DSV4_PACKED_CHOICE_SCHEMA
                or not isinstance(expected, str)
                or _HASH.fullmatch(expected) is None
                or sha256_bytes(canonical_json(unsigned)) != expected
                or row.get("bits") not in PER_TENSOR_ALLOWED_BITS
                or row.get("projection") not in PROJECTIONS):
            die("dsv4 packed-choice seal differs")
        objects = row.get("objects")
        if (not isinstance(objects, Mapping)
                or set(objects) != {"trellis", "suh", "svh", "mcg"}):
            die("dsv4 packed-choice object census differs")
        values = {n: self.objects.load_tensor(r) for n, r in objects.items()}
        if values["trellis"].dtype != torch.int16:
            die("dsv4 packed-choice trellis dtype differs")
        return row


# ---------------------------------------------------------------------------
# Sensitivity DP solver - byte-faithful port at DSV4 census
# ---------------------------------------------------------------------------


def _gain_integer(mass: Decimal, loss_floor: Decimal,
                  loss_upgrade: Decimal) -> int:
    with localcontext() as context:
        context.prec = 50
        scaled = mass * (loss_floor - loss_upgrade) * Decimal(DP_SCORE_SCALE)
        return int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))


def audit_layer_allocation(layer: int, allocation: Mapping[str, int]) -> None:
    """Fail-closed census audit (GLM's k35.audit_layer_allocation at DSV4
    geometry): full 768-name census, integer bits in (3,4), exact 2688-unit
    sum, exactly 384 K4 tensors."""
    expected = _layer_tensor_names(layer)
    if sorted(allocation) != expected:
        die(f"L{layer}: allocation census differs from {len(expected)}")
    k4 = 0
    units = 0
    for name, bits in allocation.items():
        if bits not in PER_TENSOR_ALLOWED_BITS:
            die(f"L{layer} {name}: bits {bits} outside "
                f"{PER_TENSOR_ALLOWED_BITS}")
        k4 += int(bits == 4)
        units += int(bits)
    if units != TARGET_BIT_UNITS_PER_LAYER:
        die(f"L{layer}: unit sum {units} differs from "
            f"{TARGET_BIT_UNITS_PER_LAYER}")
    if k4 != K4_TENSORS_PER_LAYER:
        die(f"L{layer}: K4 count {k4} differs from {K4_TENSORS_PER_LAYER}")


def solve_layer_dp(layer: int, loss_by_bits: Mapping[str, tuple[
        Decimal, Decimal]], mass_by_expert: Sequence[Decimal]
                   ) -> dict[str, int]:
    """Exact 2688-bit-unit allocation with exactly 384 K4 tensors.

    loss_by_bits maps the full master tensor name to (loss at 3, loss at 4)
    as Decimals parsed from .17g probe strings. mass_by_expert is the
    per-expert p2 mass over fit rows.
    """
    ordered_names = _layer_tensor_names(layer)
    if len(ordered_names) != TENSORS_PER_LAYER:
        die(f"layer tensor census differs from {TENSORS_PER_LAYER}")
    if set(loss_by_bits) != set(ordered_names):
        missing = sorted(set(ordered_names) - set(loss_by_bits))
        extra = sorted(set(loss_by_bits) - set(ordered_names))
        die(f"probe loss census differs: {len(missing)} missing "
            f"(first {missing[0] if missing else None}), {len(extra)} extra")
    if len(mass_by_expert) != NUM_EXPERTS:
        die(f"expected {NUM_EXPERTS} expert masses, got {len(mass_by_expert)}")
    curves = []
    for name in ordered_names:
        loss3, loss4 = loss_by_bits[name]
        if loss3 < 0 or loss4 < 0:
            die(f"negative probe loss for {name}")
        expert = int(name.split(".experts.")[1].split(".")[0])
        curves.append((name, mass_by_expert[expert], loss3, loss4))

    budget = K4_TENSORS_PER_LAYER  # 384 upgrades; each upgrade is 1 unit
    negative_infinity = None
    scores: list[int | None] = [0] + [negative_infinity] * budget
    parent_costs: list[list[int]] = []
    parent_bits: list[list[int]] = []

    for _name, mass, loss3, loss4 in curves:
        next_scores: list[int | None] = [negative_infinity] * (budget + 1)
        costs = [-1] * (budget + 1)
        choices = [0] * (budget + 1)
        gains = {
            3: _gain_integer(mass, loss3, loss3),
            4: _gain_integer(mass, loss3, loss4),
        }
        for prior_cost, prior_score in enumerate(scores):
            if prior_score is None:
                continue
            for bits in PER_TENSOR_ALLOWED_BITS:  # low bits first (3, 4)
                extra_units = bits - FLOOR_BITS
                new_cost = prior_cost + extra_units
                if new_cost > budget:
                    continue
                candidate_score = prior_score + gains[bits]
                incumbent = next_scores[new_cost]
                if (incumbent is None
                        or candidate_score > incumbent):
                    next_scores[new_cost] = candidate_score
                    costs[new_cost] = prior_cost
                    choices[new_cost] = bits
        scores = next_scores
        parent_costs.append(costs)
        parent_bits.append(choices)

    final_score = scores[budget]
    if final_score is None:
        die(f"layer {layer}: exact {TARGET_BIT_UNITS_PER_LAYER}-bit-unit "
            "budget is unreachable")
    selected_reversed = []
    cost = budget
    for item_index in range(len(curves) - 1, -1, -1):
        bits = int(parent_bits[item_index][cost])
        prior = int(parent_costs[item_index][cost])
        if bits not in PER_TENSOR_ALLOWED_BITS or prior < 0:
            die(f"layer {layer}: DP backpointer corruption")
        selected_reversed.append(bits)
        cost = prior
    if cost != 0:
        die(f"layer {layer}: allocation did not return to zero budget")
    selected = list(reversed(selected_reversed))
    result = {curves[i][0]: selected[i] for i in range(len(curves))}
    audit_layer_allocation(layer, result)
    return result
