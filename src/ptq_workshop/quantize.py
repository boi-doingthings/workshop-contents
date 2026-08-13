"""Auditable ModelOpt 0.46.0rc0 PTQ command construction and execution."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .artifacts import ArtifactLayout, canonical_json, sha256_file, write_json_atomic
from .config import MODELOPT_EXAMPLE, WorkshopConfig


QUANTIZED_VARIANTS = ("fp8", "nvfp4")
# ModelOpt 0.46.0rc0 serializes this derived Nemotron-H property into the
# unified-HF config.  NemotronHConfig derives it from hybrid_override_pattern
# and exposes no setter, so AutoConfig rejects the otherwise valid export.
# Keep this deliberately minimal: mtp_layers_block_type is accepted by the
# copied config class and must not be discarded.
READ_ONLY_DERIVED_CONFIG_FIELDS = ("layers_block_type",)
CONFIG_NORMALIZATION_MANIFEST = ".config_normalization.json"
QUANTIZATION_METADATA_NORMALIZATION_MANIFEST = ".quantization_metadata_normalization.json"
SOURCE_TOPOLOGY_NORMALIZATION_MANIFEST = ".source_topology_normalization.json"
PINNED_HYBRID_OVERRIDE_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
PINNED_NUM_HIDDEN_LAYERS = len(PINNED_HYBRID_OVERRIDE_PATTERN)
ATTENTION_LAYER_INDICES = (5, 12, 19, 26, 33, 42)
PRECEDING_MAMBA_LAYER_INDICES = (4, 11, 18, 25, 32, 41)
CONTROLLED_IN_MEMORY_MODULES = tuple(
    f"model.layers.{layer}.mixer.{projection}"
    for layer in ATTENTION_LAYER_INDICES
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
) + tuple(
    f"model.layers.{layer}.mixer.{projection}"
    for layer in PRECEDING_MAMBA_LAYER_INDICES
    for projection in ("in_proj", "out_proj")
)
CONTROLLED_RECIPE_EXCLUSION_PATTERNS = tuple(
    f"*{module_name}*" for module_name in CONTROLLED_IN_MEMORY_MODULES
)
CONTROLLED_IN_MEMORY_QUANTIZERS = tuple(
    f"{module_name}.{quantizer_name}"
    for module_name in CONTROLLED_IN_MEMORY_MODULES
    for quantizer_name in ("input_quantizer", "weight_quantizer")
)
# Unified-HF export reverses the Transformers in-memory `model` prefix back to
# the source checkpoint's `backbone` namespace. Verify the tensors themselves;
# `exclude_modules` alone cannot prove their serialized dtype.
CONTROLLED_EXPORT_BF16_WEIGHTS = tuple(
    module_name.replace("model.layers.", "backbone.layers.") + ".weight"
    for module_name in CONTROLLED_IN_MEMORY_MODULES
)
CONTROLLED_EXPORT_FORBIDDEN_SCALE_TENSORS = tuple(
    module_name.replace("model.layers.", "backbone.layers.") + scale_suffix
    for module_name in CONTROLLED_IN_MEMORY_MODULES
    for scale_suffix in (".input_scale", ".weight_scale", ".weight_scale_2", ".weight_scale_inv")
)
MAMBA_LAYER_INDICES = (
    0,
    2,
    4,
    7,
    9,
    11,
    14,
    16,
    18,
    21,
    23,
    25,
    28,
    30,
    32,
    35,
    37,
    39,
    41,
    44,
    46,
    48,
    50,
)
# Exact order published in NVIDIA's Nemotron-3 Nano NVFP4 ModelOpt metadata:
# lm_head, six sensitive Mamba/attention pairs (36 modules), then all 23
# Mamba Conv1d modules.  The headline experiment retains the same exclusions.
CLEAN_QUANTIZATION_EXCLUDE_MODULES = (
    "lm_head",
    *(
        f"backbone.layers.{layer}.mixer.{projection}"
        for mamba_layer, attention_layer in zip(
            PRECEDING_MAMBA_LAYER_INDICES, ATTENTION_LAYER_INDICES
        )
        for layer, projections in (
            (mamba_layer, ("in_proj", "out_proj")),
            (attention_layer, ("q_proj", "k_proj", "v_proj", "o_proj")),
        )
        for projection in projections
    ),
    *(f"backbone.layers.{layer}.mixer.conv1d" for layer in MAMBA_LAYER_INDICES),
)
if len(CLEAN_QUANTIZATION_EXCLUDE_MODULES) != 60:  # pragma: no cover - import invariant
    raise AssertionError("Controlled ModelOpt exclusion policy must contain exactly 60 modules")
POSITIVE_CONTROL_IN_MEMORY_MODULE = "model.layers.2.mixer.in_proj"
POSITIVE_CONTROL_IN_MEMORY_QUANTIZERS = tuple(
    f"{POSITIVE_CONTROL_IN_MEMORY_MODULE}.{quantizer_name}"
    for quantizer_name in ("input_quantizer", "weight_quantizer")
)
POSITIVE_CONTROL_EXPORT_MODULE = "backbone.layers.2.mixer.in_proj"
POSITIVE_CONTROL_SOURCE_WEIGHT = f"{POSITIVE_CONTROL_EXPORT_MODULE}.weight"
POSITIVE_CONTROL_SOURCE_SHAPE = (10304, 2688)


class QuantizationError(RuntimeError):
    """A PTQ process failed or produced an invalid artifact."""


@dataclass(frozen=True)
class QuantizationJob:
    variant: str
    command: tuple[str, ...]
    source_model: Path
    calibration_jsonl: Path
    recipe_path: Path
    output_dir: Path
    log_path: Path
    metadata_path: Path
    environment: Mapping[str, str]

    def manifest(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "command": list(self.command),
            "source_model": str(self.source_model),
            "calibration_jsonl": str(self.calibration_jsonl),
            "calibration_sha256": sha256_file(self.calibration_jsonl),
            "recipe_path": str(self.recipe_path),
            "recipe_sha256": sha256_file(self.recipe_path),
            "output_dir": str(self.output_dir),
            "log_path": str(self.log_path),
            "environment_overrides": dict(sorted(self.environment.items())),
        }


@dataclass(frozen=True)
class QuantizationResult:
    variant: str
    returncode: int
    output_dir: Path
    log_path: Path
    metadata_path: Path


def _json_value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _checkpoint_tensor_metadata_sha256(export: Path) -> str:
    index_path = export / "model.safetensors.index.json"
    if index_path.is_file():
        return sha256_file(index_path)
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise QuantizationError(
            "safetensors is required to fingerprint checkpoint tensor metadata"
        ) from exc
    metadata: dict[str, dict[str, Any]] = {}
    shards = sorted(export.glob("*.safetensors"))
    if not shards:
        raise QuantizationError(f"No safetensors shards found: {export}")
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in metadata:
                    raise QuantizationError(f"Duplicate tensor {key} across safetensors shards")
                tensor_slice = handle.get_slice(key)
                metadata[key] = {
                    "dtype": str(tensor_slice.get_dtype()),
                    "shape": list(tensor_slice.get_shape()),
                    "shard": shard.name,
                }
    return _json_value_sha256(metadata)


def _validated_source_topology_audit(
    audit_path: Path, *, config_path: Path
) -> dict[str, Any]:
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise QuantizationError(f"Invalid source-topology normalization audit: {audit_path}") from exc
    if (
        not isinstance(audit, dict)
        or audit.get("schema_version") != 1
        or audit.get("operation") != "restore_pinned_source_topology"
    ):
        raise QuantizationError(f"Invalid source-topology normalization audit: {audit_path}")
    if audit.get("config_after_sha256") != sha256_file(config_path):
        raise QuantizationError(f"Config changed after source-topology audit: {config_path}")
    before_sha256 = audit.get("config_before_sha256")
    if not isinstance(before_sha256, str) or before_sha256 == audit.get("config_after_sha256"):
        raise QuantizationError(
            f"Source-topology audit lacks a distinct pre-normalization config: {audit_path}"
        )
    restored = audit.get("restored_fields")
    expected = {
        "hybrid_override_pattern": PINNED_HYBRID_OVERRIDE_PATTERN,
        "num_hidden_layers": PINNED_NUM_HIDDEN_LAYERS,
    }
    if not isinstance(restored, dict) or set(restored) != set(expected):
        raise QuantizationError(f"Source-topology audit restored-field set is invalid: {audit_path}")
    for field, value in expected.items():
        record = restored[field]
        if (
            not isinstance(record, dict)
            or record.get("value") != value
            or record.get("value_sha256") != _json_value_sha256(value)
            or config.get(field) != value
        ):
            raise QuantizationError(
                f"Source-topology audit value mismatch for {field}: {audit_path}"
            )
    assert_controlled_topology(config, location=config_path)
    source_path = Path(str(audit.get("source_config_path", "")))
    if not source_path.is_file() or audit.get("source_config_sha256") != sha256_file(source_path):
        raise QuantizationError(f"Pinned source config changed after topology repair: {audit_path}")
    source_config = json.loads(source_path.read_text(encoding="utf-8"))
    assert_controlled_topology(source_config, location=source_path)
    if any(source_config.get(field) != value for field, value in expected.items()):
        raise QuantizationError(f"Topology audit no longer matches its pinned source: {audit_path}")
    tensor_metadata_sha256 = _checkpoint_tensor_metadata_sha256(config_path.parent)
    if (
        audit.get("tensor_metadata_before_sha256") != tensor_metadata_sha256
        or audit.get("tensor_metadata_after_sha256") != tensor_metadata_sha256
    ):
        raise QuantizationError(
            f"Checkpoint tensor metadata changed across source-topology repair: {audit_path}"
        )
    return audit


def _validated_quantization_metadata_audit(
    audit_path: Path, *, config_path: Path, hf_quant_config_path: Path
) -> dict[str, Any]:
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        hf_quant_config = json.loads(hf_quant_config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise QuantizationError(
            f"Invalid quantization-metadata normalization audit: {audit_path}"
        ) from exc
    if (
        not isinstance(audit, dict)
        or audit.get("schema_version") != 1
        or audit.get("operation") != "normalize_modelopt_quantization_metadata"
    ):
        raise QuantizationError(
            f"Invalid quantization-metadata normalization audit: {audit_path}"
        )
    current_config_sha256 = sha256_file(config_path)
    if audit.get("config_after_sha256") != current_config_sha256:
        topology_path = config_path.parent / SOURCE_TOPOLOGY_NORMALIZATION_MANIFEST
        if not topology_path.is_file():
            raise QuantizationError(
                f"Config changed after quantization metadata audit: {config_path}"
            )
        topology_audit = _validated_source_topology_audit(
            topology_path, config_path=config_path
        )
        if topology_audit.get("config_before_sha256") != audit.get("config_after_sha256"):
            raise QuantizationError(
                f"Source-topology audit is not chained to quantization metadata: {audit_path}"
            )
    if audit.get("hf_quant_config_after_sha256") != sha256_file(hf_quant_config_path):
        raise QuantizationError(
            f"hf_quant_config changed after quantization metadata audit: {hf_quant_config_path}"
        )
    embedded = config.get("quantization_config") if isinstance(config, dict) else None
    hf_quantization = (
        hf_quant_config.get("quantization") if isinstance(hf_quant_config, dict) else None
    )
    if not isinstance(embedded, dict) or not isinstance(hf_quantization, dict):
        raise QuantizationError(f"Normalized quantization metadata is malformed: {audit_path}")
    expected = list(CLEAN_QUANTIZATION_EXCLUDE_MODULES)
    if embedded.get("ignore") != expected or hf_quantization.get("exclude_modules") != expected:
        raise QuantizationError(
            f"Normalized quantization exclusions differ from the exact 60-module policy: {audit_path}"
        )
    if any("\x00" in item for item in expected) or hf_quantization.get("torch_dtype") != "bfloat16":
        raise QuantizationError(f"Normalized ModelOpt runtime metadata is unsafe: {audit_path}")
    clean_sha256 = _json_value_sha256(expected)
    if (
        audit.get("clean_exclusion_count") != len(expected)
        or audit.get("clean_exclusion_sha256") != clean_sha256
        or audit.get("tensor_backed_exclusion_count") != len(expected)
    ):
        raise QuantizationError(f"Quantization metadata audit policy proof mismatch: {audit_path}")
    for before_key, after_key in (
        ("config_before_sha256", "config_after_sha256"),
        ("hf_quant_config_before_sha256", "hf_quant_config_after_sha256"),
        ("embedded_quantization_before_sha256", "embedded_quantization_after_sha256"),
    ):
        before_hash = audit.get(before_key)
        after_hash = audit.get(after_key)
        if not isinstance(before_hash, str) or not isinstance(after_hash, str) or before_hash == after_hash:
            raise QuantizationError(
                f"Quantization metadata audit lacks distinct {before_key} evidence: {audit_path}"
            )
    if audit.get("embedded_quantization_after_sha256") != _json_value_sha256(embedded):
        raise QuantizationError(f"Embedded quantization metadata hash mismatch: {audit_path}")
    replaced = audit.get("replaced_lists")
    required_paths = {
        "config.quantization_config.ignore",
        "hf_quant_config.quantization.exclude_modules",
    }
    if not isinstance(replaced, dict) or set(replaced) != required_paths:
        raise QuantizationError(f"Quantization metadata audit lacks replaced-list evidence: {audit_path}")
    for field, record in replaced.items():
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("before"), list)
            or record.get("before_sha256") != _json_value_sha256(record["before"])
            or record.get("after_sha256") != clean_sha256
        ):
            raise QuantizationError(
                f"Quantization metadata audit list hash mismatch for {field}: {audit_path}"
            )
    dtype_record = audit.get("torch_dtype")
    if not isinstance(dtype_record, dict) or dtype_record.get("after") != "bfloat16":
        raise QuantizationError(f"Quantization metadata audit lacks BF16 dtype evidence: {audit_path}")
    return audit


def _validated_normalization_audit(
    audit_path: Path,
    *,
    config_path: Path,
    quantization_config: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise QuantizationError(
            f"Invalid config-normalization audit manifest: {audit_path}"
        ) from exc
    if (
        not isinstance(audit, dict)
        or audit.get("schema_version") != 1
        or audit.get("operation") != "remove_read_only_derived_config_fields"
    ):
        raise QuantizationError(
            f"Invalid config-normalization audit manifest: {audit_path}"
        )
    current_sha256 = sha256_file(config_path)
    chained_quantization_audit: dict[str, Any] | None = None
    if audit.get("after_sha256") != current_sha256:
        quantization_audit_path = config_path.parent / QUANTIZATION_METADATA_NORMALIZATION_MANIFEST
        hf_quant_config_path = config_path.parent / "hf_quant_config.json"
        if not quantization_audit_path.is_file() or not hf_quant_config_path.is_file():
            raise QuantizationError(f"Config changed after normalization audit: {config_path}")
        chained_quantization_audit = _validated_quantization_metadata_audit(
            quantization_audit_path,
            config_path=config_path,
            hf_quant_config_path=hf_quant_config_path,
        )
        if chained_quantization_audit.get("config_before_sha256") != audit.get("after_sha256"):
            raise QuantizationError(
                f"Quantization metadata audit is not chained to config normalization: {config_path}"
            )
    before_sha256 = audit.get("before_sha256")
    if not isinstance(before_sha256, str) or before_sha256 == current_sha256:
        raise QuantizationError(
            f"Config-normalization audit lacks distinct original config evidence: {audit_path}"
        )
    if chained_quantization_audit is None:
        if audit.get("quantization_config_sha256") != _json_value_sha256(quantization_config):
            raise QuantizationError(
                f"Embedded quantization_config changed after normalization: {config_path.parent}"
            )
    elif chained_quantization_audit.get("embedded_quantization_before_sha256") != audit.get(
        "quantization_config_sha256"
    ):
        raise QuantizationError(
            f"Quantization metadata audit does not preserve embedded config lineage: {config_path}"
        )
    removed_fields = audit.get("removed_fields")
    if not isinstance(removed_fields, dict) or set(removed_fields) != set(
        READ_ONLY_DERIVED_CONFIG_FIELDS
    ):
        raise QuantizationError(
            "Config-normalization audit must prove removal of exactly "
            f"{list(READ_ONLY_DERIVED_CONFIG_FIELDS)}: {audit_path}"
        )
    for field in READ_ONLY_DERIVED_CONFIG_FIELDS:
        record = removed_fields[field]
        if not isinstance(record, dict) or not record.get("value"):
            raise QuantizationError(
                f"Config-normalization audit lacks original value for {field}: {audit_path}"
            )
        if record.get("value_sha256") != _json_value_sha256(record["value"]):
            raise QuantizationError(
                f"Config-normalization removed-value hash mismatch for {field}: {audit_path}"
            )
    return audit


def normalize_export_config(
    path: Path | str, *, manifest_path: Path | str | None = None
) -> dict[str, Any]:
    """Atomically remove ModelOpt's read-only Nemotron-H derived property.

    The source structural inputs and every quantization/export field remain
    byte-for-byte equivalent at the JSON-value level.  A checkpoint-local
    manifest records the config hashes before and after the one-field repair.
    """

    export = Path(path)
    config_path = export / "config.json"
    if not config_path.is_file():
        raise QuantizationError(f"Export config is missing: {config_path}")
    try:
        original = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise QuantizationError(f"Export config is not valid JSON: {config_path}") from exc
    if not isinstance(original, dict):
        raise QuantizationError(f"Export config must be a JSON object: {config_path}")
    quantization_config = original.get("quantization_config")
    if not isinstance(quantization_config, dict) or not quantization_config:
        raise QuantizationError(
            f"Export config lacks embedded quantization_config metadata: {config_path}"
        )

    before_sha256 = sha256_file(config_path)
    quantization_sha256 = _json_value_sha256(quantization_config)
    audit_path = (
        export / CONFIG_NORMALIZATION_MANIFEST
        if manifest_path is None
        else Path(manifest_path)
    )
    normalized = dict(original)
    removed = {
        field: normalized.pop(field)
        for field in READ_ONLY_DERIVED_CONFIG_FIELDS
        if field in normalized
    }
    if not removed:
        if not audit_path.is_file():
            raise QuantizationError(
                "Export config is already normalized but lacks original removal evidence: "
                f"{audit_path}"
            )
        audit = _validated_normalization_audit(
            audit_path,
            config_path=config_path,
            quantization_config=quantization_config,
        )
        return {**audit, "manifest_path": str(audit_path.resolve())}
    if audit_path.exists():
        raise QuantizationError(
            f"Refusing to replace existing config-normalization audit: {audit_path}"
        )
    write_json_atomic(config_path, normalized, overwrite=True)

    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    remaining = sorted(set(READ_ONLY_DERIVED_CONFIG_FIELDS).intersection(persisted))
    if remaining:
        raise QuantizationError(
            f"Read-only derived config fields remain after normalization: {remaining}"
        )
    if _json_value_sha256(persisted.get("quantization_config")) != quantization_sha256:
        raise QuantizationError("Export config normalization changed quantization_config")
    expected = {
        key: value
        for key, value in original.items()
        if key not in READ_ONLY_DERIVED_CONFIG_FIELDS
    }
    if persisted != expected:
        raise QuantizationError(
            "Export config normalization changed fields other than layers_block_type"
        )

    manifest = {
        "schema_version": 1,
        "operation": "remove_read_only_derived_config_fields",
        "config_path": str(config_path.resolve()),
        "before_sha256": before_sha256,
        "after_sha256": sha256_file(config_path),
        "removed_fields": {
            field: {
                "value": value,
                "value_sha256": _json_value_sha256(value),
            }
            for field, value in sorted(removed.items())
        },
        "quantization_config_sha256": quantization_sha256,
    }
    write_json_atomic(audit_path, manifest)
    return {**manifest, "manifest_path": str(audit_path.resolve())}


def _checkpoint_tensor_names(export: Path) -> set[str]:
    index_path = export / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise QuantizationError(f"Invalid safetensors index: {index_path}") from exc
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            raise QuantizationError(f"Safetensors index lacks weight_map: {index_path}")
        return set(weight_map)
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise QuantizationError(
            "safetensors is required to prove quantization exclusions are tensor-backed"
        ) from exc
    names: set[str] = set()
    shards = sorted(export.glob("*.safetensors"))
    if not shards:
        raise QuantizationError(f"No safetensors shards found: {export}")
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            names.update(handle.keys())
    return names


def normalize_quantization_metadata(path: Path | str) -> dict[str, Any]:
    """Repair ModelOpt sentinel-corrupted exclusion metadata with an audit chain."""

    export = Path(path)
    config_path = export / "config.json"
    hf_path = export / "hf_quant_config.json"
    audit_path = export / QUANTIZATION_METADATA_NORMALIZATION_MANIFEST
    if not config_path.is_file() or not hf_path.is_file():
        raise QuantizationError(f"Quantized export metadata is incomplete: {export}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        hf_config = json.loads(hf_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise QuantizationError(f"Quantized export metadata is invalid JSON: {export}") from exc
    embedded = config.get("quantization_config") if isinstance(config, dict) else None
    hf_quantization = hf_config.get("quantization") if isinstance(hf_config, dict) else None
    if not isinstance(embedded, dict) or not isinstance(hf_quantization, dict):
        raise QuantizationError(f"Quantized export lacks ModelOpt metadata objects: {export}")

    expected = list(CLEAN_QUANTIZATION_EXCLUDE_MODULES)
    already_normalized = (
        embedded.get("ignore") == expected
        and hf_quantization.get("exclude_modules") == expected
        and hf_quantization.get("torch_dtype") == "bfloat16"
    )
    if already_normalized:
        if not audit_path.is_file():
            raise QuantizationError(
                f"Quantization metadata is normalized but lacks original audit evidence: {audit_path}"
            )
        audit = _validated_quantization_metadata_audit(
            audit_path, config_path=config_path, hf_quant_config_path=hf_path
        )
        return {**audit, "manifest_path": str(audit_path.resolve())}
    if audit_path.exists():
        raise QuantizationError(
            f"Refusing to replace inconsistent quantization metadata audit: {audit_path}"
        )

    old_ignore = embedded.get("ignore")
    old_excludes = hf_quantization.get("exclude_modules")
    if not isinstance(old_ignore, list) or not isinstance(old_excludes, list):
        raise QuantizationError(f"ModelOpt exclusion metadata is not a list: {export}")
    if not any("\x00" in str(item) for item in (*old_ignore, *old_excludes)):
        raise QuantizationError(
            f"Refusing to rewrite exclusion metadata without the known NUL sentinel defect: {export}"
        )

    tensor_names = _checkpoint_tensor_names(export)
    missing_tensors = sorted(
        f"{module}.weight"
        for module in CLEAN_QUANTIZATION_EXCLUDE_MODULES
        if f"{module}.weight" not in tensor_names
    )
    if missing_tensors:
        raise QuantizationError(
            "Clean exclusion policy is not backed by serialized tensors: "
            f"{missing_tensors}"
        )

    config_before_sha256 = sha256_file(config_path)
    hf_before_sha256 = sha256_file(hf_path)
    embedded_before_sha256 = _json_value_sha256(embedded)
    normalized_config = deepcopy(config)
    normalized_hf = deepcopy(hf_config)
    normalized_config["quantization_config"]["ignore"] = expected
    normalized_hf["quantization"]["exclude_modules"] = expected
    previous_torch_dtype = normalized_hf["quantization"].get("torch_dtype")
    normalized_hf["quantization"]["torch_dtype"] = "bfloat16"
    write_json_atomic(config_path, normalized_config, overwrite=True)
    write_json_atomic(hf_path, normalized_hf, overwrite=True)

    clean_sha256 = _json_value_sha256(expected)
    audit = {
        "schema_version": 1,
        "operation": "normalize_modelopt_quantization_metadata",
        "config_path": str(config_path.resolve()),
        "hf_quant_config_path": str(hf_path.resolve()),
        "config_before_sha256": config_before_sha256,
        "config_after_sha256": sha256_file(config_path),
        "hf_quant_config_before_sha256": hf_before_sha256,
        "hf_quant_config_after_sha256": sha256_file(hf_path),
        "embedded_quantization_before_sha256": embedded_before_sha256,
        "embedded_quantization_after_sha256": _json_value_sha256(
            normalized_config["quantization_config"]
        ),
        "clean_exclusion_count": len(expected),
        "clean_exclusion_sha256": clean_sha256,
        "tensor_backed_exclusion_count": len(expected),
        "replaced_lists": {
            "config.quantization_config.ignore": {
                "before": old_ignore,
                "before_sha256": _json_value_sha256(old_ignore),
                "after_sha256": clean_sha256,
            },
            "hf_quant_config.quantization.exclude_modules": {
                "before": old_excludes,
                "before_sha256": _json_value_sha256(old_excludes),
                "after_sha256": clean_sha256,
            },
        },
        "torch_dtype": {"before": previous_torch_dtype, "after": "bfloat16"},
    }
    write_json_atomic(audit_path, audit)
    return {**audit, "manifest_path": str(audit_path.resolve())}


def normalize_source_topology(
    path: Path | str, *, source_model: Path | str
) -> dict[str, Any]:
    """Restore only the two source topology fields omitted by unified-HF export."""

    export = Path(path)
    config_path = export / "config.json"
    source_config_path = Path(source_model) / "config.json"
    audit_path = export / SOURCE_TOPOLOGY_NORMALIZATION_MANIFEST
    quantization_audit_path = export / QUANTIZATION_METADATA_NORMALIZATION_MANIFEST
    hf_path = export / "hf_quant_config.json"
    if not config_path.is_file() or not source_config_path.is_file():
        raise QuantizationError(
            f"Topology normalization requires export and pinned source configs: {export}"
        )
    if not quantization_audit_path.is_file() or not hf_path.is_file():
        raise QuantizationError(
            f"Topology normalization requires prior quantization-metadata audit: {export}"
        )
    quantization_audit = _validated_quantization_metadata_audit(
        quantization_audit_path,
        config_path=config_path,
        hf_quant_config_path=hf_path,
    )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise QuantizationError(f"Invalid config JSON for topology normalization: {export}") from exc
    if not isinstance(config, dict) or not isinstance(source_config, dict):
        raise QuantizationError(f"Topology configs must be JSON objects: {export}")
    assert_controlled_topology(source_config, location=source_config_path)
    restored_values = {
        "hybrid_override_pattern": source_config["hybrid_override_pattern"],
        "num_hidden_layers": source_config["num_hidden_layers"],
    }
    present = {field for field in restored_values if field in config}
    if present:
        if present != set(restored_values) or any(
            config.get(field) != value for field, value in restored_values.items()
        ):
            raise QuantizationError(
                f"Export contains partial or conflicting source topology fields: {config_path}"
            )
        if not audit_path.is_file():
            raise QuantizationError(
                f"Export topology is restored but lacks original audit evidence: {audit_path}"
            )
        audit = _validated_source_topology_audit(audit_path, config_path=config_path)
        return {**audit, "manifest_path": str(audit_path.resolve())}
    if audit_path.exists():
        raise QuantizationError(f"Refusing to replace inconsistent topology audit: {audit_path}")

    config_before_sha256 = sha256_file(config_path)
    if config_before_sha256 != quantization_audit.get("config_after_sha256"):
        raise QuantizationError(
            f"Topology input is not the audited quantization-metadata output: {config_path}"
        )
    tensor_metadata_sha256 = _checkpoint_tensor_metadata_sha256(export)
    normalized = dict(config)
    normalized.update(restored_values)
    write_json_atomic(config_path, normalized, overwrite=True)
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert_controlled_topology(persisted, location=config_path)
    if {
        key: value for key, value in persisted.items() if key not in restored_values
    } != config:
        raise QuantizationError("Topology normalization changed fields beyond the two source keys")
    if _checkpoint_tensor_metadata_sha256(export) != tensor_metadata_sha256:
        raise QuantizationError("Topology normalization changed checkpoint tensor metadata")

    audit = {
        "schema_version": 1,
        "operation": "restore_pinned_source_topology",
        "config_path": str(config_path.resolve()),
        "config_before_sha256": config_before_sha256,
        "config_after_sha256": sha256_file(config_path),
        "source_config_path": str(source_config_path.resolve()),
        "source_config_sha256": sha256_file(source_config_path),
        "restored_fields": {
            field: {"value": value, "value_sha256": _json_value_sha256(value)}
            for field, value in restored_values.items()
        },
        "tensor_metadata_before_sha256": tensor_metadata_sha256,
        "tensor_metadata_after_sha256": tensor_metadata_sha256,
    }
    write_json_atomic(audit_path, audit)
    return {**audit, "manifest_path": str(audit_path.resolve())}


def load_recipe(path: Path | str) -> dict[str, Any]:
    recipe_path = Path(path)
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Recipe does not exist: {recipe_path}")
    data = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("quantize"), dict):
        raise ValueError(f"Invalid ModelOpt PTQ recipe: {recipe_path}")
    return data


def disabled_quantizers(recipe: Mapping[str, Any]) -> tuple[tuple[str | None, str, bool], ...]:
    entries = recipe.get("quantize", {}).get("quant_cfg", [])
    if not isinstance(entries, list):
        raise ValueError("recipe quantize.quant_cfg must be a list")
    result = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("enable") is False:
            result.append((entry.get("parent_class"), str(entry.get("quantizer_name")), False))
    return tuple(result)


def assert_controlled_recipes(fp8_path: Path | str, nvfp4_path: Path | str) -> None:
    """Enforce identical exclusions and absence of KV-cache quantization."""

    recipes = [load_recipe(fp8_path), load_recipe(nvfp4_path)]
    if disabled_quantizers(recipes[0]) != disabled_quantizers(recipes[1]):
        raise ValueError("FP8 and NVFP4 recipes must use an identical ordered exclusion list")
    for path, recipe in zip((fp8_path, nvfp4_path), recipes):
        entries = recipe["quantize"].get("quant_cfg", [])
        disabled_names = {
            str(entry.get("quantizer_name"))
            for entry in entries
            if isinstance(entry, dict) and entry.get("enable") is False
        }
        missing = set(CONTROLLED_RECIPE_EXCLUSION_PATTERNS) - disabled_names
        if missing:
            raise ValueError(
                f"Recipe lacks exact in-memory Nemotron high-precision exclusions: "
                f"{path}: {sorted(missing)}"
            )
        legacy_names = sorted(name for name in disabled_names if "backbone.layers." in name)
        if legacy_names:
            raise ValueError(
                f"Recipe uses checkpoint names that cannot match in-memory quantizers: "
                f"{path}: {legacy_names}"
            )
        encoded = json.dumps(entries).lower()
        if "bmm_quantizer" in encoded or "kv_cache" in encoded:
            raise ValueError(f"Headline recipe must keep the KV cache BF16: {path}")


def assert_controlled_topology(
    model_config: Mapping[str, Any], *, location: Path | str
) -> None:
    """Prove that the pinned high-precision layer indices still describe this model."""

    pattern = model_config.get("hybrid_override_pattern")
    num_layers = model_config.get("num_hidden_layers")
    if (
        pattern != PINNED_HYBRID_OVERRIDE_PATTERN
        or num_layers != PINNED_NUM_HIDDEN_LAYERS
    ):
        raise QuantizationError(
            "Nemotron hybrid topology differs from the pinned source schedule: "
            f"hybrid_override_pattern={pattern!r}, num_hidden_layers={num_layers!r}: {location}"
        )
    attention_layers = tuple(index for index, layer_type in enumerate(pattern) if layer_type == "*")
    preceding_mamba_layers = tuple(index - 1 for index in attention_layers)
    preceding_types = tuple(pattern[index] for index in preceding_mamba_layers)
    if (
        attention_layers != ATTENTION_LAYER_INDICES
        or preceding_mamba_layers != PRECEDING_MAMBA_LAYER_INDICES
        or preceding_types != ("M",) * len(PRECEDING_MAMBA_LAYER_INDICES)
    ):
        raise QuantizationError(
            "Nemotron hybrid topology no longer matches the controlled attention/preceding-Mamba "
            f"policy: attention={attention_layers}, preceding={preceding_mamba_layers}, "
            f"preceding_types={preceding_types}: {location}"
        )


def _positive_summary_variant(states: Mapping[str, str]) -> str:
    positive_states = [states[name] for name in POSITIVE_CONTROL_IN_MEMORY_QUANTIZERS]
    formats = {
        "fp8": ("(4, 3) bit", "amax=", " quant)"),
        "nvfp4": ("(2, 1) bit", "block_sizes={-1: 16", "amax=", " quant)"),
    }
    matches = [
        variant
        for variant, tokens in formats.items()
        if all(all(token in state for token in tokens) for state in positive_states)
    ]
    if len(matches) != 1:
        raise QuantizationError(
            "Layer-2 Mamba positive-control quantizers do not prove one calibrated FP8/NVFP4 "
            f"format: {dict(zip(POSITIVE_CONTROL_IN_MEMORY_QUANTIZERS, positive_states))}"
        )
    return matches[0]


def validate_quant_summary(
    path: Path | str, *, expected_variant: str | None = None
) -> dict[str, Any]:
    """Prove exact controlled exclusions and a live layer-2 Mamba quantized control."""

    summary = Path(path)
    if not summary.is_file():
        raise QuantizationError(f"ModelOpt quantizer summary is missing: {summary}")
    states: dict[str, str] = {}
    for line in summary.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            states[fields[0]] = fields[1]
    missing = sorted(set(CONTROLLED_IN_MEMORY_QUANTIZERS) - states.keys())
    if missing:
        raise QuantizationError(
            f"Quantizer summary lacks controlled Nemotron quantizers: {missing}"
        )
    enabled = sorted(
        name for name in CONTROLLED_IN_MEMORY_QUANTIZERS if "(disabled)" not in states[name]
    )
    if enabled:
        raise QuantizationError(
            f"Controlled attention/Mamba quantizers are not disabled: {enabled}"
        )
    missing_positive = sorted(set(POSITIVE_CONTROL_IN_MEMORY_QUANTIZERS) - states.keys())
    if missing_positive:
        raise QuantizationError(
            f"Quantizer summary lacks layer-2 Mamba positive controls: {missing_positive}"
        )
    positive_variant = _positive_summary_variant(states)
    if expected_variant is not None:
        if expected_variant not in QUANTIZED_VARIANTS:
            raise ValueError(f"Unknown expected variant: {expected_variant}")
        if positive_variant != expected_variant:
            raise QuantizationError(
                "Layer-2 Mamba positive-control format mismatch: "
                f"summary={positive_variant}, expected={expected_variant}: {summary}"
            )
    return {
        "path": str(summary.resolve()),
        "sha256": sha256_file(summary),
        "controlled_disabled_quantizer_count": len(CONTROLLED_IN_MEMORY_QUANTIZERS),
        "positive_control_quantizer_count": len(POSITIVE_CONTROL_IN_MEMORY_QUANTIZERS),
        "positive_control_variant": positive_variant,
    }


def validate_source_snapshot(source_model: Path | str, config: WorkshopConfig) -> Path:
    """Require a local snapshot whose config identifies the pinned source revision."""

    source = Path(source_model).expanduser().resolve()
    required = (source / "config.json", source / "model.safetensors.index.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Pinned BF16 source snapshot is incomplete: {', '.join(missing)}")

    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    assert_controlled_topology(source_config, location=source / "config.json")

    # snapshot_download returns .../snapshots/<commit>.  Resolve symlinks first, then
    # require the immutable Hub commit in the path; arbitrary model directories are
    # deliberately rejected because hf_ptq.py cannot apply a separate revision flag.
    if config.model_revision not in source.parts:
        raise ValueError(
            "source_model must be the pinned Hugging Face snapshot directory containing "
            f"revision {config.model_revision}; got {source}"
        )
    return source


def validate_calibration_jsonl(path: Path | str, expected_samples: int) -> Path:
    calibration = Path(path).expanduser().resolve()
    if not calibration.is_file():
        raise FileNotFoundError(f"Frozen calibration JSONL is missing: {calibration}")
    rows = 0
    with calibration.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {calibration}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"]:
                raise ValueError(f"Calibration row {line_number} must contain nonempty text")
            rows += 1
    if rows != expected_samples:
        raise ValueError(f"Calibration JSONL has {rows} rows; expected exactly {expected_samples}")
    return calibration


def build_quantization_job(
    config: WorkshopConfig,
    layout: ArtifactLayout,
    *,
    variant: str,
    source_model: Path | str,
    calibration_jsonl: Path | str,
    modelopt_root: Path | str,
    python_executable: Path | str = sys.executable,
    environment: Mapping[str, str] | None = None,
) -> QuantizationJob:
    if variant not in QUANTIZED_VARIANTS:
        raise ValueError(f"variant must be one of {QUANTIZED_VARIANTS}, got {variant!r}")
    source = validate_source_snapshot(source_model, config)
    calibration = validate_calibration_jsonl(
        calibration_jsonl, config.profile.calibration_samples
    )
    modelopt_script = Path(modelopt_root).expanduser().resolve() / MODELOPT_EXAMPLE
    if not modelopt_script.is_file():
        raise FileNotFoundError(f"Pinned ModelOpt example is missing: {modelopt_script}")
    fp8_recipe = config.recipe_dir / "fp8.yaml"
    nvfp4_recipe = config.recipe_dir / "nvfp4.yaml"
    assert_controlled_recipes(fp8_recipe, nvfp4_recipe)
    recipe = config.recipe_dir / f"{variant}.yaml"
    output = layout.checkpoint_dir(variant)
    log = layout.log_path(variant)
    metadata = layout.run_dir / "manifests" / f"{variant}-quantize.json"
    command = (
        str(python_executable),
        str(modelopt_script),
        "--pyt_ckpt_path",
        str(source),
        "--recipe",
        str(recipe),
        "--dataset",
        str(calibration),
        "--calib_size",
        str(config.profile.calibration_samples),
        "--calib_seq",
        str(config.profile.calibration_sequence_length),
        "--batch_size",
        str(config.profile.calibration_batch_size),
        "--export_path",
        str(output),
        "--trust_remote_code",
        "--skip_generate",
        "--verbose",
    )
    return QuantizationJob(
        variant=variant,
        command=command,
        source_model=source,
        calibration_jsonl=calibration,
        recipe_path=recipe,
        output_dir=output,
        log_path=log,
        metadata_path=metadata,
        environment={} if environment is None else dict(environment),
    )


def _selected_safetensor_metadata(
    directory: Path,
    keys: Sequence[str],
    *,
    safe_open: Any,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Read dtype/shape metadata for selected tensors without materializing them."""

    selected = set(keys)
    metadata: dict[str, tuple[str, tuple[int, ...]]] = {}
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise QuantizationError(f"No safetensors shards found: {directory}")
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in selected.intersection(handle.keys()):
                if key in metadata:
                    raise QuantizationError(f"Duplicate tensor {key} across safetensors shards")
                tensor_slice = handle.get_slice(key)
                metadata[key] = (
                    str(tensor_slice.get_dtype()),
                    tuple(tensor_slice.get_shape()),
                )
    missing = sorted(selected - metadata.keys())
    if missing:
        raise QuantizationError(f"Safetensors checkpoint {directory} lacks tensors: {missing}")
    return metadata


def validate_export(
    path: Path | str,
    *,
    expected_variant: str | None = None,
    source_model: Path | str | None = None,
) -> dict[str, Any]:
    """Validate packed format, exact exclusions, topology, lineage, and BF16-KV policy."""

    export = Path(path)
    required = ("config.json", "hf_quant_config.json")
    missing = [name for name in required if not (export / name).is_file()]
    shard_paths = sorted(export.glob("*.safetensors"))
    if missing or not shard_paths:
        detail = [*(f"missing {name}" for name in missing)]
        if not shard_paths:
            detail.append("no .safetensors files")
        raise QuantizationError(f"Invalid exported checkpoint {export}: {', '.join(detail)}")
    export_config = json.loads((export / "config.json").read_text(encoding="utf-8"))
    read_only_fields = sorted(
        set(READ_ONLY_DERIVED_CONFIG_FIELDS).intersection(export_config)
    )
    if read_only_fields:
        raise QuantizationError(
            "Export config contains read-only Nemotron-H derived properties that break "
            f"AutoConfig: {read_only_fields}: {export}"
        )
    embedded_quant_config = export_config.get("quantization_config")
    if not isinstance(embedded_quant_config, dict) or not embedded_quant_config:
        raise QuantizationError(
            f"Export config lacks embedded quantization_config metadata: {export}"
        )
    normalization_path = export / CONFIG_NORMALIZATION_MANIFEST
    if not normalization_path.is_file():
        raise QuantizationError(
            f"Export lacks config-normalization audit manifest: {normalization_path}"
        )
    normalization = _validated_normalization_audit(
        normalization_path,
        config_path=export / "config.json",
        quantization_config=embedded_quant_config,
    )
    quantization_metadata_path = export / QUANTIZATION_METADATA_NORMALIZATION_MANIFEST
    if not quantization_metadata_path.is_file():
        raise QuantizationError(
            f"Export lacks quantization-metadata normalization audit: {quantization_metadata_path}"
        )
    quantization_metadata = _validated_quantization_metadata_audit(
        quantization_metadata_path,
        config_path=export / "config.json",
        hf_quant_config_path=export / "hf_quant_config.json",
    )
    source_topology_path = export / SOURCE_TOPOLOGY_NORMALIZATION_MANIFEST
    if not source_topology_path.is_file():
        raise QuantizationError(
            f"Export lacks source-topology normalization audit: {source_topology_path}"
        )
    source_topology = _validated_source_topology_audit(
        source_topology_path, config_path=export / "config.json"
    )
    serialized_names = _checkpoint_tensor_names(export)
    unbacked_exclusions = sorted(
        f"{module}.weight"
        for module in CLEAN_QUANTIZATION_EXCLUDE_MODULES
        if f"{module}.weight" not in serialized_names
    )
    if unbacked_exclusions:
        raise QuantizationError(
            "Normalized exclusion policy is no longer tensor-backed: "
            f"{unbacked_exclusions}"
        )
    export_topology_keys = ("hybrid_override_pattern", "num_hidden_layers")
    if any(key in export_config for key in export_topology_keys):
        assert_controlled_topology(export_config, location=export / "config.json")
    elif source_model is None:
        raise QuantizationError(
            "Normalized Nemotron-H export omits raw topology inputs; source_model is required "
            f"to validate controlled topology and lineage: {export}"
        )
    quant_config = json.loads((export / "hf_quant_config.json").read_text(encoding="utf-8"))
    encoded = json.dumps(quant_config, sort_keys=True).upper()
    if '"KV_CACHE_QUANT_ALGO": "FP8"' in encoded or '"KV_CACHE_QUANT_ALGO": "NVFP4"' in encoded:
        raise QuantizationError(f"Export unexpectedly quantized the KV cache: {export}")
    algorithm = str(quant_config.get("quantization", {}).get("quant_algo", "")).upper()
    algorithm_variants = {"FP8": "fp8", "NVFP4": "nvfp4"}
    declared_variant = algorithm_variants.get(algorithm)
    if expected_variant is not None:
        if expected_variant not in QUANTIZED_VARIANTS:
            raise ValueError(f"Unknown expected variant: {expected_variant}")
        if declared_variant != expected_variant:
            raise QuantizationError(
                f"Export format is {algorithm or '<missing>'}, expected "
                f"{expected_variant.upper()}: {export}"
            )
    validation_variant = expected_variant or declared_variant
    quant_summary = validate_quant_summary(
        export / ".quant_summary.txt", expected_variant=validation_variant
    )
    validation_variant = validation_variant or str(quant_summary["positive_control_variant"])

    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise QuantizationError("torch and safetensors are required for scale validation") from exc
    scale_count = 0
    packed_count = 0
    controlled_weight_dtypes: dict[str, str] = {}
    controlled_weight_shapes: dict[str, tuple[int, ...]] = {}
    controlled_weight_names = set(CONTROLLED_EXPORT_BF16_WEIGHTS)
    controlled_module_prefixes = tuple(
        weight_name.removesuffix(".weight") + "."
        for weight_name in CONTROLLED_EXPORT_BF16_WEIGHTS
    )
    export_metadata: dict[str, tuple[str, tuple[int, ...]]] = {}
    scale_is_positive: dict[str, bool] = {}
    for shard in shard_paths:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in export_metadata:
                    raise QuantizationError(f"Duplicate tensor {key} across safetensors shards")
                tensor_slice = handle.get_slice(key)
                dtype = str(tensor_slice.get_dtype())
                shape = tuple(tensor_slice.get_shape())
                export_metadata[key] = (dtype, shape)
                lower_leaf = key.rsplit(".", 1)[-1].lower()
                if (
                    ("scale" in lower_leaf or "amax" in lower_leaf)
                    and any(key.startswith(prefix) for prefix in controlled_module_prefixes)
                ):
                    raise QuantizationError(
                        "Controlled BF16 module unexpectedly has quantization scale/amax tensor "
                        f"{key}: {shard}"
                    )
                if key in controlled_weight_names:
                    controlled_weight_dtypes[key] = (
                        "torch.bfloat16" if dtype == "BF16" else dtype
                    )
                    controlled_weight_shapes[key] = shape
                    if dtype != "BF16":
                        raise QuantizationError(
                            f"Controlled high-precision tensor {key} is {dtype}, "
                            f"expected torch.bfloat16 (BF16): {shard}"
                        )
                lower_key = key.lower()
                if "scale" in lower_key or "amax" in lower_key:
                    scale_count += 1
                    tensor = handle.get_tensor(key)
                    if not bool(torch.isfinite(tensor.float()).all()):
                        raise QuantizationError(f"Non-finite quantization scale {key} in {shard}")
                    scale_is_positive[key] = bool((tensor.float() > 0).all())
                if "weight" in lower_key and dtype not in {"BF16", "F16", "F32", "F64"}:
                    packed_count += 1
    if scale_count == 0 or packed_count == 0:
        raise QuantizationError(
            "Export lacks packed weights or explicit scale tensors: "
            f"packed={packed_count}, scales={scale_count}"
        )
    missing_controlled_weights = sorted(controlled_weight_names - controlled_weight_dtypes.keys())
    if missing_controlled_weights:
        raise QuantizationError(
            "Export is missing exact controlled BF16 attention/Mamba weights: "
            f"{missing_controlled_weights}"
        )

    positive_specs = {
        "fp8": {
            f"{POSITIVE_CONTROL_EXPORT_MODULE}.weight": ("F8_E4M3", POSITIVE_CONTROL_SOURCE_SHAPE),
            f"{POSITIVE_CONTROL_EXPORT_MODULE}.input_scale": ("F32", ()),
            f"{POSITIVE_CONTROL_EXPORT_MODULE}.weight_scale": ("F32", ()),
        },
        "nvfp4": {
            f"{POSITIVE_CONTROL_EXPORT_MODULE}.weight": (
                "U8",
                (POSITIVE_CONTROL_SOURCE_SHAPE[0], POSITIVE_CONTROL_SOURCE_SHAPE[1] // 2),
            ),
            f"{POSITIVE_CONTROL_EXPORT_MODULE}.input_scale": ("F32", ()),
            f"{POSITIVE_CONTROL_EXPORT_MODULE}.weight_scale": (
                "F8_E4M3",
                (POSITIVE_CONTROL_SOURCE_SHAPE[0], POSITIVE_CONTROL_SOURCE_SHAPE[1] // 16),
            ),
            f"{POSITIVE_CONTROL_EXPORT_MODULE}.weight_scale_2": ("F32", ()),
        },
    }
    expected_positive = positive_specs[validation_variant]
    positive_prefix = POSITIVE_CONTROL_EXPORT_MODULE + "."
    actual_positive_scales = {
        key
        for key in export_metadata
        if key.startswith(positive_prefix)
        and (
            "scale" in key.rsplit(".", 1)[-1].lower()
            or "amax" in key.rsplit(".", 1)[-1].lower()
        )
    }
    expected_positive_scales = {
        key for key in expected_positive if "scale" in key.rsplit(".", 1)[-1].lower()
    }
    if actual_positive_scales != expected_positive_scales:
        raise QuantizationError(
            "Layer-2 Mamba positive-control scale set mismatch: "
            f"actual={sorted(actual_positive_scales)}, expected={sorted(expected_positive_scales)}"
        )
    for key, expected_metadata in expected_positive.items():
        actual_metadata = export_metadata.get(key)
        if actual_metadata != expected_metadata:
            raise QuantizationError(
                f"Layer-2 Mamba positive-control tensor {key} has metadata "
                f"{actual_metadata}, expected {expected_metadata}"
            )
    nonpositive_scales = sorted(
        key for key in expected_positive_scales if not scale_is_positive.get(key, False)
    )
    if nonpositive_scales:
        raise QuantizationError(
            f"Layer-2 Mamba positive-control scales are not strictly positive: {nonpositive_scales}"
        )

    if source_model is not None:
        source = Path(source_model)
        source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        assert_controlled_topology(source_config, location=source / "config.json")
        if source_topology.get("source_config_sha256") != sha256_file(source / "config.json"):
            raise QuantizationError(
                f"Export topology audit does not match the pinned validation source: {export}"
            )
        for key in ("model_type", "architectures", "vocab_size"):
            if source_config.get(key) != export_config.get(key):
                raise QuantizationError(f"Export config lineage mismatch for {key}: {export}")
        for tokenizer_file in ("tokenizer_config.json", "tokenizer.json"):
            if (source / tokenizer_file).is_file() and not (export / tokenizer_file).is_file():
                raise QuantizationError(
                    f"Export is missing source tokenizer artifact {tokenizer_file}"
                )
        source_metadata = _selected_safetensor_metadata(
            source,
            (*CONTROLLED_EXPORT_BF16_WEIGHTS, POSITIVE_CONTROL_SOURCE_WEIGHT),
            safe_open=safe_open,
        )
        bad_source_dtypes = sorted(
            key for key, (dtype, _) in source_metadata.items() if dtype != "BF16"
        )
        if bad_source_dtypes:
            raise QuantizationError(
                f"Pinned source controlled tensors are not BF16: {bad_source_dtypes}"
            )
        shape_mismatches = {
            key: {"source": source_metadata[key][1], "export": export_metadata[key][1]}
            for key in CONTROLLED_EXPORT_BF16_WEIGHTS
            if source_metadata[key][1] != export_metadata[key][1]
        }
        if shape_mismatches:
            raise QuantizationError(
                "Controlled BF16 export shapes differ from the pinned source: "
                f"{shape_mismatches}"
            )
        if source_metadata[POSITIVE_CONTROL_SOURCE_WEIGHT][1] != POSITIVE_CONTROL_SOURCE_SHAPE:
            raise QuantizationError(
                "Pinned source layer-2 Mamba positive-control shape is unexpected: "
                f"{source_metadata[POSITIVE_CONTROL_SOURCE_WEIGHT][1]}, "
                f"expected {POSITIVE_CONTROL_SOURCE_SHAPE}"
            )
    return {
        "path": str(export.resolve()),
        "files": len(shard_paths),
        "bytes": sum(path.stat().st_size for path in shard_paths),
        "packed_tensor_count": packed_count,
        "scale_tensor_count": scale_count,
        "controlled_bf16_weight_count": len(controlled_weight_dtypes),
        "controlled_bf16_weight_dtypes": controlled_weight_dtypes,
        "controlled_bf16_weight_shapes": controlled_weight_shapes,
        "positive_control_variant": validation_variant,
        "positive_control_tensor_count": len(expected_positive),
        "quant_summary": quant_summary,
        "hf_quant_config_sha256": sha256_file(export / "hf_quant_config.json"),
        "config_normalization_sha256": sha256_file(normalization_path),
        "quantization_metadata_normalization_sha256": sha256_file(
            quantization_metadata_path
        ),
        "source_topology_normalization_sha256": sha256_file(source_topology_path),
    }


def run_quantization(
    job: QuantizationJob,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    allow_existing_valid_export: bool = False,
) -> QuantizationResult:
    """Run one job with a durable merged log; never switch recipes or dimensions."""

    if job.output_dir.exists():
        if not allow_existing_valid_export:
            raise FileExistsError(f"Refusing to overwrite checkpoint: {job.output_dir}")
        validate_export(
            job.output_dir,
            expected_variant=job.variant,
            source_model=job.source_model,
        )
        return QuantizationResult(job.variant, 0, job.output_dir, job.log_path, job.metadata_path)

    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    job.output_dir.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(job.metadata_path, {**job.manifest(), "status": "started"})
    process_environment = os.environ.copy()
    process_environment.update({str(key): str(value) for key, value in job.environment.items()})
    try:
        with job.log_path.open("x", encoding="utf-8") as log:
            result = runner(
                list(job.command),
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=process_environment,
            )
    except BaseException as exc:
        write_json_atomic(
            job.metadata_path,
            {**job.manifest(), "status": "exception", "error": repr(exc)},
            overwrite=True,
        )
        raise
    if result.returncode != 0:
        write_json_atomic(
            job.metadata_path,
            {**job.manifest(), "status": "failed", "returncode": result.returncode},
            overwrite=True,
        )
        raise QuantizationError(
            f"{job.variant} ModelOpt PTQ failed with exit code {result.returncode}; log={job.log_path}"
        )
    try:
        normalization = normalize_export_config(job.output_dir)
        quantization_metadata_normalization = normalize_quantization_metadata(job.output_dir)
        source_topology_normalization = normalize_source_topology(
            job.output_dir, source_model=job.source_model
        )
        validation = validate_export(
            job.output_dir,
            expected_variant=job.variant,
            source_model=job.source_model,
        )
    except BaseException as exc:
        write_json_atomic(
            job.metadata_path,
            {
                **job.manifest(),
                "status": "exception",
                "stage": "post_export_normalization_and_validation",
                "error": repr(exc),
            },
            overwrite=True,
        )
        raise
    write_json_atomic(
        job.metadata_path,
        {
            **job.manifest(),
            "status": "complete",
            "returncode": 0,
            "config_normalization": normalization,
            "quantization_metadata_normalization": quantization_metadata_normalization,
            "source_topology_normalization": source_topology_normalization,
            "validation": validation,
        },
        overwrite=True,
    )
    return QuantizationResult(job.variant, 0, job.output_dir, job.log_path, job.metadata_path)
