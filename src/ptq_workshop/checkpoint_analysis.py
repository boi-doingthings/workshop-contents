"""Representative error analysis from real packed ModelOpt checkpoints.

Unlike :mod:`ptq_workshop.numerics`, this module does not simulate a format.
It reads the BF16 source and the serialized packed tensors, then reconstructs
the latter through ModelOpt's public QTensor dequantizers.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import sha256_file
from .numerics import quantization_error_metrics


ANALYSIS_SCHEMA_VERSION = 1
ERROR_CDF_PERCENTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
SCALE_PERCENTILES = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(token) if token.isdigit() else token
        for token in re.split(r"(\d+)", value)
    )


def _weight_map(checkpoint: Path) -> dict[str, str]:
    index_path = checkpoint / "model.safetensors.index.json"
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid safetensors index: {index_path}") from exc
    if not isinstance(weight_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise ValueError(f"Invalid weight_map in {index_path}")
    return dict(weight_map)


def quantized_weight_candidates(
    source_weight_map: Mapping[str, str],
    export_weight_map: Mapping[str, str],
    *,
    variant: str,
) -> tuple[str, ...]:
    """Return source-backed packed weights with their required scale tensors."""

    if variant not in {"fp8", "nvfp4"}:
        raise ValueError("variant must be fp8 or nvfp4")
    required_suffixes = ("weight_scale",)
    if variant == "nvfp4":
        required_suffixes += ("weight_scale_2",)
    candidates: list[str] = []
    for key in export_weight_map:
        if not key.endswith(".weight") or key not in source_weight_map:
            continue
        prefix = key.removesuffix("weight")
        if all(f"{prefix}{suffix}" in export_weight_map for suffix in required_suffixes):
            candidates.append(key)
    return tuple(sorted(candidates, key=_natural_key))


def select_representative_tensors(
    candidates: Sequence[str], *, count: int
) -> tuple[str, ...]:
    """Choose deterministic tensors spread across the checkpoint namespace."""

    if count <= 0:
        raise ValueError("count must be positive")
    ordered = tuple(sorted(set(candidates), key=_natural_key))
    if not ordered:
        raise ValueError("No source-backed packed weight tensors were found")
    if len(ordered) <= count:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, num=count, dtype=np.int64)
    return tuple(ordered[int(position)] for position in positions)


def _quantile_map(values: np.ndarray, probabilities: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("distribution values must be nonempty and finite")
    quantiles = np.quantile(array, probabilities)
    return {
        f"p{probability * 100:g}": float(value)
        for probability, value in zip(probabilities, quantiles)
    }


def distribution_summary(
    values: np.ndarray | Sequence[float],
    *,
    probabilities: Sequence[float] = SCALE_PERCENTILES,
    logarithmic_histogram_bins: int = 20,
) -> dict[str, Any]:
    """Summarize a finite distribution with quantiles and a positive log histogram."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("distribution values must be nonempty and finite")
    positive = array[array > 0]
    histogram: dict[str, Any] | None = None
    if positive.size:
        logged = np.log10(positive)
        low, high = float(logged.min()), float(logged.max())
        if low == high:
            edges = np.asarray([low - 0.5, high + 0.5], dtype=np.float64)
        else:
            edges = np.linspace(low, high, logarithmic_histogram_bins + 1)
        counts, edges = np.histogram(logged, bins=edges)
        histogram = {
            "domain": "log10_positive_values",
            "bin_edges": [float(value) for value in edges],
            "counts": [int(value) for value in counts],
        }
    return {
        "count": int(array.size),
        "positive_count": int(positive.size),
        "zero_count": int(np.count_nonzero(array == 0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "quantiles": _quantile_map(array, probabilities),
        "positive_log10_histogram": histogram,
    }


def error_summary(reference: np.ndarray, approximation: np.ndarray) -> dict[str, Any]:
    """Return scalar metrics plus absolute and relative error CDF points."""

    reference_array = np.asarray(reference, dtype=np.float32).reshape(-1)
    approximation_array = np.asarray(approximation, dtype=np.float32).reshape(-1)
    metrics = quantization_error_metrics(reference_array, approximation_array)
    sanitized_metrics = {
        key: (float(value) if math.isfinite(value) else None)
        for key, value in metrics.items()
    }
    absolute_error = np.abs(approximation_array - reference_array).astype(np.float64)
    relative_error = absolute_error / np.maximum(
        np.abs(reference_array).astype(np.float64), 1e-8
    )
    return {
        "metrics": sanitized_metrics,
        "absolute_error_cdf": _quantile_map(absolute_error, ERROR_CDF_PERCENTILES),
        "relative_error_cdf": _quantile_map(relative_error, ERROR_CDF_PERCENTILES),
    }


def _load_tensor(
    checkpoint: Path,
    weight_map: Mapping[str, str],
    key: str,
    *,
    safe_open: Any,
) -> Any:
    try:
        shard_name = weight_map[key]
    except KeyError as exc:
        raise KeyError(f"Checkpoint {checkpoint} lacks tensor {key}") from exc
    # Hub snapshots intentionally symlink shards into the blob store, so the
    # resolved target need not remain beneath the snapshot. Reject path
    # traversal in the index while allowing that normal immutable symlink.
    if Path(shard_name).name != shard_name:
        raise ValueError(f"Invalid shard path for tensor {key}: {shard_name}")
    shard = checkpoint / shard_name
    if not shard.is_file():
        raise ValueError(f"Missing shard for tensor {key}: {shard_name}")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _sample_pair(reference: Any, approximation: Any, *, maximum_samples: int) -> tuple[Any, Any]:
    if maximum_samples <= 0:
        raise ValueError("maximum_samples must be positive")
    reference_flat = reference.reshape(-1).float()
    approximation_flat = approximation.reshape(-1).float()
    if reference_flat.numel() != approximation_flat.numel():
        raise ValueError("source and dequantized tensors have different element counts")
    if reference_flat.numel() <= maximum_samples:
        return reference_flat, approximation_flat
    import torch

    indices = torch.linspace(
        0,
        reference_flat.numel() - 1,
        steps=maximum_samples,
        dtype=torch.int64,
    )
    return reference_flat[indices], approximation_flat[indices]


def _dequantize_modelopt(
    packed: Any,
    reference: Any,
    scales: Mapping[str, Any],
    *,
    variant: str,
) -> tuple[Any, str]:
    import torch
    from modelopt.torch.quantization.qtensor import FP8QTensor, NVFP4QTensor

    if variant == "fp8":
        qtensor = FP8QTensor(reference.shape, reference.dtype, packed)
        return (
            qtensor.dequantize(dtype=torch.float32, scale=scales["weight_scale"]),
            "modelopt.torch.quantization.qtensor.FP8QTensor.dequantize",
        )
    qtensor = NVFP4QTensor(reference.shape, reference.dtype, packed)
    return (
        qtensor.dequantize(
            dtype=torch.float32,
            scale=scales["weight_scale"],
            double_scale=scales["weight_scale_2"],
            block_sizes={-1: 16},
        ),
        "modelopt.torch.quantization.qtensor.NVFP4QTensor.dequantize",
    )


def analyze_packed_checkpoint(
    source_checkpoint: str | Path,
    export_checkpoint: str | Path,
    *,
    variant: str,
    tensor_names: Sequence[str] | None = None,
    tensor_count: int = 3,
    maximum_samples_per_tensor: int = 1_000_000,
    safe_open_impl: Any | None = None,
) -> dict[str, Any]:
    """Compare representative BF16 weights with their packed reconstructions."""

    if variant not in {"fp8", "nvfp4"}:
        raise ValueError("variant must be fp8 or nvfp4")
    source = Path(source_checkpoint).expanduser().resolve()
    export = Path(export_checkpoint).expanduser().resolve()
    source_map = _weight_map(source)
    export_map = _weight_map(export)
    candidates = quantized_weight_candidates(source_map, export_map, variant=variant)
    selected = (
        tuple(tensor_names)
        if tensor_names
        else select_representative_tensors(candidates, count=tensor_count)
    )
    unknown = sorted(set(selected) - set(candidates), key=_natural_key)
    if unknown:
        raise ValueError(
            f"Requested tensors are not source-backed packed {variant} weights: {unknown}"
        )
    if safe_open_impl is None:
        try:
            from safetensors import safe_open as safe_open_impl
        except ImportError as exc:
            raise RuntimeError("Packed checkpoint analysis requires safetensors") from exc

    tensor_reports: list[dict[str, Any]] = []
    aggregate_scales: dict[str, list[np.ndarray]] = {}
    dequantizer_name: str | None = None
    for weight_name in selected:
        reference = _load_tensor(source, source_map, weight_name, safe_open=safe_open_impl)
        packed = _load_tensor(export, export_map, weight_name, safe_open=safe_open_impl)
        prefix = weight_name.removesuffix("weight")
        scale_names = ["weight_scale"]
        if variant == "nvfp4":
            scale_names.append("weight_scale_2")
        if f"{prefix}input_scale" in export_map:
            scale_names.append("input_scale")
        scales = {
            name: _load_tensor(
                export, export_map, f"{prefix}{name}", safe_open=safe_open_impl
            )
            for name in scale_names
        }
        dequantized, dequantizer_name = _dequantize_modelopt(
            packed, reference, scales, variant=variant
        )
        sampled_reference, sampled_dequantized = _sample_pair(
            reference, dequantized, maximum_samples=maximum_samples_per_tensor
        )
        reference_numpy = sampled_reference.cpu().numpy()
        dequantized_numpy = sampled_dequantized.cpu().numpy()
        scale_reports: dict[str, Any] = {}
        for name, tensor in scales.items():
            values = tensor.float().reshape(-1).cpu().numpy()
            scale_reports[name] = distribution_summary(values)
            aggregate_scales.setdefault(name, []).append(values)
        tensor_reports.append(
            {
                "name": weight_name,
                "source_shape": [int(value) for value in reference.shape],
                "packed_shape": [int(value) for value in packed.shape],
                "source_dtype": str(reference.dtype),
                "packed_dtype": str(packed.dtype),
                "total_elements": int(reference.numel()),
                "sampled_elements": int(sampled_reference.numel()),
                **error_summary(reference_numpy, dequantized_numpy),
                "scale_distributions": scale_reports,
            }
        )

    try:
        modelopt_version = importlib.metadata.version("nvidia-modelopt")
    except importlib.metadata.PackageNotFoundError:
        modelopt_version = importlib.metadata.version("modelopt")
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "kind": "bf16_vs_modelopt_packed_dequantization",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "precision": variant,
        "source_checkpoint": str(source),
        "export_checkpoint": str(export),
        "source_index_sha256": sha256_file(source / "model.safetensors.index.json"),
        "export_index_sha256": sha256_file(export / "model.safetensors.index.json"),
        "modelopt_version": modelopt_version,
        "dequantizer": dequantizer_name,
        "selection": {
            "policy": "explicit" if tensor_names else "natural_namespace_even_spacing",
            "candidate_count": len(candidates),
            "requested_tensor_count": len(selected),
            "maximum_samples_per_tensor": maximum_samples_per_tensor,
            "sampling_policy": "deterministic_evenly_spaced_flat_indices",
        },
        "tensors": tensor_reports,
        "aggregate_scale_distributions": {
            name: distribution_summary(np.concatenate(values))
            for name, values in aggregate_scales.items()
        },
    }
