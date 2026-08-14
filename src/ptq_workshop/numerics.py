"""Small, dependency-light numerical helpers for the PTQ workshop.

The routines in this module are educational simulations.  They make the
rounding and scale choices visible, but they are not replacements for NVIDIA
Model Optimizer or the native Blackwell kernels used by the workshop's real
quantization runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QuantizationResult:
    """Result of an educational quantize/dequantize operation."""

    dequantized: np.ndarray
    encoded_values: np.ndarray
    scales: np.ndarray
    metadata: dict[str, Any]


def bf16_round(values: np.ndarray | list[float]) -> np.ndarray:
    """Round float32 values to BF16 precision and return them as float32.

    NumPy does not expose a portable bfloat16 dtype.  The bit operation below
    implements round-to-nearest-even before truncating the low 16 mantissa
    bits, which is sufficient for the workshop's format visualizations.
    """

    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = ((bits + rounding_bias) & np.uint32(0xFFFF0000)).view(np.float32)
    return rounded.copy()


def fp8_e4m3fn_values() -> np.ndarray:
    """Return the finite values representable by NVIDIA-style FP8 E4M3FN.

    E4M3FN uses exponent bias 7.  The all-ones exponent remains finite for
    mantissas 0 through 6; the final encoding is reserved for NaN.  The
    resulting maximum finite magnitude is 448.
    """

    positive: list[float] = [0.0]
    # Subnormal numbers: (mantissa / 8) * 2**(1-bias).
    positive.extend((mantissa / 8.0) * 2.0**-6 for mantissa in range(1, 8))
    for exponent in range(1, 15):
        positive.extend(
            (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
            for mantissa in range(8)
        )
    positive.extend((1.0 + mantissa / 8.0) * 2.0**8 for mantissa in range(7))
    positive_array = np.asarray(sorted(set(positive)), dtype=np.float32)
    negative_array = -positive_array[positive_array > 0][::-1]
    return np.concatenate((negative_array, positive_array))


def nvfp4_e2m1_values() -> np.ndarray:
    """Return the finite E2M1 values used by NVFP4 before scaling."""

    positive = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    return np.concatenate((-positive[positive > 0][::-1], positive))


def _nearest_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Round values to the nearest point in a sorted one-dimensional grid."""

    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    grid = np.asarray(grid, dtype=np.float32)
    if grid.ndim != 1 or grid.size == 0 or np.any(grid[1:] < grid[:-1]):
        raise ValueError("grid must be a non-empty sorted one-dimensional array")

    upper_index = np.searchsorted(grid, flat, side="left")
    upper_index = np.clip(upper_index, 0, grid.size - 1)
    lower_index = np.clip(upper_index - 1, 0, grid.size - 1)
    lower = grid[lower_index]
    upper = grid[upper_index]
    # Choosing the lower value for an exact midpoint makes this deterministic;
    # native kernels may use format-specific round-to-nearest-even behavior.
    choose_upper = np.abs(flat - upper) < np.abs(flat - lower)
    rounded = np.where(choose_upper, upper, lower)
    return rounded.reshape(np.asarray(values).shape)


def quantize_fp8_e4m3fn(
    values: np.ndarray | list[float], *, scale: float | None = None
) -> QuantizationResult:
    """Simulate per-tensor FP8 E4M3FN quantization.

    When ``scale`` is omitted, the largest magnitude maps to 448.  The return
    value contains both normalized FP8 values and reconstructed float32 data.
    """

    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")

    max_abs = float(np.max(np.abs(array)))
    resolved_scale = float(scale) if scale is not None else (max_abs / 448.0 if max_abs else 1.0)
    if not np.isfinite(resolved_scale) or resolved_scale <= 0:
        raise ValueError("scale must be finite and greater than zero")

    encoded = _nearest_grid(array / resolved_scale, fp8_e4m3fn_values())
    dequantized = (encoded * resolved_scale).astype(np.float32)
    return QuantizationResult(
        dequantized=dequantized,
        encoded_values=encoded.astype(np.float32),
        scales=np.asarray([resolved_scale], dtype=np.float32),
        metadata={"format": "fp8_e4m3fn", "scale_granularity": "tensor", "max_finite": 448.0},
    )


def quantize_nvfp4(
    values: np.ndarray | list[float],
    *,
    block_size: int = 16,
    global_scale: float | None = None,
) -> QuantizationResult:
    """Simulate NVFP4 E2M1 quantization with two-level block scaling.

    Each contiguous block shares an FP8 E4M3 scale, and the complete tensor
    shares one FP32 global scale.  A final short block is supported without
    padding the returned data.
    """

    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    if block_size <= 0:
        raise ValueError("block_size must be greater than zero")

    flat = array.reshape(-1)
    max_abs = float(np.max(np.abs(flat)))
    resolved_global = (
        float(global_scale)
        if global_scale is not None
        else (max_abs / (448.0 * 6.0) if max_abs else 1.0)
    )
    if not np.isfinite(resolved_global) or resolved_global <= 0:
        raise ValueError("global_scale must be finite and greater than zero")

    fp8_positive = fp8_e4m3fn_values()
    fp8_positive = fp8_positive[fp8_positive > 0]
    fp4_grid = nvfp4_e2m1_values()
    encoded = np.empty_like(flat)
    reconstructed = np.empty_like(flat)
    block_scales: list[float] = []

    for start in range(0, flat.size, block_size):
        block = flat[start : start + block_size]
        block_amax = float(np.max(np.abs(block)))
        if block_amax == 0.0:
            block_scale = 1.0
        else:
            desired = block_amax / (6.0 * resolved_global)
            block_scale = float(_nearest_grid(np.asarray([desired]), fp8_positive)[0])
        block_scales.append(block_scale)
        normalized = block / (resolved_global * block_scale)
        block_encoded = _nearest_grid(normalized, fp4_grid)
        encoded[start : start + block.size] = block_encoded
        reconstructed[start : start + block.size] = (
            block_encoded * resolved_global * block_scale
        )

    return QuantizationResult(
        dequantized=reconstructed.reshape(array.shape).astype(np.float32),
        encoded_values=encoded.reshape(array.shape).astype(np.float32),
        scales=np.asarray(block_scales, dtype=np.float32),
        metadata={
            "format": "nvfp4_e2m1",
            "block_size": int(block_size),
            "global_scale": resolved_global,
            "block_scale_format": "fp8_e4m3fn",
        },
    )


def quantization_error_metrics(
    reference: np.ndarray | list[float], approximation: np.ndarray | list[float]
) -> dict[str, float]:
    """Compute stable scalar error metrics for equal-shaped arrays."""

    reference_array = np.asarray(reference, dtype=np.float64)
    approximation_array = np.asarray(approximation, dtype=np.float64)
    if reference_array.shape != approximation_array.shape:
        raise ValueError("reference and approximation must have the same shape")
    if reference_array.size == 0:
        raise ValueError("arrays must not be empty")
    if not np.all(np.isfinite(reference_array)) or not np.all(np.isfinite(approximation_array)):
        raise ValueError("arrays must contain only finite values")

    error = approximation_array - reference_array
    absolute_error = np.abs(error)
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    reference_rms = float(np.sqrt(np.mean(reference_array**2)))
    denominator = float(np.linalg.norm(reference_array) * np.linalg.norm(approximation_array))
    cosine = (
        float(np.dot(reference_array.reshape(-1), approximation_array.reshape(-1)) / denominator)
        if denominator
        else 1.0
    )
    signal_power = float(np.mean(reference_array**2))
    sqnr_db = float("inf") if mse == 0.0 else float(10.0 * np.log10(signal_power / mse)) if signal_power else float("-inf")
    return {
        "mae": float(np.mean(absolute_error)),
        "rmse": rmse,
        "normalized_rmse": rmse / reference_rms if reference_rms else 0.0,
        "max_abs_error": float(np.max(absolute_error)),
        "p99_abs_error": float(np.percentile(absolute_error, 99)),
        "cosine_similarity": cosine,
        "sqnr_db": sqnr_db,
    }


def estimate_weight_storage_bytes(
    total_parameters: int,
    quantized_bits: int,
    *,
    quantized_fraction: float = 1.0,
    high_precision_bits: int = 16,
    block_size: int | None = None,
    scale_bits: int = 0,
) -> float:
    """Estimate weight-plus-block-scale storage for a mixed-precision model."""

    if total_parameters < 0:
        raise ValueError("total_parameters must be non-negative")
    if quantized_bits <= 0 or high_precision_bits <= 0:
        raise ValueError("bit widths must be positive")
    if not 0.0 <= quantized_fraction <= 1.0:
        raise ValueError("quantized_fraction must be between zero and one")
    if scale_bits and (block_size is None or block_size <= 0):
        raise ValueError("a positive block_size is required when scale_bits is non-zero")

    quantized_parameters = total_parameters * quantized_fraction
    high_precision_parameters = total_parameters - quantized_parameters
    data_bits = (
        quantized_parameters * quantized_bits
        + high_precision_parameters * high_precision_bits
    )
    scale_overhead_bits = (
        quantized_parameters / block_size * scale_bits if scale_bits and block_size else 0.0
    )
    return float((data_bits + scale_overhead_bits) / 8.0)


def precision_storage_projection(
    total_parameters: int, *, quantized_fraction: float = 1.0
) -> dict[str, float]:
    """Return simple BF16, FP8, and NVFP4 weight storage projections."""

    return {
        "bf16": estimate_weight_storage_bytes(total_parameters, 16),
        "fp8": estimate_weight_storage_bytes(
            total_parameters, 8, quantized_fraction=quantized_fraction
        ),
        "nvfp4": estimate_weight_storage_bytes(
            total_parameters,
            4,
            quantized_fraction=quantized_fraction,
            block_size=16,
            scale_bits=8,
        ),
    }
