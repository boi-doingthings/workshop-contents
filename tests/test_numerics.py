from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ptq_workshop.numerics import (  # noqa: E402
    bf16_round,
    estimate_weight_storage_bytes,
    fp8_e4m3fn_values,
    nvfp4_e2m1_values,
    precision_storage_projection,
    quantization_error_metrics,
    quantize_fp8_e4m3fn,
    quantize_nvfp4,
)


def test_format_grids_have_expected_limits_and_values():
    fp8 = fp8_e4m3fn_values()
    fp4 = nvfp4_e2m1_values()
    assert np.all(fp8[1:] > fp8[:-1])
    assert fp8[0] == -448.0
    assert fp8[-1] == 448.0
    assert fp4.tolist() == [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def test_bf16_round_preserves_shape_and_reduces_precision():
    values = np.asarray([1.0, 1.0001, -3.1415927], dtype=np.float32)
    rounded = bf16_round(values)
    assert rounded.shape == values.shape
    assert rounded.dtype == np.float32
    assert rounded[0] == 1.0
    assert rounded[1] == 1.0
    assert rounded[2] != values[2]


def test_fp8_quantization_is_exact_for_grid_values_and_saturates():
    values = np.asarray([0.0, 0.5, 1.0, 448.0, 900.0], dtype=np.float32)
    result = quantize_fp8_e4m3fn(values, scale=1.0)
    assert np.allclose(result.dequantized[:4], values[:4])
    assert result.dequantized[-1] == 448.0
    assert result.metadata["format"] == "fp8_e4m3fn"


def test_nvfp4_quantization_uses_one_scale_per_partial_or_full_block():
    values = np.linspace(-10, 10, 17, dtype=np.float32)
    result = quantize_nvfp4(values, block_size=16)
    assert result.dequantized.shape == values.shape
    assert result.scales.shape == (2,)
    assert set(np.unique(result.encoded_values)).issubset(set(nvfp4_e2m1_values()))
    assert np.all(np.isfinite(result.dequantized))


def test_error_metrics_for_known_vectors():
    metrics = quantization_error_metrics([1.0, 2.0], [1.0, 3.0])
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(2**-0.5)
    assert metrics["max_abs_error"] == 1.0
    assert 0.0 < metrics["cosine_similarity"] <= 1.0
    perfect = quantization_error_metrics([1.0, -2.0], [1.0, -2.0])
    assert perfect["sqnr_db"] == float("inf")


def test_storage_estimate_accounts_for_mixed_precision_and_scale_overhead():
    bf16 = estimate_weight_storage_bytes(1_000, 16)
    fp8_half = estimate_weight_storage_bytes(1_000, 8, quantized_fraction=0.5)
    nvfp4 = estimate_weight_storage_bytes(
        1_000, 4, block_size=16, scale_bits=8
    )
    assert bf16 == 2_000
    assert fp8_half == 1_500
    assert nvfp4 == pytest.approx(562.5)
    projection = precision_storage_projection(1_000)
    assert projection["bf16"] > projection["fp8"] > projection["nvfp4"]


def test_invalid_inputs_fail_clearly():
    with pytest.raises(ValueError):
        quantize_fp8_e4m3fn([])
    with pytest.raises(ValueError):
        quantize_nvfp4([1.0], block_size=0)
    with pytest.raises(ValueError):
        quantization_error_metrics([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        estimate_weight_storage_bytes(100, 4, scale_bits=8)
