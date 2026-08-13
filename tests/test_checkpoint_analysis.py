from __future__ import annotations

import numpy as np
import pytest

from ptq_workshop.checkpoint_analysis import (
    distribution_summary,
    error_summary,
    quantized_weight_candidates,
    select_representative_tensors,
)


def test_candidates_require_source_weight_and_variant_scale_contract() -> None:
    source = {
        "backbone.layers.2.weight": "source.safetensors",
        "backbone.layers.10.weight": "source.safetensors",
    }
    export = {
        "backbone.layers.2.weight": "export.safetensors",
        "backbone.layers.2.weight_scale": "export.safetensors",
        "backbone.layers.2.weight_scale_2": "export.safetensors",
        "backbone.layers.10.weight": "export.safetensors",
        "backbone.layers.10.weight_scale": "export.safetensors",
    }
    assert quantized_weight_candidates(source, export, variant="fp8") == (
        "backbone.layers.2.weight",
        "backbone.layers.10.weight",
    )
    assert quantized_weight_candidates(source, export, variant="nvfp4") == (
        "backbone.layers.2.weight",
    )


def test_representative_selection_is_natural_and_spread() -> None:
    candidates = tuple(f"backbone.layers.{index}.weight" for index in range(12))
    selected = select_representative_tensors(tuple(reversed(candidates)), count=3)
    assert selected == (
        "backbone.layers.0.weight",
        "backbone.layers.5.weight",
        "backbone.layers.11.weight",
    )


def test_error_summary_contains_requested_metrics_and_cdfs() -> None:
    report = error_summary(
        np.asarray([1.0, 2.0, 4.0], dtype=np.float32),
        np.asarray([1.0, 3.0, 2.0], dtype=np.float32),
    )
    assert report["metrics"]["mae"] == pytest.approx(1.0)
    assert report["metrics"]["cosine_similarity"] is not None
    assert set(report["absolute_error_cdf"]) == {
        "p0",
        "p50",
        "p90",
        "p95",
        "p99",
        "p99.9",
        "p100",
    }
    assert report["absolute_error_cdf"]["p100"] == 2.0


def test_scale_distribution_has_quantiles_and_log_histogram() -> None:
    summary = distribution_summary(np.asarray([0.0, 0.25, 0.5, 1.0]))
    assert summary["count"] == 4
    assert summary["positive_count"] == 3
    assert summary["zero_count"] == 1
    assert summary["quantiles"]["p50"] == pytest.approx(0.375)
    histogram = summary["positive_log10_histogram"]
    assert sum(histogram["counts"]) == 3
