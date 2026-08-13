from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ptq_workshop.reporting import (  # noqa: E402
    ArtifactBundle,
    comparison_rows,
    create_dashboard,
    load_artifacts,
    read_records,
    summary_as_markdown,
)


def _write_csv(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def make_artifacts(root: Path):
    (root / "accuracy").mkdir(parents=True)
    (root / "performance").mkdir(parents=True)
    (root / "telemetry").mkdir(parents=True)
    (root / "accuracy" / "scores.json").write_text(
        json.dumps(
            [
                {"precision": "bf16", "task": "mmlu", "accuracy": 0.80},
                {"precision": "fp8", "task": "mmlu", "accuracy": 0.79},
                {"precision": "nvfp4", "task": "mmlu", "accuracy": 0.77},
            ]
        ),
        encoding="utf-8",
    )
    (root / "performance" / "bench.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"precision": "bf16", "scenario": "balanced", "checkpoint_size_gib": 64, "metrics": {"output_tps": 100, "ttft_ms": 50, "tpot_ms": 10}},
                {"precision": "fp8", "scenario": "balanced", "checkpoint_size_gib": 34, "metrics": {"output_tps": 150, "ttft_ms": 40, "tpot_ms": 7}},
                {"precision": "nvfp4", "scenario": "balanced", "checkpoint_size_gib": 20, "metrics": {"output_tps": 180, "ttft_ms": 35, "tpot_ms": 6}},
            ]
        ),
        encoding="utf-8",
    )
    _write_csv(
        root / "telemetry" / "power.csv",
        [
            {"precision": "bf16", "elapsed_s": 0.0, "power_w": 300, "memory_used_gib": 70},
            {"precision": "bf16", "elapsed_s": 1.0, "power_w": 320, "memory_used_gib": 72},
            {"precision": "fp8", "elapsed_s": 0.0, "power_w": 260, "memory_used_gib": 42},
            {"precision": "fp8", "elapsed_s": 1.0, "power_w": 280, "memory_used_gib": 44},
            {"precision": "nvfp4", "elapsed_s": 0.0, "power_w": 240, "memory_used_gib": 30},
            {"precision": "nvfp4", "elapsed_s": 1.0, "power_w": 250, "memory_used_gib": 32},
        ],
    )
    (root / "telemetry" / "ignored.txt").write_text("do not read me", encoding="utf-8")


def test_read_records_flattens_nested_json_and_coerces_csv(tmp_path):
    json_path = tmp_path / "record.json"
    json_path.write_text(json.dumps({"precision": "fp8", "metrics": {"ttft_ms": 12.5}}), encoding="utf-8")
    assert read_records(json_path)[0]["metrics.ttft_ms"] == 12.5

    csv_path = tmp_path / "record.csv"
    _write_csv(csv_path, [{"precision": "bf16", "power_w": "300.5"}])
    assert read_records(csv_path)[0]["power_w"] == 300.5
    inferred_path = tmp_path / "nvfp4-telemetry.csv"
    _write_csv(inferred_path, [{"elapsed_s": "0", "power_w": "250"}])
    assert read_records(inferred_path)[0]["precision"] == "nvfp4"
    with pytest.raises(ValueError):
        read_records(tmp_path / "unsupported.txt")


def test_loading_and_comparison_are_artifact_only(tmp_path):
    make_artifacts(tmp_path)
    bundle = load_artifacts(tmp_path)
    assert len(bundle.accuracy) == 3
    assert len(bundle.performance) == 3
    assert len(bundle.telemetry) == 6
    rows = comparison_rows(bundle)
    assert [row["precision"] for row in rows] == ["bf16", "fp8", "nvfp4"]
    assert rows[0]["output_tokens_per_second"] == 100
    assert rows[1]["throughput_speedup"] == pytest.approx(1.5)
    assert rows[1]["checkpoint_compression"] == pytest.approx(64 / 34)
    assert rows[2]["checkpoint_compression"] == pytest.approx(3.2)
    assert rows[2]["accuracy_delta_pp"] == pytest.approx(-3.0)
    assert rows[2]["peak_vram_gib"] == pytest.approx(32.0)


def test_loading_supports_canonical_metrics_layout(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "fp8-accuracy.json").write_text(
        json.dumps({"task": "mmlu_pro", "accuracy": 79.0}),
        encoding="utf-8",
    )
    (metrics / "fp8-benchmark.json").write_text(
        json.dumps({"scenario": "rag", "output_tps": 150}),
        encoding="utf-8",
    )
    (metrics / "fp8-validation.json").write_text(
        json.dumps({"checkpoint_size_gib": 34}),
        encoding="utf-8",
    )
    bundle = load_artifacts(tmp_path)
    assert len(bundle.accuracy) == 1
    assert len(bundle.performance) == 1
    assert len(bundle.summary) == 1
    row = comparison_rows(bundle)[0]
    assert row["checkpoint_size_gib"] == 34
    assert row["accuracy"] == pytest.approx(0.79)


def test_markdown_summary_contains_active_metrics(tmp_path):
    make_artifacts(tmp_path)
    rows = comparison_rows(load_artifacts(tmp_path))
    markdown = summary_as_markdown(rows)
    assert "| precision |" in markdown
    assert "nvfp4" in markdown
    assert "checkpoint_size_gib" in markdown
    assert "checkpoint_compression" in markdown
    assert "throughput_speedup" in markdown
    assert summary_as_markdown([]).startswith("_No comparable")


def test_unavailable_runtime_suppresses_stale_headline_metrics(tmp_path):
    make_artifacts(tmp_path)
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "fp8-all-runtime-status.json").write_text(
        json.dumps(
            {
                "precision": "fp8",
                "runtime_label": "all",
                "available": False,
                "stage": "smoke",
                "attempt_started_utc": "2026-08-12T12:00:00+00:00",
                "exception_type": "RuntimeError",
                "exception_message": "gibberish did not match sentinel",
            }
        ),
        encoding="utf-8",
    )

    bundle = load_artifacts(tmp_path)
    assert len(bundle.runtime_status) == 1
    fp8 = next(row for row in comparison_rows(bundle) if row["precision"] == "fp8")
    assert fp8["runtime_status"] == "unavailable"
    assert fp8["runtime_stage"] == "smoke"
    assert fp8["runtime_error"] == "gibberish did not match sentinel"
    for stale_metric in (
        "accuracy",
        "accuracy_delta_pp",
        "output_tokens_per_second",
        "throughput_speedup",
        "peak_vram_gib",
        "power_w",
    ):
        assert stale_metric not in fp8
    assert "unavailable" in summary_as_markdown(comparison_rows(bundle))


def test_dashboard_filters_unavailable_precision_from_runtime_plots(tmp_path):
    pytest.importorskip("matplotlib")
    make_artifacts(tmp_path)
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "fp8-all-runtime-status.json").write_text(
        json.dumps(
            {
                "precision": "fp8",
                "available": False,
                "stage": "startup",
                "attempt_started_utc": "2026-08-12T12:00:00+00:00",
                "exception_message": "server failed",
            }
        ),
        encoding="utf-8",
    )
    result = create_dashboard(tmp_path)
    assert result["unavailable_precisions"] == ["fp8"]
    fp8 = next(row for row in result["rows"] if row["precision"] == "fp8")
    assert fp8["runtime_status"] == "unavailable"
    assert "accuracy" not in fp8


def test_dashboard_creates_available_plots(tmp_path):
    pytest.importorskip("matplotlib")
    make_artifacts(tmp_path)
    report_dir = tmp_path / "figures"
    result = create_dashboard(tmp_path, report_dir)
    assert set(result["figures"]) == {
        "accuracy",
        "throughput",
        "latency",
        "resources",
        "telemetry_power",
        "pareto",
    }
    assert all(Path(path).is_file() for path in result["saved"].values())


def test_dashboard_renders_real_packed_error_and_scale_artifacts(tmp_path):
    pytest.importorskip("matplotlib")
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "fp8-tensor-analysis.json").write_text(
        json.dumps(
            {
                "precision": "fp8",
                "tensors": [
                    {
                        "name": "backbone.layers.2.mixer.in_proj.weight",
                        "absolute_error_cdf": {
                            "p0": 0.0,
                            "p50": 0.001,
                            "p90": 0.002,
                            "p100": 0.01,
                        },
                    }
                ],
                "aggregate_scale_distributions": {
                    "weight_scale": {
                        "positive_log10_histogram": {
                            "bin_edges": [-3.0, -2.0, -1.0],
                            "counts": [2, 1],
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    result = create_dashboard(tmp_path, tmp_path / "figures")
    assert "packed_tensor_error_cdf" in result["figures"]
    assert "packed_scale_distributions" in result["figures"]
    assert Path(result["saved"]["packed_tensor_error_cdf"]).is_file()
    assert Path(result["saved"]["packed_scale_distributions"]).is_file()


def test_peak_resource_uses_max_and_power_excludes_cold_load():
    bundle = ArtifactBundle(
        telemetry=(
            {
                "precision": "fp8",
                "summary.peak_memory_used_bytes": 10 * 1024**3,
                "summary.average_power_w": 500,
                "_source": "/run/telemetry/fp8-evaluate-cold-load.json",
            },
            {
                "precision": "fp8",
                "summary.peak_memory_used_bytes": 20 * 1024**3,
                "summary.average_power_w": 200,
                "_source": "/run/telemetry/fp8-throughput.json",
            },
            {
                "precision": "fp8",
                "summary.peak_memory_used_bytes": 15 * 1024**3,
                "summary.average_power_w": 300,
                "_source": "/run/telemetry/fp8-interactive.json",
            },
            {
                "precision": "fp8",
                "summary.peak_memory_used_bytes": 25 * 1024**3,
                "summary.average_power_w": 600,
                "_source": "/run/telemetry/fp8-quantize.json",
            },
        )
    )
    row = comparison_rows(bundle)[0]
    assert row["peak_vram_gib"] == 25
    assert row["power_w"] == 250


def test_workshop_notebook_is_static_and_compilable():
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "blackwell_ptq_workshop.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
    assert notebook["cells"]

    joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for required in (
        "DEV_SMOKE",
        "WORKSHOP_B200",
        "FULL",
        "B300",
        "SM103",
        "quantize_variant.py",
        "evaluate_variant.py",
        "benchmark_variant.py",
        "create_dashboard",
        "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "3.5B active parameters",
        "16 × 512 tokens",
        "128 × 512 tokens",
        "| `interactive` | 512 | 128 | 1 | 16 |",
        "| `rag_balanced` | 2,048 | 256 | 8 | 32 |",
        "| `throughput` | 1,024 | 128 | 32 | 64 |",
        "| BF16 | 78.3 |",
        "| FP8 | 78.1 |",
        "| NVFP4 | 77.4 |",
        "QAD after initial PTQ",
        "4 + 8/16 = 4.5",
        "deployment/3_unified_hf.html",
    ):
        assert required in joined

    for stale in ("3B active parameters", "16 × 128 tokens", "512 × 512 tokens"):
        assert stale not in joined

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        compile("".join(cell.get("source", [])), f"notebook-cell-{index}", "exec")
