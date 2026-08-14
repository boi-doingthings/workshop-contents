"""Artifact-only reporting helpers for the Blackwell PTQ workshop.

This module deliberately knows nothing about model loading or CUDA.  It reads
JSON, JSONL, and CSV artifacts produced by completed experiment stages and
turns them into comparison rows and optional matplotlib figures.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv"}
PRECISION_ORDER = {"bf16": 0, "fp8": 1, "nvfp4": 2}


@dataclass(frozen=True)
class ArtifactBundle:
    """Normalized records grouped by experiment stage."""

    summary: tuple[dict[str, Any], ...] = ()
    accuracy: tuple[dict[str, Any], ...] = ()
    performance: tuple[dict[str, Any], ...] = ()
    telemetry: tuple[dict[str, Any], ...] = ()
    runtime_status: tuple[dict[str, Any], ...] = ()


def _coerce_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "":
        return ""
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(stripped)
    except ValueError:
        try:
            return float(stripped)
        except ValueError:
            return value


def _flatten(record: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in record.items():
        joined = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten(value, joined))
        else:
            flattened[joined] = _coerce_scalar(value)
    return flattened


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a single JSON, JSONL, or CSV artifact into flat records."""

    artifact = Path(path)
    if artifact.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported artifact type: {artifact.suffix}")
    if artifact.suffix.lower() == ".csv":
        with artifact.open("r", encoding="utf-8", newline="") as handle:
            records = [dict(row) for row in csv.DictReader(handle)]
    elif artifact.suffix.lower() == ".jsonl":
        records = []
        with artifact.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"{artifact}:{line_number} must contain a JSON object")
                records.append(dict(value))
    else:
        with artifact.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, list):
            records = value
        elif isinstance(value, Mapping) and isinstance(value.get("attempts"), list):
            attempts = value["attempts"]
            accepted = value.get("accepted_attempt")
            selected = None
            if isinstance(accepted, int) and 1 <= accepted <= len(attempts):
                selected = attempts[accepted - 1]
            elif attempts:
                selected = attempts[-1]
            summary = selected.get("summary") if isinstance(selected, Mapping) else None
            scenario = value.get("scenario", {})
            records = [
                {
                    "scenario": scenario.get("name") if isinstance(scenario, Mapping) else scenario,
                    "stable": value.get("stable"),
                    **(dict(summary) if isinstance(summary, Mapping) else {}),
                }
            ]
        elif (
            isinstance(value, Mapping)
            and isinstance(value.get("records"), list)
            and "accuracy" in value
        ):
            # ScoreReport: retain aggregate accuracy rather than exploding into
            # per-example records that omit the top-level score.
            records = [{key: item for key, item in value.items() if key != "records"}]
        elif isinstance(value, Mapping) and isinstance(value.get("records"), list):
            records = value["records"]
        elif isinstance(value, Mapping) and isinstance(value.get("results"), list):
            records = value["results"]
        elif isinstance(value, Mapping):
            records = [value]
        else:
            raise ValueError(f"{artifact} must contain a JSON object or list of objects")

    normalized: list[dict[str, Any]] = []
    filename_tokens = artifact.stem.lower().replace("_", "-").split("-")
    inferred_precision = next(
        (precision for precision in PRECISION_ORDER if precision in filename_tokens),
        None,
    )
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"{artifact} contains a non-object record")
        flattened = _flatten(record)
        if inferred_precision is not None and not any(
            key in flattened for key in ("precision", "quantization", "dtype", "variant")
        ):
            flattened["precision"] = inferred_precision
        flattened["_source"] = str(artifact)
        normalized.append(flattened)
    return normalized


def _load_directory(root: Path, directory_name: str) -> tuple[dict[str, Any], ...]:
    directory = root / directory_name
    if not directory.exists():
        return ()
    records: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            records.extend(read_records(path))
    return tuple(records)


def load_artifacts(root: str | Path) -> ArtifactBundle:
    """Load the reporting contract beneath ``root`` without reading logs/checkpoints."""

    artifact_root = Path(root).expanduser().resolve()
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {artifact_root}")

    summary_records: list[dict[str, Any]] = []
    for filename in ("summary.csv", "summary.json", "summary.jsonl"):
        candidate = artifact_root / filename
        if candidate.is_file():
            summary_records.extend(read_records(candidate))

    # The experiment wrappers use ArtifactLayout.metrics_path(), which stores
    # stage records as ``metrics/<precision>-<kind>.json``.  Keep support for
    # human-friendly ``accuracy/`` and ``performance/`` folders as well, but
    # classify the canonical metrics directory by artifact kind.  Classification
    # uses only filenames: reporting must never import model/runtime code.
    accuracy_records = list(_load_directory(artifact_root, "accuracy"))
    performance_records = list(_load_directory(artifact_root, "performance"))
    runtime_status_records: list[dict[str, Any]] = []
    metrics_directory = artifact_root / "metrics"
    if metrics_directory.is_dir():
        for path in sorted(metrics_directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            name = path.stem.lower().replace("_", "-")
            records = read_records(path)
            if "runtime-status" in name:
                runtime_status_records.extend(records)
            elif any(token in name for token in ("energy", "cold-load")):
                performance_records.extend(records)
            elif any(token in name for token in ("accuracy", "evaluate", "evaluation", "mmlu", "gsm8k")):
                accuracy_records.extend(records)
            elif any(token in name for token in ("benchmark", "performance", "latency", "throughput")):
                performance_records.extend(records)
            elif any(token in name for token in ("summary", "validate", "validation", "footprint")):
                summary_records.extend(records)
    return ArtifactBundle(
        summary=tuple(summary_records),
        accuracy=tuple(accuracy_records),
        performance=tuple(performance_records),
        telemetry=_load_directory(artifact_root, "telemetry"),
        runtime_status=tuple(runtime_status_records),
    )


def _find(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    for key, value in record.items():
        if any(key.endswith(f".{name}") for name in names):
            return value
    return None


def _number(record: Mapping[str, Any], *names: str) -> float | None:
    value = _find(record, *names)
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _precision(record: Mapping[str, Any]) -> str | None:
    value = _find(record, "precision", "quantization", "dtype", "variant")
    return str(value).lower() if value is not None else None


def _averages(records: Iterable[Mapping[str, Any]], aliases: Sequence[str]) -> float | None:
    values = [value for record in records if (value := _number(record, *aliases)) is not None]
    return mean(values) if values else None


def _maximum(records: Iterable[Mapping[str, Any]], aliases: Sequence[str]) -> float | None:
    values = [value for record in records if (value := _number(record, *aliases)) is not None]
    return max(values) if values else None


def _average_accuracy(records: Iterable[Mapping[str, Any]]) -> float | None:
    """Return accuracy on a 0..1 scale, accepting fractional or percent artifacts."""

    values: list[float] = []
    for record in records:
        value = _number(record, "accuracy", "score", "exact_match")
        if value is None:
            continue
        if 1.0 < value <= 100.0:
            value /= 100.0
        if 0.0 <= value <= 1.0:
            values.append(value)
    return mean(values) if values else None


def _latest_runtime_status(
    records: Sequence[Mapping[str, Any]], precision: str
) -> Mapping[str, Any] | None:
    candidates = [record for record in records if _precision(record) == precision]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda record: (
            str(_find(record, "attempt_started_utc") or ""),
            str(record.get("_source", "")),
        ),
    )


def _unavailable_precisions(bundle: ArtifactBundle) -> set[str]:
    precisions = {
        precision
        for record in bundle.runtime_status
        if (precision := _precision(record)) is not None
    }
    return {
        precision
        for precision in precisions
        if (status := _latest_runtime_status(bundle.runtime_status, precision)) is not None
        and _find(status, "available") is False
    }


def comparison_rows(bundle: ArtifactBundle) -> list[dict[str, Any]]:
    """Build one compact comparison row per precision.

    Existing summary records are preserved and normalized.  Missing fields are
    filled, where possible, from accuracy, performance, and telemetry records.
    """

    precisions = {
        precision
        for records in (
            bundle.summary,
            bundle.accuracy,
            bundle.performance,
            bundle.telemetry,
            bundle.runtime_status,
        )
        for record in records
        if (precision := _precision(record)) is not None
    }
    rows: list[dict[str, Any]] = []
    for precision in sorted(precisions, key=lambda item: (PRECISION_ORDER.get(item, 99), item)):
        summary_records = [record for record in bundle.summary if _precision(record) == precision]
        accuracy_records = [record for record in bundle.accuracy if _precision(record) == precision]
        performance_records = [record for record in bundle.performance if _precision(record) == precision]
        telemetry_records = [record for record in bundle.telemetry if _precision(record) == precision]
        runtime_status = _latest_runtime_status(bundle.runtime_status, precision)
        runtime_available = (
            None if runtime_status is None else _find(runtime_status, "available") is True
        )
        if runtime_available is False:
            # Checkpoint footprint remains a valid storage result, but any
            # previously generated runtime quality/performance/telemetry is
            # diagnostic residue and must not enter headline comparisons.
            accuracy_records = []
            performance_records = []
            telemetry_records = []
        steady_telemetry_records = [
            record
            for record in telemetry_records
            if not any(
                excluded in str(record.get("_source", ""))
                for excluded in ("cold-load", "quantize")
            )
        ]

        row: dict[str, Any] = {"precision": precision}
        if summary_records:
            row.update({key: value for key, value in summary_records[-1].items() if key != "_source"})
            summary_accuracy = _number(row, "accuracy")
            if summary_accuracy is not None and 1.0 < summary_accuracy <= 100.0:
                row["accuracy"] = summary_accuracy / 100.0
        if runtime_status is not None:
            row["runtime_status"] = "available" if runtime_available else "unavailable"
            row["runtime_stage"] = _find(runtime_status, "stage")
            if runtime_available is False:
                row["runtime_error"] = _find(runtime_status, "exception_message")

        derived = {
            "accuracy": _average_accuracy(accuracy_records),
            "output_tokens_per_second": _averages(
                performance_records,
                ("output_tokens_per_second", "output_tokens_per_s", "output_token_throughput", "output_tps"),
            ),
            "checkpoint_size_gib": _averages(
                (*summary_records, *performance_records),
                ("checkpoint_size_gib", "model_size_gib", "checkpoint_gib"),
            ),
            "ttft_ms": (
                None
                if (ttft_s := _averages(performance_records, ("ttft_mean_s", "ttft_p50_s"))) is None
                else ttft_s * 1000.0
            ),
            "tpot_ms": (
                None
                if (tpot_s := _averages(performance_records, ("tpot_mean_s", "tpot_p50_s"))) is None
                else tpot_s * 1000.0
            ),
            "peak_vram_gib": _maximum(
                (*performance_records, *telemetry_records),
                ("peak_vram_gib", "memory_used_gib", "gpu_memory_gib"),
            ),
            "power_w": _averages(
                steady_telemetry_records,
                ("power_w", "power_draw_w", "average_power_w", "summary.average_power_w"),
            ),
            "energy_j_per_output_token": _averages(
                (*performance_records, *telemetry_records),
                (
                    "energy_j_per_output_token",
                    "joules_per_output_token",
                    "gross_joules_per_output_token",
                ),
            ),
        }
        if derived["peak_vram_gib"] is None:
            peak_bytes = _maximum(
                (*performance_records, *telemetry_records),
                (
                    "peak_memory_used_bytes",
                    "static_vram_bytes",
                    "peak_load_vram_bytes",
                    "summary.peak_memory_used_bytes",
                ),
            )
            if peak_bytes is not None:
                derived["peak_vram_gib"] = peak_bytes / 1024**3
        for key, value in derived.items():
            if key not in row and value is not None:
                row[key] = value
        if runtime_available is False:
            for key in (
                "accuracy",
                "accuracy_delta_pp",
                "output_tokens_per_second",
                "throughput_speedup",
                "ttft_ms",
                "tpot_ms",
                "peak_vram_gib",
                "power_w",
                "energy_j_per_output_token",
                "cold_load_duration_s",
                "static_vram_bytes",
                "peak_load_vram_bytes",
            ):
                row.pop(key, None)
        rows.append(row)

    baseline = next((row for row in rows if row["precision"] == "bf16"), None)
    if baseline:
        baseline_accuracy = _number(baseline, "accuracy")
        baseline_throughput = _number(baseline, "output_tokens_per_second")
        baseline_checkpoint_size = _number(baseline, "checkpoint_size_gib")
        for row in rows:
            accuracy = _number(row, "accuracy")
            throughput = _number(row, "output_tokens_per_second")
            checkpoint_size = _number(row, "checkpoint_size_gib")
            if baseline_accuracy is not None and accuracy is not None:
                row.setdefault("accuracy_delta_pp", (accuracy - baseline_accuracy) * 100.0)
            if baseline_throughput and throughput is not None:
                row.setdefault("throughput_speedup", throughput / baseline_throughput)
            if baseline_checkpoint_size and checkpoint_size:
                row.setdefault(
                    "checkpoint_compression", baseline_checkpoint_size / checkpoint_size
                )
    return rows


def summary_as_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render comparison rows as a compact Markdown table."""

    columns = [
        "precision",
        "runtime_status",
        "runtime_stage",
        "runtime_error",
        "checkpoint_size_gib",
        "checkpoint_compression",
        "accuracy",
        "accuracy_delta_pp",
        "output_tokens_per_second",
        "throughput_speedup",
        "ttft_ms",
        "tpot_ms",
        "peak_vram_gib",
        "power_w",
        "energy_j_per_output_token",
    ]
    active_columns = [column for column in columns if any(column in row for row in rows)]
    if not rows or not active_columns:
        return "_No comparable JSON/CSV artifacts were found._"

    def format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value) if value is not None else ""

    header = "| " + " | ".join(active_columns) + " |"
    separator = "| " + " | ".join("---" for _ in active_columns) + " |"
    body = [
        "| " + " | ".join(format_value(row.get(column, "")) for column in active_columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def plot_accuracy(records: Sequence[Mapping[str, Any]]):
    """Create a task-by-precision accuracy bar chart."""

    plt = _pyplot()
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in records:
        precision = _precision(record)
        task = _find(record, "task", "benchmark", "scenario") or "accuracy"
        value = _average_accuracy((record,))
        if precision is not None and value is not None:
            grouped.setdefault((str(task), precision), []).append(value)
    if not grouped:
        raise ValueError("no accuracy records with precision and score were found")

    tasks = sorted({task for task, _ in grouped})
    precisions = sorted(
        {precision for _, precision in grouped},
        key=lambda item: (PRECISION_ORDER.get(item, 99), item),
    )
    figure, axis = plt.subplots(figsize=(max(7, len(tasks) * 2), 4.5))
    width = 0.8 / len(precisions)
    positions = list(range(len(tasks)))
    for index, precision in enumerate(precisions):
        values = [mean(grouped.get((task, precision), [0.0])) for task in tasks]
        offsets = [position - 0.4 + width / 2 + index * width for position in positions]
        axis.bar(offsets, values, width=width, label=precision.upper())
    axis.set_xticks(positions, tasks, rotation=20, ha="right")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("Accuracy by precision")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_throughput(records: Sequence[Mapping[str, Any]]):
    """Create a scenario-by-precision output throughput chart."""

    plt = _pyplot()
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in records:
        precision = _precision(record)
        scenario = _find(record, "scenario", "workload", "task") or "default"
        value = _number(
            record, "output_tokens_per_second", "output_token_throughput", "output_tps"
        )
        if precision is not None and value is not None:
            grouped.setdefault((str(scenario), precision), []).append(value)
    if not grouped:
        raise ValueError("no performance records with precision and throughput were found")

    scenarios = sorted({scenario for scenario, _ in grouped})
    precisions = sorted(
        {precision for _, precision in grouped},
        key=lambda item: (PRECISION_ORDER.get(item, 99), item),
    )
    figure, axis = plt.subplots(figsize=(max(7, len(scenarios) * 2), 4.5))
    width = 0.8 / len(precisions)
    positions = list(range(len(scenarios)))
    for index, precision in enumerate(precisions):
        values = [mean(grouped.get((scenario, precision), [0.0])) for scenario in scenarios]
        offsets = [position - 0.4 + width / 2 + index * width for position in positions]
        axis.bar(offsets, values, width=width, label=precision.upper())
    axis.set_xticks(positions, scenarios, rotation=20, ha="right")
    axis.set_ylabel("Output tokens / second")
    axis.set_title("Steady-state inference throughput")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_telemetry(records: Sequence[Mapping[str, Any]], metric: str = "power_w"):
    """Plot a telemetry metric against timestamp for each precision."""

    plt = _pyplot()
    series: dict[str, list[tuple[float, float]]] = {}
    for record in records:
        precision = _precision(record)
        timestamp = _number(record, "elapsed_s", "timestamp_s", "time_s")
        value = _number(record, metric)
        if precision is not None and timestamp is not None and value is not None:
            series.setdefault(precision, []).append((timestamp, value))
    if not series:
        raise ValueError(f"no telemetry records contain precision, time, and {metric}")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for precision in sorted(series, key=lambda item: (PRECISION_ORDER.get(item, 99), item)):
        points = sorted(series[precision])
        axis.plot([point[0] for point in points], [point[1] for point in points], label=precision.upper())
    axis.set_xlabel("Elapsed time (s)")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_title("GPU telemetry")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_pareto(rows: Sequence[Mapping[str, Any]]):
    """Plot accuracy delta against output throughput."""

    plt = _pyplot()
    usable = [
        row
        for row in rows
        if _number(row, "accuracy_delta_pp") is not None
        and _number(row, "output_tokens_per_second") is not None
    ]
    if not usable:
        raise ValueError("comparison rows do not contain accuracy delta and throughput")
    figure, axis = plt.subplots(figsize=(6.5, 4.8))
    for row in usable:
        precision = str(row.get("precision", "unknown"))
        x = _number(row, "accuracy_delta_pp")
        y = _number(row, "output_tokens_per_second")
        memory = _number(row, "peak_vram_gib") or 8.0
        axis.scatter(x, y, s=max(memory, 1.0) * 8.0, alpha=0.75, label=precision.upper())
        axis.annotate(precision.upper(), (x, y), xytext=(5, 5), textcoords="offset points")
    axis.axvline(0, color="grey", linewidth=1, linestyle="--")
    axis.set_xlabel("Accuracy delta vs BF16 (percentage points)")
    axis.set_ylabel("Output tokens / second")
    axis.set_title("Quality-throughput Pareto view (bubble = peak VRAM)")
    figure.tight_layout()
    return figure


def plot_resource_comparison(rows: Sequence[Mapping[str, Any]]):
    """Plot checkpoint size and peak VRAM for each precision when available."""

    plt = _pyplot()
    metrics = (
        ("checkpoint_size_gib", "Checkpoint size (GiB)"),
        ("peak_vram_gib", "Peak VRAM (GiB)"),
    )
    active = [
        (key, label)
        for key, label in metrics
        if any(_number(row, key) is not None for row in rows)
    ]
    if not active:
        raise ValueError("comparison rows contain neither checkpoint size nor peak VRAM")
    ordered = sorted(
        rows,
        key=lambda row: (
            PRECISION_ORDER.get(str(row.get("precision", "")).lower(), 99),
            str(row.get("precision", "")),
        ),
    )
    figure, axes = plt.subplots(1, len(active), figsize=(5.2 * len(active), 4.2), squeeze=False)
    labels = [str(row.get("precision", "unknown")).upper() for row in ordered]
    for axis, (key, label) in zip(axes[0], active):
        values = [_number(row, key) or 0.0 for row in ordered]
        axis.bar(labels, values)
        axis.set_ylabel(label)
        axis.set_title(label)
    figure.suptitle("Model storage and memory footprint")
    figure.tight_layout()
    return figure


def plot_latency(records: Sequence[Mapping[str, Any]]):
    """Plot mean TTFT and TPOT per precision from performance artifacts."""

    plt = _pyplot()
    aliases = (
        ("TTFT", ("ttft_ms", "mean_ttft_ms", "p50_ttft_ms")),
        ("TPOT", ("tpot_ms", "mean_tpot_ms", "p50_tpot_ms")),
    )
    precisions = sorted(
        {precision for record in records if (precision := _precision(record)) is not None},
        key=lambda item: (PRECISION_ORDER.get(item, 99), item),
    )
    values_by_metric = {
        label: [
            _averages(
                [record for record in records if _precision(record) == precision], aliases_
            )
            for precision in precisions
        ]
        for label, aliases_ in aliases
    }
    if not precisions or not any(
        value is not None for values in values_by_metric.values() for value in values
    ):
        raise ValueError("no TTFT or TPOT metrics were found")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    positions = list(range(len(precisions)))
    for index, (label, values) in enumerate(values_by_metric.items()):
        offsets = [position + (index - 0.5) * width for position in positions]
        axis.bar(offsets, [value or 0.0 for value in values], width=width, label=label)
    axis.set_xticks(positions, [precision.upper() for precision in precisions])
    axis.set_ylabel("Milliseconds")
    axis.set_title("Latency components (lower is better)")
    axis.legend()
    figure.tight_layout()
    return figure


def load_tensor_analyses(artifact_root: str | Path) -> tuple[dict[str, Any], ...]:
    """Load durable packed-tensor analysis manifests without opening checkpoints."""

    manifests = Path(artifact_root) / "manifests"
    reports: list[dict[str, Any]] = []
    for path in sorted(manifests.glob("*-tensor-analysis.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid tensor analysis manifest: {path}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("tensors"), list):
            raise ValueError(f"Invalid tensor analysis manifest: {path}")
        reports.append(value)
    return tuple(reports)


def plot_tensor_error_cdf(reports: Sequence[Mapping[str, Any]]):
    """Plot stored absolute-error CDF points from real packed tensors."""

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, 4.8))
    plotted = 0
    for report in reports:
        precision = str(report.get("precision", "unknown")).upper()
        tensors = report.get("tensors", [])
        if not isinstance(tensors, list):
            continue
        for tensor in tensors:
            if not isinstance(tensor, Mapping):
                continue
            cdf = tensor.get("absolute_error_cdf")
            if not isinstance(cdf, Mapping):
                continue
            points: list[tuple[float, float]] = []
            for percentile, value in cdf.items():
                try:
                    probability = float(str(percentile).removeprefix("p"))
                    error = float(value)
                except (TypeError, ValueError):
                    continue
                points.append((probability, error))
            if not points:
                continue
            points.sort()
            name = str(tensor.get("name", "tensor"))
            short_name = ".".join(name.split(".")[-5:-1])
            axis.plot(
                [item[0] for item in points],
                [item[1] for item in points],
                marker=".",
                label=f"{precision} {short_name}",
            )
            plotted += 1
    if not plotted:
        plt.close(figure)
        raise ValueError("no packed-tensor absolute-error CDFs were found")
    if any(line.get_ydata().max() > 0 for line in axis.lines):
        axis.set_yscale("symlog", linthresh=1e-8)
    axis.set_xlabel("Percentile")
    axis.set_ylabel("Absolute BF16 reconstruction error")
    axis.set_title("Real ModelOpt packed-weight error CDF")
    axis.legend(fontsize="small")
    figure.tight_layout()
    return figure


def plot_scale_distributions(reports: Sequence[Mapping[str, Any]]):
    """Plot stored positive weight-scale histograms for each packed precision."""

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    plotted = 0
    for report in reports:
        precision = str(report.get("precision", "unknown")).upper()
        aggregate = report.get("aggregate_scale_distributions")
        if not isinstance(aggregate, Mapping):
            continue
        for scale_name, distribution in aggregate.items():
            if not isinstance(distribution, Mapping):
                continue
            histogram = distribution.get("positive_log10_histogram")
            if not isinstance(histogram, Mapping):
                continue
            edges = histogram.get("bin_edges")
            counts = histogram.get("counts")
            if not isinstance(edges, list) or not isinstance(counts, list):
                continue
            if len(edges) != len(counts) + 1 or not counts:
                continue
            total = sum(float(value) for value in counts)
            if total <= 0:
                continue
            centers = [10 ** ((float(left) + float(right)) / 2) for left, right in zip(edges, edges[1:])]
            axis.plot(
                centers,
                [float(value) / total for value in counts],
                marker=".",
                label=f"{precision} {scale_name}",
            )
            plotted += 1
    if not plotted:
        plt.close(figure)
        raise ValueError("no packed-tensor scale histograms were found")
    axis.set_xscale("log")
    axis.set_xlabel("Scale value")
    axis.set_ylabel("Fraction of observed scales")
    axis.set_title("Real packed-checkpoint scale distributions")
    axis.legend(fontsize="small")
    figure.tight_layout()
    return figure


def create_dashboard(artifact_root: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Load artifacts, build the comparison, and optionally save PNG plots."""

    bundle = load_artifacts(artifact_root)
    rows = comparison_rows(bundle)
    unavailable = _unavailable_precisions(bundle)
    valid_accuracy = tuple(
        record for record in bundle.accuracy if _precision(record) not in unavailable
    )
    valid_performance = tuple(
        record for record in bundle.performance if _precision(record) not in unavailable
    )
    valid_telemetry = tuple(
        record for record in bundle.telemetry if _precision(record) not in unavailable
    )
    tensor_analyses = load_tensor_analyses(artifact_root)
    figures: dict[str, Any] = {}
    plot_builders = {
        "accuracy": lambda: plot_accuracy(valid_accuracy),
        "throughput": lambda: plot_throughput(valid_performance),
        "latency": lambda: plot_latency(valid_performance),
        "resources": lambda: plot_resource_comparison(rows),
        "telemetry_power": lambda: plot_telemetry(valid_telemetry, "power_w"),
        "pareto": lambda: plot_pareto(rows),
        "packed_tensor_error_cdf": lambda: plot_tensor_error_cdf(tensor_analyses),
        "packed_scale_distributions": lambda: plot_scale_distributions(tensor_analyses),
    }
    for name, builder in plot_builders.items():
        try:
            figures[name] = builder()
        except ValueError:
            continue

    saved: dict[str, str] = {}
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            path = destination / f"{name}.png"
            figure.savefig(path, dpi=150, bbox_inches="tight")
            saved[name] = str(path)
    return {
        "bundle": bundle,
        "rows": rows,
        "markdown": summary_as_markdown(rows),
        "figures": figures,
        "saved": saved,
        "tensor_analyses": tensor_analyses,
        "unavailable_precisions": sorted(
            unavailable, key=lambda item: (PRECISION_ORDER.get(item, 99), item)
        ),
    }
