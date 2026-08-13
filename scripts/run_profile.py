#!/usr/bin/env python3
"""Run the pinned Blackwell PTQ experiment with fail-closed stage gates."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import time
import warnings
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection

from transformers import AutoTokenizer

from ptq_workshop.artifacts import (
    ArtifactLayout,
    initialize_run,
    require_checkpoint_validation,
    write_derived_json_atomic,
    write_json_atomic,
)
from ptq_workshop.benchmark import (
    canonical_performance_scenarios,
    energy_per_output_token,
    prepare_benchmark_plan,
    run_profile_performance_scenario,
    stream_completion_request,
    write_benchmark_result,
)
from ptq_workshop.checkpoint_analysis import analyze_packed_checkpoint
from ptq_workshop.config import POST_PREP_MINIMUM_FREE_DISK_GIB, ProfileName, make_config
from ptq_workshop.evaluate import (
    prepare_eval_manifest,
    bootstrap_accuracy_delta_ci,
    request_evaluation_prediction,
    score_predictions,
    write_raw_predictions,
    write_score_report,
)
from ptq_workshop.io import read_jsonl, sha256_json
from ptq_workshop.observability import parse_modelopt_phase_durations
from ptq_workshop.preflight import preflight_or_raise, write_preflight_manifest
from ptq_workshop.quantize import (
    build_quantization_job,
    run_quantization,
    validate_export,
)
from ptq_workshop.reporting import create_dashboard
from ptq_workshop.serving import (
    SMOKE_RESPONSE_SENTINEL,
    TensorRTLLMAutoDeployServer,
    require_exact_smoke_response,
    request_json,
    save_server_manifest,
    server_config_for_workshop,
)
from ptq_workshop.telemetry import (
    NVMLSampler,
    validate_compute_process_ownership,
    validate_telemetry_quality,
    write_telemetry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "preflight",
    "quantize",
    "validate",
    "analyze",
    "evaluate",
    "benchmark",
    "report",
    "all",
)


def load_prepared(path: Path, config: Any) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Run scripts/prepare_assets.sh first; missing {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "download_only": True,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Prepared manifest {key}={manifest.get(key)!r}; expected {expected!r}"
            )
    preparation_preflight = manifest.get("preparation_preflight", {})
    recorded_preflight = (
        preparation_preflight.get("passed") is True
        and preparation_preflight.get("profile") == config.profile.name.value
        and preparation_preflight.get("model_id") == config.model_id
        and preparation_preflight.get("model_revision") == config.model_revision
        and int(preparation_preflight.get("minimum_free_disk_gib", 0))
        == config.profile.minimum_free_disk_gib
    )
    recovery = manifest.get("recovery") or {}
    recovered_complete_cache = (
        recovery.get("recovered_complete_cache") is True
        and recovery.get("historical_145_gib_gate_recorded") is False
        and int(recovery.get("remaining_artifact_budget_gib", 0))
        == POST_PREP_MINIMUM_FREE_DISK_GIB
        and recovery.get("snapshot_verification", {}).get("verified") is True
    )
    if not recorded_preflight and not recovered_complete_cache:
        raise ValueError(
            "Prepared assets prove neither the 145 GiB pre-download gate nor the "
            "verified-complete-cache plus 55 GiB recovery path"
        )
    if manifest.get("snapshot_verification", {}).get("verified") is not True:
        raise ValueError("Prepared manifest lacks a successful pinned-snapshot verification")
    frozen_files = {
        "calibration": manifest["calibration"],
        "mmlu_pro": manifest["evaluation"]["mmlu_pro"],
        "gsm8k": manifest["evaluation"]["gsm8k"],
    }
    for name, entry in frozen_files.items():
        frozen_path = Path(entry["path"])
        rows = read_jsonl(frozen_path)
        actual_hash = sha256_json(rows)
        if actual_hash != entry["sha256"]:
            raise ValueError(
                f"Prepared {name} hash mismatch at {frozen_path}: {actual_hash} != {entry['sha256']}"
            )
        if len(rows) != int(entry["count"]):
            raise ValueError(f"Prepared {name} count changed at {frozen_path}")
    return manifest


def layout_for_args(config: Any, run_dir: Path | None, stage: str) -> ArtifactLayout:
    if run_dir is None:
        if stage not in {"preflight", "quantize", "all"} and config.profile.name is not ProfileName.DEV_SMOKE:
            raise ValueError("--run-dir is required to resume a timestamped workshop/FULL run")
        return ArtifactLayout.from_config(config)
    resolved = run_dir.expanduser().resolve()
    if resolved.parent != config.artifact_root.resolve():
        raise ValueError(f"--run-dir must be directly under {config.artifact_root}")
    fingerprint = resolved.name.rsplit("-", 1)[-1]
    return ArtifactLayout(config.artifact_root, fingerprint, resolved.name)


def ensure_layout(config: Any, layout: ArtifactLayout) -> None:
    initialize_run(config, layout)
    (layout.run_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (layout.run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (layout.run_dir / "figures").mkdir(parents=True, exist_ok=True)


def validate_checkpoints(
    source: Path,
    layout: ArtifactLayout,
    variants: tuple[str, ...] = ("fp8", "nvfp4"),
) -> dict[str, Any]:
    unknown = set(variants) - {"fp8", "nvfp4"}
    if unknown:
        raise ValueError(f"Unknown checkpoint validation variants: {sorted(unknown)}")
    from validate_checkpoint import validate_bf16

    results = {"bf16": validate_bf16(source)}
    for variant in variants:
        results[variant] = validate_export(
            layout.checkpoint_dir(variant),
            expected_variant=variant,
            source_model=source,
        )
        results[variant]["precision"] = variant
        if results[variant]["bytes"] >= results["bf16"]["bytes"]:
            raise RuntimeError(f"{variant} checkpoint did not shrink versus BF16")
    for variant, result in results.items():
        write_derived_json_atomic(
            layout.run_dir / "metrics" / f"{variant}-footprint.json",
            {
                "precision": variant,
                "checkpoint_size_gib": result["bytes"] / 1024**3,
                **result,
            },
        )
    write_derived_json_atomic(
        layout.run_dir / "manifests" / "checkpoint-validation.json", results
    )
    return results


def run_quantization_with_observability(
    job: Any,
    *,
    allow_existing_valid_export: bool,
    quantization_runner: Callable[..., Any] = run_quantization,
    sampler_factory: Callable[..., Any] = NVMLSampler,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> Any:
    """Run fresh PTQ under NVML sampling and preserve observations on failure.

    A resumable DEV_SMOKE export is validated by ``run_quantization`` but is
    not mislabeled as a newly measured PTQ run.
    """

    if job.output_dir.exists():
        return quantization_runner(
            job, allow_existing_valid_export=allow_existing_valid_export
        )

    sampler = sampler_factory(interval_s=0.1).start()
    started_utc = datetime.now(timezone.utc).isoformat()
    started = monotonic_clock()

    def persist(status: str, exception: BaseException | None = None) -> None:
        ended = monotonic_clock()
        summary = sampler.stop()
        telemetry_json = (
            job.metadata_path.parent.parent
            / "telemetry"
            / f"{job.variant}-quantize.json"
        )
        telemetry_csv = telemetry_json.with_suffix(".csv")
        write_telemetry(
            sampler.samples,
            json_path=telemetry_json,
            csv_path=telemetry_csv,
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": "modelopt_ptq_observability",
            "precision": job.variant,
            "status": status,
            "started_utc": started_utc,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "total_wall_duration_s": max(0.0, ended - started),
            "phase_durations": parse_modelopt_phase_durations(job.log_path),
            "vram_measurement_scope": (
                "whole-device NVML memory used; the preflight requires an idle selected GPU"
            ),
            "peak_vram_bytes": summary.peak_memory_used_bytes,
            "peak_vram_gib": (
                None
                if summary.peak_memory_used_bytes is None
                else summary.peak_memory_used_bytes / 1024**3
            ),
            "telemetry_summary": asdict(summary),
            "telemetry_json_path": str(telemetry_json.resolve()),
            "telemetry_csv_path": str(telemetry_csv.resolve()),
            "quantization_manifest_path": str(job.metadata_path.resolve()),
            "log_path": str(job.log_path.resolve()),
        }
        if exception is not None:
            payload.update(
                exception_type=type(exception).__name__,
                exception_message=str(exception),
            )
        write_derived_json_atomic(
            job.metadata_path.parent / f"{job.variant}-ptq-observability.json",
            payload,
        )

    try:
        result = quantization_runner(
            job, allow_existing_valid_export=allow_existing_valid_export
        )
    except BaseException as exc:
        persist("exception", exc)
        raise
    persist("complete")
    return result


def analyze_checkpoints(
    source: Path,
    layout: ArtifactLayout,
    variants: tuple[str, ...],
    *,
    analyzer: Callable[..., dict[str, Any]] = analyze_packed_checkpoint,
) -> dict[str, Any]:
    """Write durable BF16-versus-packed tensor analysis manifests."""

    unknown = sorted(set(variants) - {"fp8", "nvfp4"})
    if unknown:
        raise ValueError(f"Unknown checkpoint analysis variants: {unknown}")
    reports: dict[str, Any] = {}
    for variant in variants:
        reports[variant] = analyzer(
            source,
            layout.checkpoint_dir(variant),
            variant=variant,
        )
        write_derived_json_atomic(
            layout.run_dir / "manifests" / f"{variant}-tensor-analysis.json",
            reports[variant],
        )
    return reports


def evaluation_examples(prepared: dict[str, Any], layout: ArtifactLayout, config: Any) -> dict[str, Any]:
    manifests: dict[str, Any] = {}
    tasks = {
        "mmlu_pro": (Path(prepared["evaluation"]["mmlu_pro"]["path"]), config.profile.mmlu_samples),
        "gsm8k": (Path(prepared["evaluation"]["gsm8k"]["path"]), config.profile.gsm8k_samples),
    }
    for task, (source_path, count) in tasks.items():
        rows = read_jsonl(source_path)

        def loader(*_: Any, frozen: list[dict[str, Any]] = rows) -> list[dict[str, Any]]:
            return frozen

        manifests[task] = prepare_eval_manifest(
            task,
            layout.run_dir / "manifests" / f"{task}.jsonl",
            limit=count,
            seed=config.seed,
            dataset_loader=loader,
        )
    return manifests


def run_accuracy(
    *,
    variant: str,
    server_url: str,
    model_name: str,
    examples: dict[str, Any],
    layout: ArtifactLayout,
    seed: int,
) -> None:
    for task, task_examples in examples.items():
        max_tokens = 32 if task == "mmlu_pro" else 512
        predictions = [
            request_evaluation_prediction(
                endpoint=f"{server_url}/v1/chat/completions",
                model=model_name,
                example=example,
                max_tokens=max_tokens,
                seed=seed,
            )
            for example in task_examples
        ]
        write_raw_predictions(
            predictions,
            layout.run_dir / "predictions" / f"{variant}-{task}.jsonl",
        )
        report = score_predictions(
            task_examples,
            {prediction.example_id: prediction.text for prediction in predictions},
            seed=seed,
        )
        write_score_report(
            report,
            json_path=layout.run_dir / "metrics" / f"{variant}-{task}.json",
            csv_path=layout.run_dir / "metrics" / f"{variant}-{task}.csv",
        )


def write_accuracy_acceptance(layout: ArtifactLayout, tasks: tuple[str, ...]) -> dict[str, Any]:
    """Persist acceptance evidence, then fail if any parsed-answer rate is below 98%."""

    acceptance: dict[str, Any] = {
        "parsed_answer_threshold": 0.98,
        "quality_warning_pp": 5.0,
        "tasks": {},
        "failures": [],
    }
    for task in tasks:
        by_variant: dict[str, Any] = {}
        for variant in ("bf16", "fp8", "nvfp4"):
            path = layout.run_dir / "metrics" / f"{variant}-{task}.json"
            if path.is_file():
                by_variant[variant] = json.loads(path.read_text(encoding="utf-8"))
        if not by_variant:
            continue
        baseline_records = by_variant.get("bf16", {}).get("records", [])
        baseline_by_id = {
            row["example_id"]: bool(row["correct"]) for row in baseline_records
        }
        task_results: dict[str, Any] = {}
        for variant, report in by_variant.items():
            records = report["records"]
            if not records:
                raise ValueError(f"{variant}/{task} score report contains no records")
            parsed_rate = sum(row["normalized_prediction"] is not None for row in records) / len(records)
            result: dict[str, Any] = {
                "accuracy": report["accuracy"],
                "parsed_answer_rate": parsed_rate,
                "parsed_answer_gate_passed": parsed_rate >= 0.98,
            }
            if not result["parsed_answer_gate_passed"]:
                acceptance["failures"].append(
                    {
                        "precision": variant,
                        "task": task,
                        "parsed_answer_rate": parsed_rate,
                        "required_rate": acceptance["parsed_answer_threshold"],
                    }
                )
            if variant != "bf16" and baseline_by_id:
                candidate_by_id = {row["example_id"]: bool(row["correct"]) for row in records}
                ids = sorted(set(baseline_by_id) & set(candidate_by_id))
                interval = bootstrap_accuracy_delta_ci(
                    [baseline_by_id[item] for item in ids],
                    [candidate_by_id[item] for item in ids],
                    seed=42,
                )
                delta_pp = interval.estimate * 100.0
                result.update(
                    accuracy_delta_from_bf16_pp=delta_pp,
                    paired_delta_ci=asdict(interval),
                    quality_warning=delta_pp < -5.0,
                )
                if result["quality_warning"]:
                    warnings.warn(f"{variant}/{task} accuracy regressed {delta_pp:.2f} pp versus BF16")
            task_results[variant] = result
        acceptance["tasks"][task] = task_results
    acceptance["passed"] = not acceptance["failures"]
    write_json_atomic(
        layout.run_dir / "metrics" / "accuracy-acceptance.json",
        acceptance,
        overwrite=True,
    )
    if not acceptance["passed"]:
        detail = ", ".join(
            f"{item['precision']}/{item['task']}={item['parsed_answer_rate']:.2%}"
            for item in acceptance["failures"]
        )
        raise RuntimeError(
            "Parsed-answer acceptance failed; raw predictions, score reports, and "
            f"acceptance artifact were preserved ({detail}; required >=98%)"
        )
    return acceptance


def run_benchmarks(
    *,
    variant: str,
    server_url: str,
    model_name: str,
    tokenizer: Any,
    layout: ArtifactLayout,
    config: Any,
    allowed_compute_process_pids: Collection[int],
    scenario_filter: str = "all",
) -> None:
    scenarios = canonical_performance_scenarios(
        config.profile,
        include_full_prefill=config.profile.name is ProfileName.FULL,
        seed=config.seed,
    )
    if scenario_filter != "all":
        scenarios = tuple(item for item in scenarios if item.name == scenario_filter)
        if not scenarios:
            raise ValueError(f"Unknown scenario {scenario_filter!r}")
    for original_scenario in scenarios:
        scenario = replace(original_scenario, warmup_passes=0)
        plan = prepare_benchmark_plan(
            base_url=server_url,
            model=model_name,
            tokenizer=tokenizer,
            scenario=scenario,
        )

        def request(request_id: str) -> Any:
            return stream_completion_request(
                endpoint=plan.endpoint,
                payload=plan.request_payload,
                request_id=request_id,
                tokenizer=tokenizer,
                expected_input_tokens=scenario.input_tokens,
                expected_output_tokens=scenario.output_tokens,
            )

        # Warm-up is completed before telemetry begins, so energy covers only
        # measured repetitions.
        for warmup_pass in range(original_scenario.warmup_passes):
            with concurrent.futures.ThreadPoolExecutor(max_workers=scenario.concurrency) as executor:
                futures = [
                    executor.submit(request, f"warmup-{warmup_pass}-request-{index}")
                    for index in range(scenario.request_count)
                ]
                for future in futures:
                    future.result()
        sampler = NVMLSampler(interval_s=0.1).start()
        try:
            result = run_profile_performance_scenario(config, scenario, request)
        finally:
            sampler.stop()
        write_benchmark_result(
            result,
            json_path=layout.run_dir / "metrics" / f"{variant}-{scenario.name}-performance.json",
            csv_path=layout.run_dir / "metrics" / f"{variant}-{scenario.name}-performance.csv",
        )
        write_telemetry(
            sampler.samples,
            json_path=layout.run_dir / "telemetry" / f"{variant}-{scenario.name}.json",
            csv_path=layout.run_dir / "telemetry" / f"{variant}-{scenario.name}.csv",
        )
        telemetry_summary = validate_telemetry_quality(
            sampler.samples,
            sampling_interval_s=sampler.interval_s,
            allowed_compute_process_pids=allowed_compute_process_pids,
        )
        accepted = None if result.accepted_attempt is None else result.attempts[result.accepted_attempt - 1]
        if accepted is not None and accepted.summary is not None:
            total_output_tokens = sum(
                timing.output_tokens for attempt in result.attempts for timing in attempt.timings
            )
            energy = energy_per_output_token(
                gross_energy_j=telemetry_summary.energy_j,
                duration_s=telemetry_summary.duration_s,
                output_tokens=total_output_tokens,
            )
            write_derived_json_atomic(
                layout.run_dir / "metrics" / f"{variant}-{scenario.name}-energy.json",
                {"precision": variant, "scenario": scenario.name, **asdict(energy)},
            )
        if not result.stable:
            raise RuntimeError(
                f"Benchmark remained unstable after all attempts: {variant}/{scenario.name}"
            )


def run_runtime(
    *,
    source: Path,
    prepared: dict[str, Any],
    layout: ArtifactLayout,
    config: Any,
    do_evaluate: bool,
    do_benchmark: bool,
    precision_filter: str = "all",
    task_filter: str = "all",
    scenario_filter: str = "all",
) -> None:
    examples = evaluation_examples(prepared, layout, config) if do_evaluate else {}
    if task_filter != "all":
        if task_filter not in examples:
            raise ValueError(f"Unknown evaluation task {task_filter!r}")
        examples = {task_filter: examples[task_filter]}
    tokenizer = (
        AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        if do_benchmark
        else None
    )
    variants = {
        "bf16": source,
        "fp8": layout.checkpoint_dir("fp8"),
        "nvfp4": layout.checkpoint_dir("nvfp4"),
    }
    if precision_filter != "all":
        variants = {precision_filter: variants[precision_filter]}
    for port_offset, (variant, model_path) in enumerate(variants.items()):
        runtime_label = (
            "all" if do_evaluate and do_benchmark else "evaluate" if do_evaluate else "benchmark"
        )
        attempt_started_utc = datetime.now(timezone.utc).isoformat()
        status_path = (
            layout.run_dir
            / "metrics"
            / f"{variant}-{runtime_label}-runtime-status.json"
        )
        server_log_path = layout.log_path(variant, f"{runtime_label}-serve")
        server_manifest_path = (
            layout.run_dir
            / "manifests"
            / f"{variant}-{runtime_label}-server.json"
        )
        runtime_stage = "startup"
        smoke_observed: str | None = None
        server_config = server_config_for_workshop(
            config,
            model_path=model_path,
            port=8000 + port_offset,
        )
        # Absence of KV quantizers in both recipes plus auto runtime selection keeps
        # the source and exported variants on the same BF16 KV-cache path.
        save_server_manifest(
            server_config,
            server_manifest_path,
        )
        server = TensorRTLLMAutoDeployServer(
            server_config,
            log_path=server_log_path,
        )
        try:
            cold_sampler = NVMLSampler(interval_s=0.1).start()
            cold_started = time.monotonic()
            try:
                server.start()
                cold_duration_s = time.monotonic() - cold_started
            finally:
                cold_summary = cold_sampler.stop()
            write_telemetry(
                cold_sampler.samples,
                json_path=(
                    layout.run_dir
                    / "telemetry"
                    / f"{variant}-{runtime_label}-cold-load.json"
                ),
                csv_path=(
                    layout.run_dir
                    / "telemetry"
                    / f"{variant}-{runtime_label}-cold-load.csv"
                ),
            )
            static_vram = (
                cold_sampler.samples[-1].memory_used_bytes
                if cold_sampler.samples
                else None
            )
            write_derived_json_atomic(
                layout.run_dir / "metrics" / f"{variant}-{runtime_label}-cold-load.json",
                {
                    "precision": variant,
                    "cold_load_duration_s": cold_duration_s,
                    "static_vram_bytes": static_vram,
                    "peak_load_vram_bytes": cold_summary.peak_memory_used_bytes,
                },
            )
            runtime_stage = "smoke"
            smoke_payload = {
                "model": str(model_path),
                "messages": [
                    {
                        "role": "user",
                        "content": f"Reply with exactly: {SMOKE_RESPONSE_SENTINEL}",
                    }
                ],
                "max_tokens": 16,
                "temperature": 0.0,
                "seed": config.seed,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            smoke_response = request_json(
                f"{server_config.base_url}/v1/chat/completions",
                payload=smoke_payload,
                timeout_s=300.0,
            )
            try:
                smoke_observed = str(
                    smoke_response["choices"][0]["message"]["content"]
                ).strip()
            except (KeyError, IndexError, TypeError):
                smoke_observed = None
            smoke_text = require_exact_smoke_response(smoke_response, variant=variant)
            write_derived_json_atomic(
                layout.run_dir / "metrics" / f"{variant}-{runtime_label}-smoke.json",
                {
                    "precision": variant,
                    "request": smoke_payload,
                    "response": smoke_response,
                    "expected_text": SMOKE_RESPONSE_SENTINEL,
                    "observed_text": smoke_text,
                    "exact_match": True,
                },
            )
            if do_evaluate:
                runtime_stage = "evaluate"
                run_accuracy(
                    variant=variant,
                    server_url=server_config.base_url,
                    model_name=str(model_path),
                    examples=examples,
                    layout=layout,
                    seed=config.seed,
                )
            if do_benchmark:
                runtime_stage = "benchmark"
                process_sampler = NVMLSampler(interval_s=0.1)
                try:
                    process_sample = process_sampler.collect_once()
                finally:
                    process_sampler.stop()
                if server.process is None:
                    raise RuntimeError("TensorRT-LLM server process is unavailable")
                allowed_compute_process_pids = validate_compute_process_ownership(
                    process_sample.compute_process_pids,
                    owner_pid=int(server.process.pid),
                )
                run_benchmarks(
                    variant=variant,
                    server_url=server_config.base_url,
                    model_name=str(model_path),
                    tokenizer=tokenizer,
                    layout=layout,
                    config=config,
                    allowed_compute_process_pids=allowed_compute_process_pids,
                    scenario_filter=scenario_filter,
                )
            write_derived_json_atomic(
                status_path,
                {
                    "precision": variant,
                    "runtime_label": runtime_label,
                    "available": True,
                    "stage": "complete",
                    "attempt_started_utc": attempt_started_utc,
                    "server_log_path": str(server_log_path.resolve()),
                    "server_manifest_path": str(server_manifest_path.resolve()),
                    "smoke_expected": SMOKE_RESPONSE_SENTINEL,
                    "smoke_observed": smoke_text,
                },
            )
        except BaseException as exc:
            write_derived_json_atomic(
                status_path,
                {
                    "precision": variant,
                    "runtime_label": runtime_label,
                    "available": False,
                    "stage": runtime_stage,
                    "attempt_started_utc": attempt_started_utc,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "server_log_path": str(server_log_path.resolve()),
                    "server_manifest_path": str(server_manifest_path.resolve()),
                    "smoke_expected": SMOKE_RESPONSE_SENTINEL,
                    "smoke_observed": smoke_observed,
                },
            )
            raise
        finally:
            server.stop()
    if do_evaluate:
        write_accuracy_acceptance(layout, tuple(examples))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="WORKSHOP_B200")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--precision", choices=("all", "bf16", "fp8", "nvfp4"), default="all")
    parser.add_argument("--task", choices=("all", "mmlu_pro", "gsm8k"), default="all")
    parser.add_argument(
        "--scenario",
        choices=("all", "interactive", "rag_balanced", "throughput", "full_prefill"),
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepared-root", type=Path, default=PROJECT_ROOT / "artifacts" / "prepared")
    parser.add_argument(
        "--modelopt-root",
        type=Path,
        default=PROJECT_ROOT / ".cache" / "Model-Optimizer-0.46.0rc0",
    )
    args = parser.parse_args()
    config = make_config(args.profile, project_root=PROJECT_ROOT)
    layout = layout_for_args(config, args.run_dir, args.stage)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "profile": config.profile.name.value,
                    "stage": args.stage,
                    "run_dir": str(layout.run_dir),
                    "precision": args.precision,
                    "task": args.task,
                    "scenario": args.scenario,
                    "mutations": [],
                },
                indent=2,
            )
        )
        return
    ensure_layout(config, layout)

    if args.stage in {"preflight", "all"}:
        optional_prepared_path = args.prepared_root / "prepared_manifest.json"
        optional_prepared = (
            load_prepared(optional_prepared_path, config)
            if optional_prepared_path.is_file()
            else None
        )
        disk_budget = (
            POST_PREP_MINIMUM_FREE_DISK_GIB
            if optional_prepared is not None
            else config.profile.minimum_free_disk_gib
        )
        report = preflight_or_raise(
            config,
            modelopt_root=args.modelopt_root,
            minimum_free_disk_gib=disk_budget,
        )
        write_preflight_manifest(
            layout.run_dir / "manifests" / "environment.json",
            config,
            report,
            prepared_manifest=optional_prepared,
        )
    if args.stage == "preflight":
        print(json.dumps({"run_dir": str(layout.run_dir), "stage": args.stage}, indent=2))
        return
    prepared = load_prepared(args.prepared_root / "prepared_manifest.json", config)
    source = Path(prepared["model_snapshot"]).expanduser().resolve()
    write_json_atomic(layout.run_dir / "manifests" / "prepared-assets.json", prepared)
    if args.stage in {"quantize", "all"}:
        variants = ("fp8", "nvfp4") if args.precision == "all" else (args.precision,)
        if "bf16" in variants:
            raise ValueError("BF16 is the immutable source and is not a quantization target")
        for variant in variants:
            job = build_quantization_job(
                config,
                layout,
                variant=variant,
                source_model=source,
                calibration_jsonl=prepared["calibration"]["path"],
                modelopt_root=args.modelopt_root,
            )
            run_quantization_with_observability(
                job,
                allow_existing_valid_export=config.profile.name is ProfileName.DEV_SMOKE,
            )
    if args.stage in {"validate", "all"}:
        validation_variants = (
            ()
            if args.precision == "bf16"
            else ("fp8", "nvfp4")
            if args.precision == "all"
            else (args.precision,)
        )
        validate_checkpoints(source, layout, validation_variants)
    if args.stage in {"analyze", "all"}:
        analysis_variants = (
            ("fp8", "nvfp4")
            if args.precision == "all"
            else ()
            if args.precision == "bf16"
            else (args.precision,)
        )
        if not analysis_variants:
            raise ValueError("Packed tensor analysis requires FP8 or NVFP4 precision")
        require_checkpoint_validation(
            layout,
            source_model=source,
            variants=analysis_variants,
        )
        analyze_checkpoints(source, layout, analysis_variants)
    if args.stage in {"evaluate", "benchmark", "all"}:
        runtime_variants = (
            ("bf16", "fp8", "nvfp4") if args.precision == "all" else (args.precision,)
        )
        require_checkpoint_validation(
            layout,
            source_model=source,
            variants=runtime_variants,
        )
        run_runtime(
            source=source,
            prepared=prepared,
            layout=layout,
            config=config,
            do_evaluate=args.stage in {"evaluate", "all"},
            do_benchmark=args.stage in {"benchmark", "all"},
            precision_filter=args.precision,
            task_filter=args.task,
            scenario_filter=args.scenario,
        )
    if args.stage in {"report", "all"}:
        dashboard = create_dashboard(layout.run_dir, layout.run_dir / "figures")
        # Reports are derived exclusively from saved JSON/CSV artifacts and are
        # intentionally regenerable after a completed retry or new runtime status.
        write_derived_json_atomic(layout.run_dir / "summary.json", dashboard["rows"])
        with (layout.run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            rows = dashboard["rows"]
            fieldnames = sorted({key for row in rows for key in row}) if rows else []
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)
        (layout.run_dir / "summary.md").write_text(dashboard["markdown"] + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(layout.run_dir), "stage": args.stage}, indent=2))


if __name__ == "__main__":
    main()
