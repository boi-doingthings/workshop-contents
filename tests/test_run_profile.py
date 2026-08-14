import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ptq_workshop.artifacts import ArtifactLayout
from ptq_workshop.telemetry import TelemetrySample, summarize_telemetry


class _RuntimeSampler:
    def __init__(self, *args, **kwargs):
        self.samples = []

    def start(self):
        return self

    def stop(self):
        return SimpleNamespace(peak_memory_used_bytes=0)


def _runtime_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        project_root=tmp_path,
        model_id="nvidia/model",
        model_revision="revision",
        profile=SimpleNamespace(name=SimpleNamespace(value="DEV_SMOKE")),
        seed=17,
    )


def _patch_runtime_dependencies(module, monkeypatch) -> None:
    monkeypatch.setattr(module, "evaluation_examples", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "NVMLSampler", _RuntimeSampler)
    monkeypatch.setattr(module, "write_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "run_accuracy", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "write_accuracy_acceptance", lambda *args, **kwargs: None)


def _write_environment(layout: ArtifactLayout, capability=(12, 0)) -> None:
    path = layout.run_dir / "manifests" / "environment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"gpus": [{"compute_capability": capability}]}))


def _load_run_profile():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_profile.py"
    spec = importlib.util.spec_from_file_location("run_profile_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepared_manifest(config, tmp_path: Path) -> dict:
    entries = {}
    for name, count in (
        ("calibration", config.profile.calibration_samples),
        ("mmlu_pro", config.profile.mmlu_samples),
        ("gsm8k", config.profile.gsm8k_samples),
    ):
        path = tmp_path / f"{name}.jsonl"
        rows = [{"id": f"{name}-{index}"} for index in range(count)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        from ptq_workshop.io import sha256_json

        entries[name] = {"path": str(path), "count": count, "sha256": sha256_json(rows)}
    return {
        "download_only": True,
        "profile": config.profile.name.value,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "model_snapshot": str(tmp_path / "snapshot"),
        "snapshot_verification": {"verified": True, "weight_bytes": 123},
        "calibration": entries["calibration"],
        "evaluation": {"mmlu_pro": entries["mmlu_pro"], "gsm8k": entries["gsm8k"]},
    }


def test_prepared_assets_reject_cross_profile_reuse(tmp_path: Path) -> None:
    module = _load_run_profile()
    config = module.make_config("DEV_SMOKE", project_root=tmp_path)
    manifest = _prepared_manifest(config, tmp_path)
    manifest["profile"] = "WORKSHOP_B200"
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Prepared manifest profile"):
        module.load_prepared(path, config)


def test_prepared_assets_reject_wrong_profile_counts(tmp_path: Path, monkeypatch) -> None:
    module = _load_run_profile()
    config = module.make_config("DEV_SMOKE", project_root=tmp_path)
    manifest = _prepared_manifest(config, tmp_path)
    manifest["calibration"]["count"] -= 1
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(module, "verify_snapshot", lambda *args: {"weight_bytes": 123})
    with pytest.raises(ValueError, match="count changed"):
        module.load_prepared(path, config)


def test_checkpoint_validation_honors_selected_precision(tmp_path: Path, monkeypatch) -> None:
    module = _load_run_profile()
    source = tmp_path / "source"
    source.mkdir()
    layout = ArtifactLayout(tmp_path / "runs", "0" * 16)
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "validate_checkpoint",
        SimpleNamespace(validate_bf16=lambda _: {"bytes": 100}),
    )

    def fake_validate_export(path, *, expected_variant, source_model):
        calls.append(expected_variant)
        assert path == layout.checkpoint_dir(expected_variant)
        assert source_model == source
        return {"bytes": 50}

    monkeypatch.setattr(module, "validate_export", fake_validate_export)
    results = module.validate_checkpoints(source, layout, ("fp8",))
    assert calls == ["fp8"]
    assert set(results) == {"bf16", "fp8"}
    assert not (layout.run_dir / "metrics" / "nvfp4-footprint.json").exists()


def test_checkpoint_validation_replaces_stale_derived_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_run_profile()
    source = tmp_path / "source"
    source.mkdir()
    layout = ArtifactLayout(tmp_path / "runs", "3" * 16)
    monkeypatch.setitem(
        sys.modules,
        "validate_checkpoint",
        SimpleNamespace(validate_bf16=lambda _: {"bytes": 100}),
    )
    export_bytes = iter((50, 40))
    monkeypatch.setattr(
        module,
        "validate_export",
        lambda *args, **kwargs: {"bytes": next(export_bytes)},
    )

    module.validate_checkpoints(source, layout, ("fp8",))
    module.validate_checkpoints(source, layout, ("fp8",))

    footprint = json.loads(
        (layout.run_dir / "metrics" / "fp8-footprint.json").read_text()
    )
    validation = json.loads(
        (layout.run_dir / "manifests" / "checkpoint-validation.json").read_text()
    )
    assert footprint["bytes"] == 40
    assert validation["fp8"]["bytes"] == 40


def test_bf16_only_validation_does_not_require_quantized_exports(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_run_profile()
    source = tmp_path / "source"
    source.mkdir()
    layout = ArtifactLayout(tmp_path / "runs", "1" * 16)
    monkeypatch.setitem(
        sys.modules,
        "validate_checkpoint",
        SimpleNamespace(validate_bf16=lambda _: {"bytes": 100}),
    )
    monkeypatch.setattr(
        module,
        "validate_export",
        lambda *args, **kwargs: pytest.fail("quantized export must not be validated"),
    )
    assert set(module.validate_checkpoints(source, layout, ())) == {"bf16"}


def test_checkpoint_validation_rejects_unknown_precision(tmp_path: Path) -> None:
    module = _load_run_profile()
    layout = ArtifactLayout(tmp_path / "runs", "2" * 16)
    with pytest.raises(ValueError, match="Unknown checkpoint validation variants"):
        module.validate_checkpoints(tmp_path / "source", layout, ("int4",))


def test_runtime_startup_failure_is_persisted_and_reraised(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_run_profile()
    _patch_runtime_dependencies(module, monkeypatch)
    layout = ArtifactLayout(tmp_path / "runs", "4" * 16)
    _write_environment(layout)

    class FailingServer:
        log_path = None

        def __init__(self, config, *, log_path):
            type(self).log_path = log_path

        def start(self):
            raise LookupError("launcher imported the wrong stack")

        def stop(self):
            pass

    monkeypatch.setattr(module, "TensorRTLLMServer", FailingServer)
    with pytest.raises(LookupError, match="wrong stack"):
        module.run_runtime(
            source=tmp_path / "source",
            prepared={},
            layout=layout,
            config=_runtime_config(tmp_path),
            do_evaluate=True,
            do_benchmark=False,
            precision_filter="fp8",
        )

    status = json.loads(
        (layout.run_dir / "metrics" / "fp8-evaluate-runtime-status.json").read_text()
    )
    assert status["available"] is False
    assert status["stage"] == "startup"
    assert status["exception_type"] == "LookupError"
    assert status["exception_message"] == "launcher imported the wrong stack"
    assert status["server_log_path"].endswith("fp8-evaluate-serve.log")
    assert status["server_manifest_path"].endswith("fp8-evaluate-server.json")
    assert (layout.run_dir / "manifests" / "fp8-evaluate-server.json").is_file()
    assert FailingServer.log_path == layout.log_path("fp8", "evaluate-serve")


def test_runtime_smoke_failure_records_observed_text(tmp_path: Path, monkeypatch) -> None:
    module = _load_run_profile()
    _patch_runtime_dependencies(module, monkeypatch)
    layout = ArtifactLayout(tmp_path / "runs", "5" * 16)
    _write_environment(layout)

    class RunningServer:
        process = SimpleNamespace(pid=123)

        def __init__(self, config, *, log_path):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(module, "TensorRTLLMServer", RunningServer)
    monkeypatch.setattr(
        module,
        "request_json",
        lambda *args, **kwargs: {
            "choices": [{"message": {"content": "nonempty diagnostic gibberish"}}]
        },
    )
    with pytest.raises(RuntimeError, match="response mismatch"):
        module.run_runtime(
            source=tmp_path / "source",
            prepared={},
            layout=layout,
            config=_runtime_config(tmp_path),
            do_evaluate=True,
            do_benchmark=False,
            precision_filter="fp8",
        )
    status = json.loads(
        (layout.run_dir / "metrics" / "fp8-evaluate-runtime-status.json").read_text()
    )
    assert status["available"] is False
    assert status["stage"] == "smoke"
    assert status["smoke_observed"] == "nonempty diagnostic gibberish"
    smoke = json.loads(
        (layout.run_dir / "metrics" / "fp8-evaluate-smoke.json").read_text()
    )
    assert smoke["exact_match"] is False
    assert smoke["raw_token_diagnostic"]["request"]["skip_special_tokens"] is False
    assert (
        smoke["raw_token_diagnostic"]["response"]["choices"][0]["message"]["content"]
        == "nonempty diagnostic gibberish"
    )


def test_runtime_success_records_available_only_after_exact_smoke(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_run_profile()
    _patch_runtime_dependencies(module, monkeypatch)
    layout = ArtifactLayout(tmp_path / "runs", "6" * 16)
    _write_environment(layout)

    class RunningServer:
        process = SimpleNamespace(pid=123)

        def __init__(self, config, *, log_path):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(module, "TensorRTLLMServer", RunningServer)
    monkeypatch.setattr(
        module,
        "request_json",
        lambda *args, **kwargs: {
            "choices": [{"message": {"content": module.SMOKE_RESPONSE_SENTINEL}}]
        },
    )
    module.run_runtime(
        source=tmp_path / "source",
        prepared={},
        layout=layout,
        config=_runtime_config(tmp_path),
        do_evaluate=True,
        do_benchmark=False,
        precision_filter="fp8",
    )
    status = json.loads(
        (layout.run_dir / "metrics" / "fp8-evaluate-runtime-status.json").read_text()
    )
    assert status["available"] is True
    assert status["stage"] == "complete"
    assert status["smoke_observed"] == module.SMOKE_RESPONSE_SENTINEL


class _QuantizationSampler:
    def __init__(self, *args, **kwargs):
        self.samples = (
            TelemetrySample(
                timestamp_s=1.0,
                monotonic_s=1.0,
                device_index=0,
                memory_used_bytes=10 * 1024**3,
            ),
            TelemetrySample(
                timestamp_s=2.0,
                monotonic_s=2.0,
                device_index=0,
                memory_used_bytes=12 * 1024**3,
            ),
        )

    def start(self):
        return self

    def stop(self):
        return summarize_telemetry(self.samples)


def test_fresh_quantization_writes_duration_peak_vram_and_raw_telemetry(
    tmp_path: Path,
) -> None:
    module = _load_run_profile()
    run_dir = tmp_path / "run"
    manifests = run_dir / "manifests"
    job = SimpleNamespace(
        variant="fp8",
        output_dir=run_dir / "checkpoints" / "fp8",
        metadata_path=manifests / "fp8-quantize.json",
        log_path=run_dir / "logs" / "fp8-quantize.log",
    )

    def quantize(current_job, *, allow_existing_valid_export):
        assert allow_existing_valid_export is False
        current_job.log_path.parent.mkdir(parents=True)
        current_job.log_path.write_text(
            "16/16 [00:30<00:00, 1.9s/it]\r"
            "Quant summary saved to /export/.quant_summary.txt\n"
            "Quantized model exported to: /export. Total time used 14.5s\n",
            encoding="utf-8",
        )
        return "result"

    clocks = iter((10.0, 55.0))
    result = module.run_quantization_with_observability(
        job,
        allow_existing_valid_export=False,
        quantization_runner=quantize,
        sampler_factory=_QuantizationSampler,
        monotonic_clock=lambda: next(clocks),
    )
    assert result == "result"
    observation = json.loads(
        (manifests / "fp8-ptq-observability.json").read_text()
    )
    assert observation["total_wall_duration_s"] == 45.0
    assert observation["phase_durations"]["calibration_duration_s"] == 30.0
    assert observation["phase_durations"]["export_duration_s"] == 14.5
    assert observation["peak_vram_bytes"] == 12 * 1024**3
    assert (run_dir / "telemetry" / "fp8-quantize.json").is_file()
    assert (run_dir / "telemetry" / "fp8-quantize.csv").is_file()


def test_existing_export_is_not_mislabeled_as_new_quantization(tmp_path: Path) -> None:
    module = _load_run_profile()
    output = tmp_path / "checkpoints" / "fp8"
    output.mkdir(parents=True)
    job = SimpleNamespace(
        variant="fp8",
        output_dir=output,
        metadata_path=tmp_path / "manifests" / "fp8-quantize.json",
        log_path=tmp_path / "logs" / "fp8-quantize.log",
    )

    def forbidden_sampler(*args, **kwargs):
        pytest.fail("resumed exports must not receive fresh PTQ telemetry")

    assert module.run_quantization_with_observability(
        job,
        allow_existing_valid_export=True,
        quantization_runner=lambda *args, **kwargs: "resumed",
        sampler_factory=forbidden_sampler,
    ) == "resumed"


def test_checkpoint_analysis_writes_one_manifest_per_selected_variant(tmp_path: Path) -> None:
    module = _load_run_profile()
    layout = ArtifactLayout(tmp_path / "runs", "a" * 16)
    source = tmp_path / "source"
    calls: list[str] = []

    def analyzer(source_path, export_path, *, variant):
        assert source_path == source
        assert export_path == layout.checkpoint_dir(variant)
        calls.append(variant)
        return {"precision": variant, "tensors": [{"name": "layer.weight"}]}

    reports = module.analyze_checkpoints(
        source, layout, ("fp8", "nvfp4"), analyzer=analyzer
    )
    assert calls == ["fp8", "nvfp4"]
    assert set(reports) == {"fp8", "nvfp4"}
    assert (layout.run_dir / "manifests" / "fp8-tensor-analysis.json").is_file()
    assert (layout.run_dir / "manifests" / "nvfp4-tensor-analysis.json").is_file()


def _score_report(parsed: int, total: int = 50) -> dict:
    return {
        "accuracy": 0.5,
        "records": [
            {
                "example_id": f"item-{index}",
                "correct": index % 2 == 0,
                "normalized_prediction": "A" if index < parsed else None,
            }
            for index in range(total)
        ],
    }


def test_parsed_answer_gate_is_enforced_after_preserving_acceptance_artifact(
    tmp_path: Path,
) -> None:
    module = _load_run_profile()
    layout = ArtifactLayout(tmp_path / "runs", "b" * 16)
    metrics = layout.run_dir / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "fp8-mmlu_pro.json").write_text(
        json.dumps(_score_report(48)), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match=r"required >=98%"):
        module.write_accuracy_acceptance(layout, ("mmlu_pro",))
    acceptance = json.loads((metrics / "accuracy-acceptance.json").read_text())
    assert acceptance["passed"] is False
    assert acceptance["failures"][0]["parsed_answer_rate"] == pytest.approx(0.96)
    assert (metrics / "fp8-mmlu_pro.json").is_file()


def test_parsed_answer_gate_accepts_exactly_98_percent(tmp_path: Path) -> None:
    module = _load_run_profile()
    layout = ArtifactLayout(tmp_path / "runs", "c" * 16)
    metrics = layout.run_dir / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "fp8-gsm8k.json").write_text(
        json.dumps(_score_report(49)), encoding="utf-8"
    )
    acceptance = module.write_accuracy_acceptance(layout, ("gsm8k",))
    assert acceptance["passed"] is True
    assert acceptance["tasks"]["gsm8k"]["fp8"]["parsed_answer_rate"] == 0.98
