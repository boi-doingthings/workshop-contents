from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ptq_workshop.benchmark import (
    CVRetryPolicy,
    FixedTokenViolation,
    PerformanceScenario,
    RequestTiming,
    build_exact_token_prompt,
    canonical_performance_scenarios,
    completion_payload,
    energy_per_output_token,
    run_performance_scenario,
    summarize_timings,
    write_benchmark_result,
)
from ptq_workshop.serving import (
    SMOKE_RESPONSE_SENTINEL,
    TensorRTLLMServerConfig,
    require_exact_smoke_response,
    save_server_manifest,
    server_config_for_workshop,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


def timing(request_id, input_tokens=512, output_tokens=128, duration=2.0):
    return RequestTiming(
        request_id=request_id,
        started_s=0,
        first_token_s=0.2,
        ended_s=duration,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw_events=({"raw": True},),
    )


def test_exact_prompt_and_fixed_output_payload():
    tokenizer = CharacterTokenizer()
    prompt = build_exact_token_prompt(tokenizer, 512)
    assert len(tokenizer.encode(prompt)) == 512
    scenario = PerformanceScenario("interactive", 512, 128, 1, 2)
    payload = completion_payload(model="model", prompt=prompt, scenario=scenario)
    assert payload["max_tokens"] == payload["min_tokens"] == 128
    assert payload["ignore_eos"] is True
    assert payload["stream"] is True


def test_canonical_scenarios_preserve_dimensions_and_shorten_dev_counts():
    dev = canonical_performance_scenarios(SimpleNamespace(name="DEV_SMOKE"))
    assert [(s.input_tokens, s.output_tokens, s.concurrency, s.request_count) for s in dev] == [
        (512, 128, 1, 2),
        (2048, 256, 8, 8),
        (1024, 128, 32, 32),
    ]
    workshop = canonical_performance_scenarios(SimpleNamespace(name="WORKSHOP_B200"))
    assert [s.request_count for s in workshop] == [16, 32, 64]
    full = canonical_performance_scenarios(
        SimpleNamespace(name="FULL"), include_full_prefill=True
    )
    assert (full[-1].input_tokens, full[-1].output_tokens, full[-1].concurrency, full[-1].request_count) == (8192, 64, 8, 32)


def test_summary_reports_ttft_tpot_e2e_and_throughput():
    summary = summarize_timings(
        [timing("a"), timing("b")], wall_time_s=4, expected_requests=2
    )
    assert summary.ttft_mean_s == pytest.approx(0.2)
    assert summary.e2e_mean_s == pytest.approx(2.0)
    assert summary.tpot_mean_s == pytest.approx(1.8 / 127)
    assert summary.output_tokens_per_s == 64


def test_repetition_cv_is_gate_and_stable_run_stops_after_three(tmp_path):
    scenario = PerformanceScenario("interactive", 512, 128, 1, 2, warmup_passes=1)
    clock_values = iter([0, 4, 10, 14, 20, 24])
    result = run_performance_scenario(
        scenario,
        lambda request_id: timing(request_id),
        retry_policy=CVRetryPolicy(max_attempts=5, cv_threshold=0.05),
        minimum_repetitions=3,
        monotonic_clock=lambda: next(clock_values),
    )
    assert result.stable is True
    assert result.accepted_attempt == 3
    assert len(result.attempts) == 3
    assert result.attempts[-1].cross_repetition_cv["output_tokens_per_s"] == 0

    json_path, csv_path = tmp_path / "raw.json", tmp_path / "raw.csv"
    write_benchmark_result(result, json_path=json_path, csv_path=csv_path)
    assert json.loads(json_path.read_text())["attempts"][0]["timings"][0]["raw_events"]
    assert "raw_events_json" in csv_path.read_text().splitlines()[0]


def test_fixed_token_violation_is_recorded_not_silently_changed():
    scenario = PerformanceScenario("interactive", 512, 128, 1, 1, warmup_passes=0)
    result = run_performance_scenario(
        scenario,
        lambda request_id: timing(request_id, output_tokens=127),
        retry_policy=CVRetryPolicy(max_attempts=1),
        minimum_repetitions=1,
        monotonic_clock=iter([0, 1]).__next__,
    )
    assert result.stable is False
    assert result.attempts[0].failures[0].error_type == "FixedTokenViolation"


def test_energy_per_token_supports_idle_adjustment():
    energy = energy_per_output_token(
        gross_energy_j=500, duration_s=2, output_tokens=100, idle_power_w=50
    )
    assert energy.gross_joules_per_output_token == 5
    assert energy.idle_adjusted_joules_per_output_token == 4


def test_trtllm_command_uses_validated_serve_autodeploy_shape(tmp_path):
    config = TensorRTLLMServerConfig(
        model="checkpoint",
        config_path=tmp_path / "nano.yaml",
        hf_revision="abc123",
        trust_remote_code=True,
    )
    command = config.command()
    assert command[:3] == ("trtllm-serve", "serve", "checkpoint")
    assert command[command.index("--backend") + 1] == "_autodeploy"
    assert command[command.index("--reasoning_parser") + 1] == "nano-v3"
    assert command[command.index("--kv_cache_dtype") + 1] == "auto"
    assert command[command.index("--config") + 1] == str(tmp_path / "nano.yaml")


def test_workshop_server_config_selects_profile_yaml_and_revision(tmp_path):
    workshop = SimpleNamespace(
        project_root=tmp_path,
        model_id="nvidia/model",
        model_revision="revision",
        profile=SimpleNamespace(name=SimpleNamespace(value="DEV_SMOKE")),
    )
    config = server_config_for_workshop(workshop)
    assert str(config.config_path).endswith("configs/nano_v3_dev.yaml")
    assert config.hf_revision == "revision"
    assert config.trust_remote_code is True
    assert config.command()[:3] == (
        sys.executable,
        "/usr/local/bin/trtllm-serve",
        "serve",
    )
    assert config.command()[config.command().index("--tp_size") + 1] == "1"
    assert config.prefix_cache_reuse is False
    assert config.speculative_decoding is False


def test_rtx_blackwell_uses_autodeploy_with_sm120_kernel_policy(tmp_path):
    workshop = SimpleNamespace(
        project_root=tmp_path,
        model_id="nvidia/model",
        model_revision="revision",
        profile=SimpleNamespace(name=SimpleNamespace(value="DEV_SMOKE")),
    )
    configs = [
        server_config_for_workshop(
            workshop, precision=precision, compute_capability=(12, 0)
        )
        for precision in ("bf16", "fp8", "nvfp4")
    ]
    assert [config.backend for config in configs] == ["_autodeploy"] * 3
    assert [Path(config.config_path).name for config in configs] == [
        "nano_v3_sm120.yaml"
    ] * 3


def test_blackwell_nvfp4_moe_backends_are_hardware_specific():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    sm120 = yaml.safe_load((config_dir / "nano_v3_sm120.yaml").read_text())
    assert sm120["transforms"]["fuse_nvfp4_moe"]["backend"] == "cutlass"
    assert sm120["transforms"]["fuse_rmsnorm_quant_nvfp4"]["enabled"] is False
    assert sm120["transforms"]["fuse_relu2_quant_nvfp4"]["enabled"] is False
    for name in ("nano_v3_dev.yaml", "nano_v3_b200.yaml"):
        config = yaml.safe_load((config_dir / name).read_text())
        assert config["transforms"]["fuse_nvfp4_moe"]["backend"] == "trtllm_gen"


def test_server_manifest_records_controlled_optimization_policy(tmp_path):
    config = TensorRTLLMServerConfig(model="checkpoint")
    target = save_server_manifest(config, tmp_path / "server.json")
    payload = json.loads(target.read_text())
    assert payload["controlled_optimization_policy"] == {
        "kv_cache_dtype": "auto",
        "prefix_cache_reuse": False,
        "runtime_backend": "_autodeploy",
        "runtime_config": None,
        "speculative_decoding": False,
    }
    assert payload["launcher"] == {
        "executable": "trtllm-serve",
        "python_executable": None,
    }


def test_smoke_response_requires_exact_sentinel():
    response = {"choices": [{"message": {"content": SMOKE_RESPONSE_SENTINEL}}]}
    assert require_exact_smoke_response(response, variant="fp8") == SMOKE_RESPONSE_SENTINEL

    response["choices"][0]["message"]["content"] = "nonempty gibberish"
    with pytest.raises(RuntimeError, match="response mismatch"):
        require_exact_smoke_response(response, variant="fp8")
