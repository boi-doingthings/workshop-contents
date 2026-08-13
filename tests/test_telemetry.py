from __future__ import annotations

import json

import pytest

from ptq_workshop.telemetry import (
    NVMLSampler,
    TelemetryQualityError,
    TelemetrySample,
    integrate_power,
    summarize_telemetry,
    validate_compute_process_ownership,
    validate_telemetry_quality,
    write_telemetry,
)


def sample(at: float, power: float | None, **kwargs):
    return TelemetrySample(
        timestamp_s=1_700_000_000 + at,
        monotonic_s=at,
        device_index=0,
        power_w=power,
        **kwargs,
    )


def test_integrates_power_with_trapezoidal_rule():
    samples = [sample(0, 100), sample(1, 200), sample(3, 200)]
    assert integrate_power(samples) == pytest.approx(550.0)


def test_summary_includes_utilization_clocks_power_pstate_and_energy_counter():
    samples = [
        sample(
            0,
            100,
            total_energy_mj=10_000,
            gpu_utilization_pct=10,
            memory_utilization_pct=20,
            memory_used_bytes=100,
            temperature_c=50,
            graphics_clock_mhz=1_000,
            memory_clock_mhz=2_000,
            pstate="P2",
            clock_throttle_reasons=1,
        ),
        sample(
            2,
            200,
            total_energy_mj=310_000,
            gpu_utilization_pct=90,
            memory_utilization_pct=80,
            memory_used_bytes=200,
            temperature_c=70,
            graphics_clock_mhz=2_000,
            memory_clock_mhz=3_000,
            pstate="P0",
            clock_throttle_reasons=4,
        ),
    ]
    summary = summarize_telemetry(samples)
    assert summary.energy_j == pytest.approx(300)
    assert summary.nvml_counter_energy_j == pytest.approx(300)
    assert summary.average_power_w == 150
    assert summary.p95_power_w == pytest.approx(195)
    assert summary.p95_gpu_utilization_pct == pytest.approx(86)
    assert summary.p95_memory_utilization_pct == pytest.approx(77)
    assert summary.peak_memory_used_bytes == 200
    assert summary.average_temperature_c == 60
    assert summary.average_graphics_clock_mhz == 1_500
    assert summary.observed_pstates == ("P0", "P2")
    assert summary.clock_throttle_reasons_union == 5


class FakeBackend:
    def __init__(self):
        self.initialized = 0
        self.shutdowns = 0

    def initialize(self):
        self.initialized += 1

    def sample(self, device_index):
        return {"power_w": 123.0, "pstate": "P0"}

    def shutdown(self):
        self.shutdowns += 1


def test_sampler_supports_backend_injection_and_raw_exports(tmp_path):
    backend = FakeBackend()
    walls = iter([10.0, 11.0])
    monotonic = iter([1.0, 2.0])
    sampler = NVMLSampler(
        backend=backend,
        wall_clock=lambda: next(walls),
        monotonic_clock=lambda: next(monotonic),
    )
    sampler.collect_once()
    sampler.collect_once()
    summary = sampler.stop()
    assert backend.initialized == 1
    assert backend.shutdowns == 1
    assert summary.energy_j == 123

    json_path = tmp_path / "telemetry.json"
    csv_path = tmp_path / "telemetry.csv"
    write_telemetry(sampler.samples, json_path=json_path, csv_path=csv_path)
    payload = json.loads(json_path.read_text())
    assert len(payload["samples"]) == 2
    assert payload["samples"][0]["pstate"] == "P0"
    assert "clock_throttle_reasons" in csv_path.read_text().splitlines()[0]


def test_missing_sensors_stay_null():
    summary = summarize_telemetry([sample(0, None)])
    assert summary.energy_j == 0
    assert summary.average_power_w is None
    assert summary.p95_gpu_utilization_pct is None
    assert summary.observed_pstates == ()


def test_telemetry_quality_accepts_continuous_expected_processes():
    samples = [
        sample(0.0, 100, compute_process_pids=(123,)),
        sample(0.1, 110, compute_process_pids=(123,)),
        sample(0.2, 120, compute_process_pids=(123,)),
    ]
    summary = validate_telemetry_quality(
        samples,
        sampling_interval_s=0.1,
        allowed_compute_process_pids={123},
    )
    assert summary.maximum_sample_gap_s == pytest.approx(0.1)
    assert summary.observed_compute_process_pids == (123,)


def test_telemetry_quality_rejects_sample_gaps():
    with pytest.raises(TelemetryQualityError, match="sample gap"):
        validate_telemetry_quality(
            [sample(0.0, 100), sample(0.31, 100)],
            sampling_interval_s=0.1,
        )


@pytest.mark.parametrize(
    "reason",
    ("SwThermal", "HwThermal", "HwSlowdown", "HwPowerBrake"),
)
def test_telemetry_quality_rejects_disallowed_clock_events(reason):
    with pytest.raises(TelemetryQualityError, match=reason):
        validate_telemetry_quality(
            [
                sample(0.0, 100, clock_throttle_reasons_text="None"),
                sample(0.1, 100, clock_throttle_reasons_text=reason),
            ],
            sampling_interval_s=0.1,
        )


def test_telemetry_quality_rejects_external_compute_process():
    with pytest.raises(TelemetryQualityError, match=r"PID\(s\): 999"):
        validate_telemetry_quality(
            [
                sample(0.0, 100, compute_process_pids=(123,)),
                sample(0.1, 100, compute_process_pids=(123, 999)),
            ],
            sampling_interval_s=0.1,
            allowed_compute_process_pids={123},
        )


def test_telemetry_quality_fails_closed_when_process_sensor_is_unavailable():
    with pytest.raises(TelemetryQualityError, match="process telemetry is unavailable"):
        validate_telemetry_quality(
            [sample(0.0, 100), sample(0.1, 100)],
            sampling_interval_s=0.1,
            allowed_compute_process_pids=set(),
        )


def test_compute_process_ownership_accepts_server_descendants():
    parents = {200: 100, 201: 200, 100: 1}
    namespace_pids = {1200: 200, 1201: 201}
    assert validate_compute_process_ownership(
        (1201, 1200),
        owner_pid=100,
        parent_pid_lookup=parents.__getitem__,
        pid_namespace_lookup=namespace_pids.__getitem__,
    ) == (1200, 1201)


def test_compute_process_ownership_rejects_external_pid():
    parents = {200: 100, 999: 1, 100: 1, 1: 0}
    with pytest.raises(TelemetryQualityError, match=r"external.*999"):
        validate_compute_process_ownership(
            (200, 999),
            owner_pid=100,
            parent_pid_lookup=parents.__getitem__,
            pid_namespace_lookup=lambda pid: pid,
        )


@pytest.mark.parametrize("processes", (None, ()))
def test_compute_process_ownership_requires_process_telemetry(processes):
    with pytest.raises(TelemetryQualityError):
        validate_compute_process_ownership(processes, owner_pid=100)
