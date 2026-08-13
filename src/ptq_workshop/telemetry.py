"""Lightweight NVIDIA GPU telemetry collection for workshop benchmarks.

``pynvml`` is deliberately imported only when a real sampler is started.  Tests
and notebook dry-runs can inject a backend implementing ``initialize``,
``sample`` and ``shutdown`` without requiring a GPU or NVML installation.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Protocol, Sequence


class NVMLUnavailableError(RuntimeError):
    """Raised when real telemetry is requested but NVML cannot be loaded."""


class TelemetryQualityError(RuntimeError):
    """Raised when a measured interval is incomplete or contaminated."""


class TelemetryBackend(Protocol):
    """Small injectable boundary around NVML."""

    def initialize(self) -> None: ...

    def sample(self, device_index: int) -> Mapping[str, Any]: ...

    def shutdown(self) -> None: ...


@dataclass(frozen=True)
class TelemetrySample:
    timestamp_s: float
    monotonic_s: float
    device_index: int
    power_w: float | None = None
    total_energy_mj: float | None = None
    gpu_utilization_pct: float | None = None
    memory_utilization_pct: float | None = None
    memory_used_bytes: int | None = None
    memory_total_bytes: int | None = None
    temperature_c: float | None = None
    graphics_clock_mhz: float | None = None
    memory_clock_mhz: float | None = None
    pstate: str | None = None
    clock_throttle_reasons: int | None = None
    clock_throttle_reasons_text: str | None = None
    compute_process_pids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class TelemetrySummary:
    sample_count: int
    duration_s: float
    energy_j: float
    nvml_counter_energy_j: float | None
    average_power_w: float | None
    p95_power_w: float | None
    peak_power_w: float | None
    average_gpu_utilization_pct: float | None
    p95_gpu_utilization_pct: float | None
    peak_gpu_utilization_pct: float | None
    average_memory_utilization_pct: float | None
    p95_memory_utilization_pct: float | None
    peak_memory_utilization_pct: float | None
    peak_memory_used_bytes: int | None
    peak_temperature_c: float | None
    average_temperature_c: float | None
    p95_temperature_c: float | None
    average_graphics_clock_mhz: float | None
    peak_graphics_clock_mhz: float | None
    average_memory_clock_mhz: float | None
    peak_memory_clock_mhz: float | None
    observed_pstates: tuple[str, ...]
    clock_throttle_reasons_union: int | None
    maximum_sample_gap_s: float | None
    observed_compute_process_pids: tuple[int, ...]


def _mean(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.fmean(present) if present else None


def _max(values: Sequence[float | int | None]) -> float | int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _percentile_optional(values: Sequence[float | None], probability: float) -> float | None:
    present = sorted(float(value) for value in values if value is not None)
    if not present:
        return None
    position = (len(present) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return present[lower]
    fraction = position - lower
    return present[lower] * (1.0 - fraction) + present[upper] * fraction


def integrate_power(samples: Sequence[TelemetrySample]) -> float:
    """Integrate sampled power with the trapezoidal rule, returning joules.

    Intervals with a missing endpoint are intentionally excluded rather than
    interpolated.  Samples are sorted by monotonic timestamp so a wall-clock
    adjustment cannot corrupt the energy calculation.
    """

    ordered = sorted(samples, key=lambda item: item.monotonic_s)
    energy_j = 0.0
    for left, right in zip(ordered, ordered[1:]):
        delta_s = right.monotonic_s - left.monotonic_s
        if delta_s <= 0 or left.power_w is None or right.power_w is None:
            continue
        energy_j += delta_s * (left.power_w + right.power_w) / 2.0
    return energy_j


def summarize_telemetry(samples: Sequence[TelemetrySample]) -> TelemetrySummary:
    ordered = sorted(samples, key=lambda item: item.monotonic_s)
    duration_s = (
        max(0.0, ordered[-1].monotonic_s - ordered[0].monotonic_s)
        if len(ordered) >= 2
        else 0.0
    )
    counters = [sample.total_energy_mj for sample in ordered if sample.total_energy_mj is not None]
    counter_energy_j = None
    if len(counters) >= 2 and counters[-1] >= counters[0]:
        counter_energy_j = (counters[-1] - counters[0]) / 1000.0
    throttle_values = [
        sample.clock_throttle_reasons
        for sample in ordered
        if sample.clock_throttle_reasons is not None
    ]
    throttle_union = None
    if throttle_values:
        throttle_union = 0
        for value in throttle_values:
            throttle_union |= value
    sample_gaps = [
        right.monotonic_s - left.monotonic_s
        for left, right in zip(ordered, ordered[1:])
        if right.monotonic_s > left.monotonic_s
    ]
    process_pids = {
        int(pid)
        for sample in ordered
        for pid in (sample.compute_process_pids or ())
    }

    return TelemetrySummary(
        sample_count=len(ordered),
        duration_s=duration_s,
        energy_j=integrate_power(ordered),
        nvml_counter_energy_j=counter_energy_j,
        average_power_w=_mean([sample.power_w for sample in ordered]),
        p95_power_w=_percentile_optional([sample.power_w for sample in ordered], 0.95),
        peak_power_w=_max([sample.power_w for sample in ordered]),  # type: ignore[arg-type]
        average_gpu_utilization_pct=_mean(
            [sample.gpu_utilization_pct for sample in ordered]
        ),
        p95_gpu_utilization_pct=_percentile_optional(
            [sample.gpu_utilization_pct for sample in ordered], 0.95
        ),
        peak_gpu_utilization_pct=_max(  # type: ignore[arg-type]
            [sample.gpu_utilization_pct for sample in ordered]
        ),
        average_memory_utilization_pct=_mean(
            [sample.memory_utilization_pct for sample in ordered]
        ),
        p95_memory_utilization_pct=_percentile_optional(
            [sample.memory_utilization_pct for sample in ordered], 0.95
        ),
        peak_memory_utilization_pct=_max(  # type: ignore[arg-type]
            [sample.memory_utilization_pct for sample in ordered]
        ),
        peak_memory_used_bytes=_max(  # type: ignore[arg-type]
            [sample.memory_used_bytes for sample in ordered]
        ),
        peak_temperature_c=_max(  # type: ignore[arg-type]
            [sample.temperature_c for sample in ordered]
        ),
        average_temperature_c=_mean([sample.temperature_c for sample in ordered]),
        p95_temperature_c=_percentile_optional(
            [sample.temperature_c for sample in ordered], 0.95
        ),
        average_graphics_clock_mhz=_mean([sample.graphics_clock_mhz for sample in ordered]),
        peak_graphics_clock_mhz=_max(  # type: ignore[arg-type]
            [sample.graphics_clock_mhz for sample in ordered]
        ),
        average_memory_clock_mhz=_mean([sample.memory_clock_mhz for sample in ordered]),
        peak_memory_clock_mhz=_max(  # type: ignore[arg-type]
            [sample.memory_clock_mhz for sample in ordered]
        ),
        observed_pstates=tuple(sorted({sample.pstate for sample in ordered if sample.pstate})),
        clock_throttle_reasons_union=throttle_union,
        maximum_sample_gap_s=max(sample_gaps) if sample_gaps else None,
        observed_compute_process_pids=tuple(sorted(process_pids)),
    )


def validate_telemetry_quality(
    samples: Sequence[TelemetrySample],
    *,
    sampling_interval_s: float,
    maximum_gap_multiplier: float = 3.0,
    allowed_compute_process_pids: Collection[int] | None = None,
) -> TelemetrySummary:
    """Fail closed when a measured GPU interval is incomplete or contaminated.

    ``allowed_compute_process_pids`` should be captured after the workshop server
    is healthy and before the measured pass begins.  Omitting it disables process
    checking for callers that cannot obtain NVML process data; passing an empty
    collection requires the GPU to have no compute processes.
    """

    if sampling_interval_s <= 0:
        raise ValueError("sampling_interval_s must be positive")
    if maximum_gap_multiplier < 1:
        raise ValueError("maximum_gap_multiplier must be at least one")
    summary = summarize_telemetry(samples)
    if summary.sample_count < 2 or summary.duration_s <= 0:
        raise TelemetryQualityError("telemetry requires at least two distinct samples")
    maximum_gap_s = sampling_interval_s * maximum_gap_multiplier
    if (
        summary.maximum_sample_gap_s is None
        or summary.maximum_sample_gap_s > maximum_gap_s
    ):
        raise TelemetryQualityError(
            "telemetry sample gap "
            f"{summary.maximum_sample_gap_s!r}s exceeds {maximum_gap_s:.3f}s"
        )

    disallowed_clock_events = ("SwThermal", "HwThermal", "HwSlowdown", "HwPowerBrake")
    observed_events = {
        event
        for sample in samples
        for event in (sample.clock_throttle_reasons_text or "").split("|")
        if event
    }
    disallowed_observed = sorted(observed_events.intersection(disallowed_clock_events))
    if disallowed_observed:
        raise TelemetryQualityError(
            "disallowed GPU clock event(s): " + ", ".join(disallowed_observed)
        )

    if allowed_compute_process_pids is not None:
        if not any(sample.compute_process_pids is not None for sample in samples):
            raise TelemetryQualityError("NVML compute-process telemetry is unavailable")
        allowed = {int(pid) for pid in allowed_compute_process_pids}
        unexpected = sorted(set(summary.observed_compute_process_pids) - allowed)
        if unexpected:
            raise TelemetryQualityError(
                "unexpected GPU compute process PID(s): "
                + ", ".join(str(pid) for pid in unexpected)
            )
    return summary


def _linux_parent_pid(pid: int) -> int:
    """Read one process's parent PID without adding a psutil dependency."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        # The command name is parenthesized and may itself contain spaces. The
        # fields after the final ')' begin with state followed by parent PID.
        suffix = stat.rsplit(")", 1)[1].split()
        return int(suffix[1])
    except (FileNotFoundError, IndexError, PermissionError, ValueError) as exc:
        raise ProcessLookupError(f"cannot resolve parent PID for process {pid}") from exc


def _local_pid_for_nvml_pid(pid: int) -> int:
    """Translate an NVML/host PID into the current Linux PID namespace."""

    fallback: int | None = None
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            lines = status_path.read_text(encoding="utf-8").splitlines()
            nspid_line = next(line for line in lines if line.startswith("NSpid:"))
            namespace_pids = tuple(int(value) for value in nspid_line.split()[1:])
            local_pid = int(status_path.parent.name)
        except (FileNotFoundError, PermissionError, StopIteration, ValueError):
            continue
        if namespace_pids and namespace_pids[0] == pid:
            # NVML normally reports the host-namespace PID; prefer that exact
            # match over a coincidental inner-namespace PID with the same value.
            return local_pid
        if pid in namespace_pids:
            fallback = local_pid
    if fallback is None:
        raise ProcessLookupError(f"cannot map NVML PID {pid} into the current namespace")
    return fallback


def validate_compute_process_ownership(
    compute_process_pids: Collection[int] | None,
    *,
    owner_pid: int,
    parent_pid_lookup: Callable[[int], int] = _linux_parent_pid,
    pid_namespace_lookup: Callable[[int], int] = _local_pid_for_nvml_pid,
) -> tuple[int, ...]:
    """Return server-owned GPU PIDs, failing closed on external contamination.

    TensorRT-LLM may put CUDA work in child worker processes rather than the
    launcher itself, so ownership is checked by walking each NVML PID's parent
    chain to the ``trtllm-serve`` launcher PID.
    """

    if owner_pid <= 0:
        raise ValueError("owner_pid must be positive")
    if compute_process_pids is None:
        raise TelemetryQualityError("NVML compute-process telemetry is unavailable")
    observed = tuple(sorted({int(pid) for pid in compute_process_pids}))
    if not observed:
        raise TelemetryQualityError("no server GPU compute process was observed")

    external: list[int] = []
    unverifiable: list[int] = []
    for pid in observed:
        try:
            current = int(pid_namespace_lookup(pid))
        except (OSError, ProcessLookupError):
            unverifiable.append(pid)
            continue
        seen: set[int] = set()
        owned = False
        while current > 0 and current not in seen:
            if current == owner_pid:
                owned = True
                break
            seen.add(current)
            try:
                current = int(parent_pid_lookup(current))
            except (OSError, ProcessLookupError):
                unverifiable.append(pid)
                break
        if not owned and pid not in unverifiable:
            external.append(pid)

    if unverifiable:
        raise TelemetryQualityError(
            "could not verify GPU compute process ownership for PID(s): "
            + ", ".join(str(pid) for pid in sorted(set(unverifiable)))
        )
    if external:
        raise TelemetryQualityError(
            "external GPU compute process PID(s): "
            + ", ".join(str(pid) for pid in sorted(external))
        )
    return observed


class _PynvmlBackend:
    def __init__(self) -> None:
        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only on GPU hosts
            raise NVMLUnavailableError(
                "NVML telemetry requires the optional 'nvidia-ml-py' package"
            ) from exc
        self._nvml = pynvml
        self._handles: dict[int, Any] = {}

    def initialize(self) -> None:
        try:
            self._nvml.nvmlInit()
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise NVMLUnavailableError(f"NVML initialization failed: {exc}") from exc

    def _handle(self, device_index: int) -> Any:
        if device_index not in self._handles:
            self._handles[device_index] = self._nvml.nvmlDeviceGetHandleByIndex(device_index)
        return self._handles[device_index]

    @staticmethod
    def _optional(call: Callable[[], Any]) -> Any | None:
        try:
            return call()
        except Exception:
            return None

    def sample(self, device_index: int) -> Mapping[str, Any]:
        handle = self._handle(device_index)
        utilization = self._optional(lambda: self._nvml.nvmlDeviceGetUtilizationRates(handle))
        memory = self._optional(lambda: self._nvml.nvmlDeviceGetMemoryInfo(handle))
        throttle_reasons = self._current_clock_reasons(handle)
        return {
            "power_w": self._scaled_optional(
                lambda: self._nvml.nvmlDeviceGetPowerUsage(handle), 1000.0
            ),
            "total_energy_mj": self._optional(
                lambda: self._nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            ),
            "gpu_utilization_pct": getattr(utilization, "gpu", None),
            "memory_utilization_pct": getattr(utilization, "memory", None),
            "memory_used_bytes": getattr(memory, "used", None),
            "memory_total_bytes": getattr(memory, "total", None),
            "temperature_c": self._optional(
                lambda: self._nvml.nvmlDeviceGetTemperature(
                    handle, self._nvml.NVML_TEMPERATURE_GPU
                )
            ),
            "graphics_clock_mhz": self._optional(
                lambda: self._nvml.nvmlDeviceGetClockInfo(handle, self._nvml.NVML_CLOCK_GRAPHICS)
            ),
            "memory_clock_mhz": self._optional(
                lambda: self._nvml.nvmlDeviceGetClockInfo(handle, self._nvml.NVML_CLOCK_MEM)
            ),
            "pstate": self._pstate(handle),
            "clock_throttle_reasons": throttle_reasons,
            "clock_throttle_reasons_text": self._throttle_reasons_text(throttle_reasons),
            "compute_process_pids": self._compute_process_pids(handle),
        }

    def _pstate(self, handle: Any) -> str | None:
        value = self._optional(lambda: self._nvml.nvmlDeviceGetPerformanceState(handle))
        return None if value is None else f"P{int(value)}"

    def _current_clock_reasons(self, handle: Any) -> int | None:
        """Support both the current NVML event API and its legacy alias."""

        getter = getattr(self._nvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
        if getter is None:
            getter = getattr(self._nvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
        if getter is None:
            return None
        value = self._optional(lambda: getter(handle))
        return None if value is None else int(value)

    def _compute_process_pids(self, handle: Any) -> tuple[int, ...] | None:
        """Return compute PIDs using the newest NVML API available."""

        getter = None
        for name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            getter = getattr(self._nvml, name, None)
            if getter is not None:
                break
        if getter is None:
            return None
        processes = self._optional(lambda: getter(handle))
        if processes is None:
            return None
        return tuple(sorted({int(process.pid) for process in processes}))

    def _throttle_reasons_text(self, reasons: int | None) -> str | None:
        if reasons is None:
            return None
        known = (
            (
                "GpuIdle",
                ("nvmlClocksEventReasonGpuIdle", "nvmlClocksThrottleReasonGpuIdle"),
            ),
            (
                "ApplicationsClocks",
                (
                    "nvmlClocksEventReasonApplicationsClocksSetting",
                    "nvmlClocksThrottleReasonApplicationsClocksSetting",
                ),
            ),
            (
                "SwPowerCap",
                (
                    "nvmlClocksEventReasonSwPowerCap",
                    "nvmlClocksThrottleReasonSwPowerCap",
                ),
            ),
            (
                "HwSlowdown",
                (
                    "nvmlClocksEventReasonHwSlowdown",
                    "nvmlClocksThrottleReasonHwSlowdown",
                ),
            ),
            (
                "SyncBoost",
                ("nvmlClocksEventReasonSyncBoost", "nvmlClocksThrottleReasonSyncBoost"),
            ),
            (
                "SwThermal",
                (
                    "nvmlClocksEventReasonSwThermalSlowdown",
                    "nvmlClocksThrottleReasonSwThermalSlowdown",
                ),
            ),
            (
                "HwThermal",
                (
                    "nvmlClocksEventReasonHwThermalSlowdown",
                    "nvmlClocksThrottleReasonHwThermalSlowdown",
                ),
            ),
            (
                "HwPowerBrake",
                (
                    "nvmlClocksEventReasonHwPowerBrakeSlowdown",
                    "nvmlClocksThrottleReasonHwPowerBrakeSlowdown",
                ),
            ),
            (
                "DisplayClock",
                (
                    "nvmlClocksEventReasonDisplayClockSetting",
                    "nvmlClocksThrottleReasonDisplayClockSetting",
                ),
            ),
        )
        labels = [
            label
            for label, constants in known
            if any(
                getattr(self._nvml, constant, 0) & int(reasons)
                for constant in constants
            )
        ]
        return "None" if not labels else "|".join(labels)

    def _scaled_optional(self, call: Callable[[], Any], divisor: float) -> float | None:
        value = self._optional(call)
        return None if value is None else float(value) / divisor

    def shutdown(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass


class _DryRunBackend:
    def initialize(self) -> None:
        return None

    def sample(self, device_index: int) -> Mapping[str, Any]:
        return {}

    def shutdown(self) -> None:
        return None


class NVMLSampler:
    """Sample one GPU on a background thread.

    ``start`` and ``stop`` are idempotent.  The sampler takes an immediate
    sample at both boundaries, which gives the energy integrator endpoints for
    short benchmark runs.
    """

    def __init__(
        self,
        device_index: int = 0,
        interval_s: float = 0.1,
        *,
        backend: TelemetryBackend | None = None,
        dry_run: bool = False,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if device_index < 0:
            raise ValueError("device_index must be non-negative")
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if backend is not None and dry_run:
            raise ValueError("backend and dry_run are mutually exclusive")
        self.device_index = device_index
        self.interval_s = interval_s
        self._backend = backend or (_DryRunBackend() if dry_run else _PynvmlBackend())
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._samples: list[TelemetrySample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._initialized = False

    @property
    def samples(self) -> tuple[TelemetrySample, ...]:
        with self._lock:
            return tuple(self._samples)

    def collect_once(self) -> TelemetrySample:
        if not self._initialized:
            self._backend.initialize()
            self._initialized = True
        raw = dict(self._backend.sample(self.device_index))
        known = {field.name for field in TelemetrySample.__dataclass_fields__.values()}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"telemetry backend returned unknown fields: {unknown}")
        sample = TelemetrySample(
            timestamp_s=float(self._wall_clock()),
            monotonic_s=float(self._monotonic_clock()),
            device_index=self.device_index,
            **raw,
        )
        with self._lock:
            self._samples.append(sample)
        return sample

    def start(self) -> "NVMLSampler":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self.collect_once()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name=f"nvml-sampler-{self.device_index}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self.collect_once()

    def stop(self) -> TelemetrySummary:
        thread = self._thread
        if thread is not None and thread.is_alive():
            self._stop_event.set()
            thread.join(timeout=max(1.0, self.interval_s * 2))
            self.collect_once()
        self._thread = None
        if self._initialized:
            self._backend.shutdown()
            self._initialized = False
        return summarize_telemetry(self.samples)

    def __enter__(self) -> "NVMLSampler":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()


def write_telemetry(
    samples: Sequence[TelemetrySample],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> TelemetrySummary:
    """Persist every raw sample plus its independently derived summary."""

    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_telemetry(samples)
    with json_target.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": asdict(summary),
                "samples": [asdict(sample) for sample in samples],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    fieldnames = list(TelemetrySample.__dataclass_fields__)
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(sample) for sample in samples)
    return summary
