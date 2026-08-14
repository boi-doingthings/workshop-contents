"""Fixed-token serving benchmarks with explicit stability checks."""

from __future__ import annotations

import concurrent.futures
import csv
import json
import math
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


class FixedTokenViolation(RuntimeError):
    """Raised when the tokenizer or server changes a fixed-token scenario."""


@dataclass(frozen=True)
class PerformanceScenario:
    name: str
    input_tokens: int
    output_tokens: int
    concurrency: int
    request_count: int
    warmup_passes: int = 1
    temperature: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name must not be empty")
        for name in ("input_tokens", "output_tokens", "concurrency", "request_count"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_passes < 0:
            raise ValueError("warmup_passes must be non-negative")
        if self.request_count < self.concurrency:
            raise ValueError("request_count must be at least concurrency")


@dataclass(frozen=True)
class CVRetryPolicy:
    max_attempts: int = 5
    cv_threshold: float = 0.05
    metrics: tuple[str, ...] = ("output_tokens_per_s", "e2e_mean_s")
    require_all_requests: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.cv_threshold < 0:
            raise ValueError("cv_threshold must be non-negative")
        supported = {"output_tokens_per_s", "total_tokens_per_s", "e2e_mean_s", "ttft_mean_s", "tpot_mean_s"}
        if not self.metrics or any(metric not in supported for metric in self.metrics):
            raise ValueError(f"metrics must be a non-empty subset of {sorted(supported)}")


@dataclass(frozen=True)
class RequestTiming:
    request_id: str
    started_s: float
    first_token_s: float
    ended_s: float
    input_tokens: int
    output_tokens: int
    output_text: str = ""
    raw_events: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.started_s <= self.first_token_s <= self.ended_s:
            raise ValueError("request timestamps must be monotonic")
        if self.input_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("request token counts must be positive")

    @property
    def ttft_s(self) -> float:
        return self.first_token_s - self.started_s

    @property
    def e2e_s(self) -> float:
        return self.ended_s - self.started_s

    @property
    def tpot_s(self) -> float | None:
        if self.output_tokens <= 1:
            return None
        return (self.ended_s - self.first_token_s) / (self.output_tokens - 1)


@dataclass(frozen=True)
class RequestFailure:
    request_id: str
    error_type: str
    message: str


@dataclass(frozen=True)
class BenchmarkSummary:
    request_count: int
    successful_requests: int
    wall_time_s: float
    requests_per_s: float
    ttft_mean_s: float
    ttft_p50_s: float
    ttft_p95_s: float
    tpot_mean_s: float | None
    tpot_p50_s: float | None
    tpot_p95_s: float | None
    e2e_mean_s: float
    e2e_p50_s: float
    e2e_p95_s: float
    output_tokens_per_s: float
    total_tokens_per_s: float
    stability_metric: str
    coefficient_of_variation: float


@dataclass(frozen=True)
class BenchmarkAttempt:
    attempt: int
    stable: bool
    summary: BenchmarkSummary | None
    timings: tuple[RequestTiming, ...]
    failures: tuple[RequestFailure, ...]
    cross_repetition_cv: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    scenario: PerformanceScenario
    retry_policy: CVRetryPolicy
    stable: bool
    accepted_attempt: int | None
    attempts: tuple[BenchmarkAttempt, ...]


@dataclass(frozen=True)
class BenchmarkPlan:
    scenario: PerformanceScenario
    endpoint: str
    model: str
    request_payload: Mapping[str, Any]
    retry_policy: CVRetryPolicy


@dataclass(frozen=True)
class EnergyPerToken:
    gross_energy_j: float
    idle_adjusted_energy_j: float
    output_tokens: int
    gross_joules_per_output_token: float
    idle_adjusted_joules_per_output_token: float
    idle_power_w: float
    duration_s: float


def energy_per_output_token(
    *,
    gross_energy_j: float,
    duration_s: float,
    output_tokens: int,
    idle_power_w: float = 0.0,
) -> EnergyPerToken:
    """Derive gross and idle-adjusted energy per generated token."""

    if gross_energy_j < 0 or duration_s < 0 or idle_power_w < 0:
        raise ValueError("energy, duration, and idle power must be non-negative")
    if output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    idle_adjusted = max(0.0, gross_energy_j - idle_power_w * duration_s)
    return EnergyPerToken(
        gross_energy_j=gross_energy_j,
        idle_adjusted_energy_j=idle_adjusted,
        output_tokens=output_tokens,
        gross_joules_per_output_token=gross_energy_j / output_tokens,
        idle_adjusted_joules_per_output_token=idle_adjusted / output_tokens,
        idle_power_w=idle_power_w,
        duration_s=duration_s,
    )


def canonical_performance_scenarios(
    profile: Any,
    *,
    include_full_prefill: bool = False,
    seed: int = 42,
) -> tuple[PerformanceScenario, ...]:
    """Return the fixed workshop scenario matrix for one profile.

    DEV preserves ISL, OSL and concurrency while reducing request counts to the
    smallest useful complete batch.  Other profiles use the presentation
    request counts.  The 8K prefill case is opt-in and intended for FULL runs.
    """

    profile_name = getattr(profile.name, "value", profile.name)
    dev = profile_name == "DEV_SMOKE"
    specifications = [
        ("interactive", 512, 128, 1, 2 if dev else 16),
        ("rag_balanced", 2048, 256, 8, 8 if dev else 32),
        ("throughput", 1024, 128, 32, 32 if dev else 64),
    ]
    if include_full_prefill:
        if profile_name != "FULL":
            raise ValueError("the optional 8K prefill scenario is restricted to the FULL profile")
        specifications.append(("full_prefill", 8192, 64, 8, 32))
    return tuple(
        PerformanceScenario(
            name=name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            concurrency=concurrency,
            request_count=request_count,
            warmup_passes=1,
            seed=seed,
        )
        for name, input_tokens, output_tokens, concurrency, request_count in specifications
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def coefficient_of_variation(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    mean = statistics.fmean(values)
    if mean == 0:
        return 0.0 if all(value == 0 for value in values) else math.inf
    return statistics.pstdev(values) / abs(mean)


def summarize_timings(
    timings: Sequence[RequestTiming],
    *,
    wall_time_s: float,
    expected_requests: int,
    stability_metric: str = "e2e_s",
) -> BenchmarkSummary:
    if not timings:
        raise ValueError("cannot summarize an empty timing collection")
    if wall_time_s <= 0:
        raise ValueError("wall_time_s must be positive")
    ttft = [timing.ttft_s for timing in timings]
    e2e = [timing.e2e_s for timing in timings]
    tpot = [value for timing in timings if (value := timing.tpot_s) is not None]
    metric_values = {
        "e2e_s": e2e,
        "ttft_s": ttft,
        "tpot_s": tpot,
    }[stability_metric]
    if not metric_values:
        raise ValueError(f"stability metric {stability_metric} is undefined for this scenario")
    output_tokens = sum(timing.output_tokens for timing in timings)
    total_tokens = sum(timing.input_tokens + timing.output_tokens for timing in timings)
    return BenchmarkSummary(
        request_count=expected_requests,
        successful_requests=len(timings),
        wall_time_s=wall_time_s,
        requests_per_s=len(timings) / wall_time_s,
        ttft_mean_s=statistics.fmean(ttft),
        ttft_p50_s=_percentile(ttft, 0.50),
        ttft_p95_s=_percentile(ttft, 0.95),
        tpot_mean_s=None if not tpot else statistics.fmean(tpot),
        tpot_p50_s=None if not tpot else _percentile(tpot, 0.50),
        tpot_p95_s=None if not tpot else _percentile(tpot, 0.95),
        e2e_mean_s=statistics.fmean(e2e),
        e2e_p50_s=_percentile(e2e, 0.50),
        e2e_p95_s=_percentile(e2e, 0.95),
        output_tokens_per_s=output_tokens / wall_time_s,
        total_tokens_per_s=total_tokens / wall_time_s,
        stability_metric=stability_metric,
        coefficient_of_variation=coefficient_of_variation(metric_values),
    )


def build_exact_token_prompt(
    tokenizer: Any,
    target_tokens: int,
    *,
    seed: int = 42,
) -> str:
    """Create text whose round-trip token count is exactly ``target_tokens``.

    The function raises instead of substituting a nearby length.  This makes a
    tokenizer/version mismatch visible in the benchmark record.
    """

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    sentence = (
        f" Fixed-token inference benchmark {seed}: NVIDIA Blackwell quantization "
        "measurement uses deterministic neutral prose."
    )
    text = sentence
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    while len(token_ids) < target_tokens:
        text += sentence
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    prompt = tokenizer.decode(
        token_ids[:target_tokens],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    round_trip = tokenizer.encode(prompt, add_special_tokens=False)
    if len(round_trip) != target_tokens:
        raise FixedTokenViolation(
            f"tokenizer round-trip produced {len(round_trip)} tokens, expected {target_tokens}"
        )
    return prompt


def completion_payload(
    *,
    model: str,
    prompt: str,
    scenario: PerformanceScenario,
) -> dict[str, Any]:
    """Build an explicit TRT-LLM request that disables early EOS termination."""

    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": scenario.output_tokens,
        "min_tokens": scenario.output_tokens,
        "temperature": scenario.temperature,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": scenario.seed,
    }


def prepare_benchmark_plan(
    *,
    base_url: str,
    model: str,
    tokenizer: Any,
    scenario: PerformanceScenario,
    retry_policy: CVRetryPolicy | None = None,
) -> BenchmarkPlan:
    prompt = build_exact_token_prompt(tokenizer, scenario.input_tokens, seed=scenario.seed)
    return BenchmarkPlan(
        scenario=scenario,
        endpoint=f"{base_url.rstrip('/')}/v1/completions",
        model=model,
        request_payload=completion_payload(model=model, prompt=prompt, scenario=scenario),
        retry_policy=retry_policy or CVRetryPolicy(),
    )


def _event_text(event: Mapping[str, Any]) -> str:
    choices = event.get("choices")
    if not choices:
        return ""
    choice = choices[0]
    if choice.get("text") is not None:
        return str(choice["text"])
    delta = choice.get("delta") or {}
    return "" if delta.get("content") is None else str(delta["content"])


def stream_completion_request(
    *,
    endpoint: str,
    payload: Mapping[str, Any],
    request_id: str,
    tokenizer: Any,
    expected_input_tokens: int,
    expected_output_tokens: int,
    timeout_s: float = 300.0,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> RequestTiming:
    """Send one SSE completion request and retain every raw JSON event."""

    body = json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    started = monotonic_clock()
    first_token: float | None = None
    output_parts: list[str] = []
    events: list[Mapping[str, Any]] = []
    usage: Mapping[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            event = json.loads(line)
            events.append(event)
            if event.get("usage"):
                usage = event["usage"]
            piece = _event_text(event)
            if piece:
                if first_token is None:
                    first_token = monotonic_clock()
                output_parts.append(piece)
    ended = monotonic_clock()
    output = "".join(output_parts)
    measured_output = usage.get("completion_tokens")
    if measured_output is None:
        measured_output = len(tokenizer.encode(output, add_special_tokens=False))
    measured_input = usage.get("prompt_tokens", expected_input_tokens)
    if int(measured_input) != expected_input_tokens:
        raise FixedTokenViolation(
            f"server reported {measured_input} input tokens, expected {expected_input_tokens}"
        )
    if int(measured_output) != expected_output_tokens:
        raise FixedTokenViolation(
            f"server produced {measured_output} output tokens, expected {expected_output_tokens}"
        )
    if first_token is None:
        raise FixedTokenViolation("stream completed without a non-empty token event")
    return RequestTiming(
        request_id=request_id,
        started_s=started,
        first_token_s=first_token,
        ended_s=ended,
        input_tokens=int(measured_input),
        output_tokens=int(measured_output),
        output_text=output,
        raw_events=tuple(events),
    )


RequestFunction = Callable[[str], RequestTiming]


def _validate_timing(timing: RequestTiming, scenario: PerformanceScenario) -> None:
    if timing.input_tokens != scenario.input_tokens:
        raise FixedTokenViolation(
            f"{timing.request_id}: input_tokens={timing.input_tokens}, expected={scenario.input_tokens}"
        )
    if timing.output_tokens != scenario.output_tokens:
        raise FixedTokenViolation(
            f"{timing.request_id}: output_tokens={timing.output_tokens}, expected={scenario.output_tokens}"
        )


def run_performance_scenario(
    scenario: PerformanceScenario,
    request_fn: RequestFunction,
    *,
    retry_policy: CVRetryPolicy | None = None,
    minimum_repetitions: int = 3,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> BenchmarkResult:
    """Run unchanged scenario attempts until the configured CV is acceptable."""

    policy = retry_policy or CVRetryPolicy()
    if minimum_repetitions <= 0 or minimum_repetitions > policy.max_attempts:
        raise ValueError("minimum_repetitions must be in [1, max_attempts]")
    for warmup_pass in range(scenario.warmup_passes):
        with concurrent.futures.ThreadPoolExecutor(max_workers=scenario.concurrency) as executor:
            futures = [
                executor.submit(request_fn, f"warmup-{warmup_pass}-request-{index}")
                for index in range(scenario.request_count)
            ]
            for future in futures:
                _validate_timing(future.result(), scenario)

    attempts: list[BenchmarkAttempt] = []
    for attempt_number in range(1, policy.max_attempts + 1):
        started = monotonic_clock()
        timings: list[RequestTiming] = []
        failures: list[RequestFailure] = []

        def invoke(index: int) -> RequestTiming:
            request_id = f"attempt-{attempt_number}-request-{index}"
            result = request_fn(request_id)
            _validate_timing(result, scenario)
            return result

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=scenario.concurrency
        ) as executor:
            futures = {
                executor.submit(invoke, index): index for index in range(scenario.request_count)
            }
            for future, index in futures.items():
                try:
                    timings.append(future.result())
                except Exception as exc:
                    failures.append(
                        RequestFailure(
                            request_id=f"attempt-{attempt_number}-request-{index}",
                            error_type=type(exc).__name__,
                            message=str(exc),
                        )
                    )
        ended = monotonic_clock()
        timings.sort(key=lambda item: item.request_id)
        failures.sort(key=lambda item: item.request_id)
        summary = None
        if timings:
            summary = summarize_timings(
                timings,
                wall_time_s=max(ended - started, 1e-12),
                expected_requests=scenario.request_count,
                stability_metric="e2e_s",
            )
        all_requests = len(timings) == scenario.request_count
        completed_summaries = [
            previous.summary for previous in attempts if previous.summary is not None
        ] + ([summary] if summary is not None else [])
        cross_cv: dict[str, float] = {}
        if len(completed_summaries) >= minimum_repetitions:
            for metric in policy.metrics:
                values = [getattr(item, metric) for item in completed_summaries]
                if any(value is None for value in values):
                    cross_cv[metric] = math.inf
                else:
                    cross_cv[metric] = coefficient_of_variation(
                        [float(value) for value in values]
                    )
        stable = (
            len(completed_summaries) >= minimum_repetitions
            and bool(cross_cv)
            and all(value <= policy.cv_threshold for value in cross_cv.values())
            and (all_requests or not policy.require_all_requests)
            and all(
                not prior.failures or not policy.require_all_requests for prior in attempts
            )
        )
        attempts.append(
            BenchmarkAttempt(
                attempt=attempt_number,
                stable=stable,
                summary=summary,
                timings=tuple(timings),
                failures=tuple(failures),
                cross_repetition_cv=cross_cv,
            )
        )
        if stable:
            return BenchmarkResult(
                scenario=scenario,
                retry_policy=policy,
                stable=True,
                accepted_attempt=attempt_number,
                attempts=tuple(attempts),
            )
    return BenchmarkResult(
        scenario=scenario,
        retry_policy=policy,
        stable=False,
        accepted_attempt=None,
        attempts=tuple(attempts),
    )


def run_profile_performance_scenario(
    workshop_config: Any,
    scenario: PerformanceScenario,
    request_fn: RequestFunction,
    *,
    retry_policy: CVRetryPolicy | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> BenchmarkResult:
    """Run the repetition count fixed by the canonical workshop profile."""

    repetitions = int(workshop_config.profile.benchmark_repetitions)
    return run_performance_scenario(
        scenario,
        request_fn,
        retry_policy=retry_policy,
        minimum_repetitions=repetitions,
        monotonic_clock=monotonic_clock,
    )


def write_benchmark_result(
    result: BenchmarkResult,
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    """Write all attempts/events; rejected high-CV runs are never discarded."""

    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    with json_target.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")

    fields = (
        "attempt",
        "attempt_stable",
        "request_id",
        "status",
        "error_type",
        "message",
        "input_tokens",
        "output_tokens",
        "ttft_s",
        "tpot_s",
        "e2e_s",
        "started_s",
        "first_token_s",
        "ended_s",
        "output_text",
        "raw_events_json",
    )
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for attempt in result.attempts:
            for timing in attempt.timings:
                writer.writerow(
                    {
                        "attempt": attempt.attempt,
                        "attempt_stable": attempt.stable,
                        "request_id": timing.request_id,
                        "status": "ok",
                        "input_tokens": timing.input_tokens,
                        "output_tokens": timing.output_tokens,
                        "ttft_s": timing.ttft_s,
                        "tpot_s": timing.tpot_s,
                        "e2e_s": timing.e2e_s,
                        "started_s": timing.started_s,
                        "first_token_s": timing.first_token_s,
                        "ended_s": timing.ended_s,
                        "output_text": timing.output_text,
                        "raw_events_json": json.dumps(timing.raw_events, ensure_ascii=False),
                    }
                )
            for failure in attempt.failures:
                writer.writerow(
                    {
                        "attempt": attempt.attempt,
                        "attempt_stable": attempt.stable,
                        "request_id": failure.request_id,
                        "status": "error",
                        "error_type": failure.error_type,
                        "message": failure.message,
                    }
                )
