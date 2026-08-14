"""Auditable TensorRT-LLM server lifecycle helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


class ServerStartupError(RuntimeError):
    """Raised when the serving process exits or misses its readiness deadline."""


SMOKE_RESPONSE_SENTINEL = "BLACKWELL PTQ READY"


@dataclass(frozen=True)
class TensorRTLLMServerConfig:
    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    backend: str = "_autodeploy"
    executable: str = "trtllm-serve"
    python_executable: str | None = None
    tensor_parallel_size: int | None = None
    max_batch_size: int | None = None
    max_num_tokens: int | None = None
    max_seq_len: int | None = None
    config_path: str | Path | None = None
    hf_revision: str | None = None
    trust_remote_code: bool = False
    reasoning_parser: str | None = "nano-v3"
    kv_cache_dtype: str = "auto"
    prefix_cache_reuse: bool = False
    speculative_decoding: bool = False
    extra_args: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if self.python_executable is not None and not self.python_executable:
            raise ValueError("python_executable must not be empty when specified")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        for name in ("tensor_parallel_size", "max_batch_size", "max_num_tokens", "max_seq_len"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when specified")
        reserved = {
            "--backend",
            "--host",
            "--port",
            "--tp_size",
            "--max_batch_size",
            "--max_num_tokens",
            "--max_seq_len",
            "--config",
            "--extra_llm_api_options",
            "--hf_revision",
            "--revision",
            "--trust_remote_code",
            "--reasoning_parser",
            "--kv_cache_dtype",
        }
        duplicates = sorted(reserved.intersection(self.extra_args))
        if duplicates:
            raise ValueError(
                "extra_args duplicates explicitly modeled options: " + ", ".join(duplicates)
            )
        if self.prefix_cache_reuse or self.speculative_decoding:
            raise ValueError(
                "controlled workshop comparisons require prefix-cache reuse and "
                "speculative decoding to remain disabled"
            )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def command(self) -> tuple[str, ...]:
        """Return the exact command without invoking a shell."""

        command = [
            *(() if self.python_executable is None else (self.python_executable,)),
            self.executable,
            "serve",
            self.model,
            "--backend",
            self.backend,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        options: tuple[tuple[str, Any], ...] = (
            ("--tp_size", self.tensor_parallel_size),
            ("--max_batch_size", self.max_batch_size),
            ("--max_num_tokens", self.max_num_tokens),
            ("--max_seq_len", self.max_seq_len),
            ("--config", self.config_path),
            ("--hf_revision", self.hf_revision),
            ("--reasoning_parser", self.reasoning_parser),
            ("--kv_cache_dtype", self.kv_cache_dtype),
        )
        for flag, value in options:
            if value is not None:
                command.extend((flag, str(value)))
        if self.trust_remote_code:
            command.append("--trust_remote_code")
        command.extend(self.extra_args)
        return tuple(command)


def server_config_for_workshop(
    workshop_config: Any,
    *,
    model_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    environment: Mapping[str, str] | None = None,
    precision: str = "bf16",
    compute_capability: tuple[int, int] = (10, 0),
) -> TensorRTLLMServerConfig:
    """Build the canonical hardware-specific TensorRT-LLM launch contract.

    All supported Blackwell targets use AutoDeploy for every precision.  This
    is the current Nemotron deployment path in TensorRT-LLM rc23; in
    particular, it replaces the rc17-only native/CUTLASS workaround that
    loaded SM120 NVFP4 checkpoints but produced invalid token IDs.
    """

    profile_name = getattr(workshop_config.profile.name, "value", workshop_config.profile.name)
    if precision not in {"bf16", "fp8", "nvfp4"}:
        raise ValueError(f"Unknown precision: {precision}")
    if compute_capability == (12, 0):
        backend = "_autodeploy"
        yaml_name = "nano_v3_sm120.yaml"
    elif compute_capability[0] == 10:
        backend = "_autodeploy"
        yaml_name = "nano_v3_dev.yaml" if profile_name == "DEV_SMOKE" else "nano_v3_b200.yaml"
    else:
        raise ValueError(f"Unsupported Blackwell compute capability: {compute_capability}")
    config_path = Path(workshop_config.project_root) / "configs" / yaml_name
    return TensorRTLLMServerConfig(
        model=str(model_path or workshop_config.model_id),
        host=host,
        port=port,
        backend=backend,
        executable="/usr/local/bin/trtllm-serve",
        python_executable=sys.executable,
        config_path=config_path,
        hf_revision=(None if model_path is not None else workshop_config.model_revision),
        trust_remote_code=True,
        tensor_parallel_size=1,
        reasoning_parser="nano-v3",
        environment={} if environment is None else environment,
    )


def require_exact_smoke_response(
    response: Any, *, variant: str, expected: str = SMOKE_RESPONSE_SENTINEL
) -> str:
    """Require the deterministic serving sentinel, not merely nonempty text."""

    try:
        text = str(response["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Malformed smoke-generation response for {variant}") from exc
    if text != expected:
        raise RuntimeError(
            f"Smoke-generation response mismatch for {variant}: {text!r} != {expected!r}"
        )
    return text


def request_json(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> Any:
    """Perform a dependency-free JSON request and return the unmodified body."""

    body = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else None


class TensorRTLLMServer:
    """Own one TensorRT-LLM serving subprocess."""

    def __init__(
        self,
        config: TensorRTLLMServerConfig,
        *,
        log_path: str | Path | None = None,
        dry_run: bool = False,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        health_probe: Callable[[str, float], bool] | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.log_path = None if log_path is None else Path(log_path)
        self.dry_run = dry_run
        self._popen_factory = popen_factory
        self._health_probe = health_probe or self._default_health_probe
        self._monotonic_clock = monotonic_clock
        self._sleep = sleep
        self._process: Any | None = None
        self._log_handle: Any | None = None

    @property
    def process(self) -> Any | None:
        return self._process

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(
        self,
        *,
        wait: bool = True,
        startup_timeout_s: float = 900.0,
        poll_interval_s: float = 1.0,
    ) -> "TensorRTLLMServer":
        if startup_timeout_s <= 0 or poll_interval_s <= 0:
            raise ValueError("startup timeout and polling interval must be positive")
        if self.is_running:
            return self
        if self.dry_run:
            return self

        stdout: Any = subprocess.DEVNULL
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("a", encoding="utf-8")
            stdout = self._log_handle
        environment = os.environ.copy()
        environment.update({str(key): str(value) for key, value in self.config.environment.items()})
        self._process = self._popen_factory(
            list(self.config.command()),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
            start_new_session=True,
        )
        if wait:
            self.wait_ready(
                timeout_s=startup_timeout_s,
                poll_interval_s=poll_interval_s,
            )
        return self

    @staticmethod
    def _default_health_probe(url: str, timeout_s: float) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as response:
                return 200 <= int(response.status) < 300
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return False

    def wait_ready(self, *, timeout_s: float = 900.0, poll_interval_s: float = 1.0) -> None:
        if self.dry_run:
            return
        if self._process is None:
            raise ServerStartupError("server has not been started")
        deadline = self._monotonic_clock() + timeout_s
        health_url = f"{self.config.base_url}/health"
        while self._monotonic_clock() < deadline:
            return_code = self._process.poll()
            if return_code is not None:
                raise ServerStartupError(
                    f"trtllm-serve exited with code {return_code}; log={self.log_path}"
                )
            if self._health_probe(health_url, min(5.0, poll_interval_s)):
                return
            self._sleep(poll_interval_s)
        raise ServerStartupError(
            f"trtllm-serve did not become healthy within {timeout_s:.1f}s; log={self.log_path}"
        )

    def stop(self, *, timeout_s: float = 30.0) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout_s)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self) -> "TensorRTLLMServer":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()


# Compatibility alias for downstream imports created before the SM120 native
# runtime was added.
TensorRTLLMAutoDeployServer = TensorRTLLMServer


def save_server_manifest(
    config: TensorRTLLMServerConfig,
    destination: str | Path,
) -> Path:
    """Persist the exact launch contract for provenance and dry-run review."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "command": list(config.command()),
                "launcher": {
                    "python_executable": config.python_executable,
                    "executable": config.executable,
                },
                "base_url": config.base_url,
                "environment_overrides": dict(config.environment),
                "controlled_optimization_policy": {
                    "runtime_backend": config.backend,
                    "runtime_config": (
                        None if config.config_path is None else str(config.config_path)
                    ),
                    "prefix_cache_reuse": config.prefix_cache_reuse,
                    "speculative_decoding": config.speculative_decoding,
                    "kv_cache_dtype": config.kv_cache_dtype,
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return target
