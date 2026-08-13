"""Fail-closed hardware and environment checks for Blackwell PTQ runs."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import sha256_file, write_json_atomic
from .config import (
    MODELOPT_EXAMPLE,
    MODELOPT_GIT_COMMIT,
    MODELOPT_VERSION,
    TRTLLM_IMAGE,
    TRTLLM_VERSION,
    WorkshopConfig,
)


GIB = 1024**3
# B200/GB200 report SM100 and B300/GB300 report SM103. RTX Blackwell
# workstations report SM120. DGX Spark is SM121 on ARM64 and is deliberately
# outside this x86_64 workshop runtime contract.
SUPPORTED_COMPUTE_CAPABILITY_MAJORS = frozenset({10})
SUPPORTED_COMPUTE_CAPABILITIES = frozenset({(12, 0)})


class PreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    compute_capability: tuple[int, int]
    memory_total_mib: int
    uuid: str = "unknown"
    driver_version: str = "unknown"
    memory_used_mib: int = 0
    utilization_gpu_percent: float = 0.0
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    sm_clock_mhz: int = 0
    memory_clock_mhz: int = 0
    ecc_mode: str = "unknown"
    software_thermal_slowdown: str = "unknown"
    hardware_thermal_slowdown: str = "unknown"


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[CheckResult, ...]
    gpus: tuple[GpuInfo, ...] = ()
    versions: Mapping[str, str] = field(default_factory=dict)
    hf_token_present: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def require_ok(self) -> "PreflightReport":
        if not self.ok:
            summary = "; ".join(f"{item.name}: {item.detail}" for item in self.failures)
            raise PreflightError(f"Preflight failed: {summary}")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [asdict(check) for check in self.checks],
            "gpus": [
                {
                    **asdict(gpu),
                    "compute_capability": list(gpu.compute_capability),
                }
                for gpu in self.gpus
            ],
            "versions": dict(sorted(self.versions.items())),
            # Never persist token values.
            "hf_token_present": self.hf_token_present,
        }


def parse_compute_capability(value: str | int | float | Sequence[int]) -> tuple[int, int]:
    """Parse values such as ``10.0``, ``12.0``, or ``(12, 0)``."""

    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Invalid compute capability: {value!r}")
        major, minor = int(value[0]), int(value[1])
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("Compute capability cannot be empty")
        parts = text.split(".")
        if len(parts) == 1:
            major, minor = int(parts[0]), 0
        elif len(parts) == 2 and all(part.isdigit() for part in parts):
            major, minor = int(parts[0]), int(parts[1])
        else:
            raise ValueError(f"Invalid compute capability: {value!r}")
    if major < 0 or minor < 0:
        raise ValueError(f"Invalid compute capability: {value!r}")
    return major, minor


def is_supported_compute_capability(value: str | int | float | Sequence[int]) -> bool:
    try:
        capability = parse_compute_capability(value)
    except (TypeError, ValueError):
        return False
    return (
        capability[0] in SUPPORTED_COMPUTE_CAPABILITY_MAJORS
        or capability in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _number(value: str, cast: Callable[[float], Any]) -> Any:
    text = value.strip().replace("[N/A]", "0").replace("N/A", "0")
    return cast(float(text))


def parse_nvidia_smi_csv(output: str) -> tuple[GpuInfo, ...]:
    """Parse the full audit query, while accepting the four-column test fixture."""

    gpus: list[GpuInfo] = []
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) not in {4, 16}:
            raise ValueError(f"Malformed nvidia-smi row {line_number}: {raw_line!r}")
        index_text, name, capability_text, memory_text = fields[:4]
        try:
            base: dict[str, Any] = dict(
                index=int(index_text),
                name=name,
                compute_capability=parse_compute_capability(capability_text),
                memory_total_mib=_number(memory_text, int),
            )
            if len(fields) == 16:
                (
                    base["uuid"],
                    base["driver_version"],
                    used,
                    utilization,
                    temperature,
                    power_draw,
                    power_limit,
                    sm_clock,
                    memory_clock,
                    base["ecc_mode"],
                    base["software_thermal_slowdown"],
                    base["hardware_thermal_slowdown"],
                ) = fields[4:]
                base.update(
                    memory_used_mib=_number(used, int),
                    utilization_gpu_percent=_number(utilization, float),
                    temperature_c=_number(temperature, float),
                    power_draw_w=_number(power_draw, float),
                    power_limit_w=_number(power_limit, float),
                    sm_clock_mhz=_number(sm_clock, int),
                    memory_clock_mhz=_number(memory_clock, int),
                )
            gpus.append(GpuInfo(**base))
        except ValueError as exc:
            raise ValueError(f"Malformed nvidia-smi row {line_number}: {raw_line!r}") from exc
    if not gpus:
        raise ValueError("nvidia-smi returned no GPUs")
    return tuple(gpus)


def disk_space_check(
    path: Path | str,
    minimum_free_gib: int,
    *,
    usage: Callable[[Path | str], shutil._ntuple_diskusage] = shutil.disk_usage,
) -> CheckResult:
    if minimum_free_gib <= 0:
        raise ValueError("minimum_free_gib must be positive")
    target = Path(path)
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_bytes = usage(probe).free
    free_gib = free_bytes / GIB
    return CheckResult(
        name="disk_space",
        passed=free_bytes >= minimum_free_gib * GIB,
        detail=f"{free_gib:.1f} GiB free at {probe}; {minimum_free_gib} GiB required",
    )


def _completed_stdout(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout if isinstance(result.stdout, str) else ""


def run_preflight(
    config: WorkshopConfig,
    *,
    modelopt_root: Path | str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    package_version: Callable[[str], str] = importlib.metadata.version,
    disk_usage: Callable[[Path | str], shutil._ntuple_diskusage] = shutil.disk_usage,
    environment: Mapping[str, str] | None = None,
    maximum_idle_utilization_percent: float = 5.0,
    maximum_idle_memory_mib: int = 1024,
    maximum_temperature_c: float = 85.0,
    minimum_free_disk_gib: int | None = None,
) -> PreflightReport:
    """Run every check and return all failures instead of masking one with another."""

    checks: list[CheckResult] = []
    gpus: tuple[GpuInfo, ...] = ()
    versions: dict[str, str] = {}
    environment = os.environ if environment is None else environment
    modelopt_root = Path(modelopt_root).expanduser().resolve()
    script_path = modelopt_root / MODELOPT_EXAMPLE

    python_ok = (3, 10) <= sys.version_info[:2] < (3, 15)
    checks.append(
        CheckResult(
            "python",
            python_ok,
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}; requires >=3.10,<3.15",
        )
    )
    disk_requirement = (
        config.profile.minimum_free_disk_gib
        if minimum_free_disk_gib is None
        else minimum_free_disk_gib
    )
    checks.append(disk_space_check(config.artifact_root, disk_requirement, usage=disk_usage))
    checks.append(
        CheckResult(
            "modelopt_example",
            script_path.is_file(),
            f"expected pinned example at {script_path}",
        )
    )

    try:
        installed_version = package_version("nvidia-modelopt")
        versions["nvidia-modelopt"] = installed_version
        checks.append(
            CheckResult(
                "modelopt_package",
                installed_version == MODELOPT_VERSION,
                f"installed {installed_version}; required {MODELOPT_VERSION}",
            )
        )
    except importlib.metadata.PackageNotFoundError:
        checks.append(CheckResult("modelopt_package", False, "nvidia-modelopt is not installed"))
    except Exception as exc:  # package metadata must never make preflight disappear
        checks.append(CheckResult("modelopt_package", False, f"could not read package version: {exc}"))

    try:
        trtllm_version = package_version("tensorrt-llm")
        versions["tensorrt-llm"] = trtllm_version
        checks.append(
            CheckResult(
                "tensorrt_llm_package",
                trtllm_version == TRTLLM_VERSION,
                f"installed {trtllm_version}; required {TRTLLM_VERSION}",
            )
        )
    except importlib.metadata.PackageNotFoundError:
        checks.append(CheckResult("tensorrt_llm_package", False, "tensorrt-llm is not installed"))
    except Exception as exc:
        checks.append(CheckResult("tensorrt_llm_package", False, f"could not read version: {exc}"))

    try:
        git_result = runner(
            ["git", "-C", str(modelopt_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        git_commit = _completed_stdout(git_result).strip()
        checks.append(
            CheckResult(
                "modelopt_git_commit",
                git_result.returncode == 0 and git_commit == MODELOPT_GIT_COMMIT,
                f"checkout {git_commit or '<unavailable>'}; required {MODELOPT_GIT_COMMIT}",
            )
        )
    except (OSError, subprocess.SubprocessError) as exc:
        checks.append(CheckResult("modelopt_git_commit", False, f"git check failed: {exc}"))

    try:
        smi_result = runner(
            [
                "nvidia-smi",
                "--query-gpu=index,name,compute_cap,memory.total,uuid,driver_version,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,ecc.mode.current,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if smi_result.returncode != 0:
            detail = (smi_result.stderr or "nvidia-smi failed").strip()
            checks.append(CheckResult("blackwell_gpu", False, detail))
        else:
            gpus = parse_nvidia_smi_csv(_completed_stdout(smi_result))
            checks.append(
                CheckResult(
                    "single_visible_gpu",
                    len(gpus) == 1,
                    f"found {len(gpus)} visible GPU(s); exactly one selected GPU is required",
                )
            )
            unsupported = [
                f"GPU {gpu.index} ({gpu.name}, cc {gpu.compute_capability[0]}.{gpu.compute_capability[1]})"
                for gpu in gpus
                if not is_supported_compute_capability(gpu.compute_capability)
            ]
            detail = (
                "; ".join(unsupported)
                if unsupported
                else ", ".join(
                    f"GPU {gpu.index}: {gpu.name} cc {gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
                    for gpu in gpus
                )
            )
            checks.append(CheckResult("blackwell_gpu", not unsupported, detail))
            busy = [
                gpu
                for gpu in gpus
                if gpu.utilization_gpu_percent > maximum_idle_utilization_percent
                or gpu.memory_used_mib > maximum_idle_memory_mib
            ]
            checks.append(
                CheckResult(
                    "gpu_idle",
                    not busy,
                    (
                        "idle-state gate passed"
                        if not busy
                        else "; ".join(
                            f"GPU {gpu.index}: util={gpu.utilization_gpu_percent:.1f}%, memory={gpu.memory_used_mib} MiB"
                            for gpu in busy
                        )
                    ),
                )
            )
            thermal = [
                gpu
                for gpu in gpus
                if gpu.temperature_c >= maximum_temperature_c
                or "active" in gpu.software_thermal_slowdown.lower()
                and "not active" not in gpu.software_thermal_slowdown.lower()
                or "active" in gpu.hardware_thermal_slowdown.lower()
                and "not active" not in gpu.hardware_thermal_slowdown.lower()
            ]
            checks.append(
                CheckResult(
                    "gpu_thermal_state",
                    not thermal,
                    (
                        "no thermal slowdown"
                        if not thermal
                        else "; ".join(
                            f"GPU {gpu.index}: {gpu.temperature_c:.1f} C, sw={gpu.software_thermal_slowdown}, hw={gpu.hardware_thermal_slowdown}"
                            for gpu in thermal
                        )
                    ),
                )
            )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        checks.append(CheckResult("blackwell_gpu", False, f"GPU discovery failed: {exc}"))

    torch_probe = (
        "import json, torch; "
        "print(json.dumps({'cuda': torch.cuda.is_available(), "
        "'torch': torch.__version__, 'cuda_version': torch.version.cuda}))"
    )
    try:
        torch_result = runner(
            [sys.executable, "-c", torch_probe],
            check=False,
            capture_output=True,
            text=True,
        )
        torch_data = json.loads(_completed_stdout(torch_result)) if torch_result.returncode == 0 else {}
        if torch_data:
            versions["torch"] = str(torch_data.get("torch"))
            versions["cuda"] = str(torch_data.get("cuda_version"))
        torch_ok = torch_result.returncode == 0 and torch_data.get("cuda") is True
        detail = (
            f"torch {torch_data.get('torch')}, CUDA {torch_data.get('cuda_version')}"
            if torch_data
            else (torch_result.stderr or "PyTorch probe failed").strip()
        )
        checks.append(CheckResult("pytorch_cuda", torch_ok, detail))
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        checks.append(CheckResult("pytorch_cuda", False, f"PyTorch probe failed: {exc}"))

    return PreflightReport(
        tuple(checks),
        gpus,
        versions,
        bool(environment.get("HF_TOKEN") or environment.get("HUGGING_FACE_HUB_TOKEN")),
    )


def preflight_or_raise(config: WorkshopConfig, **kwargs: object) -> PreflightReport:
    """Convenience entry point for scripts: never continue after a failed check."""

    return run_preflight(config, **kwargs).require_ok()


def write_preflight_manifest(
    path: Path | str,
    config: WorkshopConfig,
    report: PreflightReport,
    *,
    prepared_manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Persist the environment audit without ever serializing credentials."""

    payload = report.as_dict()
    payload.update(
        {
            "configuration": config.fingerprint_payload(),
            "container_image": os.environ.get("PTQ_IMAGE", TRTLLM_IMAGE),
            "prepared_assets": dict(prepared_manifest or {}),
        }
    )
    pip_freeze = config.project_root / "pip-freeze.txt"
    if not pip_freeze.is_file():
        raise PreflightError(f"Missing bootstrap lock snapshot: {pip_freeze}")
    payload["pip_freeze"] = {
        "sha256": sha256_file(pip_freeze),
        "packages": [
            line
            for line in pip_freeze.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
    }
    return write_json_atomic(path, payload)
