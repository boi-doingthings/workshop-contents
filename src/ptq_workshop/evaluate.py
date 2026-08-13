"""Deterministic evaluation manifests and accuracy scoring.

The Hugging Face ``datasets`` dependency is optional and imported only when a
manifest is actually downloaded.  Persisted manifests contain the original
source row so workshop results remain auditable after datasets are updated.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SUPPORTED_TASKS = ("mmlu_pro", "gsm8k")
DEFAULT_SEED = 42
DEFAULT_DATASETS = {
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, "test"),
    "gsm8k": ("openai/gsm8k", "main", "test"),
}


@dataclass(frozen=True)
class EvalExample:
    example_id: str
    task: str
    prompt: str
    reference: str
    category: str | None = None
    source_index: int | None = None
    source: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreRecord:
    example_id: str
    task: str
    reference: str
    prediction: str
    normalized_reference: str | None
    normalized_prediction: str | None
    correct: bool
    category: str | None = None


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    low: float
    high: float
    confidence: float
    n_resamples: int
    seed: int


@dataclass(frozen=True)
class ScoreReport:
    task: str
    correct: int
    total: int
    accuracy: float
    confidence_interval: BootstrapCI
    records: tuple[ScoreRecord, ...]


@dataclass(frozen=True)
class RawPrediction:
    example_id: str
    task: str
    text: str
    request_payload: Mapping[str, Any]
    raw_response: Mapping[str, Any]


DatasetLoader = Callable[[str, str | None, str], Iterable[Mapping[str, Any]]]


def _default_dataset_loader(
    repository: str,
    config: str | None,
    split: str,
) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Preparing evaluation manifests requires the optional 'datasets' package"
        ) from exc
    if config is None:
        return load_dataset(repository, split=split)
    return load_dataset(repository, config, split=split)


def build_mmlu_pro_prompt(row: Mapping[str, Any]) -> str:
    question = str(row["question"]).strip()
    options = list(row["options"])
    if len(options) < 2 or len(options) > 26:
        raise ValueError("MMLU-Pro options must contain between 2 and 26 choices")
    choices = "\n".join(
        f"{chr(65 + index)}. {str(option).strip()}" for index, option in enumerate(options)
    )
    return (
        "Answer the multiple-choice question. Respond with only the option letter.\n\n"
        f"Question: {question}\n{choices}\nAnswer:"
    )


def build_gsm8k_prompt(row: Mapping[str, Any]) -> str:
    return (
        "Solve the problem. End with the exact numeric answer in the form "
        "'#### <answer>'.\n\n"
        f"Problem: {str(row['question']).strip()}\nSolution:"
    )


def _mmlu_reference(row: Mapping[str, Any]) -> str:
    if row.get("answer") is not None:
        answer = str(row["answer"]).strip().upper()
    elif row.get("answer_index") is not None:
        answer = chr(65 + int(row["answer_index"]))
    else:
        raise KeyError("MMLU-Pro row has neither 'answer' nor 'answer_index'")
    if not re.fullmatch(r"[A-Z]", answer):
        raise ValueError(f"invalid MMLU-Pro answer: {answer!r}")
    return answer


def _gsm8k_reference(row: Mapping[str, Any]) -> str:
    answer = str(row["answer"])
    match = re.search(r"####\s*([^\n]+)\s*$", answer)
    if not match:
        raise ValueError("GSM8K answer does not contain a trailing '####' reference")
    return match.group(1).strip()


def _select_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int | None,
    seed: int,
) -> list[tuple[int, Mapping[str, Any]]]:
    if limit is None:
        return list(enumerate(rows))
    if limit <= 0:
        raise ValueError("limit must be positive when specified")
    if limit > len(rows):
        raise ValueError(f"requested {limit} examples, but only {len(rows)} are available")
    selected = random.Random(seed).sample(range(len(rows)), limit)
    return [(index, rows[index]) for index in selected]


def _select_stratified_mmlu_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[tuple[int, Mapping[str, Any]]]:
    """Select exactly ``limit`` rows, balanced across available categories.

    Remainder slots are assigned after a seeded category shuffle, and examples
    within each category are sampled independently.  This makes the policy
    deterministic without favoring alphabetically early categories.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")
    indexed_by_category: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for source_index, row in enumerate(rows):
        category = str(row.get("category", ""))
        if not category:
            raise ValueError("every MMLU-Pro row must have a non-empty category")
        indexed_by_category.setdefault(category, []).append((source_index, row))
    if not indexed_by_category:
        raise ValueError("MMLU-Pro rows must not be empty")
    if limit > len(rows):
        raise ValueError(f"requested {limit} examples, but only {len(rows)} are available")

    categories = sorted(indexed_by_category)
    random.Random(seed).shuffle(categories)
    base, remainder = divmod(limit, len(categories))
    quotas = {
        category: base + (1 if position < remainder else 0)
        for position, category in enumerate(categories)
    }
    selected: list[tuple[int, Mapping[str, Any]]] = []
    deficits = 0
    for position, category in enumerate(categories):
        candidates = indexed_by_category[category]
        requested = quotas[category]
        take = min(requested, len(candidates))
        deficits += requested - take
        selected.extend(random.Random(seed + position + 1).sample(candidates, take))
    if deficits:
        already_selected = {index for index, _ in selected}
        remaining = [
            item
            for category in categories
            for item in indexed_by_category[category]
            if item[0] not in already_selected
        ]
        if len(remaining) < deficits:
            raise ValueError("MMLU-Pro categories do not contain enough rows for the requested limit")
        selected.extend(random.Random(seed + 10_000).sample(remaining, deficits))
    random.Random(seed + 20_000).shuffle(selected)
    return selected


def prepare_eval_manifest(
    task: str,
    output_path: str | Path,
    *,
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
    categories: Sequence[str] | None = None,
    dataset_loader: DatasetLoader | None = None,
) -> tuple[EvalExample, ...]:
    """Download, deterministically select and persist an evaluation manifest."""

    if task not in SUPPORTED_TASKS:
        raise ValueError(f"task must be one of {SUPPORTED_TASKS}, got {task!r}")
    if categories is not None and task != "mmlu_pro":
        raise ValueError("categories are supported only for mmlu_pro")
    repository, config, split = DEFAULT_DATASETS[task]
    loader = dataset_loader or _default_dataset_loader
    rows = [dict(row) for row in loader(repository, config, split)]
    if categories is not None:
        requested = set(categories)
        rows = [row for row in rows if str(row.get("category")) in requested]
        present = {str(row.get("category")) for row in rows}
        missing = sorted(requested - present)
        if missing:
            raise ValueError(f"requested MMLU-Pro categories were not found: {missing}")

    examples: list[EvalExample] = []
    selected_rows = (
        _select_stratified_mmlu_rows(rows, limit=limit, seed=seed)
        if task == "mmlu_pro" and limit is not None
        else _select_rows(rows, limit=limit, seed=seed)
    )
    for source_index, row in selected_rows:
        if task == "mmlu_pro":
            prompt = build_mmlu_pro_prompt(row)
            reference = _mmlu_reference(row)
            category = None if row.get("category") is None else str(row["category"])
        else:
            prompt = build_gsm8k_prompt(row)
            reference = _gsm8k_reference(row)
            category = None
        examples.append(
            EvalExample(
                example_id=f"{task}:{source_index:06d}",
                task=task,
                prompt=prompt,
                reference=reference,
                category=category,
                source_index=source_index,
                source=row,
            )
        )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "_manifest": {
                        "task": task,
                        "dataset": repository,
                        "config": config,
                        "split": split,
                        "seed": seed,
                        "limit": limit,
                        "categories": None if categories is None else list(categories),
                        "example_count": len(examples),
                    }
                },
                sort_keys=True,
            )
            + "\n"
        )
        for example in examples:
            handle.write(json.dumps(asdict(example), sort_keys=True, ensure_ascii=False) + "\n")
    return tuple(examples)


def prepare_evaluation_manifests(
    config: Any,
    output_dir: str | Path,
    *,
    dataset_loader: DatasetLoader | None = None,
) -> dict[str, tuple[EvalExample, ...]]:
    """Canonical profile-driven manifest preparation entry point.

    ``config`` is intentionally duck-typed to avoid importing project config at
    module import time.  No independent WORKSHOP/FULL size table lives here:
    sample counts and seed always come from ``WorkshopConfig``.
    """

    profile = config.profile
    seed = int(config.seed)
    output_root = Path(output_dir)
    return {
        "mmlu_pro": prepare_eval_manifest(
            "mmlu_pro",
            output_root / "mmlu_pro.jsonl",
            limit=int(profile.mmlu_samples),
            seed=seed,
            dataset_loader=dataset_loader,
        ),
        "gsm8k": prepare_eval_manifest(
            "gsm8k",
            output_root / "gsm8k.jsonl",
            limit=int(profile.gsm8k_samples),
            seed=seed,
            dataset_loader=dataset_loader,
        ),
    }


def load_eval_manifest(path: str | Path) -> tuple[EvalExample, ...]:
    examples: list[EvalExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "_manifest" in record:
                continue
            try:
                examples.append(EvalExample(**record))
            except TypeError as exc:
                raise ValueError(f"invalid manifest record on line {line_number}") from exc
    return tuple(examples)


def reasoning_off_chat_payload(
    *,
    model: str,
    example: EvalExample,
    max_tokens: int,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Create a deterministic Nemotron chat request with reasoning disabled."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    return {
        "model": model,
        "messages": [{"role": "user", "content": example.prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _chat_response_text(response: Mapping[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("response is missing choices[0].message.content") from exc


def request_evaluation_prediction(
    *,
    endpoint: str,
    model: str,
    example: EvalExample,
    max_tokens: int,
    seed: int = DEFAULT_SEED,
    timeout_s: float = 300.0,
) -> RawPrediction:
    """Request one answer while retaining the complete unmodified response."""

    payload = reasoning_off_chat_payload(
        model=model,
        example=example,
        max_tokens=max_tokens,
        seed=seed,
    )
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw_response = json.loads(response.read().decode("utf-8"))
    return RawPrediction(
        example_id=example.example_id,
        task=example.task,
        text=_chat_response_text(raw_response),
        request_payload=payload,
        raw_response=raw_response,
    )


def write_raw_predictions(
    predictions: Sequence[RawPrediction],
    destination: str | Path,
) -> Path:
    """Write request and raw response JSONL without dropping server metadata."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(
                json.dumps(asdict(prediction), sort_keys=True, ensure_ascii=False) + "\n"
            )
    return target


_MMLU_PATTERNS = (
    re.compile(r"^\s*([A-Z])(?:[\s.)]|$)", re.IGNORECASE),
    re.compile(r"\\boxed\{\s*([A-Z])\s*\}", re.IGNORECASE),
    re.compile(r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Z])\b", re.IGNORECASE),
)


def normalize_mmlu_pro_answer(text: str) -> str | None:
    for pattern in _MMLU_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return str(matches[-1]).upper()
    return None


def _normalize_decimal(text: str) -> str | None:
    cleaned = text.strip().replace(",", "").replace("$", "").replace("%", "")
    cleaned = cleaned.rstrip(". ")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"-0", ""} else normalized


def normalize_gsm8k_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    hashes = re.findall(r"####\s*([^\n]+)", text)
    candidates = hashes or boxed
    if candidates:
        value = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", candidates[-1])
        return None if value is None else _normalize_decimal(value.group(0))
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return None if not numbers else _normalize_decimal(numbers[-1])


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_accuracy_ci(
    correctness: Sequence[bool],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    if not correctness:
        raise ValueError("correctness must not be empty")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    values = [1.0 if item else 0.0 for item in correctness]
    estimate = sum(values) / len(values)
    rng = random.Random(seed)
    bootstrap = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(n_resamples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return BootstrapCI(
        estimate=estimate,
        low=_percentile(bootstrap, alpha),
        high=_percentile(bootstrap, 1.0 - alpha),
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )


def bootstrap_accuracy_delta_ci(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    """Paired bootstrap CI for ``candidate_accuracy - baseline_accuracy``."""

    if not baseline or len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must be non-empty and have equal length")
    if not 0 < confidence < 1 or n_resamples <= 0:
        raise ValueError("invalid bootstrap configuration")
    differences = [float(new) - float(old) for old, new in zip(baseline, candidate)]
    estimate = sum(differences) / len(differences)
    rng = random.Random(seed)
    bootstrap = [
        sum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(n_resamples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return BootstrapCI(
        estimate=estimate,
        low=_percentile(bootstrap, alpha),
        high=_percentile(bootstrap, 1.0 - alpha),
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )


def _prediction_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "output", "prediction", "content"):
            if value.get(key) is not None:
                return str(value[key])
        if value.get("raw_response") is not None:
            return _chat_response_text(value["raw_response"])
    raise TypeError("prediction values must be strings or mappings with a text-like field")


def score_predictions(
    examples: Sequence[EvalExample],
    predictions: Mapping[str, Any],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> ScoreReport:
    if not examples:
        raise ValueError("examples must not be empty")
    tasks = {example.task for example in examples}
    if len(tasks) != 1:
        raise ValueError("score_predictions expects a single task per call")
    task = next(iter(tasks))
    records: list[ScoreRecord] = []
    for example in examples:
        if example.example_id not in predictions:
            raise KeyError(f"missing prediction for {example.example_id}")
        prediction = _prediction_text(predictions[example.example_id])
        if task == "mmlu_pro":
            normalized_prediction = normalize_mmlu_pro_answer(prediction)
            normalized_reference = normalize_mmlu_pro_answer(example.reference)
        elif task == "gsm8k":
            normalized_prediction = normalize_gsm8k_answer(prediction)
            normalized_reference = normalize_gsm8k_answer(example.reference)
        else:
            raise ValueError(f"unsupported task in examples: {task!r}")
        records.append(
            ScoreRecord(
                example_id=example.example_id,
                task=task,
                reference=example.reference,
                prediction=prediction,
                normalized_reference=normalized_reference,
                normalized_prediction=normalized_prediction,
                correct=(
                    normalized_reference is not None
                    and normalized_prediction == normalized_reference
                ),
                category=example.category,
            )
        )
    correctness = [record.correct for record in records]
    interval = bootstrap_accuracy_ci(
        correctness,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    return ScoreReport(
        task=task,
        correct=sum(correctness),
        total=len(records),
        accuracy=interval.estimate,
        confidence_interval=interval,
        records=tuple(records),
    )


def write_score_report(
    report: ScoreReport,
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    with json_target.open("w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    fields = list(ScoreRecord.__dataclass_fields__)
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in report.records)
