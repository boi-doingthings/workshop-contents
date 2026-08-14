"""Freeze public calibration and evaluation data into immutable manifests."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import sha256_json, write_jsonl

CNN_DATASET = "abisee/cnn_dailymail"
CNN_CONFIG = "3.0.0"
MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"
GSM8K_DATASET = "openai/gsm8k"


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_indices(population: int, count: int, seed: int) -> list[int]:
    if count < 0 or count > population:
        raise ValueError(f"count={count} must be in [0, {population}]")
    rng = random.Random(seed)
    return sorted(rng.sample(range(population), count))


def stratified_sample(rows: Iterable[dict[str, Any]], count: int, key: str, seed: int) -> list[dict[str, Any]]:
    """Choose a deterministic approximately proportional sample by category."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(dict(row))
    total = sum(map(len, groups.values()))
    if count > total:
        raise ValueError(f"requested {count} rows from {total}")
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    remainders: list[tuple[float, str]] = []
    for name in sorted(groups):
        group = groups[name]
        rng.shuffle(group)
        exact = count * len(group) / total
        take = min(len(group), int(exact))
        chosen.extend(group[:take])
        groups[name] = group[take:]
        remainders.append((exact - take, name))
    for _, name in sorted(remainders, reverse=True):
        if len(chosen) >= count:
            break
        if groups[name]:
            chosen.append(groups[name].pop())
    if len(chosen) < count:
        remaining = [row for name in sorted(groups) for row in groups[name]]
        rng.shuffle(remaining)
        chosen.extend(remaining[: count - len(chosen)])
    return sorted(chosen, key=lambda row: str(row.get("sample_id", row.get("question_id", ""))))


def _load_dataset(*args: Any, **kwargs: Any) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the project HF dependencies before preparing assets") from exc
    return load_dataset(*args, **kwargs)


def freeze_calibration(output: str | Path, count: int, seed: int = 42) -> dict[str, Any]:
    dataset = _load_dataset(CNN_DATASET, CNN_CONFIG, split="train")
    indices = deterministic_indices(len(dataset), count, seed)
    rows = []
    for index in indices:
        item = dataset[index]
        text = str(item["article"]).strip()
        rows.append({"sample_id": f"train:{index}", "dataset": CNN_DATASET, "text": text, "sha256": text_digest(text)})
    path = write_jsonl(output, rows)
    return {"path": str(path), "count": len(rows), "sha256": sha256_json(rows), "sample_ids": [r["sample_id"] for r in rows]}


def freeze_evaluation(output_dir: str | Path, mmlu_count: int, gsm8k_count: int, seed: int = 42) -> dict[str, Any]:
    destination = Path(output_dir)
    mmlu_raw = _load_dataset(MMLU_PRO_DATASET, split="test")
    candidates = []
    for index, item in enumerate(mmlu_raw):
        candidates.append({
            "sample_id": f"mmlu_pro:test:{index}",
            "task": "mmlu_pro",
            "category": item.get("category", "unknown"),
            "question": item["question"],
            "options": item["options"],
            "answer": item["answer"],
        })
    mmlu_rows = stratified_sample(candidates, mmlu_count, "category", seed)
    gsm_raw = _load_dataset(GSM8K_DATASET, "main", split="test")
    gsm_indices = deterministic_indices(len(gsm_raw), gsm8k_count, seed)
    gsm_rows = [{
        "sample_id": f"gsm8k:test:{index}",
        "task": "gsm8k",
        "question": gsm_raw[index]["question"],
        "answer": gsm_raw[index]["answer"],
    } for index in gsm_indices]
    mmlu_path = write_jsonl(destination / "mmlu_pro.jsonl", mmlu_rows)
    gsm_path = write_jsonl(destination / "gsm8k.jsonl", gsm_rows)
    return {
        "mmlu_pro": {"path": str(mmlu_path), "count": len(mmlu_rows), "sha256": sha256_json(mmlu_rows)},
        "gsm8k": {"path": str(gsm_path), "count": len(gsm_rows), "sha256": sha256_json(gsm_rows)},
    }
