from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ptq_workshop.evaluate import (
    EvalExample,
    bootstrap_accuracy_ci,
    bootstrap_accuracy_delta_ci,
    load_eval_manifest,
    normalize_gsm8k_answer,
    normalize_mmlu_pro_answer,
    prepare_eval_manifest,
    prepare_evaluation_manifests,
    reasoning_off_chat_payload,
    score_predictions,
)


def mmlu_rows(categories=3, per_category=10):
    return [
        {
            "question": f"Question {category}-{index}?",
            "options": ["zero", "one", "two", "three"],
            "answer": "B",
            "category": f"category-{category}",
        }
        for category in range(categories)
        for index in range(per_category)
    ]


def gsm_rows(count=20):
    return [
        {"question": f"What is {index} + 1?", "answer": f"work\n#### {index + 1}"}
        for index in range(count)
    ]


def test_mmlu_manifest_is_exact_stratified_and_deterministic(tmp_path):
    rows = mmlu_rows()
    loader = lambda repository, config, split: rows
    first = prepare_eval_manifest(
        "mmlu_pro", tmp_path / "first.jsonl", limit=12, seed=42, dataset_loader=loader
    )
    second = prepare_eval_manifest(
        "mmlu_pro", tmp_path / "second.jsonl", limit=12, seed=42, dataset_loader=loader
    )
    assert [item.example_id for item in first] == [item.example_id for item in second]
    counts = {category: sum(item.category == category for item in first) for category in {item.category for item in first}}
    assert set(counts.values()) == {4}
    assert load_eval_manifest(tmp_path / "first.jsonl") == first
    header = json.loads((tmp_path / "first.jsonl").read_text().splitlines()[0])
    assert header["_manifest"]["seed"] == 42


def test_canonical_manifest_api_consumes_profile_counts_and_seed(tmp_path):
    config = SimpleNamespace(
        seed=42,
        profile=SimpleNamespace(mmlu_samples=6, gsm8k_samples=4),
    )

    def loader(repository, dataset_config, split):
        return mmlu_rows() if "MMLU" in repository else gsm_rows()

    manifests = prepare_evaluation_manifests(config, tmp_path, dataset_loader=loader)
    assert len(manifests["mmlu_pro"]) == 6
    assert len(manifests["gsm8k"]) == 4


@pytest.mark.parametrize(
    ("text", "expected"),
    [("B", "B"), ("The answer is option c.", "C"), (r"\boxed{D}", "D")],
)
def test_mmlu_normalization(text, expected):
    assert normalize_mmlu_pro_answer(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("work\n#### 1,200.00", "1200"), (r"\boxed{-2.50}", "-2.5"), ("final 17", "17")],
)
def test_gsm8k_normalization(text, expected):
    assert normalize_gsm8k_answer(text) == expected


def test_scoring_and_bootstrap_are_deterministic():
    examples = (
        EvalExample("a", "gsm8k", "p", "1"),
        EvalExample("b", "gsm8k", "p", "2"),
        EvalExample("c", "gsm8k", "p", "3"),
    )
    report = score_predictions(
        examples,
        {"a": "#### 1", "b": "#### 0", "c": "#### 3"},
        n_resamples=200,
    )
    assert report.accuracy == pytest.approx(2 / 3)
    assert bootstrap_accuracy_ci([True, False, True], n_resamples=200) == report.confidence_interval
    delta = bootstrap_accuracy_delta_ci(
        [True, False, False], [True, True, False], n_resamples=200
    )
    assert delta.estimate == pytest.approx(1 / 3)


def test_reasoning_off_payload_is_explicit_and_preserves_prompt():
    example = EvalExample("a", "mmlu_pro", "exact prompt", "A")
    payload = reasoning_off_chat_payload(model="model", example=example, max_tokens=16)
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 0
    assert payload["messages"][0]["content"] == "exact prompt"
