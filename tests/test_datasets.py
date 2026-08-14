import pytest

from ptq_workshop.datasets import deterministic_indices, stratified_sample


def test_deterministic_indices():
    assert deterministic_indices(100, 5, 42) == deterministic_indices(100, 5, 42)
    assert len(set(deterministic_indices(100, 5, 42))) == 5
    with pytest.raises(ValueError):
        deterministic_indices(2, 3, 42)


def test_stratified_sample_is_stable_and_sized():
    rows = [{"sample_id": str(i), "category": "a" if i < 8 else "b"} for i in range(10)]
    first = stratified_sample(rows, 5, "category", 7)
    second = stratified_sample(rows, 5, "category", 7)
    assert first == second
    assert len(first) == 5
    assert {r["category"] for r in first} == {"a", "b"}
