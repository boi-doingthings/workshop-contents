from ptq_workshop.io import atomic_write_json, read_jsonl, sha256_json, write_jsonl


def test_canonical_digest_is_order_independent():
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_json_writers(tmp_path):
    path = atomic_write_json(tmp_path / "nested" / "value.json", {"ok": True})
    assert path.exists()
    jsonl = write_jsonl(tmp_path / "rows.jsonl", [{"n": 1}, {"n": 2}])
    assert read_jsonl(jsonl) == [{"n": 1}, {"n": 2}]
