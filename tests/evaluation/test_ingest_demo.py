from pathlib import Path

from scripts.ingest_demo import iter_supported_paths


def test_ingest_demo_expands_multiple_inputs_and_deduplicates(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    text = docs / "policy.txt"
    text.write_text("制度", encoding="utf-8")
    markdown = docs / "expense.md"
    markdown.write_text("# 报销", encoding="utf-8")
    (docs / "ignore.bin").write_bytes(b"x")

    paths = iter_supported_paths([docs, text])

    assert paths == [markdown, text]
