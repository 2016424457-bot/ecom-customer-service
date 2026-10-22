from pathlib import Path

from app.knowledge.chunking import build_chunks, split_table_rows
from app.knowledge.sources import DOCS


def test_all_source_docs_exist_and_have_chunks():
    for name, content_type in DOCS.items():
        path = Path("knowledge_docs") / name
        assert path.exists(), name
        chunks = build_chunks(
            path.read_text(encoding="utf-8"),
            content_type=content_type,
            source_doc=name,
        )
        assert chunks, name
        assert all(chunk.answer for chunk in chunks), name


def test_returns_policy_has_shipping_fee_section():
    text = (
        Path("knowledge_docs") / "returns-policy.md"
    ).read_text(encoding="utf-8")

    assert "邮费" in text and "运费" in text
    chunks = build_chunks(
        text, content_type="policy", source_doc="returns-policy.md"
    )
    assert any(
        "邮费" in chunk.questions or "运费" in chunk.questions
        for chunk in chunks
    )


def test_product_specs_has_large_table():
    text = (
        Path("knowledge_docs") / "product-specs.md"
    ).read_text(encoding="utf-8")

    assert len(split_table_rows(text, max_rows=12)) > 1
