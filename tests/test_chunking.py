from app.knowledge.chunking import (
    apply_sentence_overlap,
    build_chunks,
    is_table_block,
    split_table_rows,
)


def test_heading_hierarchy_sets_category_questions_and_path():
    md = "# 退货政策\n\n## 七天无理由\n\n签收后 7 天内可退。\n"

    chunks = build_chunks(
        md, content_type="policy", source_doc="returns-policy.md"
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.category == "退货政策"
    assert chunk.questions == "七天无理由"
    assert chunk.section_path == "退货政策 / 七天无理由"
    assert chunk.answer == "签收后 7 天内可退。"
    assert chunk.content_type == "policy"
    assert chunk.source_doc == "returns-policy.md"
    assert chunk.is_key_clause is False


def test_no_heading_uses_content_type_as_category_and_question():
    chunks = build_chunks(
        "这是一段没有标题的正文。", content_type="policy", source_doc="x.md"
    )

    assert chunks[0].category == "policy"
    assert chunks[0].questions == "policy"
    assert chunks[0].section_path is None


def test_empty_section_is_skipped():
    md = "# 文档\n\n## 空节\n\n## 有内容\n\n正文。\n"

    chunks = build_chunks(md, content_type="manual", source_doc="x.md")

    assert len(chunks) == 1
    assert chunks[0].questions == "有内容"


def test_long_section_is_split_recursively_with_sentence_overlap():
    text = "这是一个句子。" * 200
    md = f"# 文档\n\n## 长文\n\n{text}\n"

    chunks = build_chunks(
        md,
        content_type="manual",
        source_doc="x.md",
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1
    assert all(chunk.category == "文档" for chunk in chunks)
    assert all(chunk.questions == "长文" for chunk in chunks)
    assert all(len(chunk.answer) <= 120 for chunk in chunks)


def test_overlap_uses_complete_trailing_sentences():
    chunks = ["第一句。第二句。第三句。", "新块内容。"]

    out = apply_sentence_overlap(chunks, overlap=6)

    assert out == ["第一句。第二句。第三句。", "第三句。新块内容。"]


def test_overlap_keeps_single_long_sentence_whole():
    chunks = ["这是一个非常长的句子没有标点", "新块。"]

    out = apply_sentence_overlap(chunks, overlap=3)

    assert out[1] == "这是一个非常长的句子没有标点新块。"


def test_large_table_repeats_header_in_every_chunk():
    rows = "\n".join(f"| 区域{i} | {i} |" for i in range(1, 16))
    md = f"| 区域 | 运费 |\n|---|---|\n{rows}\n"

    pieces = split_table_rows(md, max_rows=5)

    assert len(pieces) == 3
    assert all(
        piece.startswith("| 区域 | 运费 |\n|---|---|") for piece in pieces
    )
    assert "| 区域1 | 1 |" in pieces[0]
    assert "| 区域6 | 6 |" in pieces[1]
    assert "| 区域15 | 15 |" in pieces[2]


def test_table_split_keeps_preamble_and_trailing_prose():
    rows = "\n".join(f"| 区域{i} | {i} |" for i in range(1, 16))
    md = (
        "下表是运费标准。\n\n"
        "| 区域 | 运费 |\n|---|---|\n"
        f"{rows}\n\n"
        "以上为参考价。"
    )

    pieces = split_table_rows(md, max_rows=5)

    assert pieces[0].startswith("下表是运费标准。")
    assert pieces[-1].endswith("以上为参考价。")


def test_is_table_block_detects_header_and_separator():
    assert is_table_block("| a | b |\n|---|---|\n| 1 | 2 |")
    assert not is_table_block("普通段落")


def test_key_clause_keyword_marks_chunk():
    md = "# 售后\n\n## 不予退换\n\n定制商品不予退换。\n"

    chunks = build_chunks(md, content_type="manual", source_doc="x.md")

    assert chunks[0].is_key_clause is True
