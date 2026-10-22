import re

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.knowledge.types import Chunk

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
# 中文没有词边界：先段落/换行，再句末标点，再逗号，最后逐字。
CJK_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "!",
    "?",
    ";",
    "，",
    " ",
    "",
]
KEY_TERMS = (
    "不予",
    "不支持",
    "必须",
    "应当",
    "不得",
    "免责",
    "赔偿",
    "时效",
    "超过",
    "除外",
    "仅限",
    "责任",
    "一经",
)
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
SENTENCE_RE = re.compile(r"(?<=[。！？；!?;])\s*|(?<=\.)\s+")


def split_sections(md: str) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS,
        strip_headers=True,
    )
    return splitter.split_text(md)


def recursive_split(
    text: str, chunk_size: int, chunk_overlap: int = 0
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CJK_SEPARATORS,
    )
    return splitter.split_text(text)


def _split_sentences(text: str) -> list[str]:
    return [part for part in SENTENCE_RE.split(text) if part]


def _trailing_sentences(text: str, max_chars: int) -> str:
    """取结尾若干完整句作为重叠，总长尽量不超过 max_chars。"""
    if max_chars <= 0:
        return ""
    out: list[str] = []
    total = 0
    for sentence in reversed(_split_sentences(text)):
        if out and total + len(sentence) > max_chars:
            break
        out.insert(0, sentence)
        total += len(sentence)
    return "".join(out)


def apply_sentence_overlap(chunks: list[str], overlap: int) -> list[str]:
    if not chunks:
        return []
    out = [chunks[0]]
    for index in range(1, len(chunks)):
        previous = _trailing_sentences(chunks[index - 1], overlap)
        out.append(previous + chunks[index] if previous else chunks[index])
    return out


def _find_table_header(lines: list[str]) -> int:
    for index in range(len(lines) - 1):
        if (
            lines[index].lstrip().startswith("|")
            and TABLE_SEP_RE.match(lines[index + 1])
            and "-" in lines[index + 1]
        ):
            return index
    return -1


def is_table_block(text: str) -> bool:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return _find_table_header(lines) != -1


def split_table_rows(table_md: str, max_rows: int = 12) -> list[str]:
    lines = [line for line in table_md.strip().splitlines() if line.strip()]
    header_index = _find_table_header(lines)
    if header_index == -1:
        return [table_md.strip()]

    preamble = lines[:header_index]
    header = lines[header_index]
    separator = lines[header_index + 1]
    end = header_index + 2
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    rows = lines[header_index + 2 : end]
    trailing = lines[end:]

    if len(rows) <= max_rows:
        return [table_md.strip()]

    groups = [
        rows[start : start + max_rows]
        for start in range(0, len(rows), max_rows)
    ]
    pieces: list[str] = []
    for index, group in enumerate(groups):
        block = (preamble if index == 0 else []) + [header, separator, *group]
        if index == len(groups) - 1:
            block += trailing
        pieces.append("\n".join(block))
    return pieces


def _is_key_clause(title: str, body: str) -> bool:
    head = title + body[:40]
    return any(term in head for term in KEY_TERMS)


def build_chunks(
    md: str,
    *,
    content_type: str,
    source_doc: str,
    chunk_size: int = 700,
    overlap: int = 120,
    table_max_rows: int = 12,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in split_sections(md):
        path = [
            section.metadata[key]
            for key in ("h1", "h2", "h3", "h4")
            if section.metadata.get(key)
        ]
        section_path = " / ".join(path) if path else None
        title = path[-1] if path else content_type
        if len(path) > 1:
            category = " / ".join(path[:-1])
        else:
            category = path[0] if path else content_type

        body = section.page_content.strip()
        if not body:
            continue

        if is_table_block(body):
            pieces = split_table_rows(body, max_rows=table_max_rows)
        else:
            pieces = apply_sentence_overlap(
                recursive_split(body, chunk_size, chunk_overlap=0),
                overlap,
            )

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    category=category,
                    questions=title,
                    answer=piece,
                    section_path=section_path,
                    content_type=content_type,
                    is_key_clause=_is_key_clause(title, piece),
                    source_doc=source_doc,
                )
            )
    return chunks
