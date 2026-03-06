from __future__ import annotations
import re
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter

PAGE_HEADER_RE = re.compile(r"^# Page (\d+)\s*$", re.MULTILINE)

@dataclass(frozen=True)
class ChunkOut:
    chunk_index: int
    text: str
    pages: list[int]

def _page_spans(md: str) -> list[tuple[int, int, int]]:
    """
    Returns list of (page_num, start_char, end_char) spans in markdown.
    """
    matches = list(PAGE_HEADER_RE.finditer(md))
    spans: list[tuple[int, int, int]] = []
    if not matches:
        return spans

    for i, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        spans.append((page_num, start, end))
    return spans

def _pages_for_chunk(spans: list[tuple[int, int, int]], chunk_start: int, chunk_end: int) -> list[int]:
    pages = []
    for page_num, s, e in spans:
        # overlap check
        if chunk_end <= s:
            continue
        if chunk_start >= e:
            continue
        pages.append(page_num)
    return pages if pages else []

def chunk_markdown_with_pages(md: str, chunk_size: int = 1200, overlap: int = 150) -> list[ChunkOut]:
    md = md.strip()
    spans = _page_spans(md)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    # We want character positions too, so we do a manual sliding window over the split output.
    # LangChain split_text does not expose offsets; we reconstruct via incremental search.
    texts = splitter.split_text(md)

    chunks: list[ChunkOut] = []
    cursor = 0
    for i, t in enumerate(texts):
        # find t in md starting from cursor (best-effort)
        pos = md.find(t, cursor)
        if pos == -1:
            pos = md.find(t)  # fallback
        start = max(pos, 0)
        end = start + len(t)
        cursor = end

        pages = _pages_for_chunk(spans, start, end)
        chunks.append(ChunkOut(chunk_index=i, text=t, pages=pages))

    return chunks