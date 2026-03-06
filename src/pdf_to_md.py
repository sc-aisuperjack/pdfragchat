from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import fitz  # PyMuPDF

_whitespace_re = re.compile(r"[ \t]+")

def _clean_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = _whitespace_re.sub(" ", line).strip()
    return line

@dataclass(frozen=True)
class MdConversion:
    markdown: str
    page_count: int
    page_texts: list[str]  # raw-ish per page (cleaned)

def pdf_bytes_to_markdown(pdf_bytes: bytes) -> MdConversion:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    md_parts: list[str] = []
    page_texts: list[str] = []

    for p in range(doc.page_count):
        page = doc.load_page(p)
        text = page.get_text("text") or ""
        lines = [_clean_line(l) for l in text.splitlines()]
        lines = [l for l in lines if l]
        page_text = "\n".join(lines).strip()
        page_texts.append(page_text)

        md_parts.append("\n\n---\n\n")
        md_parts.append(f"# Page {p+1}\n\n")
        md_parts.append(page_text + "\n")

    doc.close()
    md = "".join(md_parts).strip() + "\n"
    return MdConversion(markdown=md, page_count=len(page_texts), page_texts=page_texts)

def write_doc_files(doc_dir: Path, pdf_bytes: bytes, markdown: str) -> None:
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "original.pdf").write_bytes(pdf_bytes)
    (doc_dir / "document.md").write_text(markdown, encoding="utf-8")