from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import Settings
from src.hashing import sha256_bytes, short_hash
from src.pdf_to_md import pdf_bytes_to_markdown, write_doc_files
from src.chunking import chunk_markdown_with_pages
from src.schemas import IngestResult

class DocStore:
    """
    Owns:
    - data/docs/<doc_hash>/ files + manifest.json
    - FAISS index per doc_hash persisted under data/index/faiss/<doc_hash>/
    - prefs.json (active doc, default k)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base = Path(settings.data_dir)
        self.docs_dir = self.base / "docs"
        self.index_dir = self.base / "index" / "faiss"
        self.prefs_path = self.base / "prefs.json"

        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    # ---------- prefs ----------
    def _load_prefs(self) -> dict:
        if self.prefs_path.exists():
            return json.loads(self.prefs_path.read_text(encoding="utf-8"))
        return {"active_doc": None, "default_k": 5}

    def _save_prefs(self, prefs: dict) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        self.prefs_path.write_text(json.dumps(prefs, indent=2), encoding="utf-8")

    def set_active_doc(self, doc_hash: str) -> None:
        prefs = self._load_prefs()
        prefs["active_doc"] = doc_hash
        self._save_prefs(prefs)

    def get_active_doc(self) -> str | None:
        return self._load_prefs().get("active_doc")

    def set_default_k(self, k: int) -> None:
        prefs = self._load_prefs()
        prefs["default_k"] = int(k)
        self._save_prefs(prefs)

    def get_default_k(self) -> int:
        return int(self._load_prefs().get("default_k", 5))

    # ---------- docs ----------
    def _doc_dir(self, doc_hash: str) -> Path:
        return self.docs_dir / doc_hash

    def _manifest_path(self, doc_hash: str) -> Path:
        return self._doc_dir(doc_hash) / "manifest.json"

    def _faiss_dir(self, doc_hash: str) -> Path:
        return self.index_dir / doc_hash

    def get_doc_title(self, doc_hash: str) -> str:
        m = self.get_manifest(doc_hash)
        return m.get("title", doc_hash)

    def get_manifest(self, doc_hash: str) -> dict[str, Any]:
        p = self._manifest_path(doc_hash)
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def list_documents(self) -> list[dict[str, Any]]:
        out = []
        for d in self.docs_dir.glob("*"):
            if not d.is_dir():
                continue
            mf = d / "manifest.json"
            if not mf.exists():
                continue
            m = json.loads(mf.read_text(encoding="utf-8"))
            out.append(
                {
                    "doc_hash": m.get("doc_hash"),
                    "title": m.get("title"),
                    "created_at": m.get("created_at"),
                    "chunk_count": m.get("chunk_count"),
                    "page_count": m.get("page_count"),
                }
            )
        return out

    def delete_document(self, doc_hash: str) -> None:
        # remove files
        doc_dir = self._doc_dir(doc_hash)
        if doc_dir.exists():
            for p in doc_dir.rglob("*"):
                if p.is_file():
                    p.unlink()
            for p in sorted(doc_dir.rglob("*"), reverse=True):
                if p.is_dir():
                    p.rmdir()
            doc_dir.rmdir()

        # remove faiss index dir
        faiss_dir = self._faiss_dir(doc_hash)
        if faiss_dir.exists():
            for p in faiss_dir.rglob("*"):
                if p.is_file():
                    p.unlink()
            for p in sorted(faiss_dir.rglob("*"), reverse=True):
                if p.is_dir():
                    p.rmdir()
            faiss_dir.rmdir()

        prefs = self._load_prefs()
        if prefs.get("active_doc") == doc_hash:
            prefs["active_doc"] = None
            self._save_prefs(prefs)

    # ---------- embeddings ----------
    def _embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=self.settings.embed_model,
            api_key=self.settings.openai_api_key,
        )

    # ---------- FAISS vector db ----------
    def load_vectordb(self, doc_hash: str) -> FAISS:
        faiss_dir = self._faiss_dir(doc_hash)
        embeddings = self._embeddings()
        if not faiss_dir.exists():
            raise FileNotFoundError(f"FAISS index not found for doc {doc_hash}")
        return FAISS.load_local(str(faiss_dir), embeddings, allow_dangerous_deserialization=True)

    def save_vectordb(self, doc_hash: str, db: FAISS) -> None:
        faiss_dir = self._faiss_dir(doc_hash)
        faiss_dir.mkdir(parents=True, exist_ok=True)
        db.save_local(str(faiss_dir))

    # ---------- ingestion ----------
    def ingest_pdf_bytes(self, filename: str, pdf_bytes: bytes, chunk_size: int, overlap: int) -> IngestResult:
        full_hash = sha256_bytes(pdf_bytes)
        doc_hash = short_hash(full_hash, 12)
        doc_dir = self._doc_dir(doc_hash)

        # Cache check: manifest + faiss dir exists
        mf_path = self._manifest_path(doc_hash)
        if mf_path.exists() and self._faiss_dir(doc_hash).exists():
            m = json.loads(mf_path.read_text(encoding="utf-8"))
            return IngestResult(doc_hash=doc_hash, status="cached", chunk_count=int(m.get("chunk_count", 0)))

        md_conv = pdf_bytes_to_markdown(pdf_bytes)
        write_doc_files(doc_dir, pdf_bytes, md_conv.markdown)

        chunks = chunk_markdown_with_pages(md_conv.markdown, chunk_size=chunk_size, overlap=overlap)

        texts = [c.text for c in chunks]
        metadatas = [
            {
                "doc_hash": doc_hash,
                "source_filename": filename,
                "chunk_index": c.chunk_index,
                "pages": c.pages,
            }
            for c in chunks
        ]

        embeddings = self._embeddings()
        db = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        self.save_vectordb(doc_hash, db)

        manifest = {
            "doc_hash": doc_hash,
            "title": filename,
            "created_at": int(time.time()),
            "page_count": md_conv.page_count,
            "chunk_count": len(chunks),
            "chunking": {"chunk_size": chunk_size, "overlap": overlap},
            "vectorstore": "faiss",
        }
        mf_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        prefs = self._load_prefs()
        if not prefs.get("active_doc"):
            prefs["active_doc"] = doc_hash
            self._save_prefs(prefs)

        return IngestResult(doc_hash=doc_hash, status="indexed", chunk_count=len(chunks))