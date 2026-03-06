import os
import pytest
from reportlab.pdfgen import canvas

from src.config import get_settings
from src.store import DocStore
from src.rag import RagEngine

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set (integration test)")
def test_full_ingest_and_retrieve(tmp_path, monkeypatch):
    # Override DATA_DIR and CHROMA_DIR to temp locations
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "data" / "index" / "chroma"))

    settings = get_settings()
    store = DocStore(settings)
    rag = RagEngine(settings, store)

    pdf_path = tmp_path / "doc.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "France capital is Paris.")
    c.showPage()
    c.drawString(100, 750, "Germany capital is Berlin.")
    c.save()

    res = store.ingest_pdf_bytes("doc.pdf", pdf_path.read_bytes(), chunk_size=500, overlap=50)
    assert res.chunk_count > 0

    retrieved = rag.retrieve(res.doc_hash, "What is the capital of France?", k=3)
    assert len(retrieved) > 0