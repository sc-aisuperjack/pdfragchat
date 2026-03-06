import json
from pathlib import Path
from src.config import Settings
from src.store import DocStore

def test_store_lists_documents(tmp_path):
    settings = Settings(
        openai_api_key="dummy",
        chat_model="dummy",
        embed_model="dummy",
        data_dir=str(tmp_path / "data"),
        chroma_dir=str(tmp_path / "data" / "index" / "chroma"),
    )
    store = DocStore(settings)

    doc_hash = "abc123abc123"
    doc_dir = Path(settings.data_dir) / "docs" / doc_hash
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "manifest.json").write_text(json.dumps({
        "doc_hash": doc_hash,
        "title": "file.pdf",
        "created_at": 1,
        "page_count": 2,
        "chunk_count": 10,
        "collection": f"doc_{doc_hash}",
    }), encoding="utf-8")

    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["doc_hash"] == doc_hash