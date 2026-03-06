from src.chunking import chunk_markdown_with_pages

def test_chunking_extracts_pages():
    md = """
# Page 1

Hello page one.

# Page 2

Hello page two. More content here.
""".strip()

    chunks = chunk_markdown_with_pages(md, chunk_size=50, overlap=10)
    assert len(chunks) >= 1
    assert any(1 in c.pages for c in chunks)
    assert any(2 in c.pages for c in chunks)