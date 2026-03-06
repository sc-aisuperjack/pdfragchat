from reportlab.pdfgen import canvas
from src.pdf_to_md import pdf_bytes_to_markdown

def test_pdf_to_markdown_basic(tmp_path):
    pdf_path = tmp_path / "t.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Hello PDF world")
    c.drawString(100, 730, "Second line")
    c.showPage()
    c.drawString(100, 750, "Page two text")
    c.save()

    pdf_bytes = pdf_path.read_bytes()
    res = pdf_bytes_to_markdown(pdf_bytes)

    assert res.page_count == 2
    assert "Hello PDF world" in res.markdown
    assert "# Page 1" in res.markdown
    assert "# Page 2" in res.markdown