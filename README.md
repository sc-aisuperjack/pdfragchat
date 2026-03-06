# PDF RAG Chat (PDF → Markdown → RAG + Eval)

Live app: https://pdfragchat.streamlit.app/

A practical, interview-grade RAG system.
Upload one or more PDFs, convert them to Markdown (no LLM used for conversion), chunk, embed with OpenAIEmbeddings, store vectors locally, and chat with grounded answers. Includes caching, page-aware metadata, multi-document chat, and retrieval evaluation (MRR, Hit@k, Recall, Precision, MAP).

## What this project does

1. Upload PDFs (single or multiple).
2. Convert PDF to Markdown using PyMuPDF (no LLM).
3. Chunk Markdown with LangChain text splitters.
4. Embed chunks using OpenAIEmbeddings.
5. Store vectors locally (FAISS) per document.
6. Chat with one or more selected documents (multi-doc RAG).
7. Evaluate retrieval quality using MRR, Hit@k, Recall@k, Precision@k, and MAP@k.

## Key features

- Multi-document chat (select multiple PDFs and ask one question across them).
- Page-aware citations (each chunk stores page numbers, shown in sources).
- Caching (identical PDF content is not re-ingested).
- Separate document library (list, select active, delete).
- Retrieval evaluation (upload JSONL or CSV and compute metrics).
- Clean separation between ingestion, retrieval, chat, and evaluation.

## Tech stack

Python, Streamlit, LangChain, OpenAI (embeddings and chat), FAISS, PyMuPDF, pytest.

## Project structure

- app.py: Streamlit UI (Upload, Library, Chat, Evaluate).
- src/pdf_to_md.py: PDF bytes to Markdown conversion (PyMuPDF).
- src/chunking.py: Chunking plus page-number metadata.
- src/store.py: Document caching, manifests, FAISS persistence per document.
- src/rag.py: Single-doc and multi-doc retrieval + chat.
- src/evals.py: MRR, Hit@k, Recall, Precision, MAP.
- tests/: unit tests plus an OpenAI-gated integration test.

## Setup

Create and activate a virtual environment.

Windows:
python -m venv .venv
.venv\Scripts\activate

macOS or Linux:
python -m venv .venv
source .venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Create a .env file (or set environment variables):
OPENAI_API_KEY=your_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
DATA_DIR=data

Run locally:
streamlit run app.py

## Streamlit deployment (secrets.toml)

In Streamlit Cloud, set secrets in TOML format:

OPENAI_API_KEY = "your_key_here"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
DATA_DIR = "data"

Note: Streamlit Cloud storage can be ephemeral across restarts, so indexed files may need to be re-ingested after redeploys.

## Caching behaviour

A document hash is computed from the raw PDF bytes (SHA-256 shortened).
If the same PDF content is uploaded again, ingestion is skipped and it is marked cached.

## Evaluation dataset format

Upload JSONL or CSV.

JSONL example:
{"query":"What is the purpose of this document?","expected_pages":[1]}
{"query":"Define term X","expected_pages":[2]}

CSV columns:
query,expected_pages,expected_chunk_ids

expected_pages can be a single number or a comma-separated list like "1,2".
expected_chunk_ids can be used if you want to label relevance by chunk index.

Metrics:

- Hit@k (accuracy style)
- MRR@k
- Recall@k
- Precision@k
- MAP@k

## Tests

Run unit tests:
pytest -q

Integration test (requires OPENAI_API_KEY):
pytest -q -k integration

## Roadmap ideas

- “Chat with all documents” toggle.
- Hybrid retrieval (keyword + vectors).
- Eval set builder inside the UI (click relevant chunks to generate gold data).
- Optional persistence to S3/GCS for stable indices across redeploys.

## Author

Stefanos Cunning
GitHub: https://github.com/sc-aisuperjack
