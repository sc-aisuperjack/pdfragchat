# pdfragchat

PDF Rag Chat App - multi-PDF, caching, page-aware metadata, LangChain pipeline, chat across any uploaded doc, plus retrieval evaluation (MRR, Accuracy/Hit@k, Recall@k, Precision@k, MAP).

## Setup

python -m venv .venv
source .venv/bin/activate (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

Copy .env.example to .env and set OPENAI_API_KEY.

## Run

streamlit run app.py

## Tests

pytest -q

## Evaluation Dataset Format

JSONL example:
{"query":"What is the purpose of this document?","expected_pages":[1]}
{"query":"Define term X","expected_chunk_ids":["3"]}

CSV columns:
query,expected_pages,expected_chunk_ids
"What is ...?","1,2",""
"Define ...","", "3"
