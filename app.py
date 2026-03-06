import json
import time
from pathlib import Path
import streamlit as st
import pandas as pd

from src.config import get_settings
from src.store import DocStore
from src.rag import RagEngine
from src.evals import evaluate_retrieval, load_eval_dataset

st.set_page_config(page_title="PDF RAG Chat", layout="wide")

settings = get_settings()
store = DocStore(settings)
rag = RagEngine(settings, store)

st.title("📄 PDF RAG Chat (Markdown → LangChain RAG → Eval)")

tab_upload, tab_library, tab_chat, tab_eval = st.tabs(
    ["1) Upload & Index", "2) Library", "3) Chat", "4) Evaluate Retrieval"]
)

# ----------------------------
# 1) Upload & Index
# ----------------------------
with tab_upload:
    st.subheader("Upload PDFs")
    st.caption("PDFs are converted to Markdown with PyMuPDF. Chunking, embeddings, vector store, retrieval, and chat are LangChain-based.")

    uploaded = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

    colA, colB, colC = st.columns(3)
    with colA:
        chunk_size = st.number_input("Chunk size (chars)", min_value=300, max_value=5000, value=1200, step=100)
    with colB:
        overlap = st.number_input("Overlap (chars)", min_value=0, max_value=2000, value=150, step=25)
    with colC:
        k_default = st.number_input("Default retrieval k", min_value=1, max_value=20, value=5, step=1)

    if uploaded:
        if st.button("🚀 Convert → Chunk → Embed → Index (with caching)", type="primary"):
            results = []
            for f in uploaded:
                raw = f.getvalue()
                with st.spinner(f"Processing: {f.name}"):
                    t0 = time.time()
                    out = store.ingest_pdf_bytes(
                        filename=f.name,
                        pdf_bytes=raw,
                        chunk_size=int(chunk_size),
                        overlap=int(overlap),
                    )
                    dt = time.time() - t0
                    results.append(
                        {
                            "filename": f.name,
                            "doc_hash": out.doc_hash,
                            "status": out.status,
                            "chunks": out.chunk_count,
                            "seconds": round(dt, 2),
                        }
                    )

            store.set_default_k(int(k_default))
            st.success("Done.")
            st.dataframe(pd.DataFrame(results), use_container_width=True)

# ----------------------------
# 2) Library
# ----------------------------
with tab_library:
    st.subheader("Document Library")
    docs = store.list_documents()

    if not docs:
        st.info("No documents indexed yet. Upload PDFs in the first tab.")
    else:
        df = pd.DataFrame(docs)
        df = df.sort_values(["created_at"], ascending=False)
        st.dataframe(df, use_container_width=True)

        st.markdown("### Actions")
        col1, col2 = st.columns(2)

        with col1:
            selected = st.selectbox(
                "Select active document for chat",
                options=[d["doc_hash"] for d in docs],
                format_func=lambda h: f"{h} — {store.get_doc_title(h)}",
            )
            if st.button("✅ Set active"):
                store.set_active_doc(selected)
                st.success(f"Active doc set to {selected}")

        with col2:
            to_delete = st.selectbox(
                "Delete a document (removes index + stored files)",
                options=["(none)"] + [d["doc_hash"] for d in docs],
                format_func=lambda h: h if h == "(none)" else f"{h} — {store.get_doc_title(h)}",
            )
            if to_delete != "(none)" and st.button("🗑️ Delete", type="secondary"):
                store.delete_document(to_delete)
                st.success(f"Deleted {to_delete}. Refresh the page if needed.")

# ----------------------------
# 3) Chat
# ----------------------------
with tab_chat:
    st.subheader("Chat with any indexed PDF")
    docs = store.list_documents()

    if not docs:
        st.info("Upload and index at least one PDF first.")
        st.stop()

    active = store.get_active_doc() or docs[0]["doc_hash"]
    doc_choice = st.selectbox(
        "Choose document",
        options=[d["doc_hash"] for d in docs],
        index=[d["doc_hash"] for d in docs].index(active) if active in [d["doc_hash"] for d in docs] else 0,
        format_func=lambda h: f"{h} — {store.get_doc_title(h)}",
    )
    store.set_active_doc(doc_choice)

    k = st.slider("Retrieval k", 1, 20, value=store.get_default_k(), step=1)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask something about this document...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Retrieving + answering..."):
            answer, sources = rag.ask(doc_hash=doc_choice, question=question, k=int(k))

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        with st.expander("Sources (chunk text + page numbers)"):
            for i, s in enumerate(sources, start=1):
                meta = s.get("metadata", {})
                pages = meta.get("pages", [])
                st.markdown(f"**Source {i}** — pages: {pages} — chunk: {meta.get('chunk_index')}")
                st.text(s.get("text", "")[:2500] + ("..." if len(s.get("text", "")) > 2500 else ""))

# ----------------------------
# 4) Evaluate Retrieval
# ----------------------------
with tab_eval:
    st.subheader("Evaluate retrieval (MRR, Accuracy/Hit@k, Recall@k, Precision@k, MAP)")
    st.caption("You provide an eval dataset. The app tests retrieval quality against the expected page or expected chunk id.")

    docs = store.list_documents()
    if not docs:
        st.info("Upload and index at least one PDF first.")
        st.stop()

    doc_hash = st.selectbox(
        "Document to evaluate",
        options=[d["doc_hash"] for d in docs],
        format_func=lambda h: f"{h} — {store.get_doc_title(h)}",
    )

    k_eval = st.slider("k for evaluation", 1, 20, value=5, step=1)

    st.markdown("### Upload evaluation file")
    st.caption("Supported: JSONL or CSV. Required columns/fields: query, expected_pages OR expected_chunk_ids.")
    eval_file = st.file_uploader("Upload eval dataset", type=["jsonl", "csv"])

    if eval_file is not None:
        dataset = load_eval_dataset(eval_file)
        st.write(f"Loaded {len(dataset)} evaluation queries.")
        st.dataframe(pd.DataFrame(dataset).head(20), use_container_width=True)

        if st.button("📊 Run evaluation", type="primary"):
            with st.spinner("Running retrieval evaluation..."):
                metrics, per_query = evaluate_retrieval(
                    rag=rag,
                    doc_hash=doc_hash,
                    dataset=dataset,
                    k=int(k_eval),
                )

            st.success("Done.")
            st.markdown("### Metrics")
            st.json(metrics)

            st.markdown("### Per-query results (top hits)")
            st.dataframe(pd.DataFrame(per_query), use_container_width=True)