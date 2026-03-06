from __future__ import annotations
from typing import Any

from langchain_openai import ChatOpenAI

from src.config import Settings
from src.store import DocStore

class RagEngine:
    def __init__(self, settings: Settings, store: DocStore):
        self.settings = settings
        self.store = store
        self.llm = ChatOpenAI(
            model=settings.chat_model,
            temperature=0.2,
            api_key=settings.openai_api_key,
        )

    def ask(self, doc_hash: str, question: str, k: int = 5) -> tuple[str, list[dict[str, Any]]]:
        db = self.store.load_vectordb(doc_hash)
        docs = db.similarity_search(question, k=int(k))

        # build context (simple + effective)
        context_blocks = []
        sources = []
        for d in docs:
            sources.append({"text": d.page_content, "metadata": dict(d.metadata) if d.metadata else {}})
            context_blocks.append(d.page_content)

        context = "\n\n---\n\n".join(context_blocks)

        prompt = (
            "Use the context to answer the question.\n"
            "If the answer is not in the context, say you do not know.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context}\n"
        )

        resp = self.llm.invoke(prompt)
        answer = (resp.content or "").strip()

        pages = []
        for s in sources:
            p = s.get("metadata", {}).get("pages", [])
            if isinstance(p, list):
                pages.extend(p)
        pages = sorted(set([int(x) for x in pages if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit())]))

        if pages:
            answer = f"{answer}\n\n_Cited pages: {pages}_"

        return answer, sources

    def retrieve(self, doc_hash: str, query: str, k: int = 5) -> list[dict[str, Any]]:
        db = self.store.load_vectordb(doc_hash)
        docs = db.similarity_search(query, k=int(k))
        out = []
        for d in docs:
            out.append({"text": d.page_content, "metadata": dict(d.metadata) if d.metadata else {}})
        return out