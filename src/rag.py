from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI

from src.config import Settings
from src.store import DocStore


@dataclass(frozen=True)
class RetrievedItem:
    text: str
    metadata: dict[str, Any]
    score: float


class RagEngine:
    """
    FAISS-backed RAG engine.

    - ask(): single document chat
    - ask_multi(): multi-document chat (merge top-k across selected docs)
    - retrieve(): retrieval-only (single doc) for evaluation
    - retrieve_multi(): retrieval-only (multi doc) for evaluation
    """

    def __init__(self, settings: Settings, store: DocStore):
        self.settings = settings
        self.store = store
        self.llm = ChatOpenAI(
            model=settings.chat_model,
            temperature=0.2,
            api_key=settings.openai_api_key,
        )

    # ---------------------------
    # Prompt / context building
    # ---------------------------
    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        return (
            "You are a helpful assistant.\n"
            "Use ONLY the provided CONTEXT to answer the QUESTION.\n"
            "If the answer is not in the CONTEXT, say you do not know.\n"
            "If you quote or rely on specific details, keep them faithful to the context.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context}\n"
        )

    @staticmethod
    def _normalise_pages(value: Any) -> list[int]:
        if not isinstance(value, list):
            return []
        out: list[int] = []
        for x in value:
            if isinstance(x, int):
                out.append(x)
            elif isinstance(x, str) and x.strip().isdigit():
                out.append(int(x.strip()))
        return sorted(set(out))

    @staticmethod
    def _summarise_citations(items: list[RetrievedItem]) -> dict[str, list[int]]:
        """
        Returns a dict: {doc_title: [pages...]}
        """
        cited: dict[str, set[int]] = {}
        for it in items:
            meta = it.metadata or {}
            title = meta.get("doc_title") or meta.get("doc_hash") or "unknown"
            pages = RagEngine._normalise_pages(meta.get("pages", []))
            if pages:
                cited.setdefault(str(title), set()).update(pages)
        return {k: sorted(list(v)) for k, v in cited.items()}

    @staticmethod
    def _format_context(items: list[RetrievedItem], max_chars: int = 12000) -> str:
        """
        Build a single context string with doc/page headers.
        Limit context length to reduce latency and prompt bloat.
        """
        parts: list[str] = []
        total = 0

        for it in items:
            meta = it.metadata or {}
            header = (
                f"[doc={meta.get('doc_title', meta.get('doc_hash'))} "
                f"| pages={meta.get('pages')} "
                f"| chunk={meta.get('chunk_index')} "
                f"| score={it.score}]"
            )
            block = header + "\n" + (it.text or "").strip()

            if not block.strip():
                continue

            if total + len(block) > max_chars:
                break

            parts.append(block)
            total += len(block)

        return "\n\n---\n\n".join(parts).strip()

    # ---------------------------
    # Retrieval helpers
    # ---------------------------
    def _retrieve_items_single(self, doc_hash: str, query: str, k: int) -> list[RetrievedItem]:
        db = self.store.load_vectordb(doc_hash)
        doc_title = self.store.get_doc_title(doc_hash)

        # FAISS wrapper typically supports similarity_search_with_score
        docs_scores = db.similarity_search_with_score(query, k=int(k))

        items: list[RetrievedItem] = []
        for d, score in docs_scores:
            meta = dict(d.metadata) if d.metadata else {}
            meta["doc_hash"] = doc_hash
            meta["doc_title"] = doc_title

            items.append(
                RetrievedItem(
                    text=d.page_content,
                    metadata=meta,
                    score=float(score),
                )
            )

        # For FAISS with L2 distance, LOWER score means closer.
        items.sort(key=lambda x: x.score)
        return items

    def _retrieve_items_multi(self, doc_hashes: list[str], query: str, k: int) -> list[RetrievedItem]:
        """
        Retrieve from each doc and merge globally.
        We pull per-doc candidates then take global top-k.
        """
        if not doc_hashes:
            return []

        per_doc_k = max(int(k), 3)  # pull enough from each doc to make the merge meaningful
        all_items: list[RetrievedItem] = []

        for dh in doc_hashes:
            try:
                all_items.extend(self._retrieve_items_single(dh, query, per_doc_k))
            except FileNotFoundError:
                # index missing for that doc hash; just skip it
                continue

        all_items.sort(key=lambda x: x.score)
        return all_items[: int(k)]

    # ---------------------------
    # Public API: retrieval-only
    # ---------------------------
    def retrieve(self, doc_hash: str, query: str, k: int = 5) -> list[dict[str, Any]]:
        items = self._retrieve_items_single(doc_hash, query, int(k))
        return [{"text": it.text, "metadata": it.metadata, "score": it.score} for it in items]

    def retrieve_multi(self, doc_hashes: list[str], query: str, k: int = 5) -> list[dict[str, Any]]:
        items = self._retrieve_items_multi(doc_hashes, query, int(k))
        return [{"text": it.text, "metadata": it.metadata, "score": it.score} for it in items]

    # ---------------------------
    # Public API: chat
    # ---------------------------
    def ask(self, doc_hash: str, question: str, k: int = 5) -> tuple[str, list[dict[str, Any]]]:
        """
        Single doc chat. Returns (answer, sources).
        Sources include: text, metadata, score.
        """
        items = self._retrieve_items_single(doc_hash, question, int(k))
        context = self._format_context(items)

        prompt = self._build_prompt(question=question, context=context)
        resp = self.llm.invoke(prompt)
        answer = (resp.content or "").strip()

        citations = self._summarise_citations(items)
        if citations:
            answer = f"{answer}\n\n_Cited pages by document: {citations}_"

        sources = [{"text": it.text, "metadata": it.metadata, "score": it.score} for it in items]
        return answer, sources

    def ask_multi(self, doc_hashes: list[str], question: str, k: int = 5) -> tuple[str, list[dict[str, Any]]]:
        """
        Multi-doc chat. Merge top-k across selected documents.
        Returns (answer, sources).
        """
        items = self._retrieve_items_multi(doc_hashes, question, int(k))
        if not items:
            return "I could not retrieve any context from the selected documents.", []

        context = self._format_context(items)
        prompt = self._build_prompt(question=question, context=context)
        resp = self.llm.invoke(prompt)
        answer = (resp.content or "").strip()

        citations = self._summarise_citations(items)
        if citations:
            answer = f"{answer}\n\n_Cited pages by document: {citations}_"

        sources = [{"text": it.text, "metadata": it.metadata, "score": it.score} for it in items]
        return answer, sources