from __future__ import annotations
import io
import json
from typing import Any

import numpy as np
import pandas as pd

def load_eval_dataset(uploaded_file) -> list[dict[str, Any]]:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".jsonl"):
        lines = raw.decode("utf-8").splitlines()
        rows = [json.loads(l) for l in lines if l.strip()]
        return rows

    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
        return df.to_dict(orient="records")

    raise ValueError("Unsupported eval file format. Use JSONL or CSV.")

def _is_hit(item: dict[str, Any], expected_pages: list[int] | None, expected_chunk_ids: list[str] | None) -> bool:
    meta = item.get("metadata", {}) or {}

    if expected_chunk_ids:
        # We store chunk_index but not the full chroma id. Use chunk_index string matching if user provides.
        # Accept either "chunk_index" int or string.
        ci = meta.get("chunk_index")
        if ci is not None and str(ci) in set([str(x) for x in expected_chunk_ids]):
            return True

    if expected_pages:
        pages = meta.get("pages", [])
        if isinstance(pages, list):
            return any(int(p) in set(expected_pages) for p in pages if isinstance(p, int) or str(p).isdigit())

    return False

def evaluate_retrieval(rag, doc_hash: str, dataset: list[dict[str, Any]], k: int = 5) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """
    Computes:
    - Hit@k (accuracy style)
    - MRR@k
    - Recall@k (same as hit@k when only one relevant target is defined, but we support multi-page relevance)
    - Precision@k
    - MAP@k
    """
    hits = []
    rr = []
    recalls = []
    precisions = []
    ap_scores = []
    per_query_rows = []

    for row in dataset:
        query = str(row.get("query", "")).strip()
        if not query:
            continue

        expected_pages = row.get("expected_pages", None)
        expected_chunk_ids = row.get("expected_chunk_ids", None)

        # normalise expected fields
        if isinstance(expected_pages, str):
            # allow "1,2,3"
            expected_pages = [int(x.strip()) for x in expected_pages.split(",") if x.strip().isdigit()]
        if isinstance(expected_chunk_ids, str):
            expected_chunk_ids = [x.strip() for x in expected_chunk_ids.split(",") if x.strip()]

        if expected_pages is not None and not isinstance(expected_pages, list):
            expected_pages = [int(expected_pages)]
        if expected_chunk_ids is not None and not isinstance(expected_chunk_ids, list):
            expected_chunk_ids = [str(expected_chunk_ids)]

        results = rag.retrieve(doc_hash=doc_hash, query=query, k=k)

        # Determine relevance per rank
        rel = []
        first_rel_rank = None
        for i, item in enumerate(results, start=1):
            is_rel = _is_hit(item, expected_pages, expected_chunk_ids)
            rel.append(1 if is_rel else 0)
            if is_rel and first_rel_rank is None:
                first_rel_rank = i

        hit = 1 if any(rel) else 0
        hits.append(hit)

        rrank = 0.0 if first_rel_rank is None else 1.0 / float(first_rel_rank)
        rr.append(rrank)

        # Recall: did we retrieve at least one relevant? (for multi-relevant cases, it's still "any")
        recalls.append(hit)

        # Precision@k: relevant retrieved / k
        precisions.append(float(sum(rel)) / float(len(rel) if len(rel) else k))

        # AP@k (average precision)
        # AP = average of precision@i at each i where rel_i=1, divided by number of relevant retrieved (or 0)
        if sum(rel) == 0:
            ap = 0.0
        else:
            prec_at_i = []
            running_rel = 0
            for i, r in enumerate(rel, start=1):
                if r == 1:
                    running_rel += 1
                    prec_at_i.append(running_rel / i)
            ap = float(np.mean(prec_at_i)) if prec_at_i else 0.0
        ap_scores.append(ap)

        # Log some top results for visibility
        top_hits = []
        for item in results[: min(3, len(results))]:
            meta = item.get("metadata", {}) or {}
            top_hits.append(
                {
                    "chunk_index": meta.get("chunk_index"),
                    "pages": meta.get("pages"),
                    "source_filename": meta.get("source_filename"),
                }
            )

        per_query_rows.append(
            {
                "query": query,
                "hit@k": hit,
                "mrr@k": rrank,
                "precision@k": precisions[-1],
                "ap@k": ap,
                "expected_pages": expected_pages,
                "expected_chunk_ids": expected_chunk_ids,
                "top_results": top_hits,
            }
        )

    metrics = {
        "k": float(k),
        "num_queries": float(len(hits)),
        "hit@k": float(np.mean(hits)) if hits else 0.0,
        "mrr@k": float(np.mean(rr)) if rr else 0.0,
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "precision@k": float(np.mean(precisions)) if precisions else 0.0,
        "map@k": float(np.mean(ap_scores)) if ap_scores else 0.0,
    }
    return metrics, per_query_rows