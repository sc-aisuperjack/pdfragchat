from src.evals import evaluate_retrieval

class FakeRag:
    def retrieve(self, doc_hash: str, query: str, k: int = 5):
        # Always returns 3 results with metadata
        return [
            {"text": "a", "metadata": {"chunk_index": 0, "pages": [2]}},
            {"text": "b", "metadata": {"chunk_index": 1, "pages": [1]}},
            {"text": "c", "metadata": {"chunk_index": 2, "pages": [3]}},
        ][:k]

def test_metrics_mrr_hit():
    rag = FakeRag()
    dataset = [
        {"query": "q1", "expected_pages": [1]},
        {"query": "q2", "expected_chunk_ids": ["0"]},
    ]
    metrics, rows = evaluate_retrieval(rag, doc_hash="x", dataset=dataset, k=3)

    assert metrics["num_queries"] == 2.0
    # q1: relevant at rank 2 => RR=0.5, hit=1
    # q2: relevant at rank 1 => RR=1.0, hit=1
    assert abs(metrics["hit@k"] - 1.0) < 1e-9
    assert abs(metrics["mrr@k"] - 0.75) < 1e-9