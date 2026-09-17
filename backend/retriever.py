"""
Evidence Retriever: Find relevant passages from source documents.
Hybrid BM25 + dense semantic embeddings with cross-encoder reranking.
"""
from typing import List, Tuple, Optional
from api.services.retriever import Retriever as EnhancedRetriever, chunk_document

class Retriever:
    """
    Hybrid retriever combining BM25 (lexical) and semantic search + cross-encoder reranking.
    """

    def __init__(self, source_documents: List[str]):
        self._retriever = EnhancedRetriever(source_documents)
        self.documents = source_documents

    def retrieve(self, claim: str, top_k: int = 3, rerank: bool = True) -> List[str]:
        return self._retriever.retrieve(claim, top_k=top_k, rerank=rerank)

    def retrieve_with_scores(self, claim: str, top_k: int = 3, rerank: bool = True) -> List[Tuple[str, float]]:
        return self._retriever.retrieve_with_scores(claim, top_k=top_k, rerank=rerank)

__all__ = ["Retriever", "chunk_document"]

if __name__ == "__main__":
    sources = [
        "Metformin is a medication used to treat type 2 diabetes. It should be taken with meals to reduce stomach upset.",
        "Common side effects of metformin include nausea, diarrhea, and stomach pain. Lactic acidosis is a rare but serious side effect, occurring in approximately 1 in 30,000 patient-years.",
        "Metformin can be safely combined with insulin therapy under medical supervision. This combination is commonly prescribed for patients with poorly controlled diabetes.",
        "Patients with kidney problems should not take metformin due to increased risk of lactic acidosis. Regular kidney function tests are required.",
    ]

    retriever = Retriever(sources)

    test_claims = [
        "Lactic acidosis occurs in 10% of patients",
        "Metformin should be taken on an empty stomach",
        "Metformin can be combined with insulin therapy",
    ]

    for claim in test_claims:
        print(f"\nClaim: {claim}")
        results = retriever.retrieve_with_scores(claim, top_k=2)
        for passage, score in results:
            print(f"  Score {score:.3f}: {passage[:100]}...")

