"""
Online Feature Computation — Semantic Search Path

Computes features for real-time semantic search inference:
  1. Encode query text → dense vector using a bi-encoder
  2. Search Qdrant for nearest document chunks
  3. Return ranked results with similarity scores

This module is integrate-able with the Paperless-ngx search endpoint.
"""

import os
import io
import json
import logging
import uuid
from datetime import datetime, timezone

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "document_chunks"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.4  # below this, fall back to keyword search


class RetrievalFeaturePipeline:
    """
    Online feature computation for semantic search.
    Encodes queries and retrieves matching document chunks from Qdrant.
    """

    def __init__(self):
        log.info(f"Loading encoder model: {MODEL_NAME}")
        self.encoder = SentenceTransformer(MODEL_NAME)
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._ensure_collection()

    def _ensure_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            log.info(f"Created Qdrant collection: {COLLECTION_NAME}")
        else:
            log.info(f"Qdrant collection exists: {COLLECTION_NAME}")

    def index_document(self, document_id: str, merged_text: str, chunk_size: int = 256, stride: int = 64):
        """
        Index a document's text into Qdrant.
        Splits text into overlapping chunks, encodes each, and upserts.
        
        This would be called by the document indexing service after upload.
        """
        words = merged_text.split()
        chunks = []
        for i in range(0, len(words), stride):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 10:  # skip tiny trailing chunks
                continue
            chunks.append(" ".join(chunk_words))

        if not chunks:
            log.warning(f"No chunks generated for document {document_id}")
            return

        log.info(f"Encoding {len(chunks)} chunks for document {document_id}")
        embeddings = self.encoder.encode(chunks, show_progress_bar=False)

        points = []
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}:{idx}"))
            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "document_id": document_id,
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    "is_deleted": False,
                },
            ))

        self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        log.info(f"Indexed {len(points)} chunks for document {document_id}")

    def compute_query_features(self, query_text: str) -> dict:
        """
        Online feature computation for a search query.
        
        Returns the full feature dict that would be passed to the ranking model.
        """
        import time
        start = time.time()

        # Step 1: Encode query → dense vector
        query_vector = self.encoder.encode(query_text).tolist()

        # Step 2: Search Qdrant for nearest chunks
        results = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="is_deleted", match=MatchValue(value=False))]
            ),
            limit=TOP_K * 2,  # fetch extra to allow dedup
        )

        # Step 3: Deduplicate to document level (keep best chunk per doc)
        seen_docs = {}
        for hit in results:
            doc_id = hit.payload["document_id"]
            if doc_id not in seen_docs or hit.score > seen_docs[doc_id]["similarity_score"]:
                seen_docs[doc_id] = {
                    "document_id": doc_id,
                    "chunk_index": hit.payload["chunk_index"],
                    "chunk_text": hit.payload["chunk_text"],
                    "similarity_score": round(hit.score, 4),
                }

        # Step 4: Sort by score, take top-k
        ranked_results = sorted(seen_docs.values(), key=lambda x: x["similarity_score"], reverse=True)[:TOP_K]

        # Step 5: Determine if we should fall back to keyword search
        max_score = ranked_results[0]["similarity_score"] if ranked_results else 0.0
        fallback_to_keyword = max_score < SIMILARITY_THRESHOLD

        elapsed_ms = round((time.time() - start) * 1000, 1)

        feature_output = {
            "query_text": query_text,
            "query_vector_dim": len(query_vector),
            "query_vector_sample": query_vector[:5],  # first 5 dims for inspection
            "results": ranked_results,
            "fallback_to_keyword": fallback_to_keyword,
            "max_similarity": max_score,
            "inference_time_ms": elapsed_ms,
        }

        return feature_output


def demo():
    """Run an end-to-end demo of the retrieval feature pipeline."""
    pipeline = RetrievalFeaturePipeline()

    # Step 1: Index some sample documents
    sample_docs = [
        {
            "id": "doc-001",
            "text": (
                "The fiscal year 2024 budget allocates 2.3 million dollars to laboratory "
                "equipment upgrades across three departments. The College of Engineering "
                "will receive the largest share for updating the robotics lab and the "
                "materials science testing facility. The remaining funds are split between "
                "the Chemistry department for new spectrometers and the Biology department "
                "for microscopy equipment. All purchases must be completed by June 30."
            ),
        },
        {
            "id": "doc-002",
            "text": (
                "Meeting minutes from the faculty senate session held on March 15 2025. "
                "The committee discussed the proposed changes to the tenure review process. "
                "Professor Williams presented the new evaluation criteria which emphasize "
                "both research output and teaching effectiveness. The motion was tabled "
                "for further discussion at the next meeting scheduled for April 12."
            ),
        },
        {
            "id": "doc-003",
            "text": (
                "Lab safety protocol update effective January 2025. All personnel working "
                "with hazardous chemicals must complete the updated online training module "
                "before accessing the laboratory. Emergency eyewash stations have been "
                "installed in rooms 204 and 310. New chemical waste disposal procedures "
                "require dual sign-off from the lab manager and safety officer."
            ),
        },
        {
            "id": "doc-004",
            "text": (
                "Travel reimbursement request form for the International Conference on "
                "Machine Learning held in Vancouver. Flight cost was 450 dollars and hotel "
                "was 180 dollars per night for three nights. The conference registration fee "
                "was 600 dollars. Please attach all receipts and the conference acceptance "
                "letter for the paper titled Efficient Transformers for Document Understanding."
            ),
        },
        {
            "id": "doc-005",
            "text": (
                "Student enrollment data for Fall 2024 semester. Total undergraduate "
                "enrollment increased by 3.2 percent compared to Fall 2023. The Computer "
                "Science program saw the largest growth at 12 percent. Graduate enrollment "
                "remained flat with a slight decline in the MBA program offset by growth "
                "in the Data Science masters program."
            ),
        },
    ]

    log.info("=" * 60)
    log.info("STEP 1: Indexing sample documents into Qdrant")
    log.info("=" * 60)
    for doc in sample_docs:
        pipeline.index_document(doc["id"], doc["text"])

    # Step 2: Run sample queries
    test_queries = [
        "budget report 2024",
        "meeting minutes tenure review",
        "lab safety chemical waste",
        "conference travel reimbursement",
        "student enrollment statistics",
    ]

    log.info("")
    log.info("=" * 60)
    log.info("STEP 2: Computing features for search queries")
    log.info("=" * 60)

    for query in test_queries:
        log.info(f"\n{'─' * 40}")
        log.info(f"Query: '{query}'")
        log.info(f"{'─' * 40}")

        features = pipeline.compute_query_features(query)

        log.info(f"  Vector dimension: {features['query_vector_dim']}")
        log.info(f"  Inference time: {features['inference_time_ms']}ms")
        log.info(f"  Max similarity: {features['max_similarity']}")
        log.info(f"  Fallback to keyword: {features['fallback_to_keyword']}")
        log.info(f"  Results:")
        for r in features["results"]:
            log.info(f"    [{r['similarity_score']:.4f}] doc={r['document_id']} "
                     f"chunk={r['chunk_index']} "
                     f"text='{r['chunk_text'][:60]}...'")

        # Print full JSON for the first query
        if query == test_queries[0]:
            log.info(f"\n  Full feature output (JSON):")
            print(json.dumps(features, indent=2))


if __name__ == "__main__":
    demo()
