"""Lightweight RAG advisory system for orbital debris mission planning.

Zero external dependencies beyond Python stdlib — uses TF-IDF-like
bag-of-words with BM25 scoring for document retrieval.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was were will with".split()
)


def tokenize(text: str, *, remove_stops: bool = True) -> list[str]:
    """Lowercase alphanumeric tokenization with optional stop-word removal."""
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    if remove_stops:
        tokens = [t for t in tokens if t not in _STOP_WORDS]
    return tokens


def token_counter(text: str) -> Counter[str]:
    return Counter(tokenize(text))


# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------

def cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    a_norm = math.sqrt(sum(v * v for v in a.values()))
    b_norm = math.sqrt(sum(v * v for v in b.values()))
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return dot / (a_norm * b_norm)


def bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_doc_len: float,
    doc_freq: dict[str, int],
    total_docs: int,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Okapi BM25 relevance score."""
    score = 0.0
    doc_counter = Counter(doc_tokens)
    doc_len = len(doc_tokens)

    for term in set(query_tokens):
        if term not in doc_counter:
            continue
        tf = doc_counter[term]
        df = doc_freq.get(term, 0)
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        score += idf * tf_norm
    return score


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 150,
    overlap: int = 30,
) -> list[str]:
    """Split *text* into overlapping word-level chunks."""
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        part = words[start : start + chunk_size]
        if not part:
            continue
        chunks.append(" ".join(part))
        if start + chunk_size >= len(words):
            break
    return chunks


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DocumentChunk:
    source: str
    chunk_id: int
    text: str
    tokens: list[str]
    vector: Counter[str]


# ---------------------------------------------------------------------------
# RAG Advisor
# ---------------------------------------------------------------------------

class SimpleRAGAdvisor:
    """Retrieval-augmented advisory over local mission documents.

    Uses BM25 as the primary ranker with cosine similarity as a fallback
    tie-breaker.  No external ML dependencies required.
    """

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._doc_freq: dict[str, int] = {}
        self._avg_doc_len: float = 0.0

    def index_directory(
        self,
        docs_dir: Path,
        *,
        allowed_extensions: tuple[str, ...] = (".md", ".txt"),
        chunk_size: int = 150,
        overlap: int = 30,
    ) -> None:
        """Read and index all documents in *docs_dir*."""
        if not docs_dir.exists() or not docs_dir.is_dir():
            raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

        chunks: list[DocumentChunk] = []
        for file_path in sorted(docs_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in allowed_extensions:
                continue
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for i, text in enumerate(
                chunk_text(content, chunk_size=chunk_size, overlap=overlap)
            ):
                tokens = tokenize(text)
                chunks.append(
                    DocumentChunk(
                        source=str(file_path),
                        chunk_id=i,
                        text=text,
                        tokens=tokens,
                        vector=Counter(tokens),
                    )
                )

        if not chunks:
            raise ValueError("No indexable documents found.")

        self._chunks = chunks

        # Pre-compute BM25 statistics
        self._avg_doc_len = sum(len(c.tokens) for c in chunks) / len(chunks)
        df: dict[str, int] = {}
        for c in chunks:
            for term in set(c.tokens):
                df[term] = df.get(term, 0) + 1
        self._doc_freq = df

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def retrieve(
        self, query: str, *, top_k: int = 3
    ) -> list[tuple[float, DocumentChunk]]:
        """Return the top-*k* most relevant chunks for *query*."""
        if not self._chunks:
            raise RuntimeError("Knowledge base is empty. Run index_directory first.")

        query_tokens = tokenize(query)
        query_vec = Counter(query_tokens)

        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in self._chunks:
            # Primary: BM25
            bm = bm25_score(
                query_tokens,
                chunk.tokens,
                self._avg_doc_len,
                self._doc_freq,
                len(self._chunks),
            )
            # Secondary: cosine (0-1) as tie-breaker
            cos = cosine_similarity(query_vec, chunk.vector)
            combined = bm + 0.1 * cos  # BM25 dominates
            if combined > 0.0:
                scored.append((combined, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def answer(self, query: str, *, top_k: int = 3) -> dict[str, object]:
        """Retrieve relevant chunks and format an advisory response."""
        top_chunks = self.retrieve(query, top_k=top_k)
        if not top_chunks:
            return {
                "answer": "No relevant guidance found in the indexed documents.",
                "sources": [],
                "query": query,
            }

        snippets: list[str] = []
        sources: list[dict] = []
        for rank, (score, chunk) in enumerate(top_chunks, 1):
            source_name = Path(chunk.source).name
            snippets.append(
                f"[{rank}] (relevance: {score:.3f}, source: {source_name})\n"
                f"    {chunk.text[:300]}{'...' if len(chunk.text) > 300 else ''}"
            )
            sources.append({
                "source": chunk.source,
                "source_name": source_name,
                "chunk_id": chunk.chunk_id,
                "score": round(score, 4),
            })

        response = (
            f"Advisory response for: \"{query}\"\n"
            f"{'=' * 60}\n"
            f"Retrieved {len(top_chunks)} relevant passages from mission documents:\n\n"
            + "\n\n".join(snippets)
            + f"\n\n{'=' * 60}\n"
            f"Note: This advisory is based on indexed public-domain NASA/ESA "
            f"documentation excerpts. Always verify with official sources before "
            f"operational use."
        )
        return {"answer": response, "sources": sources, "query": query}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_EXAMPLE_QUERIES = [
    "What should we do when delta-v budget is low?",
    "What are the debris mitigation guidelines for end-of-life operations?",
    "How to prioritize high-risk debris objects during removal missions?",
    "What fuel conservation protocols apply during conjunction windows?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the debris-removal RAG advisory system."
    )
    parser.add_argument("--docs", type=str, default="docs")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run pre-built example queries for demonstration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    advisor = SimpleRAGAdvisor()
    advisor.index_directory(Path(args.docs))
    print(f"Indexed {advisor.chunk_count} chunks from '{args.docs}/'.\n")

    queries = _EXAMPLE_QUERIES if args.demo else [args.query]
    if not args.demo and not args.query:
        print("Provide --query or --demo. Example:")
        print(f'  python -m rag.rag_system --query "{_EXAMPLE_QUERIES[0]}"')
        return

    for q in queries:
        result = advisor.answer(q, top_k=args.top_k)
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()
