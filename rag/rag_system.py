from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def tokenize(text: str) -> Counter[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return Counter(tokens)


def cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a).intersection(b)
    dot = sum(a[token] * b[token] for token in common)
    a_norm = math.sqrt(sum(v * v for v in a.values()))
    b_norm = math.sqrt(sum(v * v for v in b.values()))
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return dot / (a_norm * b_norm)


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        part = words[start : start + chunk_size]
        if not part:
            continue
        chunks.append(" ".join(part))
        if start + chunk_size >= len(words):
            break
    return chunks


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    chunk_id: int
    text: str
    vector: Counter[str]


class SimpleRAGAdvisor:
    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []

    def index_directory(
        self,
        docs_dir: Path,
        *,
        allowed_extensions: tuple[str, ...] = (".md", ".txt"),
        chunk_size: int = 120,
        overlap: int = 20,
    ) -> None:
        if not docs_dir.exists() or not docs_dir.is_dir():
            raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

        chunks: list[DocumentChunk] = []
        for file_path in sorted(docs_dir.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in allowed_extensions:
                continue

            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for i, text in enumerate(chunk_text(content, chunk_size=chunk_size, overlap=overlap)):
                chunks.append(
                    DocumentChunk(
                        source=str(file_path),
                        chunk_id=i,
                        text=text,
                        vector=tokenize(text),
                    )
                )

        if not chunks:
            raise ValueError("No indexable .md/.txt documents were found.")

        self._chunks = chunks

    def retrieve(self, query: str, *, top_k: int = 3) -> list[tuple[float, DocumentChunk]]:
        if not self._chunks:
            raise RuntimeError("Knowledge base is empty. Run index_directory first.")

        query_vector = tokenize(query)
        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in self._chunks:
            score = cosine_similarity(query_vector, chunk.vector)
            if score > 0.0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def answer(self, query: str, *, top_k: int = 3) -> dict[str, object]:
        top_chunks = self.retrieve(query, top_k=top_k)
        if not top_chunks:
            return {
                "answer": "No relevant guidance found in the indexed documents.",
                "sources": [],
            }

        snippets = []
        sources = []
        for score, chunk in top_chunks:
            snippets.append(f"- ({score:.3f}) {chunk.text}")
            sources.append({"source": chunk.source, "chunk_id": chunk.chunk_id, "score": score})

        response = (
            "Retrieved guidance from indexed mission documents:\n"
            + "\n".join(snippets)
        )
        return {"answer": response, "sources": sources}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a local lightweight RAG advisor.")
    parser.add_argument("--docs", type=str, default="docs")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    advisor = SimpleRAGAdvisor()
    advisor.index_directory(Path(args.docs))
    result = advisor.answer(args.query, top_k=args.top_k)
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source['source']} (chunk {source['chunk_id']}, score={source['score']:.3f})")


if __name__ == "__main__":
    main()

