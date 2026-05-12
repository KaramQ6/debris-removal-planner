"""Tests for the BM25-based RAG advisor."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rag.rag_system import (
    SimpleRAGAdvisor,
    bm25_score,
    chunk_text,
    cosine_similarity,
    token_counter,
    tokenize,
)


class TestTokenization:
    def test_lowercases(self):
        assert tokenize("Orbital DEBRIS") == ["orbital", "debris"]

    def test_strips_stop_words(self):
        assert "the" not in tokenize("the orbital debris field")

    def test_keep_stop_words_when_requested(self):
        assert "the" in tokenize("the orbital debris", remove_stops=False)

    def test_alphanumeric_only(self):
        assert tokenize("delta-v: 1500 m/s!") == ["delta", "v", "1500", "m", "s"]


class TestChunking:
    def test_returns_chunks_with_overlap(self):
        words = " ".join(str(i) for i in range(100))
        chunks = chunk_text(words, chunk_size=30, overlap=10)
        assert len(chunks) >= 3
        # First chunk should start at word "0".
        assert chunks[0].startswith("0 ")

    def test_overlap_validation(self):
        with pytest.raises(ValueError):
            chunk_text("a b c", chunk_size=5, overlap=10)

    def test_empty_text(self):
        assert chunk_text("") == []


class TestSimilarity:
    def test_cosine_identical(self):
        a = token_counter("orbital debris removal")
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_disjoint(self):
        a = token_counter("orbital debris")
        b = token_counter("ground station")
        assert cosine_similarity(a, b) == 0.0

    def test_bm25_score_higher_for_relevant_doc(self):
        query = tokenize("debris mitigation")
        relevant = tokenize("debris mitigation guidelines for spacecraft")
        irrelevant = tokenize("antenna pointing accuracy table")
        doc_freq = {"debris": 1, "mitigation": 1, "guidelines": 1,
                    "spacecraft": 1, "antenna": 1, "pointing": 1,
                    "accuracy": 1, "table": 1}
        rel_score = bm25_score(query, relevant, 5.0, doc_freq, 2)
        irr_score = bm25_score(query, irrelevant, 5.0, doc_freq, 2)
        assert rel_score > irr_score


class TestAdvisor:
    @pytest.fixture
    def populated_advisor(self, tmp_path: Path):
        (tmp_path / "deorbit.md").write_text(textwrap.dedent("""
            # End-of-life deorbit guidelines
            Spacecraft in LEO must perform a controlled deorbit within
            25 years of mission completion. Disposal orbit must guarantee
            atmospheric re-entry to limit debris accumulation.
        """).strip())
        (tmp_path / "conjunction.md").write_text(textwrap.dedent("""
            # Conjunction avoidance
            During predicted close approaches, operators should plan a
            collision avoidance maneuver if the probability exceeds 1e-4.
            Maintain fuel reserve for at least three such maneuvers per year.
        """).strip())
        advisor = SimpleRAGAdvisor()
        advisor.index_directory(tmp_path)
        return advisor

    def test_indexes_chunks(self, populated_advisor):
        assert populated_advisor.chunk_count >= 2

    def test_retrieves_relevant_chunk(self, populated_advisor):
        results = populated_advisor.retrieve(
            "What is the deorbit deadline for LEO spacecraft?", top_k=1
        )
        assert len(results) == 1
        _, chunk = results[0]
        assert "deorbit" in chunk.text.lower() or "25 years" in chunk.text

    def test_retrieves_different_chunks_for_different_queries(self, populated_advisor):
        deorbit = populated_advisor.retrieve("deorbit disposal LEO", top_k=1)
        conjunction = populated_advisor.retrieve("conjunction collision avoidance", top_k=1)
        assert deorbit[0][1].text != conjunction[0][1].text

    def test_answer_includes_sources(self, populated_advisor):
        response = populated_advisor.answer("conjunction avoidance threshold", top_k=2)
        assert response["sources"]
        assert response["query"] == "conjunction avoidance threshold"

    def test_empty_advisor_raises(self):
        with pytest.raises(RuntimeError):
            SimpleRAGAdvisor().retrieve("anything")
