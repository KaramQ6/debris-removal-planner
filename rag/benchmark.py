"""Formal retrieval evaluation benchmark for the RAG advisory system.

Evaluates Precision@1, Precision@3, Mean Reciprocal Rank (MRR), and Latency
over 30 standard operational debris guidelines queries.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rag.rag_system import SimpleRAGAdvisor

# Define the 30 structured queries and their expected ground-truth sources
BENCHMARK_SUITE = [
    {
        "query": "What is the recommended disposal timeline for Low Earth Orbit?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "What are the deorbit rules for LEO?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "How long can debris remain in orbit after EOL?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "What is the 25-year deorbit guideline?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "What should be done with on-board stored energy at end-of-life?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "How to prevent on-orbit spacecraft break-ups?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "What is the depletion timeline for stored energy?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "Why should we avoid intentional destruction of spacecraft?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "What disposal actions apply to the Geosynchronous region?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "How should a GEO spacecraft be deorbited?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "What is the collision probability threshold for scheduling avoidance maneuvers?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "How many collision avoidance maneuvers must be budgeted per year?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "What are the conjunction screening rules for spacecraft?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "What is the fuel reservation standard for yearly collision maneuvers?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "Which orbital debris objects are prioritized for active removal?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "What fuel margins should a debris removal mission budget?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "What is the fuel constraint protocol when reserve drops below 15%?",
        "expected_source": "iadc_guidelines_excerpt.md",
    },
    {
        "query": "How to optimize the target sequence to minimize total delta-v?",
        "expected_source": "debris_mitigation_best_practices.md",
    },
    {
        "query": "What are fuel conservation protocols during spacecraft transfers?",
        "expected_source": "debris_mitigation_best_practices.md",
    },
    {
        "query": "How does fuel margin management shift below 20%?",
        "expected_source": "debris_mitigation_best_practices.md",
    },
    {
        "query": "When should we abort the debris removal mission due to low fuel?",
        "expected_source": "debris_mitigation_best_practices.md",
    },
    {
        "query": "What are the average transfer and de-orbit delta-v costs in LEO?",
        "expected_source": "debris_mitigation_best_practices.md",
    },
    {
        "query": "How to calculate the debris risk priority score?",
        "expected_source": "debris_mitigation_best_practices.md",
    },
    {
        "query": "What are high-risk orbital regions for space debris?",
        "expected_source": "debris_mitigation_best_practices.md",
    },
    {
        "query": "What is the ISS orbital inclination and altitude congestion risk?",
        "expected_source": "debris_mitigation_best_practices.md",
    },
    {
        "query": "How to perform pre-maneuver TLE screening?",
        "expected_source": "debris_mitigation_best_practices.md",
    },
    {
        "query": "What are the protocols during dense conjunction windows and solar storms?",
        "expected_source": "debris_mitigation_best_practices.md",
    },
    {
        "query": "What is the passivation procedure for removal spacecraft at EOL?",
        "expected_source": "debris_mitigation_best_practices.md",
    },
    {
        "query": "What documentation is required for debris removal missions?",
        "expected_source": "debris_mitigation_best_practices.md",
    },
    {
        "query": "How do NASA and ESA emphasize mission planning and fragmentation risk?",
        "expected_source": "nasa_esa_reference_excerpt.txt",
    },
]


def run_benchmark(docs_dir: str = "docs") -> dict[str, float | list[dict[str, object]]]:
    advisor = SimpleRAGAdvisor()
    advisor.index_directory(Path(docs_dir))

    p1_hits = 0
    p3_hits = 0
    rr_sum = 0.0
    latencies = []
    detailed_results = []

    for entry in BENCHMARK_SUITE:
        query = entry["query"]
        expected = entry["expected_source"]

        start_time = time.perf_counter()
        results = advisor.retrieve(query, top_k=3)
        latency = (time.perf_counter() - start_time) * 1000.0  # in ms
        latencies.append(latency)

        p1_hit = 0
        p3_hit = 0
        reciprocal_rank = 0.0
        retrieved_sources = []

        for rank, (_, chunk) in enumerate(results, 1):
            source_filename = Path(chunk.source).name
            retrieved_sources.append(
                {"rank": rank, "source": source_filename, "text_snippet": chunk.text[:100]}
            )
            if source_filename == expected:
                p3_hit = 1
                if rank == 1:
                    p1_hit = 1
                    reciprocal_rank = 1.0
                else:
                    reciprocal_rank = 1.0 / rank
                # Once found, stop checking (single ground-truth match per query)
                break

        p1_hits += p1_hit
        p3_hits += p3_hit
        rr_sum += reciprocal_rank

        detailed_results.append(
            {
                "query": query,
                "expected": expected,
                "p1_hit": p1_hit,
                "p3_hit": p3_hit,
                "reciprocal_rank": reciprocal_rank,
                "latency_ms": latency,
                "retrieved": retrieved_sources,
            }
        )

    n_queries = len(BENCHMARK_SUITE)
    p1 = p1_hits / n_queries
    p3 = p3_hits / n_queries
    mrr = rr_sum / n_queries
    avg_latency = sum(latencies) / n_queries

    return {
        "p1": p1,
        "p3": p3,
        "mrr": mrr,
        "avg_latency_ms": avg_latency,
        "detailed": detailed_results,
    }


def main() -> None:
    print("Initializing local RAG Retrieval Benchmark (30 operational queries)...")
    metrics = run_benchmark()

    print("\n" + "=" * 60)
    print(" RAG RETRIEVAL BENCHMARK METRICS")
    print("=" * 60)
    print(f"Total Queries Evaluated : {len(BENCHMARK_SUITE)}")
    print(f"Precision@1 (P@1)       : {metrics['p1'] * 100.0:.1f}%")
    print(f"Precision@3 (P@3)       : {metrics['p3'] * 100.0:.1f}%")
    print(f"Mean Reciprocal Rank    : {metrics['mrr']:.3f}")
    print(f"Average CPU Latency     : {metrics['avg_latency_ms']:.2f} ms")
    print("=" * 60)

    # Save to file
    output_file = Path("results/rag_benchmark.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "summary": {
            "total_queries": len(BENCHMARK_SUITE),
            "precision_at_1": metrics["p1"],
            "precision_at_3": metrics["p3"],
            "mrr": metrics["mrr"],
            "avg_latency_ms": metrics["avg_latency_ms"],
        },
        "queries": metrics["detailed"],
    }
    
    output_file.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"Detailed RAG benchmark results saved to: {output_file}")


if __name__ == "__main__":
    main()
