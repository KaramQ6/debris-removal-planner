"""Evaluate the RAG system and produce verifiable results for the paper.

Runs the exact queries from Table IV and a 30-query benchmark,
saving all output to results/rag_evaluation.json.
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.rag_system import SimpleRAGAdvisor

# Initialize the advisor with the actual docs/ directory (default KB)
advisor = SimpleRAGAdvisor()
docs_dir = Path("docs")
advisor.index_directory(docs_dir)
print(f"Indexed {advisor.chunk_count} chunks from '{docs_dir}/'")

# === TABLE IV VERIFICATION ===
table_iv_queries = [
    "LEO disposal timeline",
    "Conjunction fuel protocol",
    "High-risk debris prioritization",
]

table_iv_results = []
for q in table_iv_queries:
    start = time.perf_counter()
    result = advisor.answer(q, top_k=3)
    latency = (time.perf_counter() - start) * 1000
    
    # Extract score from the retrieve method
    retrieved = advisor.retrieve(q, top_k=3)
    scores = [round(s, 2) for s, _ in retrieved]
    sources = [c.source.split("\\")[-1].split("/")[-1] for _, c in retrieved]
    
    table_iv_results.append({
        "query": q,
        "top_source": sources[0] if sources else "N/A",
        "top_score": scores[0] if scores else None,
        "all_scores": scores,
        "all_sources": sources,
        "answer_snippet": result["answer"][:300],
        "latency_ms": round(latency, 2),
    })
    print(f"\n  Query: '{q}'")
    print(f"    Source: {sources[0] if sources else 'N/A'}")
    print(f"    Score:  {scores[0] if scores else 'N/A'}")
    print(f"    Latency: {latency:.2f}ms")

# === 30-QUERY BENCHMARK ===
benchmark_queries = [
    "What is the 25-year disposal rule for LEO satellites?",
    "How should spent rocket bodies be disposed?",
    "What are post-mission disposal requirements?",
    "When must a satellite complete de-orbit after end-of-life?",
    "What altitude defines the LEO protected zone?",
    "What are the requirements for passivation of a spacecraft?",
    "How to safely dispose of a GEO satellite?",
    "What is the graveyard orbit disposal strategy?",
    "What fuel reserves must be allocated for disposal?",
    "What are the collision probability thresholds?",
    "How should a spacecraft perform a collision avoidance maneuver?",
    "What distance defines a close approach?",
    "When should operators be notified of a conjunction event?",
    "What tracking accuracy is required for conjunction assessment?",
    "How to calculate collision probability between two objects?",
    "What mitigation measures exist for solar activity effects?",
    "What are the IADC guidelines for debris mitigation?",
    "What tracking networks monitor space debris?",
    "How does atmospheric drag affect debris lifetime?",
    "What shielding is recommended against small debris impacts?",
    "How should tethered systems be designed to minimize debris?",
    "What are the break-up criteria for defunct spacecraft?",
    "What explosive device safety margins are required?",
    "How should propellant tanks be vented at end-of-life?",
    "What materials should be avoided to reduce debris creation?",
    "What battery safety procedures are required for end-of-life?",
    "What are the requirements for on-orbit servicing vehicles?",
    "How should large constellation operators manage debris risk?",
    "What is the acceptable probability of casualty from re-entry?",
    "What are the NASA requirements for orbital debris limitation?",
]

latencies = []
all_results = []
for q in benchmark_queries:
    start = time.perf_counter()
    retrieved = advisor.retrieve(q, top_k=3)
    latency = (time.perf_counter() - start) * 1000
    latencies.append(latency)
    
    scores = [round(s, 2) for s, _ in retrieved]
    sources = [c.source.split("\\")[-1].split("/")[-1] for _, c in retrieved]
    snippets = [c.text[:100] for _, c in retrieved]
    
    all_results.append({
        "query": q,
        "top_score": scores[0] if scores else 0,
        "top_source": sources[0] if sources else "N/A",
        "scores": scores,
        "sources": sources,
        "snippet": snippets[0] if snippets else "",
        "latency_ms": round(latency, 2),
    })

avg_latency = sum(latencies) / len(latencies)
n = len(benchmark_queries)

print(f"\n=== RAG BENCHMARK RESULTS ({n} queries) ===")
print(f"  Avg Latency: {avg_latency:.2f} ms")
print(f"  Min Score: {min(r['top_score'] for r in all_results):.2f}")
print(f"  Max Score: {max(r['top_score'] for r in all_results):.2f}")
print(f"  Avg Score: {sum(r['top_score'] for r in all_results)/n:.2f}")

# Save
output = {
    "table_iv_verification": table_iv_results,
    "benchmark_summary": {
        "n_queries": n,
        "avg_latency_ms": round(avg_latency, 2),
        "avg_top_score": round(sum(r['top_score'] for r in all_results)/n, 2),
    },
    "per_query_results": all_results,
}

output_path = Path("results/rag_evaluation.json")
output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"\nResults saved to {output_path}")
