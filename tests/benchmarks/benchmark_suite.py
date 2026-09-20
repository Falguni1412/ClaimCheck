"""
Benchmark suite for ClaimCheck.
Measures latency, throughput, and accuracy under various conditions.
"""
import time
import asyncio
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_per_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def measure_latency(fn, *args, **kwargs) -> float:
    """Measure execution time in milliseconds."""
    start = time.perf_counter()
    fn(*args, **kwargs)
    return (time.perf_counter() - start) * 1000


async def measure_latency_async(fn, *args, **kwargs) -> float:
    """Measure async execution time in milliseconds."""
    start = time.perf_counter()
    await fn(*args, **kwargs)
    return (time.perf_counter() - start) * 1000


def compute_stats(times_ms: List[float]) -> Dict[str, float]:
    """Compute summary statistics from a list of latencies."""
    if not times_ms:
        return {}
    sorted_times = sorted(times_ms)
    return {
        "mean": statistics.mean(sorted_times),
        "median": statistics.median(sorted_times),
        "p95": sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 1 else sorted_times[0],
        "p99": sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 1 else sorted_times[0],
        "min": min(sorted_times),
        "max": max(sorted_times),
    }


def run_decomposition_benchmark(iterations: int = 100) -> BenchmarkResult:
    """Benchmark claim decomposition."""
    from api.services.decomposer import decompose_into_claims

    sample_text = (
        "Metformin should be taken on an empty stomach. "
        "It can be combined with insulin therapy. "
        "Lactic acidosis occurs in 10% of patients. "
        "Nausea is a common side effect. "
        "The drug works by reducing glucose production in the liver."
    )

    times = []
    for _ in range(iterations):
        times.append(measure_latency(decompose_into_claims, sample_text))

    stats = compute_stats(times)
    return BenchmarkResult(
        name="decomposition",
        iterations=iterations,
        mean_ms=round(stats["mean"], 2),
        median_ms=round(stats["median"], 2),
        p95_ms=round(stats["p95"], 2),
        p99_ms=round(stats["p99"], 2),
        min_ms=round(stats["min"], 2),
        max_ms=round(stats["max"], 2),
        throughput_per_sec=round(1000 / stats["mean"], 2) if stats["mean"] > 0 else 0,
    )


def run_chunking_benchmark(iterations: int = 100) -> BenchmarkResult:
    """Benchmark document chunking."""
    from api.services.retriever import chunk_document

    sample_doc = " ".join([
        f"This is sentence number {i} in a long medical document."
        for i in range(200)
    ])

    times = []
    for _ in range(iterations):
        times.append(measure_latency(chunk_document, sample_doc, 500, 50))

    stats = compute_stats(times)
    return BenchmarkResult(
        name="chunking",
        iterations=iterations,
        mean_ms=round(stats["mean"], 2),
        median_ms=round(stats["median"], 2),
        p95_ms=round(stats["p95"], 2),
        p99_ms=round(stats["p99"], 2),
        min_ms=round(stats["min"], 2),
        max_ms=round(stats["max"], 2),
        throughput_per_sec=round(1000 / stats["mean"], 2) if stats["mean"] > 0 else 0,
    )


def run_numerical_check_benchmark(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark numerical consistency check."""
    from api.services.numerical import check_numerical_consistency

    claim = "Lactic acidosis occurs in 10% of patients."
    evidence = "Lactic acidosis is rare, occurring in 1 in 30,000 patient-years."

    times = []
    for _ in range(iterations):
        times.append(measure_latency(check_numerical_consistency, claim, evidence))

    stats = compute_stats(times)
    return BenchmarkResult(
        name="numerical_check",
        iterations=iterations,
        mean_ms=round(stats["mean"], 2),
        median_ms=round(stats["median"], 2),
        p95_ms=round(stats["p95"], 2),
        p99_ms=round(stats["p99"], 2),
        min_ms=round(stats["min"], 2),
        max_ms=round(stats["max"], 2),
        throughput_per_sec=round(1000 / stats["mean"], 2) if stats["mean"] > 0 else 0,
    )


def print_results(results: List[BenchmarkResult]) -> None:
    """Pretty-print benchmark results."""
    print("\n" + "=" * 80)
    print("  ClaimCheck Benchmark Results")
    print("=" * 80)
    print(f"{'Name':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'Throughput':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r.name:<20} {r.mean_ms:<12.2f} {r.p95_ms:<12.2f} {r.p99_ms:<12.2f} {r.throughput_per_sec:<15.1f}/s")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("Running ClaimCheck benchmarks...")

    results = [
        run_decomposition_benchmark(),
        run_chunking_benchmark(),
        run_numerical_check_benchmark(),
    ]
    print_results(results)

