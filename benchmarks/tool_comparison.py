"""Benchmark: giddyanne vs grepai search quality

Runs both tools against the same queries with known ground-truth expected files.
Measures search quality (precision/recall) and latency.

Usage:
    .venv/bin/python benchmarks/tool_comparison.py [config.yaml]

Config defaults to benchmarks/configs/giddyanne.yaml if no arg given.
Requires: giddy server running, grepai watch running (or indexed).
"""

import json
import os
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

GIDDYANNE_ROOT = Path(__file__).resolve().parent.parent

TOOLS = {
    "giddyanne": {
        "search_cmd": ["giddy", "find"],
        "search_flags": ["--json", "--limit", "5"],
    },
    "grepai": {
        "search_cmd": ["grepai", "search"],
        "search_flags": ["--json", "--limit", "5"],
    },
}

WARM_RUNS = 3


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    project_root = Path(raw["project_root"])
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()

    queries = [entry["query"] for entry in raw["queries"]]
    expected_files = {entry["query"]: entry["expected"] for entry in raw["queries"]}

    return {
        "project_root": project_root,
        "queries": queries,
        "expected_files": expected_files,
    }


def run_search(tool: str, query: str, cwd: Path) -> tuple[list[dict], float]:
    """Run a search and return (results, elapsed_seconds)."""
    cfg = TOOLS[tool]
    cmd = cfg["search_cmd"] + [query] + cfg["search_flags"]

    start = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30, cwd=cwd
    )
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"  WARN: {tool} search failed: {result.stderr.strip()}", file=sys.stderr)
        return [], elapsed

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  WARN: {tool} returned invalid JSON", file=sys.stderr)
        return [], elapsed

    # Normalize result format
    results = []
    for r in data[:5]:
        path = r.get("path") or r.get("file_path", "")
        score = r.get("score", 0)
        results.append({"path": path, "score": score})

    return results, elapsed


def score_results(results: list[dict], expected: list[str], project_root: Path) -> dict:
    """Score search results against expected files."""
    result_paths = [r["path"] for r in results]

    found = 0
    found_files = []
    missed_files = []
    for exp in expected:
        if any(rp.endswith(exp) for rp in result_paths):
            found += 1
            found_files.append(exp)
        else:
            missed_files.append(exp)

    # Reciprocal rank: rank of first expected file found
    rr = 0.0
    for i, rp in enumerate(result_paths):
        if any(rp.endswith(exp) for exp in expected):
            rr = 1.0 / (i + 1)
            break

    recall = found / len(expected) if expected else 0
    # Precision: how many of top-k are expected
    relevant_in_results = sum(
        1 for rp in result_paths if any(rp.endswith(exp) for exp in expected)
    )
    precision = relevant_in_results / len(result_paths) if result_paths else 0

    return {
        "recall": recall,
        "precision": precision,
        "reciprocal_rank": rr,
        "found": found_files,
        "missed": missed_files,
        "top_5": [
            os.path.relpath(rp, project_root)
            if rp.startswith(str(project_root)) else rp
            for rp in result_paths
        ],
    }


def benchmark_tool(tool: str, queries: list[str], expected_files: dict, project_root: Path) -> dict:
    """Run all queries against a tool, return quality and timing data."""
    print(f"\n  {tool}")
    print(f"  {'=' * len(tool)}")

    query_results = {}
    for query in queries:
        # Cold search
        results, cold_time = run_search(tool, query, project_root)
        quality = score_results(results, expected_files[query], project_root)

        # Warm searches
        warm_times = []
        for _ in range(WARM_RUNS):
            _, t = run_search(tool, query, project_root)
            warm_times.append(t)

        query_results[query] = {
            "quality": quality,
            "cold_time": cold_time,
            "warm_avg": statistics.mean(warm_times),
            "warm_times": warm_times,
        }

        status = "HIT" if quality["recall"] > 0 else "MISS"
        print(f"    {status} ({quality['recall']:.0%}) {query[:50]}  [{cold_time:.3f}s]")

    # Aggregate
    recalls = [qr["quality"]["recall"] for qr in query_results.values()]
    precisions = [qr["quality"]["precision"] for qr in query_results.values()]
    rrs = [qr["quality"]["reciprocal_rank"] for qr in query_results.values()]
    cold_times = [qr["cold_time"] for qr in query_results.values()]
    warm_avgs = [qr["warm_avg"] for qr in query_results.values()]

    summary = {
        "avg_recall": statistics.mean(recalls),
        "avg_precision": statistics.mean(precisions),
        "mrr": statistics.mean(rrs),
        "avg_cold_time": statistics.mean(cold_times),
        "avg_warm_time": statistics.mean(warm_avgs),
    }

    print("    ---")
    r = summary['avg_recall']
    p = summary['avg_precision']
    mrr = summary['mrr']
    print(f"    Recall: {r:.0%}  Precision: {p:.0%}  MRR: {mrr:.2f}")
    cold = summary['avg_cold_time']
    warm = summary['avg_warm_time']
    print(f"    Latency: {cold:.3f}s cold, {warm:.3f}s warm")

    return {"queries": query_results, "summary": summary}


def print_comparison(report: dict):
    """Print side-by-side comparison."""
    tools = list(report["tools"].keys())
    queries = report["queries"]

    print("\n" + "=" * 70)
    print("TOOL COMPARISON RESULTS")
    print(f"Project: {report['project']}")
    print("=" * 70)

    col = 25

    def row(label, *values):
        print(f"  {label:<25}", end="")
        for v in values:
            print(f"  {str(v):>{col}}", end="")
        print()

    def sep():
        print("  " + "-" * (25 + (col + 2) * len(tools)))

    print()
    row("", *tools)
    sep()
    row("Avg Recall", *[f"{report['tools'][t]['summary']['avg_recall']:.0%}" for t in tools])
    row("Avg Precision", *[f"{report['tools'][t]['summary']['avg_precision']:.0%}" for t in tools])
    row("MRR", *[f"{report['tools'][t]['summary']['mrr']:.2f}" for t in tools])
    sep()
    s = report['tools']
    row("Avg Cold Latency", *[f"{s[t]['summary']['avg_cold_time']:.3f}s" for t in tools])
    row("Avg Warm Latency", *[f"{s[t]['summary']['avg_warm_time']:.3f}s" for t in tools])
    sep()

    # Per-query detail
    print("\n  PER-QUERY DETAIL")
    print("  " + "-" * 68)
    for query in queries:
        print(f"\n  Query: \"{query}\"")
        for t in tools:
            qr = report["tools"][t]["queries"][query]
            q = qr["quality"]
            print(f"    {t}:")
            rr = q['reciprocal_rank']
            ct = qr['cold_time']
            print(
                f"      Recall: {q['recall']:.0%}  Precision: {q['precision']:.0%}"
                f"  RR: {rr:.2f}  [{ct:.3f}s]"
            )
            for i, p in enumerate(q["top_5"], 1):
                expected = report["expected_files"][query]
                marker = " *" if any(p.endswith(e) for e in expected) else ""
                print(f"      {i}. {p}{marker}")
            if q["missed"]:
                print(f"      Missed: {', '.join(q['missed'])}")


def main():
    default_config = GIDDYANNE_ROOT / "benchmarks" / "configs" / "giddyanne.yaml"
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_config
    config = load_config(config_path)

    project_name = config_path.stem
    project_root = config["project_root"]
    queries = config["queries"]
    expected_files = config["expected_files"]

    print("Tool Comparison Benchmark")
    print(f"Project: {project_root} ({project_name})")
    print(f"Queries: {len(queries)}")

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "project": str(project_root),
        "project_name": project_name,
        "queries": queries,
        "expected_files": expected_files,
        "tools": {},
    }

    for tool in TOOLS:
        report["tools"][tool] = benchmark_tool(
            tool, queries, expected_files, project_root
        )

    print_comparison(report)

    # Write JSON
    data_dir = GIDDYANNE_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = data_dir / f"tool-comparison-{project_name}-{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
