"""Benchmark: search quality across projects and search modes.

Runs giddy find against ground-truth queries from config files.
Measures recall, precision, and MRR per project and mode.

Usage:
    .venv/bin/python benchmarks/quality.py                    # all configs, default mode
    .venv/bin/python benchmarks/quality.py --mode hybrid      # specific mode
    .venv/bin/python benchmarks/quality.py --mode all         # compare all modes
    .venv/bin/python benchmarks/quality.py giddyanne          # single project
    .venv/bin/python benchmarks/quality.py gilder profai      # multiple projects

Requires: giddy server running for each project being benchmarked.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

BENCHMARKS_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = BENCHMARKS_DIR / "configs"

MODES = ["--semantic", "--full-text", "--hybrid"]


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    project_root = Path(raw["project_root"])
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()

    return {
        "project_root": project_root,
        "queries": raw["queries"],
    }


def run_query(query: str, mode: str, project_root: Path) -> list[str]:
    """Run giddy find and return result paths."""
    cmd = ["giddy", "find", query, "--json", "--limit", "5", mode]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30, cwd=project_root
    )
    if result.returncode != 0:
        return []
    try:
        data = json.loads(result.stdout)
        return [r.get("path", "") for r in data[:5]]
    except json.JSONDecodeError:
        return []


def score_query(result_paths: list[str], expected: list[str]) -> dict:
    """Score a single query result against expected files."""
    found = [exp for exp in expected if any(rp.endswith(exp) for rp in result_paths)]
    missed = [exp for exp in expected if exp not in found]

    recall = len(found) / len(expected) if expected else 0

    relevant_in_results = sum(
        1 for rp in result_paths if any(rp.endswith(exp) for exp in expected)
    )
    precision = relevant_in_results / len(result_paths) if result_paths else 0

    rr = 0.0
    for i, rp in enumerate(result_paths):
        if any(rp.endswith(exp) for exp in expected):
            rr = 1.0 / (i + 1)
            break

    return {
        "recall": recall,
        "precision": precision,
        "reciprocal_rank": rr,
        "found": found,
        "missed": missed,
    }


def benchmark_project(config_path: Path, mode: str, verbose: bool) -> dict:
    """Run all queries for a project in a given mode. Returns summary dict."""
    config = load_config(config_path)
    project_root = config["project_root"]
    project_name = config_path.stem

    recalls, precisions, rrs = [], [], []

    for entry in config["queries"]:
        query = entry["query"]
        expected = entry["expected"]

        result_paths = run_query(query, mode, project_root)
        scores = score_query(result_paths, expected)

        recalls.append(scores["recall"])
        precisions.append(scores["precision"])
        rrs.append(scores["reciprocal_rank"])

        if verbose:
            tag = "HIT" if scores["recall"] > 0 else "MISS"
            print(f"    {tag} recall={scores['recall']:.0%} prec={scores['precision']:.0%} "
                  f"rr={scores['reciprocal_rank']:.2f}  {query[:55]}")
            for i, rp in enumerate(result_paths, 1):
                pr = str(project_root)
                short = str(Path(rp).relative_to(pr)) if rp.startswith(pr) else rp
                marker = " *" if any(short.endswith(exp) for exp in expected) else ""
                print(f"          {i}. {short}{marker}")
            if scores["missed"]:
                print(f"          Missed: {', '.join(scores['missed'])}")

    n = len(recalls)
    summary = {
        "project": project_name,
        "mode": mode,
        "queries": n,
        "avg_recall": sum(recalls) / n if n else 0,
        "avg_precision": sum(precisions) / n if n else 0,
        "mrr": sum(rrs) / n if n else 0,
    }
    return summary


def print_summary_table(summaries: list[dict]):
    """Print a comparison table."""
    # Group by project
    projects = []
    seen = set()
    for s in summaries:
        if s["project"] not in seen:
            projects.append(s["project"])
            seen.add(s["project"])

    modes = []
    seen = set()
    for s in summaries:
        if s["mode"] not in seen:
            modes.append(s["mode"])
            seen.add(s["mode"])

    lookup = {(s["project"], s["mode"]): s for s in summaries}

    col = 14
    mode_labels = [m.lstrip("-") for m in modes]

    print(f"\n  {'':20s}", end="")
    for label in mode_labels:
        print(f"  {label:>{col}}", end="")
    print()
    print("  " + "-" * (20 + (col + 2) * len(modes)))

    for proj in projects:
        # Recall row
        print(f"  {proj:20s}", end="")
        for mode in modes:
            s = lookup.get((proj, mode))
            val = f"{s['avg_recall']:.0%}" if s else "N/A"
            print(f"  {val:>{col}}", end="")
        print("  recall")

        # Precision row
        print(f"  {'':20s}", end="")
        for mode in modes:
            s = lookup.get((proj, mode))
            val = f"{s['avg_precision']:.0%}" if s else "N/A"
            print(f"  {val:>{col}}", end="")
        print("  precision")

        # MRR row
        print(f"  {'':20s}", end="")
        for mode in modes:
            s = lookup.get((proj, mode))
            val = f"{s['mrr']:.2f}" if s else "N/A"
            print(f"  {val:>{col}}", end="")
        print("  mrr")

        print()


def main():
    parser = argparse.ArgumentParser(description="Search quality benchmark")
    parser.add_argument("projects", nargs="*", help="Project names (config file stems)")
    parser.add_argument(
        "--mode", default="all",
        help="Search mode: semantic, full-text, hybrid, or all",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-query details")
    args = parser.parse_args()

    # Find configs
    if args.projects:
        config_paths = []
        for name in args.projects:
            p = CONFIGS_DIR / f"{name}.yaml"
            if not p.exists():
                print(f"Config not found: {p}", file=sys.stderr)
                sys.exit(1)
            config_paths.append(p)
    else:
        config_paths = sorted(CONFIGS_DIR.glob("*.yaml"))

    if not config_paths:
        print("No config files found", file=sys.stderr)
        sys.exit(1)

    # Determine modes
    if args.mode == "all":
        modes = MODES
    else:
        mode_flag = f"--{args.mode}" if not args.mode.startswith("--") else args.mode
        if mode_flag not in MODES:
            print(
                f"Unknown mode: {args.mode}. "
                "Choose from: semantic, full-text, hybrid, all",
                file=sys.stderr,
            )
            sys.exit(1)
        modes = [mode_flag]

    print("Search Quality Benchmark")
    print(f"Projects: {', '.join(p.stem for p in config_paths)}")
    print(f"Modes: {', '.join(m.lstrip('-') for m in modes)}")

    summaries = []
    for config_path in config_paths:
        for mode in modes:
            if args.verbose:
                print(f"\n  {config_path.stem} ({mode.lstrip('-')})")
                print(f"  {'=' * 40}")
            summary = benchmark_project(config_path, mode, args.verbose)
            summaries.append(summary)

    print_summary_table(summaries)


if __name__ == "__main__":
    main()
