"""Utilities for printing evaluation results.

This module contains helpers used by the evaluation scripts to present
question-level diagnostics and an aggregate summary. The output is
plain-text and intended for console inspection during development and
evaluation runs.
"""

from typing import Dict, List


def print_results(results: List[Dict], metrics: Dict) -> None:
    """Print detailed per-question comparisons and an aggregate summary.

    Args:
        results: List of per-question result dicts produced by the
            evaluation harness. Each dict is expected to contain fields
            such as `company`, `query`, `expected_tools`, `tools_called`,
            `tools_correct`, keyword metrics, and RAG/source info.
        metrics: Aggregate metrics dictionary containing overall counts
            and percentages (e.g., tools_accuracy, tools_avg_precision,
            etc.). Values are printed with percent formatting where
            appropriate.

    The function prints a human-friendly table to stdout; it intentionally
    does not return any value so callers can redirect stdout if needed.
    """
    # Header for question-wise section
    print(f"\n{'=' * 100}")
    print("QUESTION-WISE COMPARISON")
    print(f"{'='*100}\n")

    # Print detailed results for each question
    for i, r in enumerate(results, 1):
        print(f"Question {i}/{len(results)} | Company: {r['company']}")
        print(f"Query: {r['query']}")

        # Tools: expected vs actual with pass/fail and precision/recall
        print(
            f"  Tools     expected={r['expected_tools']}  actual={r['tools_called']}  "
            f"{'PASS' if r['tools_correct'] else 'FAIL'} "
            f"(P={r['tools_precision']:.0%} R={r['tools_recall']:.0%})"
        )

        # Keywords diagnostics (only show when expected keywords exist)
        if r["expected_keywords"]:
            print(
                f"  Keywords  found={r['found_keywords']}  missing={r['missing_keywords']}  "
                f"recall={r['keywords_recall']:.0%}"
            )

        # Multi-hop & response presence information
        print(
            f"  Multi-hop {'PASS' if r['multi_hop_correct'] else 'FAIL'} | "
            f"Response: {'yes' if r['has_response'] else 'no'} ({r['response_length']} chars) | "
            f"Latency: {r['latency_s']:.2f}s"
        )

        # RAG/source checks (only print when relevant)
        if r["rag_expected"] or r["sources_used"]:
            print(
                f"  Sources   expected={r['expected_sources']}  actual={r['sources_used']}  "
                f"(P={r['sources_precision']:.0%} R={r['sources_recall']:.0%})"
            )

        print(f"{'-' * 100}\n")

    # Aggregate summary
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    print(f"Total tests:        {metrics['total_tests']}")
    print(f"Tools accuracy:     {metrics['tools_accuracy']:.0%}")
    print(f"Tools precision:    {metrics['tools_avg_precision']:.0%}")
    print(f"Tools recall:       {metrics['tools_avg_recall']:.0%}")
    print(f"Keywords recall:    {metrics['keywords_avg_recall']:.0%}")
    print(f"Multi-hop accuracy: {metrics['multi_hop_accuracy']:.0%}")
    print(f"Response rate:      {metrics['response_rate']:.0%}")

    # If RAG tests exist, print their aggregated metrics
    if metrics["rag_tests"]:
        print(f"RAG tests:          {metrics['rag_tests']}")
        print(f"Sources precision:  {metrics['sources_avg_precision']:.0%}")
        print(f"Sources recall:     {metrics['sources_avg_recall']:.0%}")

    print(f"Latency avg:        {metrics['latency_avg_s']:.2f}s")
    print(f"Latency p95:        {metrics['latency_p95_s']:.2f}s")
    print(f"Latency max:        {metrics['latency_max_s']:.2f}s")
    print(f"{'=' * 100}\n")
