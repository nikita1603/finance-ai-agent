"""Utilities for printing evaluation results.

Contains helpers used by the evaluation scripts to present question-level
diagnostics and an aggregate summary with 95% confidence intervals.
"""

from typing import Dict, List, Optional, Tuple


def _fmt_ci(ci: Optional[Tuple[float, float]]) -> str:
    """Format a (lower, upper) CI tuple as a bracket string, e.g. '[85%, 97%]'."""
    if ci is None:
        return ""
    lo, hi = ci
    return f"[{lo:.0%}, {hi:.0%}]"


def print_results(results: List[Dict], metrics: Dict) -> None:
    """Print detailed per-question comparisons and an aggregate summary.

    Args:
        results: Per-question result dicts produced by the evaluation harness.
        metrics: Aggregate metrics dict including point estimates and 95% CIs.
    """
    print(f"\n{'=' * 100}")
    print("QUESTION-WISE COMPARISON")
    print(f"{'=' * 100}\n")

    for i, r in enumerate(results, 1):
        print(f"Question {i}/{len(results)} | Company: {r['company']}")
        print(f"Query: {r['query']}")
        print(
            f"  Tools     expected={r['expected_tools']}  actual={r['tools_called']}  "
            f"{'PASS' if r['tools_correct'] else 'FAIL'} "
            f"(P={r['tools_precision']:.0%} R={r['tools_recall']:.0%})"
        )
        if r["expected_keywords"]:
            print(
                f"  Keywords  found={r['found_keywords']}  missing={r['missing_keywords']}  "
                f"recall={r['keywords_recall']:.0%}"
            )
        print(
            f"  Multi-hop {'PASS' if r['multi_hop_correct'] else 'FAIL'} | "
            f"Response: {'yes' if r['has_response'] else 'no'} ({r['response_length']} chars) | "
            f"Latency: {r['latency_s']:.2f}s"
        )
        if r["rag_expected"] or r["sources_used"]:
            print(
                f"  Sources   expected={r['expected_sources']}  actual={r['sources_used']}  "
                f"(P={r['sources_precision']:.0%} R={r['sources_recall']:.0%})"
            )
        print(f"{'-' * 100}\n")

    # ── Aggregate summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("SUMMARY  (95% confidence intervals in brackets)")
    print(f"{'=' * 100}")
    print(f"Total tests:        {metrics['total_tests']}")
    print(
        f"Tools accuracy:     {metrics['tools_accuracy']:.0%}  "
        f"{_fmt_ci(metrics.get('tools_accuracy_ci'))}"
    )
    print(
        f"Tools precision:    {metrics['tools_avg_precision']:.0%}  "
        f"{_fmt_ci(metrics.get('tools_avg_precision_ci'))}"
    )
    print(
        f"Tools recall:       {metrics['tools_avg_recall']:.0%}  "
        f"{_fmt_ci(metrics.get('tools_avg_recall_ci'))}"
    )
    print(
        f"Keywords recall:    {metrics['keywords_avg_recall']:.0%}  "
        f"{_fmt_ci(metrics.get('keywords_avg_recall_ci'))}"
    )
    print(
        f"Multi-hop accuracy: {metrics['multi_hop_accuracy']:.0%}  "
        f"{_fmt_ci(metrics.get('multi_hop_accuracy_ci'))}"
    )
    print(
        f"Response rate:      {metrics['response_rate']:.0%}  "
        f"{_fmt_ci(metrics.get('response_rate_ci'))}"
    )

    if metrics["rag_tests"]:
        print(f"RAG tests:          {metrics['rag_tests']}")
        print(
            f"Sources precision:  {metrics['sources_avg_precision']:.0%}  "
            f"{_fmt_ci(metrics.get('sources_avg_precision_ci'))}"
        )
        print(
            f"Sources recall:     {metrics['sources_avg_recall']:.0%}  "
            f"{_fmt_ci(metrics.get('sources_avg_recall_ci'))}"
        )

    print(f"Latency avg:        {metrics['latency_avg_s']:.2f}s")
    print(
        f"Latency p95:        {metrics['latency_p95_s']:.2f}s  "
        f"[{metrics['latency_p95_ci'][0]:.1f}s, {metrics['latency_p95_ci'][1]:.1f}s]"
        if metrics.get('latency_p95_ci') else f"Latency p95:        {metrics['latency_p95_s']:.2f}s"
    )
    print(f"Latency max:        {metrics['latency_max_s']:.2f}s")
    print(f"{'=' * 100}\n")
