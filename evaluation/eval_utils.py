"""Printing utilities for evaluation results."""

from typing import Dict, List


def print_results(results: List[Dict], metrics: Dict) -> None:
    """Print question-wise comparison and aggregate summary."""
    print(f"\n{'='*100}")
    print("QUESTION-WISE COMPARISON")
    print(f"{'='*100}\n")

    for i, r in enumerate(results, 1):
        print(f"Question {i}/{len(results)} | Company: {r['company']}")
        print(f"Query: {r['query']}")
        print(f"  Tools     expected={r['expected_tools']}  actual={r['tools_called']}  "
              f"{'PASS' if r['tools_correct'] else 'FAIL'} "
              f"(P={r['tools_precision']:.0%} R={r['tools_recall']:.0%})")
        if r["expected_keywords"]:
            print(f"  Keywords  found={r['found_keywords']}  missing={r['missing_keywords']}  "
                  f"recall={r['keywords_recall']:.0%}")
        print(f"  Multi-hop {'PASS' if r['multi_hop_correct'] else 'FAIL'} | "
              f"Response: {'yes' if r['has_response'] else 'no'} ({r['response_length']} chars)")
        if r["rag_expected"] or r["sources_used"]:
            print(f"  Sources   expected={r['expected_sources']}  actual={r['sources_used']}  "
                  f"(P={r['sources_precision']:.0%} R={r['sources_recall']:.0%})")
        print(f"{'-'*100}\n")

    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"Total tests:        {metrics['total_tests']}")
    print(f"Tools accuracy:     {metrics['tools_accuracy']:.0%}")
    print(f"Tools precision:    {metrics['tools_avg_precision']:.0%}")
    print(f"Tools recall:       {metrics['tools_avg_recall']:.0%}")
    print(f"Keywords recall:    {metrics['keywords_avg_recall']:.0%}")
    print(f"Multi-hop accuracy: {metrics['multi_hop_accuracy']:.0%}")
    print(f"Response rate:      {metrics['response_rate']:.0%}")
    if metrics["rag_tests"]:
        print(f"RAG tests:          {metrics['rag_tests']}")
        print(f"Sources precision:  {metrics['sources_avg_precision']:.0%}")
        print(f"Sources recall:     {metrics['sources_avg_recall']:.0%}")
    print(f"{'='*100}\n")
