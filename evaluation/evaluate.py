"""Direct in-process evaluation of the Finance AI Agent.

Calls agent.run() directly — no server or log file needed.

Usage:
    python -m evaluation.evaluate evaluation/evaluation_sample.csv
"""

import asyncio
import argparse
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from llama_index.core.agent.workflow import ToolCallResult

from backend.agent_system import agent
from evaluation.eval_utils import print_results

logger = logging.getLogger(__name__)

QUERY_TIMEOUT_S = 300  # per-query hard cap to prevent indefinite hangs

QUERY_TEMPLATE = """Date: {date}
Company: {company}
Financial Year: {financial_year}
Quarter: {quarter}
Question: {query}
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_cases(csv_path: str) -> List[Dict]:
    """Load evaluation test cases from CSV and normalise fields.

    Semicolon-separated list columns (`expected_tools_called`,
    `expected_keywords`, `expected_sources_used`) are split into Python lists.
    """
    data = pd.read_csv(csv_path)
    data["expected_tools"] = data["expected_tools_called"].fillna("").apply(
        lambda x: [t.strip() for t in x.split(";") if t.strip()]
    )
    data["expected_keywords"] = data["expected_keywords"].fillna("").apply(
        lambda x: [k.strip() for k in x.split(";") if k.strip()]
    )
    data["expected_sources"] = data["expected_sources_used"].fillna("").apply(
        lambda x: [s.strip() for s in x.split(";") if s.strip()]
    )
    data["is_multi_hop"] = data["is_multi_hop"].apply(lambda x: str(x).lower() == "true")
    return data.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------------

def compute_result(
    expected: Dict,
    tools_called: List[str],
    response: Optional[str],
    sources_used: List[str],
    latency: float = 0.0,
) -> Dict:
    """Compute per-question evaluation result dict."""
    exp = set(t.lower() for t in expected["expected_tools"])
    act = set(t.lower() for t in tools_called)
    found_kw = [k for k in expected["expected_keywords"] if k.lower() in (response or "").lower()]
    missing_kw = [k for k in expected["expected_keywords"] if k not in found_kw]

    exp_src = set(s.lower() for s in expected["expected_sources"])
    act_src = set(s.lower() for s in sources_used)
    rag_expected = bool(exp_src)
    sources_precision = len(act_src & exp_src) / len(act_src) if act_src else (1.0 if not exp_src else 0.0)
    sources_recall = len(act_src & exp_src) / len(exp_src) if exp_src else (1.0 if not act_src else 0.0)

    return {
        "company": expected["company"],
        "query": expected["query"],
        "expected_tools": expected["expected_tools"],
        "tools_called": tools_called,
        "tools_correct": act == exp,
        "tools_precision": len(act & exp) / len(act) if act else (1.0 if not exp else 0.0),
        "tools_recall": len(act & exp) / len(exp) if exp else (1.0 if not act else 0.0),
        "expected_keywords": expected["expected_keywords"],
        "found_keywords": found_kw,
        "missing_keywords": missing_kw,
        "keywords_recall": len(found_kw) / len(expected["expected_keywords"]) if expected["expected_keywords"] else 1.0,
        "is_multi_hop": expected["is_multi_hop"],
        "multi_hop_correct": (len(tools_called) > 1) == expected["is_multi_hop"],
        "has_response": response is not None,
        "response_length": len(response) if response else 0,
        "rag_expected": rag_expected,
        "expected_sources": list(exp_src),
        "sources_used": list(act_src),
        "sources_precision": sources_precision,
        "sources_recall": sources_recall,
        "latency_s": latency,
    }


# ---------------------------------------------------------------------------
# Confidence interval helpers
# ---------------------------------------------------------------------------

def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score confidence interval for a proportion.

    Appropriate for binary (0/1) metrics such as tools_accuracy.
    """
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def bootstrap_ci(
    values: List[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    statistic: str = "mean",
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval.

    Appropriate for averaged ratio metrics (precision, recall, etc.) and for
    the p95 latency estimate. Uses a fixed seed for reproducibility.
    """
    if not values:
        return 0.0, 1.0
    rng = np.random.default_rng(42)
    arr = np.array(values)
    if statistic == "mean":
        boot_stats = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    elif statistic == "p95":
        boot_stats = [np.percentile(rng.choice(arr, size=len(arr), replace=True), 95) for _ in range(n_boot)]
    else:
        raise ValueError(f"Unknown statistic: {statistic!r}")
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_stats, alpha * 100)), float(np.percentile(boot_stats, (1 - alpha) * 100))


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: List[Dict]) -> Dict:
    """Aggregate per-question results into overall metrics with 95% CIs."""
    n = len(results)
    rag_results = [r for r in results if r["rag_expected"]]
    n_rag = len(rag_results)

    tools_accuracy = sum(r["tools_correct"] for r in results) / n
    tools_avg_precision = sum(r["tools_precision"] for r in results) / n
    tools_avg_recall = sum(r["tools_recall"] for r in results) / n
    keywords_avg_recall = sum(r["keywords_recall"] for r in results) / n
    multi_hop_accuracy = sum(r["multi_hop_correct"] for r in results) / n
    response_rate = sum(r["has_response"] for r in results) / n
    latencies = [r["latency_s"] for r in results]
    latency_avg = sum(latencies) / n
    latency_p95 = float(np.percentile(latencies, 95))
    latency_max = max(latencies)

    sources_avg_precision = sum(r["sources_precision"] for r in rag_results) / n_rag if n_rag else None
    sources_avg_recall = sum(r["sources_recall"] for r in rag_results) / n_rag if n_rag else None

    return {
        "total_tests": n,

        # Point estimates
        "tools_accuracy": tools_accuracy,
        "tools_avg_precision": tools_avg_precision,
        "tools_avg_recall": tools_avg_recall,
        "keywords_avg_recall": keywords_avg_recall,
        "multi_hop_accuracy": multi_hop_accuracy,
        "response_rate": response_rate,
        "rag_tests": n_rag,
        "sources_avg_precision": sources_avg_precision,
        "sources_avg_recall": sources_avg_recall,
        "latency_avg_s": latency_avg,
        "latency_p95_s": latency_p95,
        "latency_max_s": latency_max,

        # 95% confidence intervals
        # Wilson score for binary outcomes (correct/incorrect per query)
        "tools_accuracy_ci": wilson_ci(tools_accuracy, n),
        "tools_avg_recall_ci": wilson_ci(tools_avg_recall, n),
        "multi_hop_accuracy_ci": wilson_ci(multi_hop_accuracy, n),
        "response_rate_ci": wilson_ci(response_rate, n),

        # Bootstrap for averaged ratio metrics and latency p95
        "tools_avg_precision_ci": bootstrap_ci([r["tools_precision"] for r in results]),
        "keywords_avg_recall_ci": bootstrap_ci([r["keywords_recall"] for r in results]),
        "sources_avg_precision_ci": bootstrap_ci([r["sources_precision"] for r in rag_results]) if n_rag else None,
        "sources_avg_recall_ci": bootstrap_ci([r["sources_recall"] for r in rag_results]) if n_rag else None,
        "latency_p95_ci": bootstrap_ci(latencies, statistic="p95"),
    }


# ---------------------------------------------------------------------------
# Query runner
# ---------------------------------------------------------------------------

async def run_query(query: str) -> Tuple[List[str], Optional[str], List[str]]:
    """Run a single query through the agent and collect tool usage and sources."""
    tools_called: List[str] = []
    sources_used: List[str] = []
    try:
        handler = agent.run(query)
        async for event in handler.stream_events():
            if isinstance(event, ToolCallResult):
                if event.tool_name not in tools_called:
                    tools_called.append(event.tool_name)
                if event.tool_name == "rag_tool":
                    output = str(event.tool_output)
                    m = re.search(r"\[SOURCES_USED: ([^\]]+)\]", output)
                    if m:
                        sources_used.extend(
                            s.strip() for s in m.group(1).split(";") if s.strip()
                        )
        return tools_called, str(await handler), sources_used
    except asyncio.TimeoutError:
        logger.error("Query timed out after %ds", QUERY_TIMEOUT_S)
        return tools_called, None, sources_used
    except Exception as e:
        logger.error("Query failed: %s", e)
        return tools_called, None, sources_used


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(csv_path: str) -> None:
    test_cases = load_test_cases(csv_path)
    results = []
    for i, expected in enumerate(test_cases, 1):
        logger.info("Running test case %d/%d: %s", i, len(test_cases), expected["company"])
        query = QUERY_TEMPLATE.format(**expected)
        t0 = time.perf_counter()
        try:
            tools_called, response, sources_used = await asyncio.wait_for(
                run_query(query), timeout=QUERY_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            logger.error("Test case %d timed out after %ds", i, QUERY_TIMEOUT_S)
            tools_called, response, sources_used = [], None, []
        latency = time.perf_counter() - t0
        logger.info("Latency: %.2fs", latency)
        results.append(compute_result(expected, tools_called, response, sources_used, latency))
    print_results(results, compute_metrics(results))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Direct in-process agent evaluation")
    parser.add_argument("csv_path", help="Path to CSV file with test cases")
    args = parser.parse_args()
    asyncio.run(main(args.csv_path))
