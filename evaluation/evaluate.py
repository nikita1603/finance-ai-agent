"""Direct in-process evaluation of the Finance AI Agent.

Calls agent.run() directly — no server or log file needed.

Usage:
    python -m evaluation.evaluate evaluation/evaluation_sample.csv
"""

import asyncio
import argparse
import logging
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from llama_index.core.agent.workflow import ToolCallResult

from backend.agent_system import agent
from evaluation.eval_utils import print_results

logger = logging.getLogger(__name__)

QUERY_TEMPLATE = """Date: {date}
Company: {company}
Financial Year: {financial_year}
Quarter: {quarter}
Question: {query}
"""


def load_test_cases(csv_path: str) -> List[Dict]:
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


def compute_result(expected: Dict, tools_called: List[str], response: Optional[str], sources_used: List[str]) -> Dict:
    exp = set(expected["expected_tools"])
    act = set(tools_called)
    found_kw = [k for k in expected["expected_keywords"] if k.lower() in (response or "").lower()]
    missing_kw = [k for k in expected["expected_keywords"] if k not in found_kw]

    exp_src = set(expected["expected_sources"])
    act_src = set(sources_used)
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
    }


def compute_metrics(results: List[Dict]) -> Dict:
    n = len(results)
    rag_results = [r for r in results if r["rag_expected"]]
    n_rag = len(rag_results)
    return {
        "total_tests": n,
        "tools_accuracy": sum(r["tools_correct"] for r in results) / n,
        "tools_avg_precision": sum(r["tools_precision"] for r in results) / n,
        "tools_avg_recall": sum(r["tools_recall"] for r in results) / n,
        "keywords_avg_recall": sum(r["keywords_recall"] for r in results) / n,
        "multi_hop_accuracy": sum(r["multi_hop_correct"] for r in results) / n,
        "response_rate": sum(r["has_response"] for r in results) / n,
        "rag_tests": n_rag,
        "sources_avg_precision": sum(r["sources_precision"] for r in rag_results) / n_rag if n_rag else None,
        "sources_avg_recall": sum(r["sources_recall"] for r in rag_results) / n_rag if n_rag else None,
    }


async def run_query(query: str) -> Tuple[List[str], Optional[str], List[str]]:
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
                    m = re.search(r'\[SOURCES_USED: ([^\]]+)\]', output)
                    if m:
                        sources_used.extend(
                            s.strip() for s in m.group(1).split(";") if s.strip()
                        )
        return tools_called, str(await handler), sources_used
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return tools_called, None, sources_used


async def main(csv_path: str) -> None:
    test_cases = load_test_cases(csv_path)
    results = []
    for i, expected in enumerate(test_cases, 1):
        logger.info(f"Running test case {i}/{len(test_cases)}: {expected['company']}")
        query = QUERY_TEMPLATE.format(**expected)
        tools_called, response, sources_used = await run_query(query)
        results.append(compute_result(expected, tools_called, response, sources_used))
    print_results(results, compute_metrics(results))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Direct in-process agent evaluation")
    parser.add_argument("csv_path", help="Path to CSV file with test cases")
    args = parser.parse_args()
    asyncio.run(main(args.csv_path))
