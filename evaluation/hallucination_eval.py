"""Hallucination evaluation for the RAG tool.

Bypasses the agent entirely — calls _retrieve_context and _generate_answer
directly so we evaluate the RAG pipeline in isolation.

Usage (via makefile):
    cd evaluation && make hallucination
"""

import argparse
import json
import logging
import re
from typing import Dict, List, Optional

from backend.tools.company_financial_statement_tool.rag_model import (
    _parse_query,
    _retrieve_context,
    _generate_answer,
    client,
    gemini_model,
)
from evaluation.evaluate import load_test_cases, QUERY_TEMPLATE

logger = logging.getLogger(__name__)

HALLUCINATION_PROMPT = """\
You are a hallucination detection expert evaluating an AI agent's response.

Your task: identify any claims in the FINAL ANSWER that are NOT supported by \
or directly contradict the RETRIEVED CONTEXT. These unsupported claims are hallucinations.

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
{response}

Respond ONLY with valid JSON in exactly this format (no markdown, no code fences):
{{
  "has_hallucination": true or false,
  "hallucinated_claims": ["claim1", "claim2"],
  "verdict": "one sentence summary"
}}
"""


def evaluate_hallucination(context: str, response: str, question: str) -> Dict:
    """Ask Gemini to detect hallucinations in response vs retrieved context."""
    prompt = HALLUCINATION_PROMPT.format(
        context=context,
        question=question,
        response=response,
    )
    try:
        result = client.models.generate_content(model=gemini_model, contents=prompt)
        raw = result.text.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        parsed = json.loads(raw)
        return {
            "has_hallucination": bool(parsed.get("has_hallucination", False)),
            "hallucinated_claims": parsed.get("hallucinated_claims", []),
            "verdict": parsed.get("verdict", ""),
        }
    except Exception as e:
        logger.error(f"Hallucination evaluation failed: {e}")
        return {
            "has_hallucination": None,
            "hallucinated_claims": [],
            "verdict": f"Evaluation error: {e}",
        }


def run_hallucination_eval(csv_path: str) -> None:
    test_cases = load_test_cases(csv_path)
    rows = []

    for i, case in enumerate(test_cases, 1):
        logger.info(f"[{i}/{len(test_cases)}] {case['company']} — {case['query'][:60]}")

        # Only evaluate cases that expect the rag_tool
        if "rag_tool" not in case.get("expected_tools", []):
            logger.info("  rag_tool not expected for this case — skipping")
            continue

        query = QUERY_TEMPLATE.format(**case)
        parsed = _parse_query(query)
        company = parsed["company"]
        fy = parsed["financial_year"]
        quarter = parsed["quarter"]
        question = parsed["question"]

        try:
            nodes, context = _retrieve_context(company, fy, quarter, question)
        except Exception as e:
            logger.error(f"  Context retrieval failed: {e}")
            rows.append({
                "company": case["company"],
                "query": case["query"],
                "has_hallucination": None,
                "hallucinated_claims": [],
                "verdict": f"Retrieval error: {e}",
            })
            continue

        if not nodes:
            logger.warning("  No nodes retrieved — skipping hallucination check")
            rows.append({
                "company": case["company"],
                "query": case["query"],
                "has_hallucination": None,
                "hallucinated_claims": [],
                "verdict": "No documents retrieved",
            })
            continue

        try:
            response = _generate_answer(context, question)
        except Exception as e:
            logger.error(f"  Answer generation failed: {e}")
            rows.append({
                "company": case["company"],
                "query": case["query"],
                "has_hallucination": None,
                "hallucinated_claims": [],
                "verdict": f"Generation error: {e}",
            })
            continue

        logger.info("  Running hallucination check with Gemini...")
        result = evaluate_hallucination(context, response, question)
        flag = "HALLUCINATION DETECTED" if result["has_hallucination"] else "No hallucination"
        logger.info(f"  {flag}: {result['verdict']}")

        rows.append({
            "company": case["company"],
            "query": case["query"],
            "has_hallucination": result["has_hallucination"],
            "hallucinated_claims": result["hallucinated_claims"],
            "verdict": result["verdict"],
        })

    # Summary
    evaluated = [r for r in rows if r["has_hallucination"] is not None]
    hallucinated = [r for r in evaluated if r["has_hallucination"]]

    print("\n=== Hallucination Evaluation Summary ===")
    print(f"RAG test cases evaluated : {len(rows)}")
    print(f"Successfully checked     : {len(evaluated)}")
    print(f"Hallucinations detected  : {len(hallucinated)}")
    if evaluated:
        rate = len(hallucinated) / len(evaluated) * 100
        print(f"Hallucination rate       : {rate:.1f}%")

    if hallucinated:
        print("\nQueries with hallucinations:")
        for r in hallucinated:
            print(f"  [{r['company']}] {r['query'][:70]}")
            print(f"    Verdict : {r['verdict']}")
            for claim in r["hallucinated_claims"]:
                print(f"    Claim   : {claim}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="RAG hallucination evaluator")
    parser.add_argument("csv_path", help="Path to rag_evaluation_sample.csv")
    args = parser.parse_args()
    run_hallucination_eval(args.csv_path)
