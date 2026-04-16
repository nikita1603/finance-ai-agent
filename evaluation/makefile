# Makefile for evaluation helpers.
# Targets:
#   evaluate     - Run the full evaluation suite using evaluation_sample.csv (default)
#   hallucination - Run the hallucination evaluator against rag_evaluation_sample.csv
# Usage: run targets from the `evaluation/` directory, e.g. `make evaluate`.
# You can override `CSV_PATH` for the `evaluate` target:
#   make evaluate CSV_PATH=other_file.csv

CSV_PATH ?= evaluation_sample.csv

evaluate:
	cd .. && venv/bin/python -m evaluation.evaluate evaluation/$(CSV_PATH)

hallucination:
	cd .. && venv/bin/python -m evaluation.hallucination_eval evaluation/rag_evaluation_sample.csv
