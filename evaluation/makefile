CSV_PATH ?= evaluation_sample.csv

evaluate:
	cd .. && venv/bin/python -m evaluation.evaluate evaluation/$(CSV_PATH)

hallucination:
	cd .. && venv/bin/python -m evaluation.hallucination_eval evaluation/rag_evaluation_sample.csv
