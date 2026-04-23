develop:
	rm -rf venv
	python3.13 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install --upgrade uv
	venv/bin/uv pip install -e .

build_index:
	venv/bin/python backend/build_index.py

run_server:
	venv/bin/uvicorn backend.main:app

run_ui:
	cd frontend && streamlit run app.py --server.headless true

evaluate:
	venv/bin/python -m evaluation.evaluate evaluation/evaluation_sample.csv

hallucination:
	venv/bin/python -m evaluation.hallucination_eval evaluation/rag_evaluation_sample.csv
