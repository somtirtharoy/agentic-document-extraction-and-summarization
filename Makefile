.PHONY: setup ingest extract summarize evaluate agent-demo test lint fmt

setup:
	pip install -r requirements.txt
	pre-commit install
	python -m spacy download en_core_web_sm

ingest:
	python -m scripts.01_ingest_data

extract:
	python -m scripts.02_run_extraction

summarize:
	python -m scripts.03_run_summarization

evaluate:
	python -m scripts.04_run_evaluation

agent-demo:
	python -m scripts.05_agent_demo

test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/

fmt:
	black src/ scripts/ tests/
