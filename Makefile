.PHONY: install test results results-bm25 submission

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

results:
	python scripts/run_experiments.py

results-bm25:
	python scripts/run_experiments.py --bm25-only

submission:
	python scripts/make_submission.py
