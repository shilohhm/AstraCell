.PHONY: help install test lint fmt typecheck demo notebook figures check clean

PY ?= .venv/Scripts/python.exe   # on Linux/macOS: make PY=.venv/bin/python

help:
	@echo "install    create venv and install astracell with dev extras"
	@echo "test       run the test suite"
	@echo "lint       ruff check"
	@echo "fmt        ruff format"
	@echo "typecheck  mypy over src/"
	@echo "demo       run examples/01_first_demo.py, writing reports/figures/"
	@echo "notebook   regenerate notebooks/01_identifiability_study.ipynb"
	@echo "figures    alias for demo"
	@echo "check      lint + typecheck + test  (what CI runs)"

install:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,notebook]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests examples scripts

fmt:
	$(PY) -m ruff format src tests examples scripts

typecheck:
	$(PY) -m mypy

demo figures:
	$(PY) examples/01_first_demo.py

notebook:
	$(PY) scripts/build_notebook.py

check: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache reports/figures
	find . -type d -name __pycache__ -exec rm -rf {} +
