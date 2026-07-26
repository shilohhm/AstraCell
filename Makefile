.PHONY: help install install-pybamm test lint fmt format-check typecheck demo notebook notebook-run figures check clean

PY ?= .venv/Scripts/python.exe   # on Linux/macOS: make PY=.venv/bin/python

help:
	@echo "install         create venv and install astracell with dev extras"
	@echo "install-pybamm  add the optional PyBaMM plant (for examples 06-07 / notebooks 03-04)"
	@echo "test            run the test suite (external-plant tests skip without PyBaMM)"
	@echo "lint            ruff check"
	@echo "fmt             ruff format"
	@echo "format-check    verify formatting without changing files"
	@echo "typecheck       mypy over src/"
	@echo "demo            run examples/01_first_demo.py, writing reports/figures/"
	@echo "notebook        regenerate notebook source under notebooks/ (strips outputs)"
	@echo "notebook-run    execute the notebooks in place, restoring their outputs"
	@echo "figures         alias for demo"
	@echo "check           lint + typecheck + test  (what CI runs)"

install:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,notebook]"

install-pybamm:
	$(PY) -m pip install -e ".[pybamm]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests examples scripts

fmt:
	$(PY) -m ruff format src tests examples scripts

format-check:
	$(PY) -m ruff format --check src tests examples scripts

typecheck:
	$(PY) -m mypy

demo figures:
	$(PY) examples/01_first_demo.py

# Writes source-only notebooks. The committed notebooks carry their outputs, so this discards
# them; follow with notebook-run before committing.
notebook:
	$(PY) scripts/build_notebook.py

notebook-run:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

check: lint format-check typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache reports/figures
	find . -type d -name __pycache__ -exec rm -rf {} +
