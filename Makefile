PYTHON ?= .venv/bin/python

.PHONY: check test lint setup audit

check:
	$(PYTHON) tools/check.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

setup:
	$(PYTHON) setup_check.py

# Queries public advisory metadata with package names/versions only.
audit:
	uvx --from pip-audit==2.10.1 pip-audit --path .venv/lib/python3.11/site-packages --progress-spinner off
