PYTHON ?= .venv/bin/python
PUBLISHER = $(firstword $(wildcard tools/publish_source.py publish_source.py))

.PHONY: check test lint setup audit benchmark source-preview publish

check:
	$(PYTHON) tools/check.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

benchmark:
	$(PYTHON) tools/benchmark_dashboard.py

setup:
	$(PYTHON) $(firstword $(wildcard src/your_time/setup_check.py setup_check.py))

source-preview:
	$(PYTHON) $(PUBLISHER)

publish:
	$(PYTHON) $(PUBLISHER) --push

# Queries public advisory metadata with package names/versions only.
audit:
	uvx --from pip-audit==2.10.1 pip-audit --path .venv/lib/python3.11/site-packages --progress-spinner off
