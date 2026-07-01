.PHONY: setup lint test data-check train uncertainty causal backtest app all clean

VENV      := .venv
PYTHON    := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
RUFF      := $(VENV)/bin/ruff
MYPY      := $(VENV)/bin/mypy
PYTEST    := $(VENV)/bin/pytest
STREAMLIT := $(VENV)/bin/streamlit

# ── Environment ─────────────────────────────────────────────────────────────
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(VENV)/bin/pre-commit install
	$(PIP) freeze > requirements-lock.txt
	@echo ""
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"
	@echo "Optional causal forest extras: pip install -e '.[causal]'"

# ── Quality gates ────────────────────────────────────────────────────────────
lint:
	PYTHONPATH=src $(PYTHON) -m ruff check src/ tests/
	PYTHONPATH=src $(PYTHON) -m ruff format --check src/ tests/
	PYTHONPATH=src $(PYTHON) -m mypy src/

format:
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

test:
	PYTHONPATH=src $(PYTHON) -m pytest

# ── Data pipeline ────────────────────────────────────────────────────────────
data-check:
	$(PYTHON) -m margin_of_error.data.validate

# ── Modeling phases (unlocked after approval) ────────────────────────────────
train:
	PYTHONPATH=src $(PYTHON) -m margin_of_error.models.baseline

uncertainty:
	PYTHONPATH=src $(PYTHON) -m margin_of_error.models.phase2

causal:
	PYTHONPATH=src $(PYTHON) -m margin_of_error.causal.dml

backtest:
	PYTHONPATH=src $(PYTHON) -m margin_of_error.backtest.walkforward

# ── Application ─────────────────────────────────────────────────────────────
app:
	@echo "Phase 5 not yet implemented — awaiting approval."
	@# $(STREAMLIT) run src/margin_of_error/app/underwriting.py

# ── Housekeeping ─────────────────────────────────────────────────────────────
clean:
	rm -rf .venv __pycache__ .mypy_cache .ruff_cache .pytest_cache htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

all: setup lint test
