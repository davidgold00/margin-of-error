.PHONY: setup lint test data-check train backtest app all clean

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
	@echo "To install causal extras (Phase 3): pip install -e '.[causal]'"

# ── Quality gates ────────────────────────────────────────────────────────────
lint:
	$(RUFF) check src/ tests/
	$(RUFF) format --check src/ tests/
	$(MYPY) src/

format:
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

test:
	$(PYTEST)

# ── Data pipeline ────────────────────────────────────────────────────────────
data-check:
	$(PYTHON) -m margin_of_error.data.validate

# ── Modeling phases (unlocked after approval) ────────────────────────────────
train:
	@echo "Phase 1 not yet implemented — awaiting approval."

backtest:
	@echo "Phase 4 not yet implemented — awaiting approval."

# ── Application ─────────────────────────────────────────────────────────────
app:
	@echo "Phase 5 not yet implemented — awaiting approval."
	@# $(STREAMLIT) run src/margin_of_error/app/underwriting.py

# ── Housekeeping ─────────────────────────────────────────────────────────────
clean:
	rm -rf .venv __pycache__ .mypy_cache .ruff_cache .pytest_cache htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

all: setup lint test
