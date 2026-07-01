"""Phase 5 tests: app artifacts, end-to-end underwriting, docs, and figures."""

from __future__ import annotations

import re
from pathlib import Path

from margin_of_error.app.artifacts import load_app_artifacts
from margin_of_error.app.underwriting import underwrite_property


def test_app_loads_all_required_artifacts() -> None:
    """The app should load the real saved Phase 1 + Phase 2 + defaults artifacts."""
    artifacts = load_app_artifacts()
    assert artifacts.point_model is not None
    assert artifacts.cqr_model.q_hat > 0
    assert artifacts.defaults
    assert {"Neighborhood", "GrLivArea", "OverallQual"}.issubset(artifacts.defaults)


def test_app_underwrite_end_to_end() -> None:
    """A sample property should produce a complete verdict object without errors."""
    artifacts = load_app_artifacts()
    result = underwrite_property(
        artifacts,
        {
            "Neighborhood": "NAmes",
            "GrLivArea": 1500,
            "OverallQual": 6,
            "YearBuilt": 1975,
            "FullBath": 2,
            "HalfBath": 0,
            "KitchenQual": "TA",
            "TotalBsmtSF": 900,
            "GarageCars": 2,
            "GarageFinish": "Unf",
        },
        purchase_price=140_000,
        renovation_tier="moderate",
    )
    assert result.verdict in {"APPROVE", "REFER", "DECLINE"}
    assert result.valuation.point_value > 0
    assert result.valuation.interval_low < result.valuation.interval_high
    assert result.profit_p10 <= result.expected_profit <= result.profit_p90
    assert 0.0 <= result.prob_loss <= 1.0
    assert len(result.profit_draws) == artifacts.economics.flip.monte_carlo_samples
    assert result.causal_guidance


def test_explainer_numbers_exist(repo_root: Path) -> None:
    """PROJECT_EXPLAINER should not ship with unfilled placeholder tokens."""
    path = repo_root / "docs" / "PROJECT_EXPLAINER.md"
    assert path.exists()
    text = path.read_text()
    assert "NUMBERS I COULD NOT LOCATE" not in text
    assert not re.search(r"\b(TODO|PLACEHOLDER|XXX|TBD)\b", text)
    assert "| Phase | Number | Source |" in text


def test_all_referenced_figures_exist(repo_root: Path) -> None:
    """Every reports/figures path referenced by docs resolves to a real file."""
    docs = [
        repo_root / "README.md",
        repo_root / "reports" / "memo.md",
        repo_root / "reports" / "deck_outline.md",
        repo_root / "docs" / "PROJECT_EXPLAINER.md",
    ]
    missing_docs = [path for path in docs if not path.exists()]
    assert not missing_docs

    referenced: set[str] = set()
    pattern = re.compile(r"reports/figures/[A-Za-z0-9_.-]+")
    for path in docs:
        referenced.update(pattern.findall(path.read_text()))

    assert referenced, "Expected at least one referenced figure"
    missing = [rel for rel in sorted(referenced) if not (repo_root / rel).exists()]
    assert not missing, f"Missing referenced figure files: {missing}"
