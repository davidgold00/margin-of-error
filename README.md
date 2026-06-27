# Margin of Error

> *An automated valuation model can be "accurate" by RMSE yet useless for underwriting,
> because the model's prediction error is wider than the flip's profit margin.*

---

## The Problem

In 2021, Zillow Offers lost roughly $500 million and shut down its iBuying division.
The post-mortem focused on prediction accuracy — their AVM was wrong. But the deeper
failure was epistemic: **Zillow treated a point-estimate model as a decision tool
without accounting for its own uncertainty.** A home valued at $310k ± $40k
purchased at $305k has a flip margin that is entirely inside the model's noise.

This project treats the Ames, Iowa housing dataset as a **decision-under-uncertainty**
problem rather than a leaderboard regression task. We build the asset that Zillow's
engineers arguably should have built: an uncertainty-aware valuation engine whose
output is a *profit distribution*, not a price estimate, with an explicit underwriting
decision rule.

The title has a deliberate double meaning:
- **Statistical margin of error** — the width of our prediction interval
- **Profit margin** — the economic return we are trying to protect

The central thesis: *the statistical margin of error must be smaller than the profit
margin of the deal for the deal to be underwritable by any honest model.*

---

## What We Build

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Scaffold + data validation layer | Complete |
| 1 | Baseline gradient boosting model (the strawman) | **Built; awaiting acceptance** |
| 2 | CQR prediction intervals + flip P&L simulation + underwriting rule | Pending |
| 3 | Causal estimation of renovation effects (DML) | Pending |
| 4 | Temporal backtest through 2006–2010 housing crash | Pending |
| 5 | Streamlit underwriting tool + strategy memo | Pending |

---

## Quick Start

```bash
git clone <repo>
cd margin-of-error

# 1. Set up environment (Python 3.11+ required)
make setup
source .venv/bin/activate

# 2. Obtain data — follow data/README.md, then place files in data/raw/
# 3. Validate data
make data-check

# 4. Train the Phase 1 baseline
make train

# 5. (After Phase 5 approval)
make app
```

---

## Key Design Decisions

Every non-obvious modeling or economic choice has a decision note in
[docs/decisions.md](docs/decisions.md). Every economic assumption has a rationale
and stated source in [docs/assumptions.md](docs/assumptions.md).

All assumptions live in [config/economics.yaml](config/economics.yaml) and
[config/model.yaml](config/model.yaml). No magic constants in code.

---

## Data

- **Kaggle split** (~1,460 train / ~1,459 test): random 50/50 split, used for
  cross-sectional model training (Phases 1–3). The random split is a deliberate
  limitation — flagged explicitly in Phase 4 when we switch to temporal ordering.
- **Full Ames dataset** (~2,930 rows, De Cock 2011): all sales 2006–2010, sorted
  temporally for the crash backtest (Phase 4).

See [data/README.md](data/README.md) for provenance and download instructions.
Raw data is git-ignored.

---

## Results Summary

| Metric | Value | Phase |
|--------|-------|-------|
| Dumb median CV RMSE (log scale) | 0.400 ± 0.015 | 1 |
| ElasticNet CV RMSE (log scale) | 0.126 ± 0.019 | 1 |
| LightGBM CV RMSE (log scale) | 0.135 ± 0.015 | 1 |
| LightGBM CV RMSE (dollars) | $28,500 ± $6,381 | 1 |
| LightGBM median absolute dollar error | $9,413 | 1 |
| LightGBM 80th percentile absolute dollar error | $22,193 | 1 |
| 90% prediction interval coverage | TBD | 2 |
| % of "good buys" with margin < uncertainty band | TBD | 2 |
| Causal effect of kitchen upgrade (CQR-adjusted) | TBD | 3 |
| Backtest: deals underwritten in 2007 that went negative | TBD | 4 |

### Phase 1 Baseline

Phase 1 uses the Kaggle `train.csv` random cross-sectional split (1,460 rows) and
models `log1p(SalePrice)`. This validation does **not** test temporal regime
robustness; Phase 4 pays that debt with the full Ames 2006-2010 time ordering.

All feature learning is inside sklearn pipelines/ColumnTransformers fitted within
CV folds. Dollar predictions use Duan smearing to correct log retransformation
bias. The primary strawman artifact is `models/phase1/baseline_lightgbm.joblib`;
the metric card is `reports/phase1_metric_card.json`.

**Phase 1 framing hypothesis:** A typical fix-and-flip net margin is on the order
of $10-20K; this model's typical dollar error is $9,413. If that error is
comparable to or larger than that margin, point predictions cannot safely
underwrite a flip.

---

## Repository Structure

```
margin-of-error/
├── config/          # economics.yaml, model.yaml — all versioned assumptions
├── data/            # gitignored; README explains how to obtain files
├── docs/            # decisions.md (ADR log), assumptions.md
├── notebooks/       # Exploration only; logic lives in src/
├── reports/         # Generated figures + strategy memo
├── src/margin_of_error/
│   ├── config.py    # Pydantic config loaders
│   ├── data/        # Loaders, data dictionary parser, pandera schemas, cleaning
│   ├── features/    # Feature engineering; mutable vs. fixed feature registries
│   ├── models/      # Baseline + CQR conformal interval model
│   ├── economics/   # Flip P&L simulation
│   ├── causal/      # DML estimation of renovation effects
│   ├── backtest/    # Temporal walk-forward (2006–2010)
│   ├── viz/         # Signature charts
│   └── app/         # Streamlit underwriting tool
└── tests/
```

---

## Reproducibility

- Global seeds set in `config/model.yaml` and propagated via `src/margin_of_error/config.py`
- All experiments can be reproduced from a clean clone: `make setup && make data-check && make train`
- Pinned dependencies via `requirements-lock.txt` (generated by `make setup`)

---

## License

Code: MIT. Data: see [data/README.md](data/README.md) for dataset-specific terms.
