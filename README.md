# Margin of Error

> Accurate is not underwritable. The model's margin of error must be smaller than
> the deal's profit margin.

Margin of Error is an uncertainty-aware underwriting system for fix-and-flip and
iBuyer-style home acquisitions. It starts with a normal Ames Housing price model,
then turns it into a decision engine: calibrated 90% value intervals, simulated
profit distributions, causal renovation guidance, and a walk-forward stress test
through the 2007-2010 Ames downturn. The Zillow Offers framing is deliberate:
buying homes near model value is dangerous when the model's uncertainty is wider
than the margin on the deal.

For the deep plain-English walkthrough, read
[docs/PROJECT_EXPLAINER.md](docs/PROJECT_EXPLAINER.md).

![Three strategies through the downturn](reports/figures/04b_three_strategies_pnl.png)

## Results at a Glance

| Question | Result | Source |
|---|---:|---|
| How wrong is the Phase 1 point model in dollars? | Typical absolute error $9,413; 80th percentile $22,193 | `reports/phase1_metric_card.json` |
| Is the 90% interval calibrated? | 90.4% empirical coverage on 292 held-out homes | `reports/phase2_metric_card.json` |
| How wide is the honest value range? | Median 90% interval width $64,025 | `reports/phase2_metric_card.json` |
| What does the uncertainty gate do? | Declines 164 of 292 homes; rejects 50 of the top 50 naive picks | `reports/phase2_metric_card.json` |
| Biggest causal-vs-naive renovation gap? | Exterior quality: $425 naive vs $5,634 DML causal | `reports/phase3_metric_card.json` |
| Did Ames crash hard? | Median price fell 6.1% from 2007 peak to 2010 trough | `reports/phase4_metric_card.json` |
| Thin-margin iBuyer stress test | Naive max drawdown $129,522; uncertainty-aware max drawdown $21,257 | `reports/phase4_metric_card.json` |
| Crash hit rate under iBuyer pricing | Naive 76.6%; uncertainty-aware 88.1% | `reports/phase4_metric_card.json` |

## What Is in the Repo

| Phase | Deliverable |
|---|---|
| 1 | Baseline LightGBM valuation model with dollar residual diagnostics |
| 2 | Conformalized Quantile Regression intervals, profit simulation, underwriting rule |
| 3 | Double Machine Learning estimates for renovatable features |
| 4 | Walk-forward temporal backtest with conservative flip and thin-margin iBuyer regimes |
| 5 | Streamlit underwriting tool, project explainer, strategy memo, deck outline |

Key artifacts:

- App: [src/margin_of_error/app/underwriting.py](src/margin_of_error/app/underwriting.py)
- Explainer: [docs/PROJECT_EXPLAINER.md](docs/PROJECT_EXPLAINER.md)
- Strategy memo: [reports/memo.md](reports/memo.md)
- Deck outline: [reports/deck_outline.md](reports/deck_outline.md)
- Decisions: [docs/decisions.md](docs/decisions.md)
- Assumptions: [docs/assumptions.md](docs/assumptions.md)

## Underwriting Tool

The Streamlit app loads the saved Phase 1 point model, saved Phase 2 CQR interval
model, Phase 3 causal uplift configuration, and Phase 5 feature-default profile.
It exposes the property inputs that matter most for a practical underwriting
screen: neighborhood, living area, overall quality, year built, baths, kitchen
quality, basement area, garage spaces, and garage finish. The rest of the Ames
feature vector is filled from dataset medians and modes.

The output is designed for a portfolio screenshot: point valuation, 90% interval,
profit distribution, APPROVE / REFER / DECLINE verdict, causal renovation guidance,
and an assumptions expander showing the economics config.

## Run It

```bash
# 1. Create the environment
make setup

# 2. Add raw data files per data/README.md
make data-check

# 3. Reproduce the pipeline
make train
make uncertainty
make causal
make backtest
make app-artifacts

# 4. Run quality gates
make lint
make test

# 5. Launch the app
make app
```

`make all` runs the full non-interactive pipeline: data check, model phases,
app artifact build, lint, and tests. `make app` launches the Streamlit tool.

## Project Structure

```text
margin-of-error/
├── config/          # model and economics assumptions
├── data/            # raw files are gitignored; see data/README.md
├── docs/            # explainer, decisions, assumptions
├── models/          # saved Phase 1 and Phase 2 app artifacts
├── notebooks/       # narrative notebooks; production logic lives in src/
├── reports/         # metric cards, memo, deck outline, generated figures
├── src/margin_of_error/
│   ├── app/         # Phase 5 Streamlit tool and artifact loaders
│   ├── backtest/    # Phase 4 walk-forward stress test
│   ├── causal/      # Phase 3 DML estimation
│   ├── economics/   # profit simulation and verdict rule
│   ├── models/      # Phase 1 baseline and Phase 2 CQR
│   └── viz/         # signature charts
└── tests/
```

## Data

Phases 1-3 use Kaggle's random Ames competition training split (1,460 rows).
Phase 4 uses the full De Cock Ames dataset (2,930 sales, 2006-2010) sorted by
`YrSold` and `MoSold`. Raw data is not committed; see
[data/README.md](data/README.md) for download instructions.

## Caveats

This is a decision-system portfolio project, not a live investment product. Ames
is one market, the renovation costs are documented assumptions, the causal layer
is observational, and the backtest uses synthetic acquisition prices rather than
true buy-renovate-resell pairs. Those caveats are part of the story and are
spelled out in the explainer.
