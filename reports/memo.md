# Strategy Memo: Margin of Error
## Uncertainty-Aware Valuation for Fix-and-Flip Underwriting

*[SKELETON — populated after Phase 5 completion]*

**Prepared for:** [Audience]
**Date:** [TBD]
**Status:** Draft skeleton — results are placeholders

---

## Executive Summary

*[One paragraph: the Zillow framing, the central finding, the recommendation.
Populated after Phase 4 backtest is complete.]*

**PLACEHOLDER THESIS:** A conventional AVM achieves [TBD] RMSE on the Ames
dataset, which sounds precise. For a typical $180,000 home, this implies a
$[TBD] dollar error band. An investor targeting a 10% profit margin ($18,000)
needs the model to be wrong by less than half the margin of error to underwrite
safely. We show [TBD]% of apparently profitable deals fail this test.

---

## 1. The Problem with Point Estimates

*[Expand on the Zillow Offers case study. Use Phase 1 residual analysis numbers.]*

---

## 2. Our Approach: Interval Valuation + Underwriting Rules

*[Describe CQR methodology in plain English. Reference Phase 2 chart: "good buys"
with margin inside uncertainty band.]*

---

## 3. What Can Actually Be Renovated?

*[Phase 3 causal findings: compare naive regression coefficients vs. DML estimates
for kitchen, bath, basement. Highlight confounding from OverallQual.]*

| Feature | Naive Effect ($/unit) | Causal Effect ($/unit) | Bias |
|---|---|---|---|
| Kitchen quality upgrade | TBD | TBD | TBD |
| Full bathroom addition | TBD | TBD | TBD |
| Basement finish (per sqft) | TBD | TBD | TBD |

---

## 4. How the Rule Performed in the 2006–2010 Crash

*[Phase 4 backtest narrative: what would have happened to an investor using
our underwriting rule during the housing crisis? Show equity/decision curve
over time. Compare to naive "buy everything below predicted price" strategy.]*

---

## 5. Recommendation

*[Single-slide "so what." Populated after Phase 5.]*

1. Never underwrite a flip without a calibrated prediction interval.
2. The decision rule is: underwrite only if `predicted_margin > 1.0 × uncertainty_band`.
3. [TBD additional recommendations from Phase 3 and 4 findings.]

---

## Appendix: Economic Assumptions

See [docs/assumptions.md](../docs/assumptions.md) for the full assumption table.
All parameters are reviewable in [config/economics.yaml](../config/economics.yaml).
