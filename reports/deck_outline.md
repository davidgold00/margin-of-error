# Deck Outline: Margin of Error

## Slide 1 - Accurate Is Not Underwritable

**So what:** A model can predict prices well and still be too uncertain to bet on.

**Visual:** Full-bleed thesis slide with the title line and a small callout:
"The model's margin of error must be smaller than the deal's profit margin."

## Slide 2 - The Zillow Offers Lesson

**So what:** Thin-margin automated buying turns model error into capital risk.

**Visual:** Simple setup diagram: point estimate -> purchase decision -> resale
risk, with the missing interval highlighted.

## Slide 3 - The Baseline Model Looks Respectable

**So what:** Phase 1 LightGBM is not a bad model; its dollar errors are just large
enough to matter.

**Visual:** `reports/figures/01_price_error_vs_home_price.svg`

## Slide 4 - The Interval Is the Product

**So what:** Phase 2's 90% interval covered 90.4% of test homes, but the median
range was $64,025 wide.

**Visual:** `reports/figures/02d_calibration.png` plus a one-line metric callout.

## Slide 5 - The Naive "Best Deals" Fail the Gate

**So what:** The top 50 point-estimate opportunities were all rejected once model
uncertainty was priced.

**Visual:** `reports/figures/02a_confrontation.png`

## Slide 6 - Renovation Correlations Can Mislead

**So what:** Exterior quality looked like only $425 per step in naive OLS but
$5,634 after DML controls.

**Visual:** `reports/figures/03a_confounding_gap.png`

## Slide 7 - Causal Lift Still Has to Beat Cost

**So what:** DML uplifts are decision inputs, not permission to ignore the value
interval.

**Visual:** `reports/figures/03b_renovation_decision_matrix.png`

## Slide 8 - Signature Chart: Three Strategies Through the Downturn

**So what:** Under thin-margin iBuyer pricing, uncertainty-aware underwriting cut
max drawdown from $129,522 to $21,257 and raised crash hit rate from 76.6% to
88.1%.

**Visual:** `reports/figures/04b_three_strategies_pnl.png`

## Slide 9 - When the Rule Matters

**So what:** The conservative 70% rule had $0 drawdown because margin absorbed the
mild Ames downturn; the iBuyer regime exposed model error.

**Visual:** `reports/figures/04c_conservative_regime_pnl.png` paired with a small
table comparing conservative versus iBuyer drawdowns.

## Slide 10 - Recommendation and Decision Rule

**So what:** Underwrite only when calibrated uncertainty fits inside the economics.

**Visual:** Decision rule ladder: value interval -> profit distribution ->
APPROVE / REFER / DECLINE. Include the final rule: decline if the 90% interval is
wider than $60,000, loss probability is too high, or the margin buffer is not met.

Follow-up option: this outline can be turned into a `.pptx` deck after review.
