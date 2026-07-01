# Strategy Memo: Margin of Error

**Recommendation.** Do not underwrite flips from point estimates. Use a calibrated
90% value interval, decline deals whose uncertainty is wider than the allowable
band, and use causal renovation effects only after the uncertainty gate is passed.
For thin-margin iBuyer-style purchases, this rule improved the 2008-2010 crash hit
rate from 76.6% to 88.1% and cut max drawdown from $129,522 to $21,257 in the
walk-forward backtest.

## Situation

The Zillow Offers lesson is not merely that an automated valuation model can be
wrong. The deeper failure is treating a single price estimate as if it were a
capital-allocation decision. A house valued at $180,000 may be a reasonable model
prediction, but it is not underwritable if the honest 90% range spans tens of
thousands of dollars and the deal margin is only a few thousand dollars.

This project reframes Ames housing from a leaderboard regression task into a
decision-under-uncertainty problem: value range, profit distribution, decision
rule, renovation guidance, and walk-forward stress test.

## Complication

Phase 1 produced a respectable LightGBM baseline, but its dollar errors are large
relative to flip margins. The typical absolute error was $9,413, the 80th
percentile absolute error was $22,193, and the dollar RMSE was $28,500 +/- $6,381.
Those are not academic misses; they are deal-sized misses.

Phase 2 made the model state its uncertainty. The 90% CQR interval achieved 90.4%
coverage on 292 held-out homes, but the median interval was $64,025 wide and the
mean was $83,623. The underwriting rule approved 127 homes, referred 1, and
declined 164; every decline was caused by excess model uncertainty. The top 50
homes a naive point model liked were all declined by the uncertainty gate.

## Question

How should a fix-and-flip investor underwrite acquisitions when the model is
accurate on average but uncertain on individual homes, and which renovations
should influence the decision?

## Answer

Adopt a three-part Monday rule:

1. Price with a calibrated interval, not a point estimate.
2. Buy only when the profit distribution clears the margin buffer and the 90%
   interval is no wider than the $60,000 uncertainty cap.
3. Use the Phase 3 causal renovation effects as upside guidance, but do not let a
   renovation story override a wide valuation interval.

In practice, this means a conservative 70%-rule flipper can survive a mild Ames
downturn because the gross margin is large. A thin-margin iBuyer cannot rely on
that cushion. Under the 85%-of-ARV iBuyer regime, the uncertainty-aware rule bought
fewer homes than the naive point rule, earned lower total profit because it passed
on many deals, but materially improved downside quality: max drawdown fell by
$108,265 and crash-window hit rate rose by 11.5 percentage points.

## Evidence

**Phase 1 - point-model risk.** LightGBM's typical absolute dollar error was
$9,413 and its 80th percentile error was $22,193. Figure:
`reports/figures/01_price_error_vs_home_price.svg`.

**Phase 2 - uncertainty gate.** The calibrated 90% interval hit 90.4% empirical
coverage, but the median interval width was $64,025. The gate declined 56.2% of
test homes and rejected 50 of the top 50 naive picks. Figures:
`reports/figures/02a_confrontation.png`,
`reports/figures/02b_zillow_trap.png`,
`reports/figures/02d_calibration.png`.

**Phase 3 - renovation causality.** Exterior quality is the clearest confounding
example: naive OLS estimated $425 per step, while DML estimated $5,634, a $5,208
understatement. Kitchen quality was $4,146 naive versus $4,450 causal, and
basement full baths were $9,581 naive versus $8,520 causal. Figures:
`reports/figures/03a_confounding_gap.png`,
`reports/figures/03b_renovation_decision_matrix.png`.

**Phase 4 - temporal stress test.** Ames prices fell mildly, from a 2007 median of
$165,125 to a 2010 median of $155,000 (-6.1%). Coverage moved from 90.1% pre-crash
to 89.5% in-crash. Under conservative 70%-rule pricing, both naive and
uncertainty-aware rules had $0 max drawdown. Under the iBuyer regime, naive point
underwriting had $129,522 max drawdown and 76.6% crash hit rate; the
uncertainty-aware rule had $21,257 max drawdown and 88.1% crash hit rate. Figures:
`reports/figures/04a_coverage_over_time.png`,
`reports/figures/04b_three_strategies_pnl.png`,
`reports/figures/04c_conservative_regime_pnl.png`.

## Risks & Limitations

Ames is a single small market, and the 2008-2010 decline in the data was only
6.1%, not a national crash proxy. The backtest uses synthetic acquisition prices
and observed sale price as realized ARV because the dataset does not contain true
buy-renovate-resell pairs. Phase 3 is observational DML, not a randomized
renovation experiment, so unobserved owner quality, maintenance, and micro-location
can still bias effects. Renovation costs are documented assumptions, not local
contractor bids.

The recommendation survives those caveats because it is a governance rule, not a
claim about one magic model: when uncertainty is larger than margin, do not buy.

Follow-up option: this markdown memo and the deck outline can later be exported to
`.docx` and `.pptx` if presentation-ready files are needed.
