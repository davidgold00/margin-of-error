# Strategy Memo: Margin of Error

**Audience:** an investment committee deciding how to underwrite fix-and-flip
or iBuyer-style home acquisitions with a valuation model in the loop.
**Format:** situation → complication → question → answer → evidence. Every
number is traceable to a metric card under `reports/`; the deep walkthrough is
`docs/PROJECT_EXPLAINER.md`.

---

**Recommendation.** Do not underwrite home purchases from point estimates.
Price every deal with a calibrated 90% value interval, decline any deal whose
interval is wider than the allowable uncertainty band, and treat causal
renovation effects as upside guidance only after the uncertainty gate is
passed. In the walk-forward 2007–2010 backtest, this rule cut a thin-margin
iBuyer's maximum drawdown from $129,522 to $21,257 (−84%) and raised the
crash-window hit rate from 76.6% to 88.1% — at the cost of buying fewer homes
and earning less total profit in a mild downturn. That trade is insurance,
and it is the right trade for any buyer operating near model value.

## Situation

The Zillow Offers lesson is not merely that an automated valuation model can
be wrong. The deeper failure is treating a single price estimate as if it
were a capital-allocation decision. A house "worth $180,000" may be a
perfectly reasonable model prediction — and still not underwritable if the
honest 90% range spans tens of thousands of dollars while the deal margin is
a few thousand. There is also a structural trap: when you offer to buy at
model price, the sellers who accept fastest are the ones whose homes the
model has overvalued. The model's mistakes select themselves into the
portfolio.

This project reframes the classic Ames housing dataset from a leaderboard
regression exercise into a decision-under-uncertainty problem: value range,
profit distribution, decision rule, causal renovation guidance, and a
walk-forward stress test through a real downturn.

## Complication

**The model is respectable and still dangerous.** The Phase 1 LightGBM
baseline misses a typical house by $9,413; one house in five is missed by
more than $22,193, and dollar RMSE is $28,500 ± $6,381. Against a
$10,000–$20,000 flip profit target, these are not academic misses — they are
deal-sized.

**Forced to state its uncertainty, the model confesses.** The Phase 2
conformal interval is honest — a promised 90% interval covered 90.4% of 292
held-out homes — but the median interval is $64,025 wide (mean $83,623),
nearly 40% of the median Ames price. The underwriting rule approved 127
homes, referred 1, and declined 164; every decline had the same cause, model
uncertainty exceeding the $60,000 cap. Most striking: the 50 homes a naive
point-estimate ranking liked best were declined 50 out of 50 times.

## Question

How should an investor underwrite acquisitions when the model is accurate on
average but uncertain on individual homes — and which renovations should be
allowed to influence the decision?

## Answer

Adopt a three-part Monday-morning rule:

1. **Price with a calibrated interval, not a point estimate.** Uncertainty
   claims must be audited on held-out data, not asserted.
2. **Buy only when the profit distribution clears the bar.** Approval
   requires ≥65% probability of clearing a $15,000 profit buffer, ≤20%
   probability of loss, and a 90% value interval no wider than $60,000. A
   too-wide interval is an automatic decline regardless of expected profit.
3. **Use causal renovation effects as upside guidance, never as a rescue.**
   The Phase 3 estimates improve renovation planning, but no renovation story
   overrides a wide valuation interval — in testing, swapping causal for
   correlational assumptions flipped zero verdicts.

Who needs this rule depends on posture. A conservative 70%-rule flipper
survives a mild Ames downturn on gross margin alone — the backtest shows $0
drawdown with or without the gate. A thin-margin buyer at 85% of predicted
value has no such cushion: there, the gate bought fewer homes (418 vs 597),
earned less total profit ($4.95M vs $12.65M) because it passed on marginal
deals, and in exchange cut maximum drawdown by $108,265 and raised hit rate
by roughly 11 percentage points. The closer you buy to model value, the more
the model's margin of error is your capital at risk.

## Evidence

**Phase 1 — point-model risk.** Typical absolute dollar error $9,413; 80th
percentile $22,193. Figure: `reports/figures/01_price_error_vs_home_price.svg`.
Source: `reports/phase1_metric_card.json`.

**Phase 2 — the uncertainty gate.** 90.4% empirical coverage on a 90%
promise; median interval width $64,025; 56.2% of test homes declined, all for
excess uncertainty; 50 of the top 50 naive picks rejected. Figures:
`reports/figures/02a_confrontation.png`, `02b_zillow_trap.png`,
`02d_calibration.png`. Source: `reports/phase2_metric_card.json`.

**Phase 3 — renovation causality.** Exterior quality is the flagship
confounding example: $425 per step naive vs $5,634 causal (a 13× hidden
effect). Kitchen quality $4,146 vs $4,450; basement full bath $9,581 vs
$8,520. Zero verdict flips across representative homes. Figures:
`reports/figures/03a_confounding_gap.png`,
`03b_renovation_decision_matrix.png`. Source:
`reports/phase3_metric_card.json`.

**Phase 4 — temporal stress test.** Ames declined mildly (median $165,125 in
2007 → $155,000 in 2010; −6.1%). Interval coverage held: 90.1% pre-downturn
→ 89.5% in-downturn. Conservative 70% regime: $0 drawdown for naive and
gated alike. iBuyer regime: naive $129,522 max drawdown and 76.6%
crash-window hit rate; gated $21,257 and 88.1%. Figures:
`reports/figures/04a_coverage_over_time.png`,
`04b_three_strategies_pnl.png`, `04c_conservative_regime_pnl.png`. Source:
`reports/phase4_metric_card.json`.

## Risks & Limitations

Ames is a single small market whose 2007–2010 decline was 6.1%, not a
national-crash proxy. The backtest uses synthetic acquisition prices and
observed sale price as realized resale value, because the dataset contains
sales, not true buy-renovate-resell projects — Phase 4 stress-tests the
underwriting rule, not a literal P&L. Phase 3 is observational DML, not a
randomized experiment; unobserved owner quality, maintenance, and
micro-location can bias effects. Renovation costs are documented
national-average assumptions, not local bids. Every assumption is catalogued
with a sensitivity flag in `docs/assumptions.md`.

The recommendation survives these caveats because it is a governance rule,
not a claim about one model: **when uncertainty is larger than margin, do not
buy.** That rule gets more valuable — not less — as markets, models, and
assumptions get shakier.

---

*Follow-up option: this memo and `reports/deck_outline.md` can be exported to
`.docx` / `.pptx` if presentation-ready files are needed.*
