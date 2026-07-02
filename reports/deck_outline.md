# Deck Outline: Margin of Error

A ten-slide presentation skeleton. Each slide has the point it must land
("so what"), the visual, and speaker notes with the exact numbers to say out
loud. All numbers trace to the metric cards under `reports/`.

---

## Slide 1 — Accurate Is Not Underwritable

**So what:** A model can predict prices well and still be too uncertain to
bet on.

**Visual:** Full-bleed thesis slide with the title line and one callout:
"The model's margin of error must be smaller than the deal's profit margin."

**Speaker notes:** Open with the punchline, not the dataset. This talk is not
about a better price model; it is about the difference between a prediction
and a decision. Promise the audience one number to remember: a model that
misses a typical house by $9,413 produced an honest value range $64,025 wide.

## Slide 2 — The Zillow Offers Lesson

**So what:** Thin-margin automated buying turns model error into capital
risk.

**Visual:** Simple flow diagram: point estimate → purchase decision → resale
risk, with the missing uncertainty interval highlighted in red.

**Speaker notes:** Zillow Offers (shut down 2021) lost hundreds of millions
not because its model was unusually bad, but because a single number was
treated as a buying decision. Mention adverse selection: sellers whose homes
the model *overvalues* accept offers most eagerly, so the model's mistakes
select themselves into the portfolio. This project rebuilds that decision the
right way on Ames, Iowa data.

## Slide 3 — The Baseline Model Looks Respectable

**So what:** Phase 1's LightGBM is not a bad model; its dollar errors are
just large enough to matter.

**Visual:** `reports/figures/01_price_error_vs_home_price.svg`

**Speaker notes:** Log-RMSE 0.135 — respectable by leaderboard standards. In
dollars: typical absolute miss $9,413; one house in five missed by more than
$22,193; one in twenty by more than $45,283. Against a $10k–$20k flip profit
target, the typical error is the size of the entire profit. "Accurate" and
"safe to buy" are different properties.

## Slide 4 — The Interval Is the Product

**So what:** The calibrated 90% interval covered 90.4% of held-out homes —
the honesty was audited — and the median range was $64,025 wide.

**Visual:** `reports/figures/02d_calibration.png` plus one metric callout.

**Speaker notes:** Explain Conformalized Quantile Regression in one breath:
quantile models draw the range, a held-out calibration set corrects it, and
the corrected promise is then verified on a test set — 90% promised, 90.4%
delivered. Then the shock: the honest range is nearly 40% of the median Ames
house price. The Phase 1 model was never lying; it was never asked to tell
the whole truth.

## Slide 5 — The Naive "Best Deals" Fail the Gate

**So what:** The 50 best point-estimate opportunities were rejected 50 out
of 50 times once uncertainty was priced.

**Visual:** `reports/figures/02a_confrontation.png`

**Speaker notes:** The underwriting rule — ≥65% chance of clearing a $15k
buffer, ≤20% chance of loss, interval no wider than $60k — declined 164 of
292 homes, every one for excess uncertainty. The deals a point model loves
most are where predicted value most exceeds price, which is as often model
error as genuine bargain. This is the Zillow trap, quantified.

## Slide 6 — Renovation Correlations Can Mislead

**So what:** Exterior quality looked worth $425 per step in naive OLS but
$5,634 after causal (DML) controls — confounding hid a real effect 13-fold.

**Visual:** `reports/figures/03a_confounding_gap.png`

**Speaker notes:** One-breath DML: predict price from fixed traits, predict
the renovation feature from the same traits, regress leftover on leftover —
compare like with like. Note the direction surprise: confounding can *mask*
value, not just inflate it (basement baths went the other way: $9,581 naive
vs $8,520 causal). Caveat honestly: observational, not a randomized trial.

## Slide 7 — Causal Lift Still Has to Beat Cost — and the Gate

**So what:** Better renovation math flipped zero underwriting verdicts;
valuation uncertainty dominates.

**Visual:** `reports/figures/03b_renovation_decision_matrix.png`

**Speaker notes:** Swapping causal for correlational uplifts across
representative homes changed 0 of 10 verdicts. A few thousand dollars of
smarter renovation assumptions cannot rescue a deal wrapped in a $64k value
range. Causality guides the renovation plan; the uncertainty gate governs
the purchase.

## Slide 8 — Signature Chart: Three Strategies Through the Downturn

**So what:** Under thin-margin iBuyer pricing, the uncertainty gate cut max
drawdown from $129,522 to $21,257 and raised crash hit rate from 76.6% to
88.1%.

**Visual:** `reports/figures/04b_three_strategies_pnl.png`

**Speaker notes:** Walk-forward backtest, 2,930 sales, 2007–2010, no time
travel: train on the past only, retrain annually, 43 monthly periods. Say the
uncomfortable part out loud: the gated strategy earned less total profit
($4.95M vs $12.65M) because it declined marginal deals that mostly worked out
*in a 6% downturn*. That is insurance — an 84% drawdown reduction — and in a
Phoenix-2008-sized decline those marginal deals are what bankrupts you.

## Slide 9 — When the Rule Matters

**So what:** The conservative 70% rule showed $0 drawdown with or without the
gate; the 85%-of-value iBuyer regime is where discipline becomes existential.

**Visual:** `reports/figures/04c_conservative_regime_pnl.png` paired with a
small conservative-vs-iBuyer drawdown table.

**Speaker notes:** This contrast is the thesis, not a footnote: uncertainty
discipline is worth almost nothing to a fat-margin flipper in a mild downturn
and everything to a thin-margin buyer. The value of the gate scales with how
close your purchase price sits to model value. Also note calibration held
through the downturn: coverage 90.1% → 89.5%.

## Slide 10 — Recommendation and Decision Rule

**So what:** Underwrite only when calibrated uncertainty fits inside the
deal's economics.

**Visual:** Decision ladder: calibrated value interval → simulated profit
distribution → APPROVE / REFER / DECLINE. Final rule on screen: decline if
the 90% interval exceeds $60,000, if loss probability exceeds the cap, or if
the margin buffer is not cleared.

**Speaker notes:** Close on governance, not modeling: this rule survives
every caveat (one market, mild downturn, synthetic acquisitions,
observational causality) because it does not depend on any one model being
right — only on refusing to bet when the model admits it doesn't know. End
with the tagline: the margin of error must be smaller than the margin on the
deal.

---

*Follow-up option: this outline can be turned into a `.pptx` deck after
review.*
