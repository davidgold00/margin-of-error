# Margin of Error — The Complete Project Explainer

This document is the deep, plain-English walkthrough of the entire project. It
assumes you are curious and technical but takes nothing for granted: every
concept is explained from scratch, every number is interpreted, and every
claim points to the file that proves it. If you read only one document in this
repository, read this one.

**How this document is organized:**

- **Section A** — the whole project in one paragraph.
- **Section B** — the problem, restated so a non-specialist gets it.
- **Section C** — the concepts you need (log targets, conformal prediction,
  Monte Carlo, confounding, walk-forward testing), each in plain English.
- **Section D** — what each phase did: the question, the technique, the real
  numbers, what they mean, and the interview question you should expect.
- **Section E** — how to access every result yourself, file by file.
- **Section F** — the throughline: how five phases compose into one argument.
- **Section G** — the honest limitations, framed as attack-and-response.
- **Section H** — the numbers sheet (every figure with its source file).
- **Section I** — glossary.

---

## SECTION A — The One-Paragraph Version

I built an uncertainty-aware home-flip underwriting system inspired by the
Zillow Offers failure: instead of asking only "can a model predict sale
price?", I asked "is the model certain enough to bet capital on this house?"
On Ames housing data, the Phase 1 LightGBM model looks respectable — but its
typical dollar error is $9,413, and when forced to state an honest, calibrated
90% value range, that range is $64,025 wide at the median: several times most
flip profit targets. The project turns that insight into a decision engine:
audited value intervals, simulated profit distributions, a causal renovation
layer, and a walk-forward backtest through the 2007–2010 downturn showing
that the uncertainty discipline cut a thin-margin buyer's worst losing streak
by 84%. The memorable line: **an accurate price model is not an underwriting
system until its margin of error is smaller than the deal's profit margin.**

---

## SECTION B — The Problem, Restated Plainly

Most people use the Ames dataset as a leaderboard exercise: minimize the error
metric, brag about the model, stop. That is the trap this project is built
around.

An investor does not earn RMSE. An investor earns or loses **dollars on a
purchase**. Suppose a model says a house is worth $180,000. Is that a good
deal at $160,000? You cannot know from the number alone, because the number
carries no information about *confidence*. The model might be nearly certain,
or wildly guessing — the output looks identical either way. If the honest 90%
range is roughly $150,000 to $214,000, then the point estimate can be
directionally useful and still far too uncertain to underwrite.

That is what happened to Zillow Offers in 2021. Zillow used its valuation
model to buy homes directly, with thin operating margins, and lost hundreds of
millions of dollars — not primarily because the model was inaccurate on
average, but because a single price estimate was treated as a
capital-allocation decision. Worse, there is a built-in trap called **adverse
selection**: when you publicly offer to buy at model price, the sellers who
eagerly accept are disproportionately the ones whose homes your model has
*overvalued*. Your "best deals" are systematically your worst ones.

Reframing valuation as **decision-making under uncertainty** means the model
must output three things a point estimate cannot provide: a calibrated value
*range*, a *profit distribution*, and a *rule for when to walk away*. Building
exactly that — and stress-testing it through a real downturn — is this
project.

---

## SECTION C — The Five Concepts You Need

You can understand every result in this project with five ideas. Each is
worth being able to explain to someone else.

### C.1 Why the model predicts log-price, not price

A $10,000 miss on a $100,000 house (10% off) is much worse than a $10,000 miss
on a $500,000 house (2% off). Housing errors scale like *percentages*, so the
model is trained on `log1p(SalePrice)`: in log space, errors behave like
percentage misses, which is both statistically better-behaved and economically
meaningful. There is a subtle trap on the way back, though: converting an
average log prediction to dollars with `exp()` produces estimates that are
systematically **biased low** (the average of logs is not the log of the
average). The classic fix is **Duan's smearing estimator**, which uses the
model's own residuals to correct the retransformation. This project applies
it; many leaderboard solutions do not.

### C.2 Conformal prediction: uncertainty you can audit

A prediction interval is a statement like "we are 90% sure the true value is
between $150k and $214k." Any model can *claim* that; the question is whether
the claim is true. **Conformalized Quantile Regression (CQR)** builds honest
intervals in two steps:

1. **Quantile regression** — train one model to predict the 5th percentile of
   price and another for the 95th. Together they form a raw interval. Raw
   quantile models are usually overconfident: their "90% interval" might cover
   the truth only 80% of the time.
2. **Conformal calibration** — hold out a *calibration set* the models never
   saw. Measure how badly the raw intervals miss on it, compute the exact
   correction needed, and widen every interval by that amount. Conformal
   prediction carries a finite-sample mathematical guarantee: on future data
   that behaves like the calibration data ("exchangeability"), the corrected
   intervals hit the promised coverage rate.

The key vocabulary: **nominal coverage** is what the interval promises (90%);
**empirical coverage** is what it delivers on held-out data. When the two
match, the model's uncertainty statements are *calibrated* — audited, not
asserted.

One honest caveat to internalize: the guarantee is **marginal** — 90% on
average over homes like the calibration set — not a promise for each
individual house, each neighborhood, or each month. That limitation is exactly
why Phase 4 exists.

### C.3 Monte Carlo profit simulation: from a value range to a decision

A value interval still is not a decision. The bridge is simulation: for each
candidate house, the project simulates the flip **10,000 times**. Each
simulation draws a plausible resale value from the model's uncertainty range,
draws a random holding period (a truncated normal around 4 months — flips run
late), and subtracts realistic costs: the purchase price, ~6% round-trip
transaction costs, 0.8%-per-month carrying costs, and the renovation budget.
The output is not a profit number but a **profit distribution**: "62% chance
of clearing $15k, 18% chance of losing money." Distributions can be compared
against risk rules; single numbers cannot.

### C.4 Confounding and Double Machine Learning

A flipper needs to know: "if I upgrade this kitchen one quality step, how much
does resale value rise?" The naive answer — regress price on kitchen quality —
is poisoned by **confounding**: houses with great kitchens are also in better
neighborhoods, bigger, newer, and better built. The kitchen coefficient
absorbs credit for all of that.

**Double Machine Learning (DML)** de-poisons the estimate in three moves:

1. Use a flexible model to predict **price** from everything *except* the
   renovation feature. Keep the leftover (the part of price the fixed traits
   cannot explain).
2. Use another model to predict the **renovation feature itself** from those
   same fixed traits. Keep that leftover too.
3. Regress leftover on leftover. Whatever relationship survives is the
   association between renovation and price *after* stripping out everything
   explainable by neighborhood, size, age, and build quality — a
   compare-like-with-like estimate.

Two supporting techniques keep it honest: **cross-fitting** (the nuisance
models only predict rows they did not train on, preventing overfitting from
leaking into the causal estimate) and robust (HC3) standard errors for
truthful confidence intervals. DML removes *observed* confounding only —
unmeasured things like owner wealth or contractor quality can still bias the
result, and the docs say so.

### C.5 Walk-forward backtesting: no time travel allowed

A random train/test split quietly assumes the future looks like the past —
data from 2010 helps predict a sale in 2007. Real investing happens *through
time*. A **walk-forward backtest** sorts all sales chronologically, trains
only on data available *before* each month, prices and gates that month's
homes, records realized outcomes, then steps forward (retraining annually
here). The model only ever knows what an investor could have known at the
time. This is the gold standard for "would this have actually worked?"

---

## SECTION D — What Each Phase Actually Did

### Phase 1 — The Baseline Price Model

**The question.** How good is a normal "predict the sale price" model, and —
the part leaderboards never report — how large are its errors *in dollars*?

**What was built.** A fold-safe modeling pipeline on the Kaggle Ames training
set (1,460 sales, median price $163,000) with three models of increasing
sophistication: a dumb median predictor (the sanity floor), ElasticNet (a
regularized linear model), and LightGBM (the gradient-boosted tree ensemble
that is the realistic stand-in for an industry automated valuation model).
Validation was 5-fold cross-validation, repeated 3 times, stratified by
neighborhood, with all imputation learned inside folds to prevent leakage.
The saved artifact is `models/phase1/baseline_lightgbm.joblib`; the numbers
live in `reports/phase1_metric_card.json`.

**The real result, number by number:**

| Number | Plain meaning |
|---|---|
| log-RMSE 0.135 ± 0.015 | Roughly a "typical ~13–14% miss" — respectable by Kaggle standards |
| Dollar RMSE $28,500 ± $6,381 | The same error in money terms (RMSE punishes large misses extra hard) |
| **Typical absolute error $9,413** | On a random house, expect to be about $9k off |
| **80th percentile error $22,193** | One house in five is missed by *more than* $22k |
| 95th percentile error $45,283 | One house in twenty is missed by more than $45k |

**Why it matters for the decision.** A typical flip in a market like Ames
targets perhaps $10,000–$20,000 of net profit. The model's *typical* error is
roughly the size of that entire profit, and its *routine* bad miss ($22k, one
house in five) can erase the deal completely. Phase 1 is deliberately not a
great model — it is a **respectable model that is still dangerous**, which is
exactly the Zillow situation.

**An honest wrinkle to know cold.** ElasticNet — the linear model — actually
beat LightGBM slightly on log-RMSE (0.126 vs 0.135). The project kept LightGBM
anyway and says so in writing. Why is that defensible? LightGBM was marginally
better in dollar terms, it is the realistic industry-AVM strawman the project
brief required, and the entire point is that leaderboard rank is not the
question. Reporting the inconvenient comparison honestly is itself part of the
portfolio signal.

**Likely interview question.** *"Why use LightGBM if ElasticNet had lower
log-RMSE?"* — I reported that honestly. LightGBM was the required nonlinear
strawman and slightly better on dollar RMSE, and ElasticNet's strength is
evidence that Ames is small and structured enough for linear regularization.
The project's point is not leaderboard victory; it is that even a strong model
carries deal-sized dollar error.

### Phase 2 — Uncertainty-Aware Underwriting (the heart of the project)

**The question.** What happens when the price model is forced to admit its
uncertainty *before* a buy decision is made?

**What was built.** The Phase 1 point model was wrapped with a Conformalized
Quantile Regression 90% value interval (concept C.2), a Monte Carlo flip
profit simulation (concept C.3) using the economics documented in
`config/economics.yaml`, and a three-verdict underwriting rule applied to
every held-out home. The data split: **949 train / 219 calibration / 292
test**.

**Did the interval keep its promise?** Yes, and this is checkable: on the 292
test homes, the "90%" interval contained the true sale price **90.4%** of the
time. The metric card records the whole calibration curve — a 50% interval
covered 53.4%, an 80% interval covered 81.5%, a 95% interval covered 93.2%.
The model's uncertainty statements were audited and passed.

**The devastating finding.** The calibrated 90% interval has a **median width
of $64,025** (mean $83,623). Sit with that: the median Ames house sells for
$163,000, and the truthful answer to "what is it worth?" is a range nearly
40% of the price wide. The Phase 1 model was never lying — it just was never
asked to tell the whole truth.

**The decision rule.** Each home's 10,000-draw profit distribution is scored
against thresholds that all live in `config/economics.yaml`:

- **APPROVE** — at least a 65% chance of clearing a $15,000 profit buffer,
  AND at most a 20% chance of any loss, AND the 90% value interval no wider
  than **$60,000**.
- **REFER** — the gray zone (≥50% chance of clearing the buffer, ≤30% loss
  probability): a human should look.
- **DECLINE** — everything else. Critically, an interval wider than $60k is
  an *automatic* decline regardless of how attractive the expected profit
  looks. That is the anti-Zillow guardrail: **if the model does not know, we
  do not buy.**

**What the gate did.** Of 292 test homes: **127 approved (43.5%), 1 referred,
164 declined (56.2%)** — and every single decline had the same recorded
reason: model uncertainty exceeded the cap. The showstopper: the **50 homes a
naive point-estimate ranking liked best were declined 50 out of 50 times**.
The deals a point model finds most attractive are precisely those where the
gap between predicted value and price is large — which is as often a symptom
of model error as of genuine bargain. That is adverse selection, quantified
on real data.

**Likely interview question.** *"Does conformal prediction guarantee each
individual house is covered?"* — No. It gives finite-sample *marginal*
coverage under exchangeability: the population of future cases should behave
like the calibration cases. It does not promise every neighborhood or every
month is equally calibrated — which is exactly why Phase 4 re-tests the
system through time.

### Phase 3 — The Causal Renovation Layer

**The question.** Which renovation-related features actually appear to *cause*
value lift, rather than merely correlating with nicer houses?

**What was built.** Cross-fitted DML (concept C.4) with LightGBM nuisance
models and HC3 robust standard errors, estimating the per-step dollar effect
of nine renovatable Ames features (kitchen quality, exterior quality, baths,
garage finish, and so on) while controlling for fixed traits: neighborhood,
size, age, overall quality and condition. The estimates were written into
`config/economics.yaml` as `causal_renovation_uplifts`, replacing the Phase 2
correlational priors, and the underwriting comparison was re-run.

**The real results** (naive OLS vs DML causal, dollars per one quality step):

| Renovation | Naive says | DML says | 95% CI (DML) | Story |
|---|---:|---:|---|---|
| **Exterior quality** | $425 | **$5,634** | $2,063–$9,204 | Confounding *hid* a real effect — understated 13× |
| Kitchen quality | $4,146 | $4,450 | $1,824–$7,076 | Roughly honest either way |
| Basement full bath | $9,581 | $8,520 | $5,717–$11,323 | Naive slightly *over*stated |
| Garage finish | $2,641 | $3,524 | $1,619–$5,429 | Moderately understated |

The exterior-quality row is the flagship: confounding does not always inflate
estimates — here it *masked* a real ~$5,600-per-step effect down to $425.
Using the naive number, a flipper would skip a genuinely profitable
improvement. Naive correlations can be wrong in *either direction*, by
thousands of dollars.

**The second finding — a null result that carries the argument.** When the
causal uplifts replaced the correlational ones across 10 representative Phase
2 homes, **zero underwriting verdicts flipped**. Why report a null? Because it
establishes a hierarchy: **valuation uncertainty dominates renovation
assumptions.** When the value interval is $64k wide, a few thousand dollars of
better renovation math cannot rescue a deal. Causality matters for *planning
the renovation*; it does not override the uncertainty gate. The gate rules.

**Likely interview question.** *"How do you know the causal estimates are
right?"* — I do not claim randomized-trial truth. Cross-fitted DML with
flexible nuisance models reduces *observed* confounding; the honest limitation
is the conditional independence assumption. Unobserved owner wealth,
maintenance history, contractor quality, and micro-location can still bias the
estimates, and `docs/assumptions.md` says so explicitly.

### Phase 4 — The Walk-Forward Crash Backtest

**The question.** Does the uncertainty-aware rule behave differently from
naive point underwriting when the market is experienced *through time* —
including a downturn — rather than as shuffled rows?

**What was built.** The full De Cock Ames dataset (2,930 sales, 2006–2010)
sorted by sale year and month, run through a walk-forward backtest (concept
C.5): expanding-window training on past data only, annual retraining, 43
evaluation periods from 2007 through 2010. Three buy gates were raced —
**buy-all** (no gate), **naive point-estimate** underwriting, and the
**uncertainty-aware** rule — under two acquisition regimes:

- **Conservative flip (70% rule):** buy at 70% of predicted after-repair
  value. A ~30% gross cushion — the traditional flipper posture.
- **iBuyer (85% of predicted value):** buy near model price. After the ~9%
  round-trip of transaction and carry costs, only a thin sliver of margin
  remains — the Zillow Offers posture.

**Finding 1 — Ames barely crashed, and the report says so.** Median price
fell from $165,125 (2007) to $155,000 (2010): **−6.1%** peak to trough. Ames
2008 is not Phoenix 2008, and honestly reporting a mild downturn instead of
manufacturing a "crash" is a deliberate credibility choice.

**Finding 2 — calibration held.** Interval coverage was 90.1% before the
downturn and 89.5% during it — a 0.5-point dip below the 90% target, which is
stability, not collapse. (Pedantic note: the metric card's automated
`coverage_collapsed` flag trips because 89.5% is technically below target;
the human-readable conclusion everywhere is that a half-point dip is noise.)
The conformal machinery survived a regime change essentially intact.

**Finding 3 — the money result.** Under the conservative 70% rule, *every*
strategy sailed through with **$0 max drawdown** (max drawdown = the deepest
dip in cumulative profit from any previous peak — the "worst losing streak"
statistic). A 30% cushion trivially absorbs a 6% market move. But under the
thin-margin iBuyer regime:

| Metric | Naive point rule | Uncertainty-aware rule |
|---|---:|---:|
| Deals bought (2007–2010) | 597 | 418 |
| Total profit | $12.65M | $4.95M |
| **Max drawdown** | **$129,522** | **$21,257** |
| Overall hit rate (share of deals profitable) | 77.4% | 88.3% |
| Crash-window hit rate | 76.6% | 88.1% |

**How to interpret this honestly — including the uncomfortable part.** The
uncertainty-aware rule made *less total money* in this particular history. Do
not hide that; it is the mature centerpiece of the story. The naive rule's
extra profit came from also buying 179 riskier homes that *mostly* worked out
this time — in a 6.1% downturn. The uncertainty rule is buying **insurance**:
it cut the worst losing streak by **$108,265 (an 84% reduction)** and raised
the share of winning deals by roughly 11 percentage points. In a
Phoenix-2008-style 30% decline, those extra marginal deals are exactly what
bankrupts a thin-margin buyer — which is, historically, what happened to
Zillow. Risk management always looks like foregone profit until the year it
saves the firm.

**Finding 4 — the regime comparison IS the thesis.** Uncertainty discipline
was worth nearly nothing to the fat-margin flipper and existential to the
thin-margin buyer. The value of the gate depends on how close your purchase
price sits to the model's estimate. A weaker project would have claimed the
gate helps everyone; this one shows precisely *when* it earns its keep.

**Likely interview question.** *"Did the model's coverage collapse in the
crash?"* — Not dramatically: 90.1% to 89.5%, technically below target but
essentially stable. The defensible Phase 4 result is economic, not
statistical: under thin-margin buying, the gate selected fewer deals, raised
hit rate by ~11 points, and cut max drawdown from $129,522 to $21,257.

### Phase 5 — Shipping It

**The question.** Can a person actually use this?

**What was built.** A Streamlit underwriting app
(`src/margin_of_error/app/underwriting.py`, launched with `make app`) that
loads the saved Phase 1 point model, the saved Phase 2 CQR interval model, the
Phase 3 causal uplifts, and a Phase 5 feature-default profile — it never
retrains at runtime. You enter the property facts an underwriter would
actually have (neighborhood, living area, overall quality, year built, baths,
kitchen quality, basement area, garage spaces, garage finish); the remaining
Ames features fill from dataset medians and modes. The output is a complete
underwriting screen: point valuation, 90% interval, profit distribution,
**APPROVE / REFER / DECLINE** verdict with its reason, causal renovation
guidance, and an expander disclosing every economic assumption. Missing
artifacts produce a clear "run these make targets" message, not a stack
trace. Phase 5 also produced this explainer, the strategy memo
(`reports/memo.md`), and the deck outline (`reports/deck_outline.md`).

---

## SECTION E — How to Access Every Result Yourself

Nothing in this project asks to be taken on faith. Here is where each claim
lives and how to open it. All paths are relative to the repo root; none of
these steps require running the pipeline.

**Step 1 — read the metric cards.** Each phase writes one JSON card with its
headline numbers, the exact config snapshot it ran under, and a timestamp:

```bash
python3 -m json.tool reports/phase1_metric_card.json   # dollar-error percentiles
python3 -m json.tool reports/phase2_metric_card.json   # coverage, widths, verdicts
python3 -m json.tool reports/phase3_metric_card.json   # naive vs causal effects
python3 -m json.tool reports/phase4_metric_card.json   # backtest strategies & regimes
```

Reading guide: in the Phase 2 card, `cqr.calibration_curve` is the honesty
audit (promised vs delivered coverage at six levels) and `headline` holds the
verdict counts and the 50-of-50 rejection. In the Phase 4 card,
`strategies.ibuyer` and `strategies.conservative_flip` each contain
`buy_all` / `naive_point` / `uncertainty_aware` blocks with deals, profit,
drawdown, and hit rates — the two-regime contrast in Section D is read
directly off those blocks.

**Step 2 — look at the figures** (`reports/figures/`, numbered by phase):
`01_…` shows dollar errors across the price range; `02a–02d` show the interval
confrontation, the Zillow trap, neighborhood approvals, and the calibration
audit; `03a–03c` show the confounding gap, the renovation decision matrix, and
why verdicts did not flip; `04a–04c` show coverage through time, the signature
three-strategies P&L race, and the fat-margin regime where nothing draws down.

**Step 3 — drill into the per-home tables** (`reports/*.csv`):
`phase1_oof_residuals.csv` (every out-of-fold error behind the percentiles),
`phase2_test_underwriting.csv` (all 292 homes with intervals, profit summary
statistics, verdicts, and decline reasons), `phase3_causal_effects.csv` (the
full DML table with confidence intervals), and `phase4_backtest_periods.csv`
(the month-by-month backtest ledger).

**Step 4 — check the assumptions behind any number.** Every economic
parameter lives in `config/economics.yaml` with an inline rationale, is
catalogued with source and sensitivity flag in `docs/assumptions.md`, and the
judgment calls behind the design are logged chronologically in
`docs/decisions.md` (the ADR log). If you disagree with an assumption, change
the YAML value and re-run — a test enforces that no dollar constants hide in
code.

**Step 5 — reproduce it.** `make setup`, add the two data files per
`data/README.md`, then `make all` (or phase by phase: `train`, `uncertainty`,
`causal`, `backtest`, `app-artifacts`). Runs are seeded and dependencies
pinned, so the metric cards regenerate to the same values. `make app`
launches the underwriting tool.

---

## SECTION F — The Throughline

The five phases compose into a single argument:

1. **Phase 1** shows that a respectable price model still misses by ~$9k on a
   typical house and >$22k on one house in five — deal-sized errors.
2. **Phase 2** forces the model to state honest, audited uncertainty and finds
   the median 90% range is $64k wide — several times a flip margin — so a
   disciplined rule declines 56% of homes, including all 50 of the naive
   model's favorite picks.
3. **Phase 3** shows renovation ROI needs causal discipline (naive correlation
   understated exterior-quality value 13-fold) — and simultaneously that no
   renovation story overrides a wide valuation interval.
4. **Phase 4** replays history without time travel and locates exactly where
   the discipline pays: nowhere for a fat-margin flipper in a mild downturn,
   and decisively for a thin-margin iBuyer, where it cut max drawdown 84% at
   the cost of lower total profit — insurance, priced and demonstrated.
5. **Phase 5** ships it as a tool a human can use, with every assumption on
   display.

The final product is not a better Kaggle model. It is a **safer decision
system**, governed by one rule: *when the model's margin of error exceeds the
deal's profit margin, do not buy.*

---

## SECTION G — The Honest Limitations (attack and response)

| Attack | Disarming response |
|---|---|
| Ames is one small market, not the national iBuyer market. | Correct. The results are a decision-system demonstration, not a national housing law. The temporal dataset has 2,930 sales and a mild 6.1% decline, reported plainly. |
| The Kaggle split is random, so Phases 1–3 may look too stable. | Correct, and by design: Phases 1–3 answer cross-sectional questions on the random 1,460-row split, and Phase 4 explicitly switches to the full 2006–2010 temporal ordering to answer the through-time question. |
| The causal layer depends on the conditional independence assumption. | Correct. DML reduces *observed* confounding; it cannot eliminate unobserved maintenance, owner wealth, contractor quality, or micro-location. The estimates are decision-grade observational effects, not randomized-trial truth. |
| Renovation costs are estimates. | Correct. Figures like $27,000 for a minor kitchen remodel are documented national-average assumptions (Remodeling Magazine), not Ames contractor invoices, and are flagged as placeholders in `docs/assumptions.md`. |
| The backtest uses synthetic acquisition prices, not true buy-resell pairs. | Correct. Ames records sale transactions, not flip projects, so Phase 4 is an underwriting-*rule* stress test using observed sale price as realized resale value. A production backtest needs real flip transaction data. |
| The iBuyer regime's 85% purchase factor is arbitrary. | It is a documented scenario representing thin-margin buying near model value (ADR-025 explains why 0.90 was too thin and 0.70 too fat to be informative). The conservative 70% regime is shown separately and does not break. |
| Coverage did not dramatically collapse in Phase 4. | Correct, and reported as such: 90.1% → 89.5%. The defensible Phase 4 result is drawdown and hit-rate improvement under thin margins, not a manufactured calibration disaster. |
| The uncertainty rule earned less total profit. | Correct, and central to the honest story: it declined risky deals that mostly worked out *in a 6% downturn*. The purchase is insurance — an 84% drawdown reduction and +11-point hit rate — whose value scales with how bad the downturn could have been. |
| The Streamlit tool is not a production lending system. | Correct. It is a portfolio-grade prototype that loads saved artifacts, discloses assumptions, and degrades with clear remediation messages. |

---

## SECTION H — The Numbers Sheet

Every number, with its source of truth:

| Phase | Number | Source |
|---|---:|---|
| 1 | 1,460 Kaggle training rows; median sale price $163,000 | `reports/phase1_metric_card.json` |
| 1 | LightGBM log-RMSE 0.135 ± 0.015 (ElasticNet: 0.126) | `reports/phase1_metric_card.json` |
| 1 | LightGBM dollar RMSE $28,500 ± $6,381 | `reports/phase1_metric_card.json` |
| 1 | Typical absolute dollar error $9,413 | `reports/phase1_metric_card.json` |
| 1 | 80th / 95th percentile absolute error $22,193 / $45,283 | `reports/phase1_metric_card.json` |
| 2 | Split: 949 train / 219 calibration / 292 test | `reports/phase2_metric_card.json` |
| 2 | 90% interval empirical coverage 90.4% | `reports/phase2_metric_card.json` |
| 2 | Median / mean 90% interval width $64,025 / $83,623 | `reports/phase2_metric_card.json` |
| 2 | Verdicts: 127 approve / 1 refer / 164 decline (43.5% / 0.3% / 56.2%) | `reports/phase2_metric_card.json` |
| 2 | All 164 declines: interval wider than the $60,000 cap | `reports/phase2_metric_card.json` |
| 2 | Top 50 naive picks declined: 50 of 50 | `reports/phase2_metric_card.json` |
| 3 | Exterior quality: $425 naive vs $5,634 causal (gap $5,208; CI $2,063–$9,204) | `reports/phase3_metric_card.json` |
| 3 | Kitchen quality: $4,146 naive vs $4,450 causal (CI $1,824–$7,076) | `reports/phase3_metric_card.json` |
| 3 | Basement full bath: $9,581 naive vs $8,520 causal (CI $5,717–$11,323) | `reports/phase3_metric_card.json` |
| 3 | Garage finish: $2,641 naive vs $3,524 causal (CI $1,619–$5,429) | `reports/phase3_metric_card.json` |
| 3 | Representative verdict flips: 0 of 10 | `reports/phase3_metric_card.json` |
| 4 | Full Ames rows: 2,930 sales, 2006–2010, 43 evaluation periods | `reports/phase4_metric_card.json` |
| 4 | Median price peak-to-trough −6.1% ($165,125 → $155,000) | `reports/phase4_metric_card.json` |
| 4 | Coverage: 90.1% pre-downturn → 89.5% in-downturn | `reports/phase4_metric_card.json` |
| 4 | Conservative 70% rule: $0 max drawdown for naive AND uncertainty-aware | `reports/phase4_metric_card.json` |
| 4 | iBuyer naive: 597 deals, $12.65M profit, $129,522 max drawdown, 77.4% hit rate (76.6% in crash window) | `reports/phase4_metric_card.json` |
| 4 | iBuyer uncertainty-aware: 418 deals, $4.95M profit, $21,257 max drawdown, 88.3% hit rate (88.1% in crash window) | `reports/phase4_metric_card.json` |
| 4 | Drawdown reduction from the gate: $108,265 (−84%) | `reports/phase4_metric_card.json` |
| 5 | App artifacts: Phase 1 point model, Phase 2 CQR model, Phase 5 feature defaults | `models/phase1/`, `models/phase2/`, `models/phase5/` |

---

## SECTION I — Glossary

| Term | Plain-English definition |
|---|---|
| log-RMSE | Root mean squared error computed after log-transforming price, so errors behave like percentage misses rather than raw dollar misses. |
| retransformation bias | Converting an average log prediction back to dollars with `exp` is biased low unless corrected (the average of logs isn't the log of the average). |
| Duan smearing | The classic correction for retransformation bias, using the model's own residuals. |
| quantile regression | A model trained to predict a percentile (e.g., the 5th or 95th) instead of the average. |
| conformal prediction | A wrapper that calibrates a model's intervals on held-out data so they hit a promised coverage rate, with a finite-sample guarantee. |
| Conformalized Quantile Regression (CQR) | Quantile models for the interval's edges, then conformal calibration to make the interval's promise true. |
| nominal vs empirical coverage | What the interval promises (90%) vs what it delivers on held-out data (here, 90.4%). Matching = calibrated. |
| exchangeability | The conformal guarantee's assumption: future cases behave statistically like the calibration cases. Random splits satisfy it; time drift threatens it. |
| marginal coverage | Coverage that holds on average over the population, not per individual house or neighborhood. |
| Monte Carlo simulation | Running a deal thousands of times with randomized inputs to obtain an outcome *distribution* instead of a single estimate. |
| confounding | When a feature looks valuable because it travels with other valuable traits, not because changing it would create value. |
| DML (Double Machine Learning) | A causal method that uses flexible ML to strip predictable confounding out of both treatment and outcome before estimating the effect. |
| cross-fitting | Splitting data so nuisance models only predict rows they didn't train on, keeping overfit out of the causal estimate. |
| conditional independence assumption | The untestable premise of observational causal inference: after controlling for what you measured, treatment is as good as random. |
| walk-forward backtest | Time-ordered evaluation where the model trains only on the past — no time travel. |
| ARV | After-repair value: expected resale value once the renovation plan is done. |
| MAO | Maximum allowable offer: the purchase-price ceiling implied by the investor's margin rule. |
| 70% rule | The traditional flip heuristic: pay at most 70% of ARV minus repair costs (~30% gross cushion). |
| iBuyer | A company buying homes directly using automated valuation with thin operating margins — the Zillow Offers model. |
| adverse selection | When offering to buy at model price, sellers of *overvalued* homes accept most eagerly — your model's mistakes select themselves into your portfolio. |
| hit rate | The share of purchased deals that ended with positive realized profit. |
| max drawdown | The largest drop from a previous cumulative-profit peak to a later trough — the "worst losing streak." |
| metric card | This project's per-phase JSON artifact recording headline numbers plus the exact config snapshot that produced them. |
