# Margin of Error Project Explainer

## SECTION A — The One-Paragraph Version.

I built an uncertainty-aware home-flip underwriting system inspired by the Zillow
Offers failure: instead of asking only "can a model predict sale price?", I asked
"is the model certain enough to bet capital on this house?" On Ames housing data,
the Phase 1 LightGBM model looks respectable, but its typical dollar error is
$9,413 and its 90% conformal interval is $64,025 wide at the median, which is much
larger than many flip profit targets. The project turns that insight into a
decision engine: calibrated value ranges, simulated profit distributions, a
causal renovation layer, and a walk-forward backtest through 2007-2010. The
memorable line is: an accurate price model is not an underwriting system until
its margin of error is smaller than the deal's profit margin.

## SECTION B — The Problem, Restated Plainly.

Most people use the Ames dataset as a leaderboard exercise: minimize RMSE, brag
about the model, and stop. That is the trap. An investor does not earn RMSE; an
investor earns or loses dollars on a purchase. If a model says a house is worth
$180,000 but its honest 90% range is roughly $150,000 to $214,000, the single
number can be directionally useful and still too uncertain to underwrite. A
decision-under-uncertainty reframing means the model must output a range, a profit
distribution, and a rule for when to walk away. Zillow Offers is the cautionary
anchor: buying homes near model value with thin margins leaves no room for model
error, slow renovations, or a softer resale market.

## SECTION C — What Each Phase Actually Did.

### Phase 1 - Baseline Price Model

**The question this phase answered** - How good is a normal "predict the sale
price" model, and how large are its dollar errors?

**What I actually built** - I trained a fold-safe baseline modeling pipeline on
the Kaggle Ames training set: a dumb median model, ElasticNet, and LightGBM. The
final saved artifact is `models/phase1/baseline_lightgbm.joblib`, and the metric
card is `reports/phase1_metric_card.json`.

**The key technique, explained simply** - The target was `log1p(SalePrice)`,
which makes percentage-style errors easier to learn. When converting predictions
back to dollars, I used Duan smearing so the dollar estimates are not biased low
by the log transform.

**The real result** - LightGBM produced a log-RMSE of 0.135 +/- 0.015 and a dollar
RMSE of $28,500 +/- $6,381. The typical absolute dollar error was $9,413; the
80th percentile absolute error was $22,193 and the 95th percentile was $45,283.
ElasticNet actually had the best log-RMSE at 0.126, but LightGBM was kept as the
gradient-boosting strawman required by the project. The data had 1,460 rows and a
median sale price of $163,000.

**Why it matters for the decision** - A $9,413 typical error is already close to a
small flip's profit target, and a $22,193 80th-percentile error can wipe out a
$10,000 to $20,000 margin. Phase 1 proves that "accurate" is not the same as
"safe to buy."

**The likely interview question and how to answer it** - "Why use LightGBM if
ElasticNet had lower log-RMSE?" Answer: I reported that honestly. LightGBM was the
required nonlinear strawman and slightly better on dollar RMSE, while ElasticNet's
strong log-RMSE is evidence that Ames is small and structured enough for linear
regularization. The point of the project is not model leaderboard victory; it is
that even a strong model still has deal-sized dollar error.

### Phase 2 - Uncertainty-Aware Underwriting

**The question this phase answered** - What happens when the price model is forced
to admit its uncertainty before making a buy decision?

**What I actually built** - I wrapped the Phase 1 point model with a
Conformalized Quantile Regression value interval, simulated flip profit under
the economics in `config/economics.yaml`, and produced an APPROVE / REFER /
DECLINE rule for each held-out home.

**The key technique, explained simply** - Conformalized Quantile Regression means
that instead of one guess, the model gives a value range that has been checked on
held-out data to contain the true price about 90% of the time. Then the
underwriter asks whether the profit distribution clears the required margin and
whether the model's interval is too wide to trust.

**The real result** - The split was 949 train rows, 219 calibration rows, and 292
test rows. The 90% interval achieved 90.4% empirical coverage, so it was honestly
calibrated on the random test set. The median 90% interval width was $64,025 and
the mean width was $83,623. The rule approved 127 homes (43.5%), referred 1
home (0.3%), and declined 164 homes (56.2%). All 164 declines were because the
model uncertainty exceeded the $60,000 interval-width cap. The top 50 homes a
naive point model liked were 50 out of 50 declined by the uncertainty gate.

**Why it matters for the decision** - The model's median value range is several
times wider than a normal flip profit buffer. Phase 2 is the core anti-Zillow
move: if the model is too unsure, the answer is not "buy at the point estimate,"
it is "do not put capital at risk."

**The likely interview question and how to answer it** - "Does conformal
prediction guarantee each individual house is covered?" Answer: no. It gives
finite-sample marginal coverage under exchangeability, which means the set of
future cases should behave like the calibration cases. It does not promise that
every neighborhood or every month is equally calibrated; that limitation is why
Phase 4 tests the rule through time.

### Phase 3 - Causal Renovation Layer

**The question this phase answered** - Which visible renovation-related features
actually appear to cause value lift, rather than merely correlate with nicer
houses?

**What I actually built** - I estimated treatment effects for renovatable Ames
features, populated those causal uplifts into `config/economics.yaml`, and
compared causal versus correlational renovation assumptions on representative
underwriting cases.

**The key technique, explained simply** - Double Machine Learning strips out the
"halo" from fixed property quality, neighborhood, size, and age before estimating
the renovation effect. In plain English: it tries to compare like with like, so a
kitchen upgrade is not accidentally getting credit for the whole house being in a
better neighborhood or built better in the first place.

**The real result** - Exterior quality had the largest naive-versus-causal gap:
naive OLS said $425 per quality step, while DML estimated $5,634, a $5,208
understatement, with a 95% interval from $2,063 to $9,204. Kitchen quality was
$4,146 naive versus $4,450 causal, with a 95% interval from $1,824 to $7,076.
Basement full baths were $9,581 naive versus $8,520 causal, with a 95% interval
from $5,717 to $11,323. Garage finish was $2,641 naive versus $3,524 causal, with
a 95% interval from $1,619 to $5,429. In the 10 representative Phase 2 homes,
causal-vs-correlational assumptions caused 0 verdict flips.

**Why it matters for the decision** - Renovation ROI should not be inferred from
plain correlations. Phase 3 says some effects are real but modest relative to
renovation costs and valuation uncertainty. The decision rule still dominated:
if the value interval is too wide, better renovation assumptions do not rescue the
deal.

**The likely interview question and how to answer it** - "How do you know the
causal estimates are right?" Answer: I do not claim randomized-trial truth. I
used cross-fitted DML with LightGBM nuisance models and HC3 robust standard
errors to reduce observed confounding. The honest limitation is the conditional
independence assumption: unobserved owner wealth, maintenance quality, contractor
quality, and micro-location can still bias the estimates.

### Phase 4 - Walk-Forward Crash Backtest

**The question this phase answered** - Does the uncertainty-aware rule behave
differently from naive point underwriting when the market is evaluated through
time rather than random rows?

**What I actually built** - I downloaded the full De Cock Ames dataset with 2,930
sales from 2006-2010, sorted it by sale year and month, trained expanding-window
models on past data only, and scored each month from 2007 through 2010. I compared
three buy gates under two acquisition regimes: conservative 70%-rule flips and a
thin-margin iBuyer regime at 85% of predicted ARV.

**The key technique, explained simply** - A walk-forward backtest means the model
only knows what an investor would have known at the time. It trains on the past,
buys or passes on the current month, records what would have happened, then moves
forward.

**The real result** - Ames had a mild downturn: median sale price went from
$165,125 in 2007 to $155,000 in 2010, a 6.1% peak-to-trough decline. Coverage was
90.1% before the downturn and 89.5% in the downturn, so it dipped 0.5 percentage
points below the 90% target but did not theatrically collapse. Under the
conservative 70% rule, naive point underwriting had $0 max drawdown and 97.6% hit
rate; uncertainty-aware underwriting also had $0 max drawdown and 98.3% hit rate.
Under the thin-margin iBuyer regime, naive point underwriting had a $129,522 max
drawdown and 77.4% hit rate. The uncertainty-aware rule had a $21,257 max
drawdown and 88.3% hit rate, reducing drawdown by $108,265 and improving hit rate
by 10.9 percentage points, while buying fewer homes.

**Why it matters for the decision** - The backtest sharpens the story: uncertainty
discipline matters most when the business model buys near fair value. A
conservative flipper has so much margin that a mild Ames downturn does not break
the strategy. An iBuyer-style buyer has thin enough margins that model uncertainty
becomes capital risk.

**The likely interview question and how to answer it** - "Did the model's
coverage collapse in the crash?" Answer: not dramatically. It moved from 90.1% to
89.5%, which is technically below the 90% target but basically stable. The stronger
finding is economic, not statistical: under thin-margin buying, the interval-aware
gate selected fewer deals, improved hit rate, and cut max drawdown from $129,522
to $21,257.

## SECTION D — The Throughline.

The four phases compose into one argument. Phase 1 shows that a respectable
price model still has dollar errors large enough to threaten a flip. Phase 2
turns the point estimate into a calibrated value range and proves that the
uncertainty band is usually wider than the deal margin. Phase 3 says renovation
assumptions need causal discipline because naive correlations can misstate value
lift by thousands of dollars. Phase 4 moves from random validation to time and
shows when the rule matters economically: conservative 70%-rule flips survive a
mild Ames downturn, but thin-margin iBuyer underwriting is exposed to model error,
and the uncertainty gate materially reduces drawdown. The final product is not a
better Kaggle model; it is a safer decision system.

## SECTION E — The Honest Limitations.

| Attack | Disarming response |
|---|---|
| Ames is one small market, not the national iBuyer market. | Correct; the results are a decision-system demonstration, not a national housing law. The full temporal dataset has 2,930 sales and a mild 6.1% decline, which I report plainly. |
| The Kaggle split is random, so early phases may look too stable. | Correct; Phases 1-3 use the random 1,460-row training set, and Phase 4 explicitly switches to the full 2006-2010 temporal ordering. |
| The causal layer depends on the conditional independence assumption. | Correct; DML reduces observed confounding but cannot eliminate unobserved maintenance, owner wealth, contractor quality, or micro-location. |
| Renovation costs are estimates. | Correct; costs such as $27,000 for a minor kitchen remodel and $49,000 for a bathroom addition are documented assumptions, not measured Ames project invoices. |
| The backtest uses synthetic acquisition prices, not true buy-resell pairs. | Correct; Ames records sale transactions, not flip projects, so Phase 4 is an underwriting-rule stress test using observed sale price as realized ARV. |
| The iBuyer regime uses an 85% ARV purchase factor. | Correct; it is a scenario to represent thin-margin buying near model value. The 70% conservative regime is shown separately and does not break. |
| Coverage did not dramatically collapse in Phase 4. | Correct; coverage moved from 90.1% to 89.5%. The defensible Phase 4 result is drawdown and hit-rate improvement under thin margins, not a fabricated calibration disaster. |
| The Streamlit tool is not a production lending system. | Correct; it is a portfolio-ready underwriting prototype that loads saved artifacts, shows assumptions, and degrades clearly when inputs are missing. |

## SECTION F — The Numbers Sheet.

| Phase | Number | Source |
|---|---:|---|
| 1 | 1,460 Kaggle training rows; median sale price $163,000 | `reports/phase1_metric_card.json` |
| 1 | LightGBM log-RMSE 0.135 +/- 0.015 | `reports/phase1_metric_card.json` |
| 1 | LightGBM dollar RMSE $28,500 +/- $6,381 | `reports/phase1_metric_card.json` |
| 1 | Typical absolute dollar error $9,413 | `reports/phase1_metric_card.json` |
| 1 | 80th percentile absolute dollar error $22,193 | `reports/phase1_metric_card.json` |
| 1 | 95th percentile absolute dollar error $45,283 | `reports/phase1_metric_card.json` |
| 2 | CQR split: 949 train / 219 calibration / 292 test | `reports/phase2_metric_card.json` |
| 2 | 90% interval empirical coverage 90.4% | `reports/phase2_metric_card.json` |
| 2 | Median 90% interval width $64,025 | `reports/phase2_metric_card.json` |
| 2 | Mean 90% interval width $83,623 | `reports/phase2_metric_card.json` |
| 2 | Verdicts: 127 approve / 1 refer / 164 decline | `reports/phase2_metric_card.json` |
| 2 | Approval / refer / decline rates: 43.5% / 0.3% / 56.2% | `reports/phase2_metric_card.json` |
| 2 | Top 50 naive picks declined: 50 out of 50 | `reports/phase2_metric_card.json` |
| 3 | Exterior quality: $425 naive vs $5,634 causal; gap $5,208 | `reports/phase3_metric_card.json` |
| 3 | Kitchen quality: $4,146 naive vs $4,450 causal | `reports/phase3_metric_card.json` |
| 3 | Basement full bath: $9,581 naive vs $8,520 causal | `reports/phase3_metric_card.json` |
| 3 | Garage finish: $2,641 naive vs $3,524 causal | `reports/phase3_metric_card.json` |
| 3 | Representative verdict flips: 0 of 10 | `reports/phase3_metric_card.json` |
| 4 | Full Ames rows: 2,930 sales from 2006-2010 | `reports/phase4_metric_card.json` |
| 4 | Median price peak-to-trough: -6.1% ($165,125 to $155,000) | `reports/phase4_metric_card.json` |
| 4 | Coverage: 90.1% pre-crash to 89.5% in-crash | `reports/phase4_metric_card.json` |
| 4 | Conservative 70% rule: naive max drawdown $0; uncertainty-aware max drawdown $0 | `reports/phase4_metric_card.json` |
| 4 | iBuyer naive: 597 deals, $12.65M profit, $129,522 max drawdown, 77.4% hit rate | `reports/phase4_metric_card.json` |
| 4 | iBuyer uncertainty-aware: 418 deals, $4.95M profit, $21,257 max drawdown, 88.3% hit rate | `reports/phase4_metric_card.json` |
| 4 | iBuyer drawdown reduction from uncertainty gate: $108,265 | `reports/phase4_metric_card.json` |
| 5 | App artifacts: Phase 1 point model, Phase 2 CQR model, Phase 5 feature defaults | `models/phase1/`, `models/phase2/`, `models/phase5/` |

## SECTION G — Glossary.

| Term | Plain-English definition |
|---|---|
| log-RMSE | The model's root mean squared error after taking the log of sale price, so errors behave more like percentage misses than raw dollar misses. |
| retransformation bias | The problem that converting an average log prediction back to dollars with `exp` tends to be biased unless you correct for the log-scale residuals. |
| conformal prediction | A way to wrap a model with a prediction interval that is calibrated on held-out data to hit a promised coverage rate. |
| quantile regression | A model that predicts a percentile, such as the 5th or 95th percentile, rather than the average value. |
| Conformalized Quantile Regression | The Phase 2 method that trains lower and upper quantile models, then adjusts them with conformal calibration to produce a reliable interval. |
| confounding | When a feature looks valuable because it travels with other valuable traits, not because changing that feature itself would create value. |
| DML | Double Machine Learning, a causal method that uses flexible models to remove predictable confounding before estimating a treatment effect. |
| cross-fitting | Splitting the data so nuisance models predict rows they did not train on, reducing overfit leakage in causal estimation. |
| walk-forward backtest | A time-ordered evaluation where the model trains only on past data and is tested on later periods. |
| calibration | Whether a stated probability or interval promise actually happens at the advertised rate on held-out data. |
| max drawdown | The largest drop from a previous cumulative profit peak to a later trough. |
| ARV | After-repair value, the expected resale value after the renovation plan. |
| MAO | Maximum allowable offer, the purchase price ceiling implied by the investor's margin rule. |
| hit rate | The share of bought deals that ended with positive realized profit. |
| iBuyer | A company that buys homes directly using automated valuation and thin-margin operating assumptions. |
