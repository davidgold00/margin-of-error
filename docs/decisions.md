# Architecture Decision Records (ADR Log)

This is the project's decision journal: a chronological log of every
non-obvious modeling, infrastructure, and economic judgment call, written down
at the moment it was made. Where `docs/assumptions.md` records *parameter
values* (what a kitchen remodel costs), this file records *choices* (why CQR
over symmetric intervals, why the backtest uses an 85% iBuyer factor rather
than 90%, why EconML was swapped for a hand-rolled DML implementation).

**How to read an entry:** each ADR states the **context** (what question was
on the table), the **decision** (what was chosen), and the **tradeoff** (what
was given up, and how that cost is mitigated). Entries are numbered and
appended chronologically, never rewritten — when a decision is reversed, a
later ADR supersedes it and the original is annotated with a pointer (see
ADR-003 → ADR-015 for an example). That means this file doubles as an honest
history: you can watch the project change its mind and see why.

**Why keep it:** every "why did you do X instead of Y?" question about this
project should have a written answer here, dated and reasoned, rather than a
reconstructed one.

---

## Phase 0 Decisions

### ADR-001: Package manager — venv + pip (not uv)

**Context:** The project brief prefers `uv` for speed and reproducibility.
`uv` was not installed on the development machine at scaffold time.

**Decision:** Use stdlib `venv` + `pip` with a pinned `requirements-lock.txt`
generated at setup time. Makefile exposes `make setup` which creates `.venv`,
installs all deps, and installs pre-commit hooks.

**Tradeoff:** Slightly slower installs than `uv`; no automatic lockfile format.
Mitigated by pinning versions in `pyproject.toml` and generating a lock via
`pip freeze > requirements-lock.txt` as part of `make setup`. If `uv` becomes
available, migration is straightforward: `uv pip sync requirements-lock.txt`.

---

### ADR-002: Conformal prediction — MAPIE + custom CQR implementation

**Context:** The brief requires Conformalized Quantile Regression (CQR), not
naive symmetric intervals. Two options: (a) MAPIE's `MapieQuantileRegressor`,
(b) a from-scratch split-conformal CQR implementation.

**Decision:** Both. MAPIE is the production path (well-tested, pip-installable,
handles edge cases). A clean educational CQR implementation lives in
`src/margin_of_error/models/conformal.py` alongside it — used in the notebook
to explain the mechanic and to cross-validate MAPIE's output. If results differ,
MAPIE wins.

**Tradeoff:** Slight code duplication. Benefit: the educational implementation
makes the "how it works" auditable without reading MAPIE source; also serves as
a correctness check.

---

### ADR-003: Causal library — EconML (LinearDML + CausalForestDML)

**Context:** The brief lists EconML and DoubleML as alternatives for the DML
causal layer (Phase 3).

**Decision:** EconML. Reasons: (1) it wraps both `LinearDML` (for interpretable
linear-in-treatment effects) and `CausalForestDML` (for heterogeneous treatment
effect estimation) in one package with a consistent API; (2) it has scikit-learn
compatibility; (3) it directly computes confidence intervals on CATE estimates,
which we need to contrast with naive regression coefficients.

**Tradeoff:** EconML is a heavy install (~500 MB, pulls PyTorch). Isolated in the
optional `[causal]` extra so the default install stays light.

**Phase 3 update:** Superseded by ADR-015. EconML was not installed in the active
project environment, so the accepted Phase 3 path is a manual cross-fitted DML
implementation with LightGBM nuisance models and statsmodels inference.

---

### ADR-004: Conformal calibration split strategy

**Context:** CQR requires a calibration set that the underlying quantile models
have never seen. Options: (a) a static hold-out from training data, (b) CV+ (cross-
conformal), (c) a separate time-ordered hold-out.

**Decision:** Static hold-out: 15% of the Kaggle training set (≈ 219 rows) is
withheld before any model fitting and used exclusively for conformal calibration.
The remaining 85% is used for k-fold CV to train the quantile models.

**Tradeoff:** CV+ would be more data-efficient and give tighter intervals at the
cost of significant complexity. Given the Kaggle set is only ~1,460 rows (already
small), the 15% hold-out is a known and auditable cost. 219 calibration points
are sufficient for stable empirical quantiles at α = 0.10. We log CV+ as a backlog
improvement item for Phase 2.

---

### ADR-005: Kaggle train/test split is random, not temporal

**Context:** The Kaggle competition split is a random 50/50 split of the 2,930 Ames
sales. This is by design for leaderboard fairness, but it creates a problem for any
honest backtest.

**Decision:** Use the Kaggle data for Phases 1–3 (cross-sectional modeling, conformal
calibration, causal estimation). Switch to the full De Cock dataset sorted by
`YrSold` / `MoSold` for Phase 4 (crash backtest). This is flagged in code comments,
the README, and data/README.md to prevent accidental conflation.

**Tradeoff:** Two separate data pipelines with slightly different schemas. Managed
via `src/margin_of_error/data/loaders.py` which normalizes column names before
passing data downstream. Residual risk: any Phase 1–3 result is cross-sectional
and cannot be naively compared to Phase 4 temporal results.

---

## Backlog

- CV+ (cross-conformal) for tighter intervals with small training sets
- Spatial cross-validation (block-group hold-out) to replace neighborhood stratification
- Time-varying hard-money rate series for Phase 4 financing assumptions
- Iowa-specific renovation cost adjustments (Remodeling Mag national avg is too high)
- Quantile calibration plots (reliability diagrams) per neighborhood

---

## Phase 1 Decisions

### ADR-006: Log retransformation uses Duan smearing

**Context:** Phase 1 models `log1p(SalePrice)`. A naive `expm1(prediction)` is
biased low because `E[exp(error)] != exp(E[error])`.

**Decision:** Estimate Duan's smearing factor on each training fold using
`mean(exp(y_train_log - y_train_pred_log))`, then apply that factor to the
held-out fold predictions before dollar metrics are computed.

**Tradeoff:** Smearing uses in-fold fitted predictions, so it is still an
estimate and can vary by fold. It avoids leaking holdout targets and gives a
more honest dollar-scale error than naive exponentiation.

---

### ADR-007: Fold-safe preprocessing with explicit missingness policy

**Context:** Ames NA values are semantic. Some mean structural absence (no
garage, no basement, no pool); others are true missing values. Lot frontage is
missing often enough that global imputation would be both crude and potentially
leaky if fit before CV.

**Decision:** Build a sklearn preprocessing pipeline that runs inside every CV
fold. Structural categorical NAs become `"None"`, structural numeric NAs become
`0`, and true-missing values are imputed by fold-local median/mode. Lot frontage
uses fold-local neighborhood medians with a fold-local global median fallback.

**Tradeoff:** The policy is more verbose than one `SimpleImputer`, but it keeps
the data dictionary semantics intact and makes leakage tests possible.

---

### ADR-008: Primary Phase 1 booster is LightGBM

**Context:** The brief allows LightGBM or XGBoost. Both require OpenMP on macOS.
The project already had a LightGBM config block from Phase 0.

**Decision:** Use LightGBM as the primary gradient-boosting strawman, with
internal train-fold early stopping and compact nested-CV tuning over tree shape
and L2 regularization. XGBoost was verified after installing `libomp`, but adding
a second booster would widen Phase 1 rather than sharpen the baseline.

**Tradeoff:** In this run, ElasticNet slightly beat LightGBM on log RMSE while
LightGBM slightly beat ElasticNet on dollar RMSE. We keep LightGBM as the
primary strawman because the phase requires a tuned gradient booster, and we
report the ElasticNet comparison plainly instead of hiding it.

---

### ADR-009: Repeated CV stratifies by neighborhood with rare-level bucketing

**Context:** `Neighborhood` is the strongest spatial proxy, but the Kaggle
training split includes at least one neighborhood with fewer than five rows,
which cannot be split across five stratified folds.

**Decision:** Use 5-fold CV repeated 3 times. Before each repeated stratified
split, bucket the smallest neighborhood levels into a rare bucket until every
stratum has at least five rows.

**Tradeoff:** Rare neighborhoods lose individual fold-stratification identity,
but the alternative is fully random folds. This is still not spatial CV; Phase 4
handles temporal robustness and spatial CV remains backlog.

---

## Phase 2 Decisions

### ADR-010: Three-way split (65 / 15 / 20) instead of the Phase 0 single hold-out

**Context:** ADR-004 planned an 85/15 train/calibration split. But CQR coverage
must be *measured* on data touched by neither the quantile models nor the
conformal calibration. With only train+calibration, there is no honest test set.

**Decision:** Split the 1,460 labeled Kaggle rows three ways with no overlap:
train 65% (≈949, fits the quantile arms), calibration 15% (≈219, computes Q̂),
test 20% (≈292, measures coverage and width). Sizes live in `config.conformal`
(`calibration_split`, `test_split`); the split is seeded by `global_seed`.

**Tradeoff:** The CQR quantile models see fewer rows than Phase 1's 85%. That is
the correct price for an honest, untouched test set — coverage measured on the
training or calibration folds would be optimistic. The CQR quantile models are
separate estimators from the Phase 1 point model; they do not need identical
training rows, only clean splits. The feature pipeline is **refit on the CQR
training fold** (not inherited from the Phase 1 artifact) so no cross-split
information leaks.

---

### ADR-011: CQR implemented directly, not via MAPIE; the conformal-rank bug

**Context:** ADR-002 planned MAPIE as the production path with a custom CQR as a
cross-check. In implementation, the split-conformal CQR math is ~5 lines and is
more auditable written out than wrapped.

**Decision:** Implement CQR directly in `models/conformal.py`. The quantile arms
are LightGBM (`objective="quantile"`) reusing the Phase 1 feature pipeline and
hyperparameters; the conformal correction is the Romano et al. (2019) rank.
MAPIE remains an optional future cross-check, not a dependency of the result.

**The bug the calibration gate caught:** the scaffolded `compute_conformal_quantile`
computed `level = ceil((1-alpha)(1 + 1/n)) / n`, which applies `ceil(.)` to a
fraction ≈0.9 → 1, collapsing the level to `1/n`. That produced a hugely negative
Q̂, inverted intervals (negative widths) and ~3% empirical coverage. The
non-negotiable coverage assertion stopped the run before any economics. Fixed to
the correct rank `k = ceil((1-alpha)(n+1))`, the k-th smallest score; empirical
coverage on the test set is now 90.4% at the 90% level and tracks the diagonal at
every level (see `reports/figures/02d_calibration.png`). This is exactly why the
calibration gate exists.

**Back-transformation:** all interval arithmetic is in log space; each bound is
back-transformed with `expm1` **separately**. The Duan smearing factor is applied
**only** to the point estimate (a mean correction); the bounds are quantiles, which
are median-unbiased and receive no smearing.

---

### ADR-012: Acquisition price uses the industry "70% rule" (MAO)

**Context:** The dataset-wide pass needs a purchase price per home. Buying at the
model's predicted value makes every flip a guaranteed loss (Ames home values vs.
the renovation-uplift priors), which would make the decision rule decline homes
for the *wrong* reason (upside-down economics, not uncertainty).

**Decision:** Acquire under the standard flip "70% rule" — Maximum Allowable Offer
= `acquisition_arv_factor × ARV − renovation_cost` (factor 0.70 in config). This
targets ~30% of ARV as gross margin before costs, so a genuinely profitable
baseline exists and the binding question becomes whether the model's *uncertainty*
fits inside that margin.

**Tradeoff / open assumption for review:** the 70% rule already embeds a margin of
safety. Under it, the probabilistic margin/loss checks rarely bind — the verdict is
driven mainly by the `max_acceptable_interval_width_usd` cap. That is a defensible
finding ("even with the industry 30% safety margin, model uncertainty alone
disqualifies the majority of homes"), but it makes REFER rare. Alternatives
(a fixed discount to value; a tighter buffer) are logged for Phase 5 sensitivity.

---

### ADR-013: The APPROVE / REFER / DECLINE thresholds, and the width override

**Context:** The verdict needs to encode "expected margin must clear the model's
own uncertainty by a buffer." The brief specifies probability thresholds.

**Decision (all in `config.flip.underwriting`):** APPROVE requires
P(profit > \$15K buffer) ≥ 0.65 **and** P(loss) ≤ 0.20 **and** interval width ≤ \$60K.
REFER relaxes to ≥ 0.50 / ≤ 0.30. Otherwise DECLINE, reporting the binding reason.
Crucially, an interval **wider than the \$60K cap is a hard DECLINE that overrides
REFER** — a deliberate strengthening of the brief's REFER rule, because a model that
cannot value a home within \$60K should not have capital bet on it regardless of the
point estimate. This is the explicit anti-Zillow guardrail and is what the named
test `test_wide_interval_declines_despite_positive_point_estimate` locks in.

**Why these numbers:** they are defensible starting points, not sacred — 65% is a
"more likely than not, with room to spare" bar for committing months-long illiquid
capital; 20% loss tolerance reflects that a flipper cannot survive losing on one in
four deals. All six are config-driven and swept in Phase 5.

---

### ADR-014: Profit Monte Carlo — Normal ARV, truncated-Normal hold

**Context:** The CQR interval is distribution-free; the profit simulation needs a
sampling distribution for ARV and holding period.

**Decision:** Sample ARV ~ Normal(point, (U−L)/(2·1.645)) — mapping the 90% interval
half-width to a working std — and holding period ~ truncated Normal(base=4, std=1.5,
[1, 12]). 10,000 draws per property per tier; only summary statistics are stored
(never the raw draws), keeping the dataset-wide pass memory-light.

**Tradeoff:** Approximating a distribution-free interval with a Normal is an
acknowledged simplification (it ignores any skew the conformal interval captures).
It is transparent, reproducible, and conservative enough for an underwriting screen;
a fuller treatment (sampling the empirical conformal distribution) is backlog.

---

## Phase 3 Decisions

### ADR-015: Manual cross-fitted DML instead of EconML

**Context:** The Phase 3 brief preferred EconML `LinearDML` if available, with a
manual cross-fitted DML fallback if installation was unavailable. The repo's venv
did not have EconML installed, and adding the heavy optional dependency would have
expanded the environment for no statistical benefit.

**Decision:** Implement DML manually in `src/margin_of_error/causal/dml.py`:
5-fold cross-fitting, LightGBM nuisance models for both `E[Y|W]` and `E[T|W]`,
then HC3-robust OLS of residualized `log1p(SalePrice)` on residualized treatment.
The implementation persists fold records so tests can verify no test fold was used
to fit the nuisance model that generated its residuals.

**Tradeoff:** We do not get EconML's convenience wrappers or future CausalForestDML
API for free. In exchange, the estimator is short, auditable, reproducible, and
does not require a large new dependency.

### ADR-016: Cross-fitting is mandatory for the causal stage

**Context:** DML uses flexible ML models to partial out confounders. If the same
rows are used to fit and residualize, regularized learners can overfit the
nuisance functions, biasing residuals toward zero and attenuating the final
coefficient.

**Decision:** Use 5-fold cross-fitting seeded by `global_seed=42`. For each fold,
the outcome and treatment nuisance models are fit on the other four folds only and
predict the held-out fold. Residuals are stacked, and only then is the final OLS
stage fit.

**Tradeoff:** Five folds multiply nuisance-model fits, but the Ames data is small
enough that the full Phase 3 run completes quickly. This is not optional plumbing;
it is what makes the DML coefficient honest enough to report.

### ADR-017: Estimate one treatment at a time

**Context:** A joint multivariate treatment regression could estimate all
renovation effects at once, but it requires stronger assumptions about treatment
dependence and interpretation. Kitchen, bath, basement, and garage improvements
are not independent in investor behavior.

**Decision:** Run the full DML pipeline independently per treatment. The comparison
table therefore answers "what is the marginal effect of this treatment, controlling
for fixed confounders?" without asking the reader to parse a joint treatment graph.

**Tradeoff:** Separate models are less statistically efficient than a correctly
specified joint model. The interpretation is cleaner for a portfolio project and
reduces the risk of reporting spurious precision.

### ADR-018: Registry wins over the recommended treatment list

**Context:** The Phase 3 brief recommended `BsmtQual`, `Fireplaces`, and
`GarageCars` as candidate treatments. The Phase 1 registry tags all three fixed.
It also tags `OverallCond` mutable, while the Phase 3 brief explicitly says
`OverallCond` should be a confounder, not a treatment.

**Decision:** Treat the registry as the source of truth for treatment eligibility,
with the Phase 3 brief's `OverallQual`/`OverallCond` rule as an explicit override.
Included treatments are `KitchenQual`, `BsmtFinType1`, `HeatingQC`, `FireplaceQu`,
`GarageFinish`, `ExterQual`, `FullBath`, `HalfBath`, and `BsmtFullBath`. Excluded
brief candidates are reported in `reports/phase3_metric_card.json`.

**Tradeoff:** The final treatment set is narrower than the brief's wish list, but
it preserves the precommitted mutable/fixed boundary and avoids inventing tags
after seeing results.

### ADR-019: Figure 3A omits naive OLS error bars

**Context:** The naive OLS coefficients have standard errors, but Figure 3A's job
is to make the confounding gap legible: naive estimate versus DML estimate, with
uncertainty around the causal estimate.

**Decision:** Plot DML 95% confidence intervals only. The full CSV still includes
the underlying inferential fields; the visual keeps attention on the causal
estimate and the investor-relevant bias gap.

**Tradeoff:** The naive estimates may look visually more certain than they are.
The chart title, docs, and CSV make clear that naive OLS is a foil, not a
recommended estimator.

### ADR-020: Heterogeneous treatment effects are backlog

**Context:** The brief listed CausalForestDML as an optional extension to estimate
heterogeneous kitchen effects by neighborhood or home size. EconML was not present,
and a manual causal forest implementation would be a separate modeling project.

**Decision:** Skip HTE for Phase 3 and backlog it. Phase 3 ships the linear DML
layer, comparison table, underwriting integration, and signature visuals.

**Tradeoff:** We do not yet know whether kitchen effects vary by neighborhood
median price or `GrLivArea`. Phase 4's temporal robustness question is the next
approved phase; HTE can return after that if needed.

### ADR-021: No verdict flips is a result, not a failure

**Context:** The Phase 3 brief expected homes where correlational and causal
renovation assumptions change the underwriting verdict. Under the current
70%-rule purchase model and the hard interval-width override, the representative
10-home comparison produced zero verdict flips.

**Decision:** Report zero flips honestly and still plot Figure 3C for the largest
profit-distribution deltas. The causal uplifts materially change expected profit,
but not enough to cross the APPROVE/DECLINE thresholds in the representative
sample.

**Tradeoff:** The demonstration is less dramatic than a flip table. It is more
credible: Phase 2 already showed the interval-width guardrail dominates decisions,
so Phase 3 appropriately changes renovation economics without pretending it
overrides uncertainty.

---

## Phase 4 Decisions

### ADR-022: Phase 4 uses the full De Cock temporal dataset

**Context:** The Kaggle split is random and cannot test regime drift. The full
De Cock Ames file has 2,930 sales from 2006-2010 with `YrSold` and `MoSold`.

**Decision:** Download and convert the public JSE `AmesHousing.xls` file to
`data/raw/ames/AmesHousing.csv`, normalize its columns, and sort by year/month.
Every walk-forward model trains only on past sales.

**Tradeoff:** Raw data remains gitignored, so a clean clone must download it before
reproducing Phase 4. The generated metric card, periods CSV, and figures are
committed so the story is reviewable without rerunning the full backtest.

### ADR-023: Annual expanding-window retraining

**Context:** Monthly retraining would be more granular but much slower and noisier
for a small dataset. The downturn is measured at year/month resolution.

**Decision:** Retrain once per sale year on all prior years. Evaluation starts in
2007, so the first generation trains on 2006 only; later generations expand the
training window.

**Tradeoff:** Annual retraining may understate a nimble operator's adaptation, but
it keeps the no-look-ahead invariant simple and testable.

### ADR-024: Phase 4 excludes renovation uplift from realized P&L

**Context:** Ames records sale transactions, not buy-renovate-resell pairs. Adding
a renovation uplift to the observed sale price would invent the counterfactual
resale price.

**Decision:** Treat observed sale price as realized ARV and set synthetic
acquisition prices from predicted ARV. P&L subtracts purchase price, transaction
cost, and expected holding cost; renovation is excluded from the backtest.

**Tradeoff:** The backtest is an underwriting-rule stress test, not a full flip
project simulator. That limitation is explicit in the metric card and explainer.

### ADR-025: Compare conservative flip and thin-margin iBuyer regimes

**Context:** Under the industry 70% rule, Ames's 6.1% median price decline did not
create drawdown; the margin was too padded to expose model risk.

**Decision:** Report two acquisition regimes: `conservative_flip` at 70% of
predicted ARV and `ibuyer` at 85% of predicted ARV. The iBuyer regime represents
buying near model value with a thin margin after transaction and carry costs.

**Tradeoff:** The iBuyer factor is a scenario, not a direct Zillow Offers contract
term. Showing both regimes prevents overstating the finding: uncertainty
discipline matters most when margins are thin.

### ADR-026: Phase 4 disciplined gate uses interval width plus loss probability

**Context:** The Phase 2 $15,000 profit-buffer probability is flip-specific. In a
thin-margin iBuyer regime, requiring the same buffer would make the rule decline
nearly every deal and hide the risk-selection question.

**Decision:** For Phase 4 strategy comparison, the uncertainty-aware gate buys
only when interval width is below the configured cap and modeled loss probability
is within the APPROVE tolerance. The naive point gate buys when point-estimated
profit clears the configured buffer.

**Tradeoff:** This adapts the decision rule for regime comparison while preserving
the core anti-Zillow principle: do not buy when the model is too uncertain or loss
probability is too high.

---

## Phase 5 Decisions

### ADR-027: Streamlit app loads saved artifacts only

**Context:** The Phase 5 tool needs to be fast and reproducible. Re-training CQR
inside a user interaction would be slow and would blur the line between analysis
and product.

**Decision:** Add `make app-artifacts`, which persists `models/phase2/cqr_90.joblib`
and `models/phase5/feature_defaults.json`. The app loads Phase 1, Phase 2, and
feature-default artifacts with clear errors if any are missing.

**Tradeoff:** The app has one extra build step after model runs. In exchange, app
startup is deterministic and tests can verify the real artifact path.

### ADR-028: Expose a small underwriting feature set, default the rest

**Context:** Ames has roughly 80 raw columns. A usable underwriting tool should
not force a user to populate every field.

**Decision:** Expose the most decision-relevant inputs: neighborhood, living area,
overall quality, year built, baths, kitchen quality, basement area, garage spaces,
and garage finish. All other columns use medians/modes from the training data.

**Tradeoff:** The app is a screening tool rather than a full appraisal intake. The
defaults artifact documents the hidden values so the model input is still
auditable.

### ADR-029: Test-time readline shim for the local Python 3.13 environment

**Context:** In this local venv, importing the macOS `readline` extension segfaults.
Pytest imports `readline` during capture startup before running any tests.

**Decision:** Add a tiny no-op `tests/support/readline.py` and prepend
`tests/support` to `PYTHONPATH` only in `make test`.

**Tradeoff:** This is environment compatibility plumbing, not application logic.
Normal runtime imports still use the system modules.
