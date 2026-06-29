# Architecture Decision Records (ADR Log)

Lightweight log of non-obvious modeling, infrastructure, and economic choices.
Each entry: context → decision → tradeoff. New entries appended chronologically.

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
optional `[causal]` extra so Phase 0–2 installs stay light. Phase 3 requires
`pip install -e '.[causal]'`.

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
