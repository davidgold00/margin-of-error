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
