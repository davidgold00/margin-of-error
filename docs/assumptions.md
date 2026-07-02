# Modeling and Economic Assumptions

Every result in this project sits on top of assumptions — what a renovation
costs, how long a flip takes, what transaction fees run, how the model is
validated. This file is the complete catalog: **every non-default assumption
used in any phase, with its rationale, its source, and a sensitivity flag**
indicating how much the conclusions would move if the assumption is wrong.

Three things make this catalog more than documentation:

1. **No buried constants.** Every economic parameter lives in
   `config/economics.yaml` or `config/model.yaml`, never hard-coded — a test
   in the suite enforces this. If you disagree with an assumption, change the
   YAML value and re-run; the metric cards will regenerate under your number.
2. **Placeholders are labeled.** Values marked **PLACEHOLDER** are documented
   estimates (usually national averages) that a production deployment must
   replace with local data. They are flagged, not hidden.
3. **Sensitivity flags tell you where to push.** "High" means the headline
   conclusions are meaningfully exposed to this value; "Low" means they are
   not. If you want to attack the project, start with the High rows.

**How to read each entry:** Assumption → Rationale → Source → Sensitivity
flag. Design decisions (as opposed to parameter values) live in the companion
ADR log, `docs/decisions.md`.

---

## Economic Assumptions (`config/economics.yaml`)

### Transaction Costs

| Parameter | Value | Rationale | Source | Sensitivity |
|---|---|---|---|---|
| `buy_side_pct` | 1% | Buyer closing costs (title, escrow, inspection, origination fees). Conservative mid-point of 0.5–2% range. | Stated estimate | Low — varies by state but narrow range |
| `sell_side_pct` | 6% | Total commission (buyer + seller agents). Traditional split. Post-2024 NAR settlement may lower this in practice. | NAR historical average | Medium — test at 4.5% and 6% |

### Holding Costs

| Parameter | Value | Rationale | Source | Sensitivity |
|---|---|---|---|---|
| `monthly_cost_pct` | 0.5% of purchase price / month | Property taxes + hazard insurance + utilities + minor maintenance. Iowa property tax ~1.5% / year; insurance ~0.5%; utilities vary. | Stated estimate — **PLACEHOLDER, Iowa-specific** | High — drives total hold cost |
| `typical_hold_months` | 6 months | Industry rule of thumb for light-to-medium renovation. Does not account for permit delays (which can add 2–4 months in some markets). | Stated assumption | High — test at 4 and 9 months |

### Financing

| Parameter | Value | Rationale | Source | Sensitivity |
|---|---|---|---|---|
| `hard_money_rate` | 10% annual | U.S. hard-money/bridge loan rates in 2023–2024 ranged 9–13%. Mid-range. The 2006–2010 era rates were lower (~8–10%); Phase 4 should use a time-varying series. | Stated estimate — **PLACEHOLDER for Phase 4** | High |
| `ltv` | 80% | Optimistic hard-money LTV. Conservative underwriting uses 65–70%. | Stated assumption | Medium — test at 70% |

### Renovation Costs

**CRITICAL PLACEHOLDER:** All renovation cost estimates are national averages from
Remodeling Magazine (2023). Iowa labor and material costs are typically 10–25% below
national averages. These values must be adjusted with local data before any real
investment decision.

| Renovation | Estimated Cost | Source |
|---|---|---|
| Minor kitchen remodel | $27,000 | Remodeling Mag Cost vs. Value 2023, national avg — **PLACEHOLDER** |
| Full bathroom addition | $49,000 | Remodeling Mag 2023 — **PLACEHOLDER** |
| Minor bathroom remodel | $11,000 | Remodeling Mag 2023 — **PLACEHOLDER** |
| Basement finish | $35/sqft | Wide range: $25–75/sqft depending on finish level — **PLACEHOLDER** |

### Profit Thresholds

| Parameter | Value | Rationale | Source |
|---|---|---|---|
| `minimum_margin_pct` | 10% of ARV | After all costs, a 10% margin of ARV provides headroom for ~2 standard deviations of typical cost overrun at the project scale. The "70% rule" (buy at 70% of ARV minus repairs) implies ~30% gross margin, but after transaction and holding costs a 10% net is realistic. | Stated assumption — **PLACEHOLDER; subject to Phase 2 sensitivity analysis** |
| `min_absolute_usd` | $20,000 | Floor to ensure deals make economic sense irrespective of home price. Roughly compensates for 6–8 months of management time at a modest opportunity cost. | Stated assumption — **PLACEHOLDER** |

---

## Phase 2 Fix-and-Flip Economics (`config/economics.yaml` → `flip`)

The `flip` block is the self-contained parameter set for the Phase 2 profit Monte
Carlo and underwriting rule. It is intentionally separate from the legacy
`transaction` / `holding` / `financing` blocks (which Phase 1 documented) so Phase 2
has one coherent, reviewable source of truth. See ADR-012/013/014 in decisions.md.

### Acquisition and costs

| Parameter | Value | Rationale | Source | Sensitivity |
|---|---|---|---|---|
| `acquisition_arv_factor` | 0.70 | Industry "70% rule": Maximum Allowable Offer = 0.70 × ARV − renovation cost. Establishes a ~30% gross margin so a profitable baseline exists and the binding question is whether model *uncertainty* fits inside it. | Industry standard | **High** — the result's sensitivity to this is the key open assumption (ADR-012) |
| `transaction_cost_pct` | 6% of purchase price | Selling commission + transfer taxes + closing, applied per the Phase 2 brief's profit formula. | Industry estimate (iBuyer disclosures, NAR) | Medium |
| `holding_cost_monthly_pct` | 0.8% / month | Property tax + insurance + utilities + maintenance + opportunity. Cash deal, so no debt carry. | Stated estimate | Medium |
| `financing_assumption` | cash | Cleanest capital structure for a portfolio project. Leverage would amplify both return and downside; modeled as a Phase 5 case. | Design choice | — |

### Holding period (modeled as a distribution, not a point)

| Parameter | Value | Rationale | Sensitivity |
|---|---|---|---|
| `holding_period_months_base` | 4 | Expected hold from acquisition to resale close. | High |
| `holding_period_months_std` | 1.5 | Flips rarely close on schedule; the carry cost has real variance. | Medium |
| `holding_period_months_min` / `max` | 1 / 12 | Truncation bounds for the Normal. | Low |

> **Note on the legacy `holding` block:** Phase 1's `holding.typical_hold_months`
> was 6 (a deterministic point). Phase 2 instead models hold time as a truncated
> Normal centered at 4 with a tail extending past 6 — a more honest treatment for
> an uncertainty-focused phase. The legacy block is retained for Phase 1
> reproducibility; Phase 2 uses only the `flip` block. **Flagged for review.**

### Renovation tiers (cost + uplift PRIORS)

| Tier | `cost_usd` | `value_uplift_pct` | Scope | Source |
|---|---|---|---|---|
| minimal | $8,000 | 4% | Cosmetic — paint, fixtures, landscaping, minor repairs | HomeAdvisor / Remodeling Mag ranges — **PRIOR, not a finding** |
| moderate | $25,000 | 10% | Kitchen/bath refresh, flooring, lighting | same |
| substantial | $60,000 | 18% | Structural, full kitchen/bath gut, HVAC, systems | same |

> The `value_uplift_pct` figures are **conservative priors**, explicitly *not*
> findings. **Phase 3 (causal/DML) replaces them with data-derived treatment
> effects.** In the dataset-wide pass the verdict is dominated by the value interval
> and the 70%-rule margin, not the uplift; the uplift priors matter most for the
> Phase 5 what-if app.

### Underwriting thresholds (`flip.underwriting`)

| Parameter | Value | Rationale | Sensitivity |
|---|---|---|---|
| `minimum_underwrite_margin_buffer_usd` | $15,000 | Profit floor below which a months-long illiquid bet is not worth it; P(profit > buffer) drives APPROVE/REFER. | High |
| `max_acceptable_interval_width_usd` | $60,000 | If the 90% CQR interval is wider than this, DECLINE on uncertainty regardless of the point estimate (the anti-Zillow guardrail). | **High** — near the empirical median width, so it is the dominant gate |
| `approve_prob_above_min_margin` | 0.65 | "More likely than not, with room to spare" before committing capital. | High |
| `approve_prob_loss_max` | 0.20 | A flipper cannot survive losing on one in four deals. | High |
| `refer_prob_above_min_margin` / `refer_prob_loss_max` | 0.50 / 0.30 | Borderline band routed to a human. | Medium |
| `monte_carlo_samples` | 10,000 | Draws per property per tier; summary stats stored only. | Low |
| `arv_normal_z` | 1.645 | z₀.₉₅, maps a 90% interval half-width to a working Normal std for ARV sampling. | Low |

---

## Modeling Assumptions (`config/model.yaml`)

### Target Variable

- **Assumption:** Model `log1p(SalePrice)` rather than `SalePrice` directly.
- **Rationale:** SalePrice is right-skewed; log transform stabilizes variance and
  converts absolute errors to percentage errors, which are more economically
  interpretable. `log1p` rather than `log` avoids domain issues if SalePrice = 0
  (which shouldn't occur in a valid dataset, but is defensive programming).
- **Implication:** All RMSE figures reported in log scale must be converted back
  via `expm1()` for dollar interpretation. Phase 1 will report both.

### Cross-Validation

- **Assumption:** 5-fold CV, stratified by Neighborhood.
- **Rationale:** Neighborhood is the strongest spatial proxy. Stratifying ensures
  each neighborhood appears proportionally in each fold, reducing optimistic bias
  from spatial clustering.
- **Limitation:** Not full spatial CV — nearby homes in the same neighborhood can
  still share information across fold boundaries. Logged as backlog.

### Conformal Calibration

- **Assumption:** 15% static hold-out from training data for conformal calibration.
- **Rationale:** See ADR-004 in docs/decisions.md.
- **Coverage guarantee:** CQR provides finite-sample marginal coverage at level
  `1 - alpha` = 90%, provided calibration data is exchangeable with test data.
  The random Kaggle split makes this reasonable for cross-sectional predictions.
  It does NOT hold for temporal distribution shift (Phase 4 backtest).

### Feature Treatment

- **Assumption:** `OverallQual` is treated as an ordinal integer (1–10) and used
  as a predictor in the point model.
- **Causal note (Phase 3):** `OverallQual` is a *confounder* in any renovation
  effect estimation — it reflects build quality that correlates with renovation
  choice but cannot itself be renovated. The causal layer (Phase 3) explicitly
  controls for it in the DML model. Naive regression coefficients on
  "renovatable" features will be confounded by `OverallQual`.

### Missingness Treatment

- **Assumption:** Ames NA values are interpreted from `data_description.txt`
  before imputation.
- **Structural absence:** `Garage*`, `Bsmt*`, `PoolQC`, `FireplaceQu`, `Fence`,
  `Alley`, `MiscFeature`, and `MasVnrType` NAs mean the feature is absent and are
  filled as `"None"` or `0` inside the sklearn preprocessing pipeline.
- **True missingness:** `LotFrontage` is imputed with fold-local neighborhood
  medians and a fold-local global fallback. Other true-missing numeric/categorical
  values use fold-local median/mode.
- **Leakage control:** These values are learned inside CV folds only; no imputer
  is fit on the full dataset before validation.

### Phase 3 Causal Identification

- **Estimand:** each DML coefficient is the average causal effect of a one-unit
  treatment increase on `log1p(SalePrice)`, reported in dollars as
  `coefficient × median(SalePrice)`. This is a local approximation around the
  median Ames sale price, not an exact global dollar transformation.
- **Conditional independence assumption (CIA):** conditional on the fixed feature
  registry plus `OverallQual` and `OverallCond`, treatment assignment is assumed
  plausibly as-good-as-random. The argument is that renovation choices in Ames are
  strongly tied to property quality, age, size, structural constraints, and
  neighborhood, all of which are represented in W.
- **Remaining threats:** unobserved owner wealth, undocumented deferred
  maintenance, contractor quality, permit constraints, and micro-location effects
  may still affect both treatment quality and sale price. Phase 3 estimates are
  therefore decision-grade observational estimates, not randomized-trial truth.
- **Cross-sectional scope:** Phase 3 uses the Kaggle random cross-section. It does
  not prove the same renovation effects held during every month of the 2006-2010
  crash regime; Phase 4 tests temporal robustness.

### Phase 3 Treatment Definitions

Every treatment below is included only if it appears in the Phase 1 mutable
registry. One-unit meanings are the actual units used by DML.

| Treatment | One-unit meaning | Renovatable rationale | Caveat |
|---|---|---|---|
| `KitchenQual` | One ordinal quality step | Kitchen quality is directly changed by a kitchen renovation. | Ordinal spacing is assumed equal. |
| `BsmtFinType1` | One basement-finish step | Basement finish can be improved. | Finish type also reflects existing basement layout. |
| `HeatingQC` | One ordinal quality step | HVAC quality/condition can be upgraded. | May proxy unobserved system age. |
| `FireplaceQu` | One ordinal quality step | Fireplace finish can be improved. | NA is structural absence and encoded as 0. |
| `GarageFinish` | One garage-finish step | Garage interior finish can be improved. | Garage size remains a fixed confounder. |
| `ExterQual` | One ordinal quality step | Exterior finish/materials can be upgraded. | Ambiguous: partly cosmetic, partly structural; included with caution. |
| `FullBath` | One full bathroom | Bathroom additions/remodels are investor scope. | Adding plumbing may be structurally constrained. |
| `HalfBath` | One half bathroom | Half-bath additions/remodels are investor scope. | CI crosses zero in Phase 3. |
| `BsmtFullBath` | One basement full bathroom | Basement bath additions are renovation scope. | Depends on basement feasibility. |

Brief-recommended `BsmtQual`, `Fireplaces`, and `GarageCars` were excluded as
treatments because the registry tags them fixed. `OverallQual` and `OverallCond`
are not treatments; they are confounders in the DML specification.

### Phase 3 Sensitivity Check

The informal sensitivity ratio in `reports/phase3_metric_card.json` is
`DML causal estimate / naive OLS estimate`. It is a rough Kling-Manski-style
screen, not a formal bound. The top three absolute causal effects were:

| Feature | DML causal | Naive OLS | Ratio |
|---|---:|---:|---:|
| `BsmtFullBath` | $8,520 | $9,581 | 0.89 |
| `ExterQual` | $5,634 | $425 | 13.25 |
| `KitchenQual` | $4,450 | $4,146 | 1.07 |

### Phase 1 Baseline Results

- **Data:** Kaggle `train.csv`, 1,460 rows, random cross-sectional split.
- **Validation:** 5-fold CV repeated 3 times; folds are stratified by neighborhood
  after rare-neighborhood bucketing.
- **Primary strawman:** LightGBM on `log1p(SalePrice)` with Duan-smearing dollar
  retransformation.
- **Result:** LightGBM RMSE is `0.135 ± 0.015` in log space and `$28,500 ± $6,381`
  in dollar space. Median absolute OOF dollar error is `$9,413`; the 80th
  percentile absolute error is `$22,193`.

---

## Data Assumptions

- **Assumption:** Kaggle competition data is a random 50/50 sample of the full
  2,930-record De Cock dataset.
- **Assumption:** Sale prices in the Ames dataset represent arm's-length market
  transactions. Non-arm's-length sales (family transfers, foreclosures) are not
  explicitly filtered in Phase 0; flagged for Phase 1 cleaning.
- **Assumption:** Square footage columns (GrLivArea, TotalBsmtSF, etc.) are in
  square feet, as stated in data_description.txt.

---

## Phase 4 Backtest Assumptions

### Temporal Data

- **Assumption:** `data/raw/ames/AmesHousing.csv` is the full De Cock Ames dataset
  with 2,930 sales from 2006-2010, sorted by `YrSold` and `MoSold`.
- **Rationale:** The Kaggle split is random and cannot test time drift.
- **Observed market move:** Ames median sale price peaked at $165,125 in 2007 and
  reached $155,000 in 2010, a 6.1% peak-to-trough decline.
- **Sensitivity flag:** High. A market with a larger drawdown would likely produce
  stronger regime-shift stress.

### Acquisition Regimes

| Regime | ARV factor | Purpose |
|---|---:|---|
| `conservative_flip` | 0.70 | Industry 70% rule; tests whether a fat-margin flipper needs the uncertainty gate in a mild downturn. |
| `ibuyer` | 0.85 | Thin-margin buyer near model value; tests the Zillow-style failure mode where model error becomes capital risk. |

These are methodology scenarios. They are not claims that every investor or Zillow
Offers used exactly these factors.

### Realized P&L Basis

- **Assumption:** Observed sale price is treated as realized ARV.
- **Rationale:** Ames does not contain true buy-renovate-resell pairs, so this is
  the observable resale value available for a stress test.
- **Excluded:** Renovation cost and renovation uplift are excluded from Phase 4
  realized P&L because adding a renovated resale counterfactual would require
  inventing an outcome.
- **Sensitivity flag:** High. A real flip transaction dataset would be the right
  production backtest.

### Coverage Interpretation

- **Assumption:** Phase 4 reports weighted monthly empirical coverage against the
  90% nominal CQR target.
- **Observed:** Coverage moved from 90.1% pre-crash to 89.5% in the crash window.
- **Interpretation:** This is a small dip below target, not a dramatic coverage
  failure. The headline Phase 4 finding is economic drawdown reduction under thin
  margins.

---

## Phase 5 App Assumptions

### Feature Defaults

- **Assumption:** The app user supplies a compact set of practical inputs and all
  other Ames columns are filled from Kaggle training medians/modes stored in
  `models/phase5/feature_defaults.json`.
- **Exposed fields:** neighborhood, living area, overall quality, year built,
  full baths, half baths, kitchen quality, basement area, garage spaces, and
  garage finish.
- **Rationale:** A screening tool should be usable without an 80-field intake
  form, but the model still needs the full raw feature vector.
- **Sensitivity flag:** Medium. A production tool should capture more fields or
  provide confidence penalties for default-heavy inputs.

### Runtime Artifacts

- **Assumption:** The app loads saved model artifacts:
  `models/phase1/baseline_lightgbm.joblib`, `models/phase2/cqr_90.joblib`, and
  `models/phase5/feature_defaults.json`.
- **Rationale:** The app is a product surface, not a training script. Runtime
  retraining would be slow and unreproducible.
- **Failure mode:** If an artifact is missing, the app shows a clear remediation
  message instructing the user to run `make train uncertainty app-artifacts`.
