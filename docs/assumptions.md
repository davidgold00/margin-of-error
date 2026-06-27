# Modeling and Economic Assumptions

Every non-default assumption used in any phase of this project is catalogued here
with its rationale and source. If you disagree with an assumption, change the value
in the relevant config file and re-run — no buried constants.

**Format:** Assumption → Rationale → Source → Sensitivity flag

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

---

## Data Assumptions

- **Assumption:** Kaggle competition data is a random 50/50 sample of the full
  2,930-record De Cock dataset.
- **Assumption:** Sale prices in the Ames dataset represent arm's-length market
  transactions. Non-arm's-length sales (family transfers, foreclosures) are not
  explicitly filtered in Phase 0; flagged for Phase 1 cleaning.
- **Assumption:** Square footage columns (GrLivArea, TotalBsmtSF, etc.) are in
  square feet, as stated in data_description.txt.
