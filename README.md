# Margin of Error

> **Accurate is not underwritable.** A model can predict house prices well and
> still be too uncertain to bet money on. The model's margin of error must be
> smaller than the deal's profit margin — otherwise you are gambling, not
> investing.

## The story in one minute

In 2021, Zillow shut down **Zillow Offers**, a business that used the company's
price-prediction algorithm to buy houses directly, renovate them lightly, and
resell them. It lost hundreds of millions of dollars. The crucial detail: the
model was not unusually *bad* at predicting prices. The failure was treating a
**single predicted number** as if it were a **safe buying decision**. If your
model says a house is worth $180,000 but could honestly be off by $30,000 in
either direction, and your profit plan is $15,000, then every purchase is a
coin flip dressed up as analytics.

This project takes the most famous "predict the house price" dataset in data
science — Ames, Iowa — and refuses to play the usual leaderboard game. Instead
of asking *"how low can the error metric go?"*, it asks the Zillow question:

> **"Is the model certain enough about THIS house to bet capital on it?"**

The answer is a five-phase decision system: a strong baseline price model, a
calibrated 90% value interval wrapped around it, a Monte Carlo profit
simulation with an APPROVE / REFER / DECLINE rule, a causal analysis of which
renovations actually create value, a walk-forward stress test through the
2007–2010 Ames downturn, and finally a Streamlit underwriting app that puts
the whole thing behind a form.

For the full plain-English walkthrough — every concept explained from scratch,
every number interpreted — read
**[docs/PROJECT_EXPLAINER.md](docs/PROJECT_EXPLAINER.md)**. It is the single
best place to start.

![Three strategies through the downturn](reports/figures/04b_three_strategies_pnl.png)

## Results at a glance — and what each one means

Every number below is computed by the pipeline and stored in a machine-readable
"metric card" (a JSON file under `reports/`). Nothing here is typed in by hand.

| Question | Result | What it actually means | Source |
|---|---|---|---|
| How wrong is the point model in dollars? | Typical absolute error **$9,413**; 80th percentile **$22,193** | On a random Ames house, expect to be ~$9k off. One house in five is missed by more than $22k — roughly the entire profit of a typical flip. | `reports/phase1_metric_card.json` |
| Is the 90% interval honest? | **90.4%** empirical coverage on 292 held-out homes | When the model says "90% sure the value is in this range," it is right 90.4% of the time. The uncertainty claim has been audited, not just asserted. | `reports/phase2_metric_card.json` |
| How wide is the honest value range? | Median 90% interval width **$64,025** | For a market with a $163,000 median price, the truthful answer to "what is this house worth?" spans nearly 40% of the price. This is the project's central shock. | `reports/phase2_metric_card.json` |
| What does the uncertainty gate do? | Declines **164 of 292** homes; rejects **50 of 50** top naive picks | Every decline had the same cause: model uncertainty wider than the $60k cap. The homes a point-estimate model likes *best* are exactly the ones the gate refuses — the Zillow adverse-selection trap, quantified. | `reports/phase2_metric_card.json` |
| Do naive renovation numbers mislead? | Exterior quality: **$425** naive vs **$5,634** causal per step | Plain correlation understated a real renovation effect 13-fold. Confounding can hide value, not just inflate it. | `reports/phase3_metric_card.json` |
| Did Ames actually crash? | Median price fell **6.1%** from 2007 peak to 2010 trough | A mild downturn, reported honestly. Ames 2008 is not Phoenix 2008, and the docs never pretend otherwise. | `reports/phase4_metric_card.json` |
| Does the discipline pay under thin margins? | Max drawdown: naive **$129,522** → uncertainty-aware **$21,257** | In the iBuyer-style regime (buying near model value), the uncertainty gate cut the worst losing streak by 84% and raised the crash-window hit rate from 76.6% to 88.1%. | `reports/phase4_metric_card.json` |
| Does it matter for fat-margin flippers? | Both rules: **$0** max drawdown under the 70% rule | No. A ~30% purchase discount trivially absorbs a 6% market move. Uncertainty discipline matters in proportion to how close you buy to model value — that nuance *is* the thesis. | `reports/phase4_metric_card.json` |

## The five phases, briefly

| Phase | Question it answers | Deliverable |
|---|---|---|
| 1 | How large are a good price model's errors, in dollars? | LightGBM baseline with Duan-smearing dollar retransformation and residual diagnostics |
| 2 | What happens when the model must state its uncertainty before a buy decision? | Conformalized Quantile Regression 90% intervals, Monte Carlo profit simulation, APPROVE / REFER / DECLINE rule |
| 3 | Which renovations *cause* value, rather than merely correlating with nicer houses? | Cross-fitted Double Machine Learning effects for nine renovatable features |
| 4 | Would the rule have actually helped, replayed through 2006–2010 with no time travel? | Walk-forward backtest, two acquisition regimes (conservative 70% flip vs thin-margin iBuyer) |
| 5 | Can a person use this? | Streamlit underwriting tool, project explainer, strategy memo, deck outline |

## How to access the results — a guided tour

You do not need to run any code to inspect the findings. Everything the
pipeline produces is committed under `reports/` and `models/`.

### 1. Metric cards (`reports/phase*_metric_card.json`)

These are the ground truth for every claim in every document. Each phase writes
one JSON card containing its headline numbers, the full config snapshot it ran
under, and a UTC timestamp — so any figure in the memo or explainer can be
traced back to its source. Pretty-print one from the repo root:

```bash
python3 -m json.tool reports/phase2_metric_card.json
```

What to look for in each card:

- **`phase1_metric_card.json`** — cross-validated log-RMSE and dollar RMSE for
  the median/ElasticNet/LightGBM ladder, plus the out-of-fold absolute dollar
  error percentiles (the $9,413 / $22,193 / $45,283 numbers).
- **`phase2_metric_card.json`** — the `cqr.calibration_curve` block shows
  promised vs delivered coverage at 50/60/70/80/90/95% (this is the audit of
  the interval's honesty); the `headline` block shows verdict counts, interval
  widths, and the 50-of-50 naive-picks rejection; `economics_config_snapshot`
  records exactly which assumptions the run used.
- **`phase3_metric_card.json`** — naive OLS vs DML causal dollar effects per
  renovatable feature, 95% confidence intervals, and the verdict-flip count.
- **`phase4_metric_card.json`** — the richest card: per-strategy,
  per-regime results (`strategies.conservative_flip` and `strategies.ibuyer`,
  each with buy-all / naive-point / uncertainty-aware entries), coverage
  before vs during the downturn, and the yearly median-price path.

### 2. Figures (`reports/figures/`)

Each figure is numbered by phase. In reading order:

| Figure | What it shows |
|---|---|
| `01_price_error_vs_home_price.svg` | Phase 1 dollar errors across the price range — the "respectable model, deal-sized misses" picture. |
| `02a_confrontation.png` | Point estimates vs the calibrated 90% intervals — the moment the model is forced to confess. |
| `02b_zillow_trap.png` | Why the naive model's favorite deals are the most dangerous ones. |
| `02c_approval_by_neighborhood.png` | Where the uncertainty gate approves and declines across Ames neighborhoods. |
| `02d_calibration.png` | Promised vs delivered coverage — the honesty audit as a chart. |
| `03a_confounding_gap.png` | Naive vs causal renovation effects side by side (the $425 → $5,634 exterior-quality gap). |
| `03b_renovation_decision_matrix.png` | Which renovations clear their cost once effects are causal. |
| `03c_verdict_flip_distributions.png` | Why better renovation math still flips zero verdicts: the interval dominates. |
| `04a_coverage_over_time.png` | Interval coverage month by month through the downturn (90.1% → 89.5%). |
| `04b_three_strategies_pnl.png` | **The signature chart.** Cumulative profit for buy-all, naive, and uncertainty-aware rules under thin-margin iBuyer pricing. |
| `04c_conservative_regime_pnl.png` | The same race under the fat-margin 70% rule, where nothing draws down — the contrast that makes the thesis. |

### 3. Supporting tables (`reports/*.csv`)

- `phase1_oof_residuals.csv` — per-home out-of-fold prediction errors behind
  the Phase 1 percentiles.
- `phase2_calibration.csv` / `phase2_test_underwriting.csv` — per-home
  intervals, simulated profit summaries, and verdicts for the 292 test homes.
- `phase3_causal_effects.csv` / `phase3_underwriting_comparison.csv` — the
  full DML effect table and the causal-vs-correlational verdict comparison.
- `phase4_backtest_periods.csv` — month-by-month backtest results feeding the
  P&L and coverage charts.

### 4. Narrative documents

Read them in this order:

1. **[docs/PROJECT_EXPLAINER.md](docs/PROJECT_EXPLAINER.md)** — the deep
   walkthrough. Concepts from scratch, every result interpreted, likely
   interview questions with answers, a numbers cheat-sheet, and a glossary.
2. **[reports/memo.md](reports/memo.md)** — the one-page executive strategy
   memo (situation → complication → question → answer → evidence).
3. **[reports/deck_outline.md](reports/deck_outline.md)** — a ten-slide
   presentation skeleton with speaker notes.
4. **[docs/assumptions.md](docs/assumptions.md)** — every economic and
   modeling assumption with rationale, source, and sensitivity flag.
5. **[docs/decisions.md](docs/decisions.md)** — the ADR log: every non-obvious
   judgment call, in chronological order, with tradeoffs.

### 5. Notebooks (`notebooks/`)

The notebooks are the narrative companions to each phase (`00_eda` through
`05_backtest`). All real logic lives in `src/margin_of_error/` and is imported
into the notebooks — they show the story, the library does the work.

## The underwriting app

The Streamlit app is the Phase 5 product surface. It loads the saved Phase 1
point model, the saved Phase 2 CQR interval model, the Phase 3 causal uplift
configuration, and a Phase 5 feature-default profile — it never retrains at
runtime. You enter the property facts an underwriter would actually have
(neighborhood, living area, overall quality, year built, baths, kitchen
quality, basement area, garage spaces, garage finish); the remaining Ames
features are filled from dataset medians and modes.

The output is a full underwriting screen: point valuation, calibrated 90%
interval, simulated profit distribution, an **APPROVE / REFER / DECLINE**
verdict with its reason, causal renovation guidance, and an expander showing
every economic assumption behind the numbers. If a model artifact is missing,
the app tells you exactly which `make` targets to run rather than failing
mysteriously.

```bash
make app
```

## Run it yourself

```bash
# 1. Create the environment (venv + pinned deps + pre-commit hooks)
make setup

# 2. Add the two raw data files — see data/README.md for download commands
make data-check

# 3. Reproduce the pipeline, phase by phase
make train          # Phase 1: baseline models + metric card
make uncertainty    # Phase 2: CQR intervals + profit simulation + verdicts
make causal         # Phase 3: DML causal effects
make backtest       # Phase 4: walk-forward crash backtest
make app-artifacts  # Phase 5: feature defaults for the app

# 4. Quality gates
make lint
make test

# 5. Launch the underwriting tool
make app
```

`make all` runs the whole non-interactive pipeline end to end: data check, all
model phases, app artifact build, lint, and tests. Runs are seeded and
dependencies pinned (`requirements-lock.txt`), so the metric cards reproduce.

## Project structure

```text
margin-of-error/
├── config/          # model.yaml + economics.yaml — every assumption lives here, never in code
├── data/            # raw files are git-ignored; data/README.md explains how to get them
├── docs/            # explainer, ADR log, assumptions catalog
├── models/          # saved artifacts: Phase 1 LightGBM, Phase 2 CQR, Phase 5 defaults
├── notebooks/       # narrative notebooks; production logic lives in src/
├── reports/         # metric cards (JSON), tables (CSV), figures, memo, deck outline
├── src/margin_of_error/
│   ├── app/         # Phase 5 Streamlit tool and artifact loaders
│   ├── backtest/    # Phase 4 walk-forward stress test
│   ├── causal/      # Phase 3 cross-fitted DML estimation
│   ├── economics/   # profit Monte Carlo and the verdict rule
│   ├── models/      # Phase 1 baseline and Phase 2 CQR
│   └── viz/         # signature charts
└── tests/           # per-phase test suites, including a test that bans magic economic constants in code
```

Two engineering rules hold everywhere: **every economic parameter comes from
`config/economics.yaml`** (a test enforces that no magic dollar constants hide
in code), and **each phase loads the previous phase's saved artifact** rather
than silently retraining — so Phase 2 genuinely analyzes the same model
Phase 1 built.

## Data

- **Phases 1–3** use Kaggle's Ames competition training split: 1,460 sales,
  ~80 features, median price $163,000. The split is *random*, which is fine
  for cross-sectional questions and explicitly wrong for temporal ones.
- **Phase 4** switches to the full De Cock Ames dataset: 2,930 sales spanning
  2006–2010, sorted by `YrSold` and `MoSold`, so the backtest experiences the
  downturn in real order.

Raw data is not committed. **[data/README.md](data/README.md)** has copy-paste
download commands for both sources.

## Honest caveats

This is a decision-system portfolio project, not a live investment product,
and it says so at every turn:

- **Ames is one small Midwestern market**, and its 2007–2010 decline was a
  mild 6.1% — a demonstration setting, not a national housing law.
- **The backtest uses synthetic acquisition prices** (a factor times predicted
  value) and observed sale price as realized resale value, because the dataset
  records sales, not real buy-renovate-resell projects. Phase 4 stress-tests
  the *underwriting rule*, not a literal P&L.
- **The causal layer is observational.** DML removes confounding you can
  measure; unobserved owner wealth, maintenance history, and contractor
  quality can still bias the effects.
- **Renovation costs are documented national-average assumptions**, not Iowa
  contractor bids.

The recommendation survives the caveats because it is a governance rule rather
than a claim about one magic model: **when the model's uncertainty is wider
than the deal's margin, do not buy.** Every caveat above is catalogued with a
sensitivity flag in [docs/assumptions.md](docs/assumptions.md).
