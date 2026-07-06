# The Underwriting App — A Complete Guide

This is the deep guide to the Margin of Error Streamlit tool: what it is, how
to launch it, what every input and output means, how to read the two charts,
and — most importantly — how to *use* it to see the project's findings with
your own hands. You don't need to have read anything else first; where a
concept comes from one of the project's phases, this guide explains it in
place and points to `docs/PROJECT_EXPLAINER.md` for the full story.

**Contents**

1. [What the app is (and isn't)](#1-what-the-app-is-and-isnt)
2. [Launching it](#2-launching-it)
3. [What happens when you change an input](#3-what-happens-when-you-change-an-input)
4. [The inputs, one by one](#4-the-inputs-one-by-one)
5. [The verdict banner](#5-the-verdict-banner)
6. [The three checks — the heart of the tool](#6-the-three-checks--the-heart-of-the-tool)
7. [Reading the value chart](#7-reading-the-value-chart)
8. [Reading the profit chart](#8-reading-the-profit-chart)
9. [The renovation table](#9-the-renovation-table)
10. [The assumptions panel](#10-the-assumptions-panel)
11. [Experiments that teach the project's findings](#11-experiments-that-teach-the-projects-findings)
12. [Changing the rules yourself](#12-changing-the-rules-yourself)
13. [Troubleshooting](#13-troubleshooting)
14. [How the app is built (for reviewers)](#14-how-the-app-is-built-for-reviewers)

---

## 1. What the app is (and isn't)

The app is the final product of a five-phase project about a single idea: **a
price prediction is not a buying decision.** You describe a house in Ames,
Iowa and a deal (what you'd pay, how much renovation you'd do). The app then:

1. **Prices the house** with a model trained on 1,460 real Ames sales — and,
   crucially, makes the model state an honest *90% value range*, not just one
   number.
2. **Simulates the flip 10,000 times**, drawing a plausible resale value from
   that range each time, plus a realistic holding period, and subtracting
   every cost: purchase, renovation, selling fees, carrying costs.
3. **Issues a verdict** — APPROVE, REFER, or DECLINE — from three explicit
   checks, and shows you exactly which checks passed or failed and by how
   much.

What it is **not**: a real estate product. The data is Ames, Iowa, 2006–2010;
the renovation costs are documented national-average assumptions; and the
whole thing is a portfolio demonstration of uncertainty-aware
decision-making, not investment advice. The app says this on its face, and it
is worth repeating here.

## 2. Launching it

From the repository root:

```bash
make setup          # once: create the venv and install pinned dependencies
# add the raw data files — data/README.md has copy-paste download commands
make data-check     # verify the data is in place
make train uncertainty app-artifacts   # once: build the saved models the app loads
make app            # launch — opens at http://localhost:8501
```

If the models are already built (they ship with the repo under `models/`),
`make app` alone is enough. The app **never trains anything at runtime** — it
loads the saved Phase 1 point model, the saved Phase 2 interval model, and a
saved feature-default profile, so what you interact with is exactly what the
project's reported results were produced with.

## 3. What happens when you change an input

Every time you touch a control, the whole pipeline re-runs live, in order:

```
your inputs
   → full feature vector          (your 12 inputs + typical Ames values for the rest)
   → point valuation              (Phase 1 LightGBM model, bias-corrected to dollars)
   → 90% value range              (Phase 2 conformalized quantile regression)
   → renovation uplift            (Phase 3 causal estimates for your chosen plan)
   → 10,000 profit simulations    (purchase, reno, selling, carrying costs; random hold time)
   → three checks → verdict       (thresholds from config/economics.yaml)
```

Nothing is cached per-house and nothing is hand-tuned: the verdict you see is
the same function the project's Phase 2 analysis and Phase 4 backtest used.

## 4. The inputs, one by one

The sidebar has two groups. Every field also has a `?` tooltip in the app.

**1 · The house** — the facts an underwriter would actually have:

| Input | What it means | Why it matters |
|---|---|---|
| Neighborhood | Ames district (codes like `NAmes` = North Ames) | The strongest location signal. Also the main driver of how *certain* the model is — it has seen many sales in some districts, few in others. |
| Living area | Finished above-ground square footage | One of the model's top two value drivers. |
| Overall quality | Assessor's 1–10 material/finish rating | The other top driver. 5–6 is average, 8+ is high-end. |
| Year built | Construction year (up to 2010, the dataset's end) | Age shapes value and renovation risk. |
| Full / half baths | Above-ground bathroom counts | Visible to buyers; also part of the causal renovation analysis. |
| Kitchen quality | Ames codes: `Po` poor · `Fa` fair · `TA` typical · `Gd` good · `Ex` excellent | A renovatable feature with a causally-estimated value lift. |
| Basement area | Total basement square footage (0 = none) | Usable space and renovation options. |
| Garage spaces / finish | Car capacity; `Unf` unfinished · `RFn` rough · `Fin` finished | Garage finish is one of the statistically clearest renovation effects. |

Everything else the model needs (roughly 70 more columns) is filled with the
most typical value in the dataset — medians for numbers, most-common values
for categories. That's a deliberate design choice: a screening tool should be
usable without an 80-field intake form. It also means the app's valuations
are for a "typical Ames house with your twelve facts," which is the honest
way to describe any quick-screen estimate.

**2 · The deal:**

- **Purchase price** — what you'd actually pay, before renovation. This is
  the single most consequential input; the whole verdict hinges on it.
- **Renovation plan** — three tiers (`minimal` ≈ $8,000 cosmetic; `moderate`
  ≈ $25,000 kitchen/bath refresh; `substantial` ≈ $60,000 full gut). Each
  tier's budget is subtracted as a cost, and each adds value using the
  project's *causal* Phase 3 estimates of what those upgrades are actually
  worth — not wishful percentages.

## 5. The verdict banner

The colored banner at the top is the answer. It shows three things: the
verdict word, what it means in plain English, and (in the caption underneath)
the decision engine's own note with the exact numbers that drove it.

- **✓ APPROVE — "Worth a serious look."** All three checks pass: the model is
  confident enough about the value, a good profit is likely enough, and a
  loss is unlikely enough — *on these assumptions*.
- **⚠ REFER — "Borderline — needs a human."** The numbers don't clearly fail
  but aren't strong enough to approve automatically. In a real shop this deal
  would go to a person. The thresholds for "borderline" are looser than for
  approval (see the next section).
- **✕ DECLINE — "Walk away."** At least one check fails. The tool's whole
  discipline is that nothing overrides a failed check — not a great point
  estimate, not an exciting renovation story.

A note on spirit: DECLINE does not mean "the model thinks you'll lose money."
Often it means **the model doesn't know** — and the project's central lesson
(learned the expensive way by Zillow Offers) is that not-knowing is itself a
reason to walk away.

## 6. The three checks — the heart of the tool

Every verdict comes from exactly three checks, shown as PASS / BORDERLINE /
FAIL rows directly under the banner — with this house's actual numbers
against the configured thresholds. There is no hidden scoring; this panel and
the decision engine literally read the same config values.

**Check 1 — Uncertainty: is the model sure enough about what the house is
worth?** The model's 90% value range must be no wider than **$60,000**. If it
is wider, the deal is declined *automatically*, even if the other two checks
look spectacular. This is the anti-Zillow guardrail and the project's
signature move: a model that cannot pin a home's value down to a $60k band
has no business betting capital on that home. (In the project's test set,
this one check alone declined 56% of homes — including all 50 of the deals a
naive point-estimate model ranked as its best opportunities.)

**Check 2 — Profit: is a genuinely good outcome likely enough?** At least
**65%** of the 10,000 simulations must clear a **$15,000** profit floor.
Between 50% and 65% is borderline (REFER territory). The floor exists because
a months-long, illiquid, hands-on project that *might* net $4,000 isn't worth
doing even when it's "profitable."

**Check 3 — Loss: is losing money unlikely enough?** At most **20%** of
simulations may end below break-even; between 20% and 30% is borderline. A
flipper who loses on one deal in four doesn't survive long, whatever the
average says.

APPROVE requires all three at their strict thresholds. REFER means nothing
failed outright but at least one check only cleared its looser borderline
bar. Everything else is DECLINE, and the failing check is highlighted.

All five numbers above come from `config/economics.yaml` — the app hardcodes
none of them, and section 12 shows you how to change them.

## 7. Reading the value chart

The horizontal blue band under "What is the house worth?" is the single most
important picture in the app:

- **The blue band** is the model's calibrated 90% value range *after* your
  chosen renovation. "Calibrated" is a checkable promise: ranges built this
  way contained the true sale price 90.4% of the time on 292 held-out homes
  the model had never seen. Wide band = the model honestly doesn't know.
- **The black dot** is the single best guess — the number a naive tool would
  stop at. Notice how much band surrounds it.
- **The dashed line ("You pay …")** is your purchase price. Your margin is
  the gap between that line and wherever the true value actually lands —
  *anywhere in the band, with 90% confidence*. If the line sits inside the
  band, part of the plausible world has you paying more than the house is
  worth.

The three tiles above the chart repeat the numbers: best guess, the range
itself, and the **range width** — the number Check 1 gates on.

## 8. Reading the profit chart

The histogram under "If you pay …, how does the flip go?" shows all 10,000
simulated outcomes of your exact deal:

- Each simulation draws a resale value from the value range, draws a holding
  period (about 4 months on average, sometimes much longer — flips run
  late), and subtracts every cost: your purchase price, the renovation
  budget, ~6% selling costs, and monthly carrying costs.
- **Red bars** are simulations that lose money; **blue bars** make money. The
  solid line at $0 is break-even; the dashed line marks the average outcome.
- The tiles above give the summary: the average, the **bad case (P10** — one
  simulation in ten ends *worse* than this**)**, the **good case (P90)**, and
  the **chance of losing money** — the number Check 3 gates on.

The histogram is why the tool outperforms a point estimate: two deals can
have the same *average* profit while one is a tight, safe hump and the other
is a wide smear with a fat red tail. The caption also shows what the
traditional "70% rule" would cap your offer at for this house — a useful
sanity anchor from the flipping trade.

## 9. The renovation table

"Which upgrades actually pay?" lists renovatable features with:

- **Value added** — the project's Phase 3 *causal* estimate of what one step
  of that upgrade adds to resale value. Causal matters: nice kitchens sit in
  nice houses, so naive correlations mis-state what the upgrade itself does.
  The project's flagship example: simple regression said exterior-quality
  work adds $425 per step; the causal estimate is $5,634 — a real effect that
  correlation *hid* thirteen-fold.
- **Typical cost** — a documented national-average assumption (not an Iowa
  quote), **Net gain**, and **Added per $1 spent**.
- **Bottom line** — ✓ pays for itself on these estimates, ✕ costs more than
  it adds, or — the house is already top quality on that feature.

Two honest caveats, also shown in the app: these are market-wide averages,
not quotes for your specific house; and the project's core finding is that
**no upgrade story rescues a deal the uncertainty check has declined** — in
testing, swapping causal for naive renovation assumptions flipped zero
verdicts, because the value range dwarfs any plausible renovation delta.

## 10. The assumptions panel

The expander at the bottom ("Every assumption behind these numbers") has two
tabs:

- **Plain English** — deal costs (6% selling, 0.8%/month carrying, all-cash),
  the random holding period, the three renovation tiers with budgets and
  scopes, and the exact verdict thresholds.
- **Raw config** — the literal loaded configuration, so you can verify the
  app runs on `config/economics.yaml` and nothing else.

Every value has a written rationale and a sensitivity flag in
`docs/assumptions.md`. If a number looks wrong to you, that's a feature: see
section 12.

## 11. Experiments that teach the project's findings

The app is at its best as a hands-on demonstration. Each experiment below
takes under a minute and reproduces one of the project's real findings.

**Experiment 1 — find the price where the verdict flips.** Leave the default
house alone and move only the purchase price. Walk it up in $5,000 steps and
watch the checks flip one by one — profit check first, then the loss check
(around the defaults, roughly $110k is an APPROVE and $128k already fails on
profit). *Lesson: the verdict is a function of price, not of the house. There
are no good houses, only good deals — and the margin between "buy" and "walk
away" is a few percent of the price, which is exactly why model error the
same size is lethal.*

**Experiment 2 — watch the model become uncertain.** Keep the deal fixed and
switch Neighborhood through several districts. The value range widens and
narrows dramatically — the model has seen hundreds of sales in some
neighborhoods and a handful in others, and honest uncertainty reflects that.
With an unusual combination (a very large or high-quality house in a modest
district), you can push the range past the $60,000 cap and watch Check 1
decline the deal *even at a price that simulates profitably*. *Lesson: this
is the anti-Zillow guardrail firing — a profitable-looking point estimate
means nothing when the model admits it doesn't know the value.*

**Experiment 3 — try to renovate your way out.** Take any declined deal and
upgrade everything: kitchen to `Ex`, more baths, finished garage, top overall
quality. The valuation rises, but watch the range width — better *inputs*
don't necessarily make the model more *certain*, and if Check 1 was the
problem, no renovation fixes it. *Lesson: the project's Phase 3 null result —
zero verdicts flipped when renovation assumptions changed — is a statement
about hierarchy: uncertainty dominates upgrades.*

**Experiment 4 — feel the tiers.** On a marginal deal, switch between
`minimal`, `moderate`, and `substantial` renovation plans. The substantial
tier adds the most value but costs $60,000 and stretches the same value
range, so it frequently makes marginal deals *worse*. *Lesson: renovation is
a cost with an uncertain payoff, not a free upside lever.*

**Experiment 5 — read a REFER.** Find a price where the banner turns amber
(around the defaults there is usually a narrow window under the APPROVE
price). Look at which check is borderline. *Lesson: real underwriting systems
have a human-escalation band, not just a binary; the thresholds encode who
decides.*

## 12. Changing the rules yourself

Everything the tool believes lives in `config/economics.yaml`, with a written
rationale next to each value. To run the app under *your* worldview:

1. Open `config/economics.yaml`.
2. Change, say, `flip.underwriting.max_acceptable_interval_width_usd` from
   `60000` to `40000` (a stricter uncertainty gate), or
   `holding_cost_monthly_pct` from `0.008` to `0.012` (a harsher carry).
3. Restart the app (`make app`). The checks panel, the simulations, and the
   assumptions expander all update — there is no second copy of any number to
   fall out of sync.

This is the project's "no buried constants" rule paying off: the app, the
Phase 2 analysis, and the Phase 4 backtest all read the same file, so an
assumption change propagates everywhere or nowhere.

## 13. Troubleshooting

- **"Missing app artifact(s) …" on launch** — the saved models aren't built.
  Run the command the error shows: `make train uncertainty app-artifacts`
  (requires the data files; see `data/README.md`).
- **`make app` fails to start** — the venv may be missing or stale; run
  `make setup` first. If tools misbehave when run by hand, invoke them
  through the venv's Python (`PYTHONPATH=src .venv/bin/python -m …`) rather
  than relying on shell activation.
- **A different port** — Streamlit defaults to 8501; if it's taken, pass
  `--server.port` via `streamlit run src/margin_of_error/app/underwriting.py`.
- **The verdict seems wrong** — read the three check rows first; one of them
  will name the binding constraint with its numbers. If you disagree with the
  *threshold*, that's section 12. If you disagree with the *valuation*,
  remember ~70 features are defaulted to typical Ames values (section 4).

## 14. How the app is built (for reviewers)

For anyone assessing the engineering rather than the economics:

- **Pure core, thin shell.** All decision logic lives in pure, typed,
  docstringed functions (`underwrite_property`, `gate_checks`,
  `renovation_guidance` in `src/margin_of_error/app/underwriting.py`), unit
  tested in `tests/test_phase5.py`. Streamlit code only renders their output.
- **One source of truth for the rule.** The PASS/FAIL panel is computed from
  the same `EconomicsConfig` object the decision engine uses
  (`classify_verdict` in `src/margin_of_error/economics/underwriter.py`), so
  the explanation can't drift from the decision.
- **Artifacts, not retraining.** The app loads the Phase 1 model, Phase 2
  CQR model, and feature defaults from `models/`; a missing artifact produces
  a remediation message, not a stack trace.
- **Deliberate visual design.** Charts use a colorblind-checked palette
  (blue #2a78d6 / red #e34948 for outcomes — worst-case CVD ΔE 74.6, both
  ≥3:1 contrast on the surface), status colors always paired with an icon and
  a word rather than color alone, direct labels on single-series charts, a
  legend on the two-series histogram, and quiet hairline axes. The Streamlit
  theme lives in `.streamlit/config.toml`.
- **Determinism.** Simulations are seeded; the same inputs always produce
  the same verdict.
