# Data Directory

This directory is **git-ignored** (except this README): raw data never enters
version control. To reproduce the project you need exactly two datasets, both
free, placed in the subdirectories described below. Once they are in place,
run `make data-check` from the repo root — it validates that both files exist,
parse, and match the expected schema before any phase will run.

**Why two datasets?** They are two cuts of the same underlying Ames, Iowa
sales records, and the difference between them is a core methodological point
of the project:

- The **Kaggle competition split** is a *random* half of the data. Random
  splits are fine for cross-sectional questions ("how accurate is the model
  on a typical house?") — Phases 1–3 use it.
- The **full De Cock dataset** contains all 2,930 sales with their real
  2006–2010 dates. Only time-ordered data can answer "would this rule have
  worked, month by month, through the downturn?" — Phase 4 uses it.

Using the random split for the backtest would be silent time travel (training
on 2010 sales to price a 2007 house). See `docs/decisions.md` for the ADR.

---

## 1. Kaggle Ames Housing Competition → `data/raw/kaggle/`

**Used by:** Phases 1, 2, and 3.
**Source:** [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Files needed:

```
data/raw/kaggle/
├── train.csv            (~1,460 rows, 81 columns, includes SalePrice)
├── test.csv             (~1,459 rows, 80 columns, no SalePrice)
├── data_description.txt (feature documentation — shared with the full dataset)
└── sample_submission.csv
```

**How to obtain (pick one):**

```bash
# Option A: Kaggle CLI (requires an API key in ~/.kaggle/kaggle.json)
pip install kaggle
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/raw/kaggle/
unzip data/raw/kaggle/house-prices-advanced-regression-techniques.zip -d data/raw/kaggle/

# Option B: Manual download from the Kaggle competition page (requires login).

# Option C: OpenML mirror — used for Phase 1 reproducibility if Kaggle auth is absent.
python3 -c "
from sklearn.datasets import fetch_openml
from pathlib import Path
data = fetch_openml(name='house_prices', as_frame=True, parser='auto')
out = Path('data/raw/kaggle/train.csv')
out.parent.mkdir(parents=True, exist_ok=True)
data.frame.to_csv(out, index=False)
print(data.frame.shape)
"
```

**Caveat worth repeating:** the Kaggle train/test split is **random, not
temporal**. It cannot be used for the Phase 4 crash backtest.

---

## 2. Full Ames Housing Dataset → `data/raw/ames/`

**Used by:** Phase 4 (the walk-forward backtest).
**Source:** De Cock, Dean (2011). "Ames, Iowa: Alternative to the Boston
Housing Data as an End of Year Statistics Project." *Journal of Statistics
Education*, 19(3).
Dataset: [https://jse.amstat.org/v19n3/decock/AmesHousing.xls](https://jse.amstat.org/v19n3/decock/AmesHousing.xls)

Files needed:

```
data/raw/ames/
├── AmesHousing.csv      (~2,930 rows, 82 columns, all sales 2006–2010)
└── data_description.txt (same file as the Kaggle version)
```

**How to obtain:**

```bash
mkdir -p data/raw/ames
curl -o /tmp/AmesHousing.xls https://jse.amstat.org/v19n3/decock/AmesHousing.xls
python3 -c "
import pandas as pd
df = pd.read_excel('/tmp/AmesHousing.xls')
df.to_csv('data/raw/ames/AmesHousing.csv', index=False)
print(f'Saved {len(df)} rows')
"
```

**Column-name differences:** the full dataset names some columns differently
from the Kaggle version (spaces vs. no spaces; `PID` instead of `Id`).
`src/margin_of_error/data/loaders.py` normalizes both to a common schema
before validation, so you do not need to rename anything by hand.

---

## Data Provenance

| File | Rows | Time span | Split | Used in |
|---|---|---|---|---|
| `kaggle/train.csv` | ~1,460 | 2006–2010 | Random 50% of the full dataset | Phases 1, 2, 3 |
| `kaggle/test.csv` | ~1,459 | 2006–2010 | Random 50% (no SalePrice) | Phase 1 (leaderboard only) |
| `ames/AmesHousing.csv` | ~2,930 | 2006–2010 | Full population, time-ordered by `YrSold`/`MoSold` | Phase 4 backtest |

## Licensing

- **Kaggle dataset:** redistributed under Kaggle competition terms — do not
  redistribute the raw files; see the competition page.
- **Full Ames dataset:** made available by the Journal of Statistics
  Education for educational/research use; see De Cock (2011) for terms.
