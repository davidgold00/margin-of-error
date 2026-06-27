# Data Directory

This directory is **git-ignored** (except this README). Raw data files must be
obtained separately and placed in the subdirectories below.

---

## Required Data Files

### 1. Kaggle Ames Housing Competition (`data/raw/kaggle/`)

Source: [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Files needed:
```
data/raw/kaggle/
├── train.csv            (~1,460 rows, 81 columns, includes SalePrice)
├── test.csv             (~1,459 rows, 80 columns, no SalePrice)
├── data_description.txt (feature documentation — shared with Ames full dataset)
└── sample_submission.csv
```

**How to obtain:**
```bash
# Option A: Kaggle CLI (requires API key in ~/.kaggle/kaggle.json)
pip install kaggle
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/raw/kaggle/
unzip data/raw/kaggle/house-prices-advanced-regression-techniques.zip -d data/raw/kaggle/

# Option B: Manual download from the Kaggle competition page (requires login).
```

**Important caveat:** The Kaggle train/test split is **random**, not temporal. This
makes it unsuitable for the crash backtest (Phase 4). Use the full Ames dataset for
that. See `docs/decisions.md` § Data split for details.

---

### 2. Full Ames Housing Dataset (`data/raw/ames/`)

Source: De Cock, Dean (2011). "Ames, Iowa: Alternative to the Boston Housing Data
as an End of Year Statistics Project." *Journal of Statistics Education*, 19(3).
Dataset: [https://jse.amstat.org/v19n3/decock/AmesHousing.xls](https://jse.amstat.org/v19n3/decock/AmesHousing.xls)

Files needed:
```
data/raw/ames/
├── AmesHousing.csv      (~2,930 rows, 82 columns, all sales 2006-2010)
└── data_description.txt (same file as Kaggle version, or download separately)
```

**How to obtain:**
```bash
mkdir -p data/raw/ames
# Download XLS and convert:
curl -o /tmp/AmesHousing.xls https://jse.amstat.org/v19n3/decock/AmesHousing.xls
python3 -c "
import pandas as pd
df = pd.read_excel('/tmp/AmesHousing.xls')
df.to_csv('data/raw/ames/AmesHousing.csv', index=False)
print(f'Saved {len(df)} rows')
"
```

**Column name differences:** The full Ames dataset uses slightly different column
names from the Kaggle version in some cases (e.g., spaces vs. no spaces, "PID"
instead of "Id"). The `src/margin_of_error/data/loaders.py` normalizes both to a
common schema before validation.

---

## Licensing

- **Kaggle dataset:** Redistributed under Kaggle competition terms. Do not
  redistribute the raw files. See the competition page for full terms.
- **Full Ames dataset:** Made available by the Journal of Statistics Education
  for educational/research use. See De Cock (2011) for terms.

---

## Data Provenance

| File | Rows | Time span | Random split? | Phase used |
|---|---|---|---|---|
| `kaggle/train.csv` | ~1,460 | 2006–2010 | Yes (random 50%) | 1, 2, 3 |
| `kaggle/test.csv` | ~1,459 | 2006–2010 | Yes (random 50%) | 1 (leaderboard) |
| `ames/AmesHousing.csv` | ~2,930 | 2006–2010 | No (full population) | 4 (backtest) |
