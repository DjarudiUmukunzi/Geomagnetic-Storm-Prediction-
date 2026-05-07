# Cross-Year Geomagnetic Storm Prediction Using Machine Learning
## Under Distributional Shift — 1995 to 2017

---

## Research Objective

Develop machine learning models capable of predicting geomagnetic storms across multiple years (1995–2017) despite rare event sparsity, nonstationary time-series behaviour, solar-cycle variation, and distributional shift between training and deployment years.

---

## Core Research Questions

1. Can models trained on one year successfully predict storms in other years?
2. Which model families generalise best across years?
3. How severe is distributional shift between years, and does it correlate with prediction failure?
4. Do tree-based models (XGBoost, RF) outperform sequence models (LSTM, GRU) under shift?
5. Does solar-cycle phase (minimum / rising / maximum / declining) affect cross-year transfer?

---

## Study Period

**Years covered:** 1995 – 2017 (23 years)

| Solar Cycle | Phase | Years |
|---|---|---|
| Cycle 22 tail | Declining / minimum | 1995 – 1996 |
| Cycle 23 | Rising → maximum → declining | 1997 – 2007 |
| Cycle 23/24 minimum | Deep minimum | 2008 – 2009 |
| Cycle 24 | Rising → maximum → declining | 2010 – 2017 |

Notable storms included in the dataset:

- **1998-05-04** — Dst −205 nT
- **2000-07-15** — Bastille Day storm, Dst −301 nT
- **2001-03-31** — Dst −387 nT
- **2003-10-29** — Halloween storm, Dst −383 nT
- **2004-11-07** — Dst −373 nT
- **2015-03-17** — St. Patrick's Day storm, Dst −223 nT (primary case study)

---

## Storm Window Construction

For each identified storm event, extract:

```
[  7 days pre-onset  ]  [  storm duration  ]  [  3 days recovery  ]
       quiet / precursor    main phase + SSC        gradual recovery
```

This captures quiet background, precursor solar wind signatures, sudden storm commencement (SSC), main phase injection, and ring current decay — giving the model realistic deployment context rather than only peak conditions.

Each yearly dataset contains both storm windows and background quiet periods to preserve the natural class imbalance.

---

## Data Pipeline

```
Step 1 — Storm Catalog
    Source Dst/Kp indices (OMNIWeb, WDC Kyoto)
    Threshold: Dst < −50 nT  (moderate)  |  Dst < −100 nT  (intense)
    Output: catalog of storm onset / peak / recovery timestamps per year

Step 2 — Yearly Dataset Extraction
    For each year: extract storm windows + matched quiet periods
    Save as:  data/yearly/storm_YYYY.csv

Step 3 — Preprocessing
    Parse Date_UTC, sort chronologically
    Remove cross-hemisphere columns
    Interpolate short gaps (< 10 min) linearly
    Flag and record longer gaps

Step 4 — Feature Engineering
    (see Feature Engineering section below)

Step 5 — Cross-Year Experiment Framework
    Train on year A → evaluate on all other years B
    506 total train/test year-pair evaluations
```

---

## Features

### Raw solar wind and IMF inputs

| Feature | Description | Unit |
|---|---|---|
| `B_Total` | IMF total magnitude | nT |
| `BX_GSE` | IMF X component (GSE) | nT |
| `BY_GSM` | IMF Y component (GSM) | nT |
| `BZ_GSM` | IMF Z component (GSM) — primary storm driver | nT |
| `flow_speed` | Solar wind bulk speed | km/s |
| `Pressure` | Solar wind dynamic pressure | nPa |
| `E_Field` | Interplanetary electric field | mV/m |
| `SYM_H` | High-resolution ring current index (≈ Dst) | nT |
| `AE_INDEX` | Auroral electrojet activity | nT |
| `dBHt` | Rate of change of horizontal field — **primary target** | nT/min |
| `sinMLT`, `cosMLT` | Magnetic local time (circular encoding) | — |

### Engineered features

```python
# Autoregressive lag features (physical momentum / curvature)
dBHt_t1       = dBHt.shift(1)     # momentum: where was the field 1 min ago?
dBHt_t2       = dBHt.shift(2)     # curvature: is the disturbance accelerating?

# Rolling statistics
dBHt_roll_mean_5   = dBHt.rolling(5).mean()
dBHt_roll_std_15   = dBHt.rolling(15).std()
dBHt_roll_max_30   = dBHt.abs().rolling(30).max()

# Solar wind coupling functions
Ey_coupling    = -flow_speed * BZ_GSM / 1000        # dawn-dusk E field (mV/m)
Ey_south       = Ey_coupling.clip(lower=0)           # only southward drives storm
BZ_cumint      = BZ_GSM.clip(upper=0).cumsum()       # cumulative southward BZ
BZ_roll_min_30 = BZ_GSM.rolling(30).min()

# Ring current state
dSYMH_dt       = SYM_H.diff()                        # injection rate
dSYMH_dt_lag1  = dSYMH_dt.shift(1)

# Prediction targets
dBHt_t_plus1   = dBHt.shift(-1)                      # 1-min-ahead regression
extreme_next   = (dBHt_t_plus1.abs() > p95).astype(int)   # classification
```

---

## Experimental Framework

### Cross-year train/test design

Train on a single year → test on all other 22 years. Repeat for every year.

```
Train 1995  →  Test: 1996 1997 1998 ... 2017       (22 evaluations)
Train 1996  →  Test: 1995 1997 1998 ... 2017       (22 evaluations)
...
Train 2017  →  Test: 1995 1996 1997 ... 2016       (22 evaluations)

Total: 23 × 22 = 506 train/test evaluations
```

**Critical rule:** No random shuffling. All splits are strictly chronological within and across years.

### Validation within a training year

```
|-- 70% train --|-- 15% val --|-- 15% holdout --|
                                 (chronological)
```

---

## Models

### Baseline (tree-based and linear)

| Model | Key hyperparameter | Notes |
|---|---|---|
| Logistic Regression | `C`, `class_weight='balanced'` | Linear boundary baseline |
| Decision Tree | `max_depth`, `min_samples_leaf` | Interpretable; §8.1 |
| Random Forest | `n_estimators`, `max_features='sqrt'` | §8.2.2 |
| XGBoost | `n_estimators`, `learning_rate`, `max_depth` | Often best on tabular |
| LightGBM | `num_leaves`, `min_data_in_leaf` | Faster than XGBoost |
| CatBoost | `depth`, `l2_leaf_reg` | Handles categorical natively |

### Deep learning (sequence models)

| Model | Input window | Notes |
|---|---|---|
| LSTM | 30–60 min | Captures temporal dependencies |
| Bidirectional LSTM | 30–60 min | Sees past and future context (non-causal) |
| GRU | 30–60 min | Lighter than LSTM, similar performance |
| CNN-LSTM | 30–60 min | CNN extracts local patterns, LSTM integrates |
| Attention-LSTM | 30–60 min | Learn which minutes matter most |

**Note on deep learning for this task:** Tree-based models often match or outperform LSTMs on 1-min tabular data when lag features are properly engineered. Train both and compare.

---

## Handling Class Imbalance

Extreme storm minutes are typically < 5% of all observations. Strategies:

```python
# Option 1 — Class weights (all sklearn/XGBoost models)
model = XGBClassifier(scale_pos_weight = n_negatives / n_positives)

# Option 2 — SMOTE oversampling (training set only, never test)
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Option 3 — Threshold tuning (post-training)
# Move decision boundary from 0.5 to optimise F1 or recall on val set

# Option 4 — PR-AUC as primary metric (better than ROC-AUC under imbalance)
from sklearn.metrics import average_precision_score
pr_auc = average_precision_score(y_test, y_prob)
```

---

## Evaluation Metrics

### Classification (extreme event prediction)

| Metric | Why it matters here |
|---|---|
| **PR-AUC** | Primary metric — robust to class imbalance |
| **F1-score** | Balance of precision and recall |
| Recall (storm detection rate) | Missing a storm is costly — prioritise high recall |
| Precision | Controls false alarm rate |
| False alarm rate | Operational cost — too many alarms cause alert fatigue |
| ROC-AUC | Secondary; can be misleading under imbalance |

### Regression (dBHt magnitude prediction)

| Metric | Notes |
|---|---|
| RMSE (overall) | General accuracy |
| RMSE (extremes only, \|dBHt\| > p95) | Performance where GIC risk is real |
| Skill score vs persistence | Does model beat "predict same as last minute"? |

---

## Distributional Shift Analysis

For each train/test year pair, measure how different the feature distributions are.

```python
from scipy.stats import wasserstein_distance
from scipy.special import kl_div

# For each feature f, for year pair (A, B):
# 1. Histogram comparison — visual inspection
# 2. KL divergence       — how much info is lost encoding B using A's distribution
# 3. Wasserstein distance — "earth mover's" distance between distributions
# 4. PSI (Population Stability Index):
#    PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
#    PSI < 0.1  : stable
#    PSI 0.1–0.2: slight shift
#    PSI > 0.2  : significant shift

# Correlate shift magnitude with prediction performance drop:
# Does high Wasserstein(BZ_GSM, year_A, year_B) → low PR-AUC?
```

---

## Solar Cycle Analysis

Group all 506 experiments by the solar cycle phase of the **test year**:

```
Phase           Years               Expected behaviour
──────────────────────────────────────────────────────────────────
Solar minimum   1995–1996, 2008–2009   Few storms, quiet baseline easy
Rising phase    1997–2000, 2010–2012   Increasing storm frequency
Solar maximum   2000–2002, 2013–2015   Most intense storms
Declining phase 2003–2007, 2016–2017   High-speed stream storms
```

Key question: does training on a maximum year transfer well to a minimum year, and vice versa?

---

## Validation Strategy

```
1. Chronological splitting only
   — Never shuffle rows from a time series

2. Rolling validation (within a year)
   — Expand training window forward month by month

3. Walk-forward validation (across years)
   — Train 1995 → test 1996
   — Train 1995+1996 → test 1997
   — Continue accumulating

4. Leave-one-year-out
   — Full 506 cross-year matrix
```

---

## Project Structure

```
geomagnetic_storm_project/
│
├── data/
│   ├── raw/                        # downloaded OMNI / WDC files
│   ├── yearly/                     # storm_YYYY.csv per year
│   └── processed/                  # feature-engineered datasets
│
├── notebooks/
│   ├── 01_eda_phases.ipynb         # EDA + storm phase segmentation
│   ├── 02_extreme_values.ipynb     # Extreme value analysis (POT + GPD)
│   ├── 03_features_ccf.ipynb       # Feature engineering + CCF
│   ├── 04_baseline_models.ipynb    # Tree-based models + §8 chapter exercises
│   ├── 05_deep_learning.ipynb      # LSTM / GRU / CNN-LSTM
│   ├── 06_cross_year_experiments.ipynb   # 506-pair evaluation matrix
│   └── 07_solar_cycle_analysis.ipynb    # Cycle-phase transfer analysis
│
├── src/
│   ├── data_loader.py              # Download + parse OMNI data
│   ├── storm_catalog.py            # Build storm event catalog
│   ├── features.py                 # All feature engineering functions
│   ├── models.py                   # Model definitions and wrappers
│   ├── evaluate.py                 # Metrics and distributional shift
│   └── experiment_runner.py        # 506-pair cross-year loop
│
├── results/
│   ├── cross_year_matrix/          # PR-AUC matrix (23 × 23)
│   ├── shift_analysis/             # KL / Wasserstein per year pair
│   └── figures/                    # All publication figures
│
├── requirements.txt
└── README.md                       # this file
```

---

## Timeline

| Week | Task |
|---|---|
| 1–3 | Data preparation: download OMNI, build storm catalog, extract yearly CSVs |
| 4–6 | Baseline models: logistic regression, decision tree, random forest on 2015 |
| 7–9 | Cross-year experiments: run 506 evaluations, build result matrix |
| 10–12 | Deep learning: LSTM, GRU, CNN-LSTM |
| 13–14 | Distributional shift analysis + solar cycle grouping |
| 15–16 | Final reporting, figures, interpretation |

---

## Final Deliverables

### Technical

- Cleaned yearly datasets (`data/yearly/storm_YYYY.csv`)
- Preprocessing and feature engineering pipeline (`src/features.py`)
- Trained model weights for all 23 years × 6+ model types
- 23 × 23 cross-year PR-AUC evaluation matrix
- Distributional shift distance matrix (Wasserstein, KL, PSI)

### Research

- Literature review on geomagnetic storm prediction and transfer learning
- Methodology writeup: storm window construction, feature rationale, experimental design
- Model comparison: tree-based vs deep learning under distributional shift
- Distribution shift findings: which feature drifts most across the solar cycle?
- Explainability analysis: SHAP values, feature importance across years
- Final report and presentation figures

---

## Key References

- Burton, R. K. et al. (1975) — Ring current injection model (dDst/dt equation)
- Dst and Kp indices — WDC for Geomagnetism, Kyoto: http://wdc.kugi.kyoto-u.ac.jp
- OMNI solar wind data — NASA OMNIWeb: https://omniweb.gsfc.nasa.gov
- Pulkkinen et al. (2013) — GIC review and dBHt as primary risk metric
- Camporeale (2019) — ML review for space weather prediction

