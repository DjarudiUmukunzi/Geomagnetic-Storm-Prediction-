# 🌌 Cross-Year Geomagnetic Storm Prediction

### Using Machine Learning, Deep Learning, and Self-Supervised Representation Learning

> Forecasting geomagnetic storm intensity across solar cycles (1995–2017) using dBHt as the primary regression target

---

## 📋 Project Overview

This project develops and benchmarks a progression of machine learning, deep learning, and self-supervised learning models to forecast geomagnetic storm intensity across 22 years of solar-cycle variability. The dataset spans Solar Cycles 22–24 (1995–2017), covering approximately **12 million rows** of multivariate time-series observations across **24 features**.

The primary regression target is **dBHt** — the time derivative of the horizontal geomagnetic field intensity (nT/min). Models are evaluated not only on forecast accuracy but on cross-year generalization, distributional shift robustness, and operational stability during geomagnetically quiet periods.

---

## 🔬 Core Research Questions

1. Can machine learning models predict geomagnetic storm intensity across unseen years?
2. How does distributional shift affect forecasting performance across solar-cycle phases?
3. Which model architectures generalize best under solar-cycle variability?
4. Can models remain operationally stable during non-storm quiet conditions?
5. Can self-supervised representation learning (JEPA) improve robustness under temporal drift?

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Size** | ~12 million rows |
| **Features** | 24 multivariate time-series inputs |
| **Temporal span** | 1995–2017 (Solar Cycles 22, 23, 24) |
| **Structure** | Continuous temporal observations |
| **Target variable** | `dBHt` (nT/min) |

---

## 🎯 Primary Target Variable

| Variable | Description |
|---|---|
| `dBHt` | Rate of change of horizontal geomagnetic field intensity (nT/min) — primary regression target |

---

## 📡 Input Features

| Feature | Description | Category |
|---|---|---|
| `dBHt_t1` | dBHt lagged 1 step | Autoregressive lag |
| `dBHt_t2` | dBHt lagged 2 steps | Autoregressive lag |
| `BZ_GSM` | North-south IMF component (GSM frame) | Interplanetary magnetic field |
| `flow_speed` | Solar wind bulk flow speed (km/s) | Solar wind |
| `Pressure` | Solar wind dynamic pressure (nPa) | Solar wind |
| `E_Field` | Interplanetary electric field (mV/m) | Solar wind coupling |
| `SYM_H` | Symmetric ring current index — proxy for Dst | Geomagnetic index |
| `AE_INDEX` | Auroral electrojet activity index | Geomagnetic index |
| `sinMLT` | Sine of magnetic local time | Temporal/cyclic encoding |
| `cosMLT` | Cosine of magnetic local time | Temporal/cyclic encoding |

> **Note on MLT encoding:** Magnetic Local Time is cyclic. Encoding as `sin(MLT)` / `cos(MLT)` ensures models
> correctly understand that 23:00 and 01:00 are adjacent — not 22 hours apart.

---

## 🌡️ Storm Severity Classification (NOAA-Inspired dBHt Scale)

| Category | dBHt Range (nT/min) | NOAA Equivalent | Frequency per Solar Cycle |
|---|---:|---|---|
| **Minor** | 0 – 100 | G1 | ~1700 events |
| **Moderate** | 100 – 250 | G2 | ~600 events |
| **Strong** | 250 – 400 | G3 | ~200 events |
| **Severe / Extreme** | > 400 | G4–G5 | ~100–4 events |

These categories support both regression benchmarking and operational storm classification tasks.

> **Key challenge:** The 1995 training anchor is 98.98 nT/min (Minor). The model must generalize to
> 761.12 nT/min (2003 Halloween Storm — Severe/Extreme): an ~8× extrapolation beyond training range.

---

## 🌀 Storm Window Structure

Each extracted event window contains four consecutive phases:

```
|← 10-day quiet →|← 48h pre-storm →|← Storm duration →|← 48h recovery →|
```

This structure preserves operational context and sequential temporal behavior surrounding each event.

---

## 🏗️ Pipeline Architecture

All models are **trained once on the 1995 storm event**. From that single anchor, **22 independent year-pipelines** are executed — one per year from 1996 to 2017. Every pipeline runs both evaluation tracks.

```
Training Anchor: 1995 storm (dBHt = 98.98 nT/min — Minor / G1)
        │
        ├── Year Pipeline 1996
        │       ├── Track A — Cross-Year Storm Prediction
        │       └── Track B — Quiet-Period Operational Stability
        │
        ├── Year Pipeline 1997
        │       ├── Track A — Cross-Year Storm Prediction
        │       └── Track B — Quiet-Period Operational Stability
        │
        ├── ... (1998 through 2016)
        │
        └── Year Pipeline 2017
                ├── Track A — Cross-Year Storm Prediction
                └── Track B — Quiet-Period Operational Stability
```

### Track A — Cross-Year Storm Prediction

| Component | Details |
|---|---|
| **Model** | Trained on 1995 storm (fixed — no retraining) |
| **Test data** | Storm window from the target year |
| **Window** | 10-day quiet + 48h pre-storm + storm duration + 48h post-storm recovery |
| **Objective** | Measure generalization to unseen storm years |
| **Output** | RMSE, MAE, R² per year per model |

### Track B — Quiet-Period Operational Stability

| Component | Details |
|---|---|
| **Model** | Same 1995-trained model (no retraining) |
| **Test data** | Quietest January–April interval from the target year |
| **Objective** | Confirm model does not produce false storm alerts during quiet conditions |
| **Output** | False positive rate, prediction variance, quiet-period stability score |

---

## ♻️ Backtesting Framework

The project uses **rolling-origin backtesting** to simulate real-world operational deployment:

| Training Period | Testing Period |
|---|---|
| 1995 | 1996 |
| 1995–1996 | 1997 |
| 1995–1997 | 1998 |
| … | … |
| 1995–2016 | 2017 |

This prevents temporal leakage and faithfully reproduces operational forecasting conditions.

---

## 📅 Year-by-Year Pipeline Overview

| Year Pipeline | Peak Storm Date | Peak dBHt | Severity | NOAA | Solar Phase |
|---|---|---:|---|:---:|---|
| *1995 (train)* | *1995-04-07* | *98.98* | *Minor* | *G1* | *Declining (Cycle 22)* |
| **Pipeline 1996** | 1996-10-23 | 86.05 | Minor | G1 | Solar minimum |
| **Pipeline 1997** | 1997-11-23 | 102.82 | Moderate | G2 | Rising phase |
| **Pipeline 1998** | 1998-05-04 | 619.99 | Severe/Extreme | G5 | Rising phase |
| **Pipeline 1999** | 1999-09-22 | 243.86 | Moderate | G3 | Rising phase |
| **Pipeline 2000** | 2000-07-15 | 466.08 | Severe/Extreme | G4 | Solar maximum |
| **Pipeline 2001** | 2001-03-31 | 385.32 | Strong | G3 | Solar maximum |
| **Pipeline 2002** | 2002-04-17 | 121.90 | Moderate | G2 | Solar maximum |
| **Pipeline 2003** | 2003-10-29 | **761.12** | **Severe/Extreme** ⬅ Max | **G5** | Solar maximum |
| **Pipeline 2004** | 2004-07-27 | 418.75 | Severe/Extreme | G4 | Declining phase |
| **Pipeline 2005** | 2005-05-15 | 414.42 | Severe/Extreme | G4 | Declining phase |
| **Pipeline 2006** | 2006-12-15 | 160.21 | Moderate | G2 | Declining phase |
| **Pipeline 2007** | 2007-05-24 | 51.34 | Minor | G1 | Solar minimum |
| **Pipeline 2008** | 2008-11-24 | 37.32 | Minor | Below G1 | Solar minimum |
| **Pipeline 2009** | 2009-06-24 | **28.14** | Minor ⬅ Min | Below G1 | Solar minimum |
| **Pipeline 2010** | 2010-08-04 | 62.95 | Minor | G1 | Solar minimum |
| **Pipeline 2011** | 2011-10-25 | 80.10 | Minor | G1 | Rising phase |
| **Pipeline 2012** | 2012-10-09 | 111.25 | Moderate | G2 | Rising phase |
| **Pipeline 2013** | 2013-10-02 | 112.47 | Moderate | G2 | Solar maximum |
| **Pipeline 2014** | 2014-09-12 | 76.44 | Minor | G1 | Solar maximum |
| **Pipeline 2015** | 2015-03-17 | 323.52 | Strong | G3 | Declining phase |
| **Pipeline 2016** | 2016-07-19 | 81.13 | Minor | G1 | Declining phase |
| **Pipeline 2017** | 2017-09-08 | 107.27 | Moderate | G2 | Declining phase |

### Severity Distribution Across 22 Test Years

| Category | dBHt Range | Count | Years |
|---|---|---:|---|
| Minor | 0–100 | 9 | 1996, 2007, 2008, 2009, 2010, 2011, 2014, 2016 |
| Moderate | 100–250 | 6 | 1997, 1999, 2002, 2006, 2012, 2013, 2017 |
| Strong | 250–400 | 2 | 2001, 2015 |
| Severe/Extreme | > 400 | 5 | 1998, 2000, 2003, 2004, 2005 |

---

## 🤖 Modeling Roadmap

The project follows a 5-phase progression from interpretable baselines to advanced self-supervised architectures.

### Phase 1 — Baseline Statistical Models
> Establish interpretable forecasting baselines

| Model | Framework |
|---|---|
| Linear Regression | scikit-learn |
| Ridge Regression | scikit-learn |
| Elastic Net | scikit-learn |

### Phase 2 — Tree-Based Machine Learning
> Develop strong non-linear baselines; expected best performance on tabular data

| Model | Framework |
|---|---|
| Decision Tree Regressor | scikit-learn |
| Random Forest Regressor | scikit-learn |
| XGBoost Regressor | XGBoost |
| LightGBM Regressor | LightGBM |
| CatBoost Regressor | CatBoost |

> **XGBoost with cross-year backtesting is the expected strongest initial benchmark.**
> Tree-based models excel under heterogeneous feature distributions, non-linear dynamics,
> and mixed temporal signals typical of space-weather tabular data.

### Phase 3 — Sequential Deep Learning
> Capture temporal dependencies, sequential storm dynamics, and storm evolution patterns

| Model | Architecture | Framework |
|---|---|---|
| Simple LSTM | Single-layer recurrent | PyTorch |
| Stacked LSTM | Multi-layer deep recurrent | PyTorch |
| Bidirectional LSTM | Forward + backward passes | PyTorch |
| GRU | Gated Recurrent Unit | PyTorch |
| CNN-LSTM | 1D convolution + LSTM | PyTorch |
| Attention-LSTM | LSTM + learned attention weights | PyTorch |

### Phase 4 — Transformer-Based Models
> Improve long-context temporal learning and attention-based sequence modeling

| Model | Architecture | Framework |
|---|---|---|
| Transformer Encoder | Multi-head self-attention | PyTorch |
| Temporal Attention Networks | Time-aware attention | PyTorch |
| Temporal Fusion Transformer (TFT) | Multi-horizon gated attention | PyTorch |

### Phase 5 — JEPA Self-Supervised Framework
> Learn robust latent representations of geomagnetic temporal dynamics without label supervision

Rather than predicting raw dBHt directly, JEPA learns latent embeddings that model temporal structure and sequential context, then fine-tunes for downstream tasks.

```
Input sequence (solar wind + geomagnetic measurements + temporal context)
        │
        ▼
   Encoder → Latent storm representations + hidden temporal dynamics
        │
        ▼
   Predictor → Future latent representations
        │
        ▼
   Downstream fine-tuning:
        ├── dBHt regression
        ├── Storm severity classification
        └── Operational forecasting
```

**Why JEPA adds scientific value:**
- Stronger latent temporal modeling than traditional LSTMs
- Reduced overfitting to specific storm events
- Improved generalization across solar-cycle phases
- Better robustness to temporal drift and distributional shift
- Enhanced rare-event signal extraction

---

## 📊 Evaluation Metrics

### Regression Metrics (Track A — Storm Prediction)

| Metric | Description |
|---|---|
| **RMSE** | Root Mean Squared Error |
| **MAE** | Mean Absolute Error |
| **R² Score** | Coefficient of Determination |

### Classification Metrics (Severity Benchmarking)

| Metric | Description |
|---|---|
| **Precision** | True positive rate among predicted storm events |
| **Recall** | Coverage of actual storm events |
| **F1 Score** | Harmonic mean of precision and recall |
| **PR-AUC** | Area under Precision–Recall curve |
| **Confusion Matrix** | Full severity class breakdown |

### Operational Metrics (Track B — Quiet-Period Stability)

| Metric | Description |
|---|---|
| **False Positive Rate** | Spurious storm alerts during quiet periods |
| **Quiet-Time Stability** | Prediction variance during non-storm intervals |
| **Storm Detection Rate** | Coverage of actual storm onset events |
| **Cross-Year Generalization** | Performance degradation across test years |

---

## 📉 Distributional Shift Analysis

Geomagnetic activity changes seasonally, across solar cycles, and across storm intensities. The project quantifies temporal drift per year-pipeline:

| Method | What it Measures |
|---|---|
| **Histogram Comparison** | Visual frequency distribution drift vs. 1995 training data |
| **KL Divergence** | Information-theoretic distance between distributions |
| **Wasserstein Distance** | Earth Mover's Distance — robust to non-overlapping distributions |
| **PSI** | Model health: < 0.1 stable · 0.1–0.25 monitor · > 0.25 significant shift |

---

## 🧪 Statistical Validation

### Pairwise Model Comparison — t-tests

- XGBoost vs. Random Forest
- LSTM vs. XGBoost
- JEPA vs. Transformer Encoder

### Multi-Model Comparison — ANOVA

- All models compared across storm severity groups
- All models compared across solar-cycle phases (minimum, rising, maximum, declining)

---

## 🗺️ Research Timeline

| Stage | Activity |
|---:|---|
| **1** | Data preprocessing and storm window extraction |
| **2** | Feature engineering and baseline statistical models |
| **3** | Tree-based model benchmarking |
| **4** | Cross-year backtesting experiments |
| **5** | Quiet-period operational testing |
| **6** | Sequential deep learning implementation (PyTorch) |
| **7** | Transformer-based forecasting |
| **8** | JEPA self-supervised representation learning |
| **9** | Statistical validation and comparative analysis |
| **10** | Research documentation and publication preparation |

---

## 🚀 Expected Research Contributions

1. Comprehensive cross-year benchmark of classical, deep learning, and self-supervised models under real operational constraints
2. Quantified model degradation profiles driven by solar-cycle distributional shift across 22 year-pipelines
3. Operational false-positive analysis during geomagnetically quiet periods
4. A reproducible JEPA-based self-supervised framework for rare-event prediction in nonstationary geophysical time series
5. Evidence-based guidance for operational space-weather forecasting system design

---

## 📁 Project Structure

```
geomagnetic-storm-prediction/
├── data/
│   ├── raw/                              # Raw OMNI/NOAA data (1995–2017)
│   ├── processed/
│   │   ├── storm_windows/               # Per-year storm event windows (Track A)
│   │   └── quiet_windows/               # Per-year quiet-period windows (Track B)
│   └── shift/                            # Distributional shift outputs
│
├── yearly_training/
│   ├── 1995_notebook.ipynb               # Training anchor
│   ├── 1996_notebook.ipynb
│   ├── ...
│   └── 2017_notebook.ipynb
│
├── storm_severity/
│   ├── minor_0_100.ipynb
│   ├── moderate_100_250.ipynb
│   ├── strong_250_400.ipynb
│   └── severe_gt_400.ipynb
│
├── pipelines/
│   ├── year_pipeline.py                  # Shared class: Track A + Track B
│   └── run_all_years.py                  # Iterates 1996–2017
│
├── baseline_models/
│   ├── linear_regression.ipynb
│   ├── random_forest.ipynb
│   ├── xgboost.ipynb
│   └── lightgbm.ipynb
│
├── deep_learning/
│   ├── lstm.ipynb                        # PyTorch LSTM variants
│   ├── gru.ipynb
│   ├── transformer.ipynb
│   └── jepa.ipynb                        # JEPA self-supervised framework
│
├── models/
│   ├── ml/                               # scikit-learn / XGBoost / LightGBM / CatBoost
│   └── dl/                              # PyTorch model definitions
│
├── evaluation/
│   ├── rmse_analysis.ipynb
│   ├── statistical_tests.ipynb
│   ├── distribution_shift.ipynb
│   └── quiet_period_analysis.ipynb
│
├── results/
│   ├── track_a/                          # Per-year storm prediction results
│   ├── track_b/                          # Per-year quiet-period results
│   └── figures/
│
└── README.md
```

---

## 🔧 Framework & Environment

| Component | Choice |
|---|---|
| **Deep learning** | PyTorch |
| **Tree-based ML** | scikit-learn, XGBoost, LightGBM, CatBoost |
| **Self-supervised** | PyTorch (JEPA custom implementation) |
| **Data processing** | pandas, NumPy |
| **Evaluation & stats** | scipy, scikit-learn metrics |
| **Visualization** | matplotlib, seaborn |

---

## 📚 References & Data Sources

- [NOAA Space Weather Scales](https://www.spaceweather.gov/noaa-scales-explanation) — G1–G5 scale definitions
- Solar Cycle reference: Cycles 22, 23, and 24 span the 1995–2017 study period

---

*22 year-pipelines · 5-phase modeling roadmap · 10-stage research timeline · Training anchor: April 1995 · Framework: PyTorch*
