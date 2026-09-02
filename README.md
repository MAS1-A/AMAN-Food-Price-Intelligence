# 🇪🇬 AMAN: Egypt Food Price Intelligence Platform
### Enterprise Real-Time Market Analytics, Fair-Pricing Auditing & Machine Learning Forecasting

An end-to-end academic machine learning capstone platform for monitoring, benchmarking, auditing, and forecasting food commodity prices across 23 Egyptian governorates.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Production Model](https://img.shields.io/badge/Production%20Model-Linear%20Regression-007ACC?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
![Test R²](https://img.shields.io/badge/Test%20R²-98.93%25-059669)
![Test MAE](https://img.shields.io/badge/Test%20MAE-6.67%20EGP-blue)
![Generalization Gap](https://img.shields.io/badge/Generalization%20Gap-%2B0.06%25-brightgreen)
![Market](https://img.shields.io/badge/Market-Egypt-C8102E)
![Project Type](https://img.shields.io/badge/Project%20Type-University%20Capstone-8B5CF6)

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Academic Capstone Team](#academic-capstone-team)
- [The Problem Statement](#the-problem-statement)
- [Platform Objectives](#platform-objectives)
- [System Architecture](#system-architecture)
- [Dataset & Feature Pipeline](#dataset--feature-pipeline)
- [Comprehensive Model Benchmark (11 Models)](#comprehensive-model-benchmark-11-models)
- [Production Deployment Rationale](#production-deployment-rationale)
- [Fair-Price Auditing Intelligence](#fair-price-auditing-intelligence)
- [Forward Forecasting Engine](#forward-forecasting-engine)
- [Spatial & Geospatial Intelligence](#spatial--geospatial-intelligence)
- [Interactive Dashboard Architecture](#interactive-dashboard-architecture)
- [Bilingual (Arabic / English) Engine](#bilingual-arabic--english-engine)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Getting Started & Local Deployment](#getting-started--local-deployment)
- [Reproducibility & Serialization Artifacts](#reproducibility--serialization-artifacts)
- [Monitoring & Production Reliability](#monitoring--production-reliability)
- [Project Impact & Strategic Vision](#project-impact--strategic-vision)

---

## Executive Summary

**AMAN** is an enterprise-grade food price intelligence platform built as an academic graduation capstone project. The system translates raw market transactions across **23 Egyptian governorates** and **22 core commodities** into actionable economic intelligence.

By incorporating spatial logistics, seasonal surges, and macroeconomic inflation, the platform determines fair equilibrium commodity pricing. Following a rigorous empirical evaluation of **11 distinct machine learning architectures**, an optimized **Linear Regression pipeline with logarithmic target scaling** was selected for production deployment. It achieves an explanatory power of **Test R² = 98.93%**, an average error of **Test MAE = 6.67 EGP**, and an ultra-tight **Generalization Gap of +0.06%** with zero-latency inference.

---

## Academic Capstone Team

This project was developed through a specialized division of responsibilities across the machine learning lifecycle:

| Team Member | Project Role | Engineering Scope & Core Responsibilities | LinkedIn Profile |
| :--- | :--- | :--- | :---: |
| **Rowaida Amr Ali** | **Data Engineering & EDA Lead** | • Data schema audit, hygiene, and missing-value imputation.<br>• Feature engineering (cyclic temporal harmonic signals).<br>• Bivariate correlation, distribution skewness, and spatial variance analysis. | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/rowida-amr-324047382/) |
| **Basmala El-Husseiny Ismail** | **Pipeline & Preprocessing Lead** | • Strict leak-free temporal splitting (`Train` vs `Test`).<br>• Scaler calibration via Robust & Standard scaling techniques.<br>• Categorical encoding and matrix feature alignment.<br>• Baseline model initialization and parameter setup. | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/basmalla-elhoseny-4604b7396/) |
| **Mariam Yasser Arafat** | **Advanced ML & Neural Specialist** | • Training tree ensembles (`Random Forest`, `Decision Trees`).<br>• Gradient boosting tuning (`CatBoost`, `XGBoost`, `LightGBM`).<br>• Neural architecture design (`MLPRegressor` multi-layer perceptron).<br>• Regularization penalty calibration and parameter search. | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/maryam-yasser-00bb66429/) |
| **Abdelrahman Emad Ahmed** | **Evaluation & Benchmark Lead** | • 11-Model diagnostic evaluation matrix on real currency scales.<br>• Generalization gap analysis ($\Delta R^2 = R^2_{\text{train}} - R^2_{\text{test}}$).<br>• Residual distribution analysis, bias-variance trade-offs, and error metrics.<br>• Ensemble model construction (`VotingRegressor`, `StackingRegressor`). | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/abdelrahman-zayan/) |
| **Mohamed Abdullah Sabry** | **System Architecture & Deployment Lead** | • Full-stack Streamlit bilingual dashboard development.<br>• Real-time model inference pipeline and single-row matrix alignment.<br>• Geospatial mapping engine restricted to Egypt coordinates.<br>• Git architecture, model serialization (`.pkl`), and UI state logic. | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/mohammed-abdullah-sabry-94bb99396/) |

---

## The Problem Statement

Food commodity pricing in developing markets like Egypt is exposed to structural volatility driven by interacting economic factors:
* **Logistics Overhead:** Significant transportation cost variations between import ports (Alexandria, Port Said) and inland agricultural or Upper Egypt governorates.
* **Seasonal Demand Surges:** High consumer buying pressure during the holy month of Ramadan causing transient retail price markups.
* **Information Asymmetry:** Consumers and retailers lack data-driven references to evaluate whether shelf prices represent true economic equilibrium or excessive markups.

AMAN answers key practical questions:
1. **What is the statistically fair market price for a given commodity today?**
2. **Does an observed retail price represent a fair deal, a bargain, or an overcharge?**
3. **How do transportation costs and urbanization rates affect prices across governorates?**
4. **Where will the commodity price move over the next 90 days?**

---

## Platform Objectives

* **Market Monitoring:** Continuously track commodity prices across Egyptian governorates using columnar `.parquet` storage.
* **Equilibrium Fair-Price Auditing:** Calculate dynamic reference prices and categorize retail price deviations into clear fairness tiers.
* **Forward-Looking Forecasting:** Generate 3-month price projections based on seasonal demand trends.
* **Geospatial Intelligence:** Visualize governorate-level price distributions and logistics costs across Egypt.

---

## System Architecture

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                       DATA PREPARATION & WRANGLING                     │
 │   Food_Prices_in_Egypt.parquet ──► Temporal Features ──► GIS Logistics │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
 ┌───────────────────────────────────▼────────────────────────────────────┐
 │                     FEATURE ENGINEERING & PIPELINE                     │
 │   Cyclic Trigo Encoding ──► Log Transform [log1p] ──► Feature Scaling  │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
 ┌───────────────────────────────────▼────────────────────────────────────┐
 │                  COMPREHENSIVE 11-MODEL BENCHMARK                      │
 │   Neural Nets (MLP) ──► Gradient Boosters ──► Tree Ensembles ──► Linear│
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
 ┌───────────────────────────────────▼────────────────────────────────────┐
 │               PRODUCTION SERIALIZATION (DEPLOYED ENGINE)               │
 │       Linear Regression Artifact (R²: 98.93%, MAE: 6.67, Gap: +0.06%)  │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
 ┌───────────────────────────────────▼────────────────────────────────────┐
 │                       STREAMLIT ENTERPRISE UI                          │
 │   ├── Egypt Spatial Scatter Map       ├── Real-Time Fair Price Audit   │
 │   ├── Historical Time-Series Viewer   └── 3-Month Dynamic Trajectory   │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## Dataset & Feature Pipeline

The platform is evaluated on historical Egyptian market observations covering 23 administrative governorates and 22 staple food commodities.

| Feature Name | Data Type | Analytical Role & Modeling Scope |
| :--- | :---: | :--- |
| `Date` | `datetime64` | Observation timestamp (YYYY-MM-DD). |
| `Governorate` | `category` | Egyptian governorate (23 administrative divisions). |
| `Region` | `category` | Macro geographic cluster (Greater Cairo, Northern, Upper Egypt, Canal, Sinai). |
| `Market_Type` | `category` | Retail Grocery, Central Wholesale Market, or Modern Supermarket. |
| `Commodity` | `category` | Staple commodity name (e.g., Rice, Beef, Onion, Poultry). |
| `Category` | `category` | High-level commodity classification (Vegetables, Protein, Fruits, Grains). |
| `Unit` | `category` | Standard metric of sale (KG, Liter, Dozen). |
| `Inflation_Index` | `float64` | Relative Consumer Price Index (CPI) macroeconomic baseline. |
| `Supply_Level` | `float64` | Quantified commodity market availability score. |
| `Demand_Level` | `float64` | Quantified consumer purchase volume and demand pressure score. |
| `Transport_Cost` | `float64` | Estimated logistics overhead per unit delivered to the target market. |
| `Urbanization` | `float64` | Proportion of urbanized population within the governorate. |
| **`Price_EGP`** | `float64` | **Target Variable**: Observed real transaction unit price in Egyptian Pounds. |


---

## Comprehensive Model Benchmark (11 Models)

All 11 candidates were trained on identical training splits and evaluated against an out-of-sample held-out test split. All metrics reflect **real-scale currency (EGP)**:

| Rank | Model Architecture | Train $R^2$ (%) | Test $R^2$ (%) | Generalization Gap ($\Delta R^2$) | Test MAE (EGP) | Test RMSE (EGP) | Status / Role |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 1 | **MLP Regressor** | 99.40% | **99.37%** | **+0.03%** | **5.03** | **8.98** | *Research Elite Benchmark* |
| 🥈 2 | **Voting Regressor** | 99.40% | **99.09%** | +0.31% | 6.00 | 10.77 | *Hybrid Ensemble* |
| 🥉 3 | **Linear Regression** | 98.99% | **98.93%** | **+0.06%** | **6.67** | **11.66** | **⚡ Production Deployed** |
| 4 | **Stacking Regressor** | 99.44% | 98.80% | +0.64% | 7.07 | 12.37 | Meta-Learner Ensemble |
| 5 | **CatBoost Regressor** | 99.45% | 98.69% | +0.76% | 7.31 | 12.93 | Gradient Boosted Trees |
| 6 | **XGBoost Regressor** | 99.41% | 98.58% | +0.84% | 7.63 | 13.47 | Gradient Boosted Trees |
| 7 | **Random Forest Regressor** | 99.91% | 98.40% | +1.52% | 7.99 | 14.29 | Bagging Ensemble |
| 8 | **K-Nearest Neighbors** | 99.55% | 98.26% | +1.29% | 8.38 | 14.90 | Instance-Based Baseline |
| 9 | **LightGBM Regressor** | 99.35% | 98.06% | +1.29% | 8.42 | 15.71 | Histogram Boosting |
| 10 | **Decision Tree Regressor** | 100.00% | 97.81% | +2.19% | 9.28 | 16.71 | Memorization Overfit |
| 11 | **Gradient Boosting (sklearn)** | 97.91% | 94.50% | +3.41% | 13.95 | 26.46 | Suboptimal Optimization |

---

## Production Deployment Rationale

Although the **Multi-Layer Perceptron (MLP)** achieved the highest test score ($R^2 = 99.37\%$), **Linear Regression** was selected as the **Production Model** for deployment based on the following engineering trade-offs:

1. **Near-Zero Overfitting Gap ($\Delta R^2 = +0.06\%$):** The log-transformed feature space linearized macroeconomic drivers (inflation, logistics, supply/demand), allowing the linear model to generalize without memorizing noise.
2. **Sub-Millisecond Inference Latency:** Linear matrix multiplication evaluates in under $0.5\text{ ms}$, ensuring responsive UI updates during real-time dashboard operations.
3. **Interpretability:** Model weights provide transparent elasticity coefficients, suitable for public audits.
4. **Lightweight Deployment Footprint:** Eliminates heavy C++ compilation libraries and deep learning runtimes, reducing memory consumption on production servers.

---

## Fair-Price Auditing Intelligence

AMAN evaluates real-world market fairness by computing the percentage deviation between observed shelf prices and model equilibrium prices:

$$\text{Deviation (\%)} = \frac{\text{Observed Price} - \text{Fair Price}}{\text{Fair Price}} \times 100$$

```text
 ┌───────────────────────────┬───────────────────────────────┬───────────────────────────────┐
 │     Bargain / Discount    │        Fair Equilibrium       │      Overpriced / Premium     │
 │     Deviation < -7.0%     │     -7.0% ≤ Dev ≤ +7.0%       │      Deviation > +7.0%        │
 │ Optimal consumer purchase │ Reflects genuine market forces│ Flagged for potential markup  │
 └───────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

---

## Forward Forecasting Engine

The platform forecasts a **3-Month Ahead Price Trajectory** by combining:
* Rolling future calendar months and cyclical harmonic encodings.
* Seasonal supply/demand adjustments calculated from historical monthly medians.
* Projected inflation increments.
* Symmetrical **$\pm 5\%$ statistical confidence bounds** reflecting historical error variance.

```text
 Current Fair Price ──► Month +1 Forecast ──► Month +2 Forecast ──► Month +3 Forecast
       (t_0)                 (t_1)                 (t_2)                 (t_3)
     [Point]               [±5% CI]              [±5% CI]              [±5% CI]
```

---

## Spatial & Geospatial Intelligence

AMAN features an interactive Plotly map restricted to the geographic boundaries of the **Arab Republic of Egypt**:

* **Geographic Focus:** Latitude $21.5^\circ\text{N} \dots 32.0^\circ\text{N}$, Longitude $24.5^\circ\text{E} \dots 37.0^\circ\text{E}$.
* **Map Center:** Lat $26.8206^\circ$, Lon $30.8025^\circ$, Zoom: `5.2`.
* **Telemetry Markers:** Marker sizes and colors map to real-time commodity prices and logistics transport overhead.
* **Cross-Version Plotly Support:** Dynamic fallback handling between `px.scatter_map` (Plotly v6.x+) and `px.scatter_mapbox` (Plotly v5.x).

---

## Interactive Dashboard Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│  AMAN Food Price Intelligence Platform                                 │
├────────────────────────────────────────────────────────────────────────┤
│  Navigation: [🏠 Home]  [📊 Geospatial Map]  [🎯 Fair Pricing]  [ℹ️ About]│
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [ Select Commodity ]      [ Select Governorate ]    [ Market Channel ]│
│  ┌───────────────────┐     ┌────────────────────┐    ┌────────────────┐│
│  │ 🥩 Beef           │     │ Alexandria         │    │ Retail Grocery ││
│  └───────────────────┘     └────────────────────┘    └────────────────┘│
│                                                                        │
│  ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────┐│
│  │    Fair Equilibrium    │  │ Observed Market Price │  │ Audit Badge ││
│  │       85.38 EGP        │  │       90.00 EGP       │  │   FAIR ✅   ││
│  └────────────────────────┘  └───────────────────────┘  └─────────────┘│
│                                                                        │
│  ┌─────────────────────────────────┐ ┌────────────────────────────────┐│
│  │   Interactive Gauge Meter       │ │   3-Month Future Trajectory    ││
│  │   [====|====▲========]          │ │   📈 Projected Curve (±5% CI)  ││
│  └─────────────────────────────────┘ └────────────────────────────────┘│
└────────────────────────────────────────────────────────────────────────┘
```

---

## Bilingual (Arabic / English) Engine

The application includes full native support for both **العربية (Egyptian context)** and **English**:
* Instant language toggle in the sidebar with session state persistence.
* Dynamic bidirectional typography adjustments (Cairo font family).
* Localized numerical, currency (EGP / جنيه مصري), and calendar strings.

---

## Technology Stack

| Layer | Primary Framework / Tool | Usage & Scope |
| :--- | :--- | :--- |
| **Language** | `Python 3.10+` | Core programming language across pipeline. |
| **Data Engine** | `Pandas`, `NumPy`, `FastParquet`, `PyArrow` | Columnar storage, vector math, data cleaning. |
| **Machine Learning** | `Scikit-Learn`, `CatBoost`, `XGBoost`, `LightGBM` | Estimator implementations, benchmark, and scaling. |
| **Web Dashboard** | `Streamlit` | Full-stack interactive reactive web interface. |
| **Data Visualization** | `Plotly Express`, `Plotly Graph Objects`, `Seaborn` | Geospatial maps, gauge indicators, regression diagnostics. |
| **Artifact Storage** | `Joblib` | Production model and preprocessing serialization. |

---

## Repository Structure

```text
AMAN-Food-Price-Intelligence/
│
├── app/
│   └── app.py                                         # Streamlit bilingual interactive platform
├── assets/
│   └── amanfood.png                                   # Platform identity and navigation branding
├── data/
│   └── Food_Prices_in_Egypt.parquet                   # Columnar compressed historical dataset
├── models/
│   └── egypt_food_price_LinearRegression_model.pkl    # Serialized production pipeline artifact
├── notebooks/
│   └── food_price_modeling_pipeline.ipynb             # Full 11-model training & benchmark suite
│
├── .gitignore                                         # Cache and environment exclusions
├── README.md                                          # Master technical documentation
└── requirements.txt                                   # Production dependencies
```

---

## Getting Started & Local Deployment

### 1. Clone the Repository
```bash
git clone [https://github.com/MAS1-A/AMAN-Food-Price-Intelligence.git](https://github.com/MAS1-A/AMAN-Food-Price-Intelligence.git)
cd AMAN-Food-Price-Intelligence
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Platform
```bash
streamlit run app/app.py
```
The interface will initialize locally at `http://localhost:8501`.

---

## Reproducibility & Serialization Artifacts

Production predictions rely on a serialized pipeline dictionary (`egypt_food_price_LinearRegression_model.pkl`) that guarantees design-matrix compatibility:

```python
production_artifacts = {
    'model': lr_model,
    'scaler': robust_scaler,
    'feature_names': list(X_train.columns),
    'original_features': FEATURES,
    'categorical_features': ['Governorate', 'Region', 'Market_Type', 'Commodity', 'Category', 'Unit', 'Season']
}
```

During web inference, dynamic inputs are converted to dummy variables and realigned to match `feature_names` exactly, eliminating shape-mismatch errors during single-row execution.

---

## Monitoring & Production Reliability

Food commodity distributions in Egypt are sensitive to structural shifts. The following monitoring metrics are established for operational continuity:
1. **Input Data Drift:** Track statistical divergence in feature distributions (e.g., Inflation Index or Transport Costs) via Kolmogorov-Smirnov tests.
2. **Residual Stability:** Continuously evaluate incoming transaction records to ensure error variance stays within the established benchmark ($\text{MAE} \le 7.50\text{ EGP}$).
3. **Fallback Resiliency:** If real-time inputs lack regional telemetry, the platform defaults gracefully to the historical median for that commodity.

---

## Project Impact & Strategic Vision

Developed as a university capstone project, **AMAN** transitions price analytics from passive observation to an active decision-support system:

> **From "What is the price?"**
> **To "What should the price be based on regional logistics and inflation, and where will it trend next?"**

---

<div align="center">
  <p><b>AMAN: Egypt Food Price Intelligence Platform</b></p>
  <p><i>Graduation Capstone Project • Faculty of Computers and Information • 2026</i></p>
</div>