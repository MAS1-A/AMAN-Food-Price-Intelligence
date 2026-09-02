# 🇪🇬 AMAN | Egypt Food Price Intelligence Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/Model-CatBoost%20Regressor-brightgreen)](https://catboost.ai/)
[![R2 Score](https://img.shields.io/badge/R2%20Score-99.04%25-success)](#)

An enterprise-grade AI observatory and forecasting platform for Egyptian commodity markets. It tracks supply-demand dynamics, models logistics overhead across Egyptian governorates, discovers fair equilibrium baseline prices, and projects dynamic 3-month inflation trajectories.

---

## 🌟 Key Capabilities

- **Real-Time Fair Pricing:** Quantifies supply-demand elasticities and retail markups to flag market deviations.
- **Automated Macro Telemetry:** Derives macro inflation indices, Ramadan consumption peaks, and seasonal harvest cycles directly from ground-truth data.
- **Spatial Map Density:** Interactive Egyptian geospatial visualization covering 23 governorates and 22 core commodities.
- **Multi-Month Trajectories:** Projects 3-month dynamic price trends bounded by $\pm 5\%$ statistical confidence intervals.
- **Bi-Lingual Architecture:** Full, seamless runtime localization (Arabic / English) with state preservation.

---

## 📊 Model Evaluation Benchmark

| Model | $R^2$ Score (%) | MAE (EGP) | MSE | RMSE (EGP) | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CatBoost Regressor** | **99.040%** | **5.736** | **122.100** | **11.050** | **Production Champion 🥇** |
| Stacking Regressor | 98.939% | 6.422 | 134.894 | 11.614 | Benchmark Ensemble |
| XGBoost Regressor | 98.935% | 6.093 | 135.409 | 11.637 | Single Estimator |
| Voting Regressor | 98.920% | 6.152 | 137.285 | 11.717 | Hybrid Ensemble |
| Linear Regression | 98.895% | 6.695 | 140.482 | 11.853 | Baseline Benchmark |

---

## 🚀 Quickstart & Local Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/aman-food-price-intelligence.git
cd aman-food-price-intelligence