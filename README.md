# ⚡ IEX Day-Ahead Market Price Forecasting

**Forecasting the 15-minute electricity price (MCP) on the Indian Energy Exchange, one day ahead.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-tuned-orange)
![Kaggle](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Status-complete-success)

A data mining project that compares **14 forecasting approaches**, from classical statistics (ARIMA, SARIMAX, Prophet, AutoTS) to machine learning (XGBoost, LightGBM) and deep learning (LSTM, GRU), on real IEX Day-Ahead Market data. It then goes beyond accuracy: explainability, spike-risk detection, prediction intervals, and a buyer cost-saving simulation.

---

## 🏆 Headline result

On a fully held-out test period of **93 days (1 Jul – 1 Oct 2025)**, the tuned XGBoost model is **34% more accurate than the seasonal-naive benchmark** ("same time yesterday"), and the difference is statistically significant (Diebold-Mariano test, p < 0.001).

| | Hourly MAE (Rs/kWh) | Hourly RMSE | 15-min MAE |
|---|---|---|---|
| **XGBoost (tuned)** | **0.502** | **0.877** | **0.518** |
| Seasonal naive (1 day) | 0.765 | 1.424 | 0.772 |

It also beat the seasonal-naive benchmark in **7 out of 7 rolling out-of-sample quarters** (Jan 2024 – Oct 2025), with an average MAE gain of about 20%.

---

## 📊 All models, one table

Test period, hourly level. Every model uses the same day-ahead protocol and the same clipping to the valid price range.

| Model | Family | MAE | RMSE |
|---|---|---|---|
| **XGBoost (tuned)** | Machine learning | **0.502** | **0.877** |
| XGBoost (walk-forward, monthly refit) | Machine learning | 0.497 | 0.876 |
| XGBoost with extra features (v2) | Machine learning | 0.517 | 0.887 |
| Ensemble (XGBoost + v2 + LightGBM) | Machine learning | 0.525 | 0.923 |
| GRU | Deep learning | 0.755 | 1.230 |
| Seasonal naive (1 day) | Benchmark | 0.765 | 1.424 |
| Decomposition (multiplicative) | Classical | 0.820 | 1.256 |
| LSTM | Deep learning | 0.898 | 1.398 |
| SARIMAX (weekend + yesterday's bid ratio) | Classical | 0.913 | 1.323 |
| SARIMA | Classical | 0.963 | 1.397 |
| Holt-Winters | Classical | 1.052 | 1.522 |
| AutoTS | AutoML | 1.078 | 1.471 |
| Prophet | Classical | 1.781 | 2.208 |
| ARIMA | Classical | 2.065 | 2.640 |

> **Takeaway:** for this market, tree models on carefully built day-ahead features beat every classical and deep learning model by a wide margin. Most classical models cannot even beat "same time yesterday".

---

## 🔍 The problem

In the Day-Ahead Market, electricity prices for each 15-minute block of tomorrow are decided today. Buyers (distribution companies, industries) and sellers need a forecast of tomorrow's price *before* the auction.

- **Target:** MCP (Market Clearing Price) in Rs/kWh, for 96 blocks per day
- **Horizon:** the whole next day, forecast using only information available up to 23:45 the day before
- **Hard parts:** a regulatory price ceiling (Rs 12, later Rs 10) that creates spikes, a strong solar-driven midday dip, and a price level that drifted from a mean of Rs 4.99 (train) to Rs 3.92 (test)

## 🗂️ Data

| | |
|---|---|
| Source | IEX Day-Ahead Market data (MCP, demand, supply, cleared volume) |
| Size | 96,480 15-minute observations, 1 Jan 2023 – 1 Oct 2025 |
| Quality | no missing or duplicate timestamps, no missing values, every day has all 96 blocks |
| Train | 1 Jan 2023 – 31 Mar 2025 (821 days) |
| Validation | 1 Apr 2025 – 30 Jun 2025 (91 days) |
| Test | 1 Jul 2025 – 1 Oct 2025 (93 days) |

The split is strictly chronological. Hyperparameters and model orders are chosen on the validation period only; the test period is touched once. The raw dataset is **not** included in this repository.

---

## 🧠 What is inside the notebook

| Part | What it does |
|---|---|
| 0–1 | Setup, loading and cleaning, data quality report |
| 2 | Chronological split, metrics (MAE, RMSE, MAPE, sMAPE, MASE), baselines |
| 3 | Stationarity tests (ADF, KPSS), ACF/PACF |
| 4–6 | ARIMA, SARIMA, Holt-Winters, seasonal decomposition |
| 7–8 | Comparison, residual diagnostics, error by hour of day |
| 9–10 | Day-ahead feature table (no leakage), XGBoost with random-search tuning |
| 11 | Prophet |
| 12–13 | LSTM and GRU |
| 14 | Fair comparison, stability by month, walk-forward test, paired tests |
| 15 | Volatile days, ceiling blocks, spike-risk classifier |
| 16 | SHAP explainability |
| 17 | Reusable forecast function, buyer savings simulation, deployment model and model card |
| 18 | Diebold-Mariano tests, prediction intervals, rolling-origin evaluation |
| 19 | Improvement experiments: extra features, delta target, LightGBM, ensemble |
| 20 | AutoTS |
| 20B | SARIMAX |
| 21 | Final comparison, market insights, worst-error days |

---

## 💡 Key findings

**1. Yesterday's price is the backbone, but calendar and exchange data matter too.**
SHAP shows that same-block price history (1–7 days ago) drives about **72%** of the model's output, calendar features about **12%**, and yesterday's demand/supply/volume about **12%** combined.

**2. Spikes are predictable.**
Ceiling blocks make up 13% of the test set. A model forecast of price ≥ Rs 8 flags them with **precision 0.88, recall 0.86 (F1 0.87)**. A dedicated XGBoost ceiling-risk classifier reaches ROC AUC 0.99 and the same F1.

**3. The forecast is useful for buyers.**
In a simulation where a buyer shifts flexible load to the forecast's cheapest hours over the 93 test days:

| Flexible hours per day | Saving vs random hours | Share of the maximum possible saving captured |
|---|---|---|
| 6 hours, any time | 58.1% | 99.3% (fixed "start at 10:00" rule: 97.5%) |
| 9 hours, any time | 52.1% | 99.8% (fixed rule: 77.8%) |
| 12 hours, any time | 43.9% | 99.2% (fixed rule: 3.2%) |
| 2 hours, evening only (17:00–23:00) | 42.6% | 99.3% (fixed rule: 93.2%) |

A fixed rule is almost as good when the buyer only needs a few midday hours. The forecast pays off when the load is large or the window is narrow.

**3b. Honest uncertainty.**
80% prediction intervals (quantile regression with conformal calibration) covered **84.9%** of test blocks with a mean width of Rs 2.8/kWh.

**4. Where the model is weakest.**
Error is highest when the market changes regime from one day to the next. The tuned XGBoost has an hourly MAE of **0.405** on stable days but **0.716** on "big-change" days (days whose average price moves sharply versus the previous day), and **0.753** on ceiling hours. A model built on "yesterday" lags a sudden shift by one day. Weather and fuel-price inputs are the obvious missing ingredient.

---

## 🧪 What did *not* work (reported honestly)

| Experiment | Result |
|---|---|
| 12 extra features (momentum, ceiling counts, same-hour averages, etc.) | Slightly worse on test (MAE 0.517 vs 0.502) |
| Predicting the change from yesterday instead of the price | Worse on test |
| Recency-weighted training | Small gain over the same model without it (0.541 → 0.532), still behind the tuned XGBoost |
| LightGBM and a simple ensemble | Statistically worse than the tuned XGBoost (p < 0.01) |
| LSTM / GRU | GRU only ties the naive benchmark; LSTM is worse |
| AutoTS (univariate, weekly refit) | Worse than the seasonal-naive benchmark |
| SARIMAX | Better than SARIMA (0.913 vs 0.963); `is_weekend` is highly significant, yesterday's bid ratio is borderline (p = 0.04); still behind the naive benchmark |

These variants looked better on the validation period but did not carry over to the test period, a reminder that ~90 days is a small window and that selecting models on it is noisy.

---

## ▶️ How to run

1. Open the notebook on Kaggle, or download it.
2. Add the IEX Day-Ahead Market dataset (`IEX_Market_Data.csv`) as an input. The notebook searches for this file name under the input folder.
3. Turn **Internet ON** (needed once for `pip install autots`).
4. Run all cells from top to bottom.

Main libraries: `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `xgboost`, `lightgbm`, `prophet`, `tensorflow`, `shap`, `autots`, `scipy`, `matplotlib`.

## ⚠️ Limitations

- The test period is 93 days, so rankings between close models are noisy.
- The data has no weather, fuel-price, or outage information, which limits accuracy on sudden regime changes.
- Prices are bounded by a regulatory ceiling that changed over time; results may not transfer to a different market design.
- This is a research and learning project, not trading advice.

---

## 👤 Author

**Kumar Saurav**
[GitHub](https://github.com/Saurav021406) · [LinkedIn](https://www.linkedin.com/in/kumar-saurav-8a31b633b/)

If you found this useful, a ⭐ on the repo is appreciated.
