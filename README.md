# Demand Forecasting & Inventory Optimization for Consumer Products

Forecast SKU-level weekly demand using ARIMA and XGBoost, then translate those forecasts into actionable inventory reorder recommendations under real-world constraints — lead time, safety stock, and working capital budget.

---

## Results

| Model | Avg MAPE |
|---|---|
| Naive | 31.7% |
| 4-week Moving Average | 27.8% |
| ARIMA(2,1,2) | 30.0% |
| **XGBoost** | **19.2%** |

**XGBoost achieved an 8.6 percentage point improvement over the best baseline (~22% relative improvement)**, with the largest gain on SKU_006 (20.1 pp) — a high-promotion SKU where ARIMA's inability to consume external features cost it the most.

---

## Project Structure

```
consumer-goods-demand-forecasting/
│
├── demand_forecasting_notebook.ipynb   # Main notebook — fully executable
└── README.md
```

The notebook is self-contained. It generates synthetic data, runs all models, and produces all charts and reorder recommendations in a single execution.

---

## Notebook Walkthrough

### 1. Synthetic Data Generation
Simulates 3 years of weekly sales (156 weeks, Jan 2021 – Dec 2023) for **8 SKUs across 4 categories** — Beverages, Snacks, Personal Care, and Household. Each SKU has a distinct base demand, growth trend, seasonal pattern, promotional lift (20–40%), and noise level. Holiday spikes are injected at Thanksgiving and Christmas. The result is 1,248 rows of realistic consumer goods demand data.

### 2. Exploratory Data Analysis
- Weekly sales trends per SKU with promo weeks shaded
- Average monthly demand by category to visualize seasonality
- Promo lift analysis: measured average uplift per SKU during promotional weeks

### 3. Feature Engineering
18 features built for the XGBoost model:

| Feature Type | Features |
|---|---|
| Lag features | Sales 1, 2, 3, 4, 8, 12, 52 weeks ago |
| Rolling stats | 4-week and 12-week rolling mean & std |
| Calendar | Week of year, month, quarter, year |
| Event flags | `promo_flag`, `holiday_flag` |
| Encoding | Label-encoded SKU and category |

All lag and rolling features use `shift(1)` or greater — no data leakage. Train/test split is strictly time-based (weeks 1–130 train, weeks 131–156 test).

### 4. ARIMA Modeling
Fits an independent **ARIMA(2,1,2)** model per SKU. The d=1 differencing addresses non-stationarity confirmed via ADF tests. Forecasts 26 weeks ahead on the held-out test set.

### 5. XGBoost Modeling
Trains an **XGBRegressor** (300 estimators, max depth 4, learning rate 0.05) using the full feature set. XGBoost substantially outperforms ARIMA on high-promotion SKUs because it directly consumes `promo_flag` and `holiday_flag` — features ARIMA cannot use.

### 6. Baseline Models
Two baselines for honest benchmarking:
- **Naive**: next week = last observed week
- **4-week Moving Average**: next week = mean of prior 4 weeks

### 7. Model Benchmarking
Side-by-side MAPE and RMSE comparison across all four models for each SKU, plus an average summary row.

### 8. Inventory Reorder Recommendations
Translates forecasts into procurement decisions using:

$$\text{Safety Stock} = Z \times \sigma_{demand} \times \sqrt{L}$$

$$\text{Reorder Point} = \bar{d} \times L + \text{Safety Stock}$$

$$\text{Reorder Qty} = \min(\text{EOQ},\ \text{Budget Limit})$$

Where $L$ = lead time in weeks, $Z$ = 1.65 (95% service level), and EOQ = $\sqrt{2DS/H}$. Reorder quantities are capped by a per-SKU share of a $150,000 working capital budget. SKUs with current stock at or below their reorder point are flagged for immediate procurement.

---

## Tech Stack

- **Python 3.10+**
- `pandas`, `numpy` — data manipulation
- `statsmodels` — ARIMA modeling
- `xgboost` — gradient boosting
- `scikit-learn` — metrics, label encoding
- `matplotlib` — visualization

---

## Setup

```bash
git clone https://github.com/dhruvi002/demand-forecast-inventory-optimizer.git
cd demand-forecast-inventory-optimizer
pip install -r requirements.txt
jupyter notebook demand_forecasting_notebook.ipynb
```

**requirements.txt**
```
pandas
numpy
matplotlib
statsmodels
xgboost
scikit-learn
jupyter
```

No external data needed — the notebook generates all data from scratch with `numpy.random.seed(42)` for reproducibility.

---

## Key Design Decisions

**Why XGBoost over ARIMA for the primary model?** ARIMA operates solely on the demand series itself — it cannot incorporate known future events like promotions. XGBoost treats `promo_flag` as a first-class feature, letting it anticipate demand spikes rather than react to them. This is the core driver of the performance gap, especially on high-promotion SKUs.

**Why MAPE as the evaluation metric?** MAPE is scale-independent, making it valid for cross-SKU comparison where base demand ranges from ~155 to ~650 units/week. Its limitation — asymmetric penalization of over-forecasting — is noted, and in a production setting would be complemented with a directional bias metric.

**Why a time-based train/test split?** Random splitting would leak future demand into training, artificially inflating performance. The strict chronological cutoff (week 130) mirrors real deployment conditions where the model only ever trains on the past.

---

## Potential Extensions

- **Auto-ARIMA** (pmdarima) for per-SKU order selection instead of a fixed (2,1,2)
- **SARIMA(p,d,q)(P,D,Q,52)** to explicitly model annual seasonality
- **ARIMAX** to incorporate `promo_flag` as an exogenous variable in ARIMA
- **Walk-forward validation** for more robust out-of-sample evaluation
- **Price elasticity features** using `base_price` and week-over-week price changes
- **Airflow DAG** for weekly automated retraining and reorder generation
