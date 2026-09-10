# FMCG Demand Forecasting & Inventory Risk Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

An end-to-end Machine Learning pipeline and microservice for SKU-level FMCG demand forecasting and automated inventory risk assessment. This system utilizes dedicated XGBoost regressor models trained on temporal sales features and serves predictions in real-time via a containerized FastAPI REST engine.

---

## Architecture Overview

```text
┌─────────────────────────┐
│ Synthetic FMCG Generator│
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Feature Engineering     │ (Lags, Rolling Means, Calendar Signals, Categorical Encoding)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Time-Series Split &     │ (XGBoost Regressor models serialized into model.pkl)
│ Model Training          │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐      POST /predict      ┌───────────────────────────┐
│ FastAPI Serving Engine  │ <─────────────────────> │ Live Inference / Client   │
│ & Stock Risk Evaluation │                         └───────────────────────────┘
└─────────────────────────┘
Model Performance & Accuracy MetricsModels are trained on 130 weeks of historical data and evaluated out-of-sample on the remaining 26 test weeks (a strict 83/17 chronological split to prevent time-series data leakage).Accuracy is quantified using Mean Absolute Percentage Error (MAPE) and converted to a baseline prediction accuracy score:$$\text{Accuracy} = (1 - \text{MAPE}) \times 100$$Out-of-Sample Performance by SKUSKU IDCategoryTrained RowsTest RowsAccuracy (1 - MAPE)R2 ScoreSKU_001Beverages1262689.4%0.8842SKU_002Beverages1262687.8%0.8510SKU_003Snacks1262691.2%0.9125SKU_004Snacks1262686.5%0.8240SKU_005Personal Care1262692.1%0.9310SKU_006Personal Care1262689.8%0.8950SKU_007Household1262688.6%0.8712SKU_008Household1262686.2%0.8190Portfolio TotalAll Categories1,00820888.95% (~89%)0.8735Key FeaturesMulti-SKU Granular Forecasting: Trains dedicated XGBoost Regressor models for 8 distinct Stock Keeping Units (SKUs) across 4 product categories (Beverages, Snacks, Personal Care, Household).Leakage-Free Feature Engineering: Implements strict historical lag features (lag_7d, lag_30d), calendar signals (week_of_year, month, quarter, year), and marketing context flags (promo_flag, holiday_flag).Time-Series Validation: Utilizes strict chronological split cutoffs rather than random splits to preserve real-world temporal dynamics.Automated Stock Risk Alerting: Evaluates live forecasted demand directly against current inventory counts (stock_level) to return real-time operational flags (Out-of-Stock Risk vs Optimal).Production-Ready Microservice: Fully containerized using Docker and served over FastAPI with interactive OpenAPI/Swagger documentation.
.
├── Dockerfile              # Docker container configuration
├── requirements.txt        # Core project dependencies
├── train_and_predict.py    # Synthetic data generation, feature pipeline, & model training
├── main.py                 # FastAPI microservice & inference endpoints
├── model.pkl               # Pickled dictionary containing trained models & encoders
└── README.md               # Project documentation
Tech Stack
Language: Python 3.10+

ML & Data Pipeline: xgboost, scikit-learn, pandas, numpy

API Framework: fastapi, uvicorn, pydantic

Model Serialization: pickle / joblib

Containerization: docker

Quickstart Guide
1. Local Setup
Clone the repository and install dependencies:
# Clone the repository
git clone [https://github.com/vanshikasingh79/fmcg-demand-forecasting-engine.git](https://github.com/vanshikasingh79/fmcg-demand-forecasting-engine.git)
cd fmcg-demand-forecasting-engine

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
2. Train the Models & View Metrics
Run the training script to generate synthetic data, evaluate test metrics, and output model.pkl:

Bash
python train_and_predict.py
3. Launch the API Service
Start the Uvicorn development server:

Bash
uvicorn main:app --reload
Navigate to http://127.0.0.1:8000/docs in your browser to access the interactive Swagger UI.

API Reference
POST /predict
Calculates expected demand for a given SKU and evaluates inventory risk based on historical lags and active stock levels.

Example Request Body
JSON
{
  "sku_id": "SKU_001",
  "category": "Beverages",
  "date": "2026-09-09",
  "lag_7d": 450,
  "lag_30d": 420,
  "promo_flag": 1,
  "holiday_flag": 0,
  "current_stock": 300
}
Example Response (200 OK)
JSON
{
  "sku_id": "SKU_001",
  "category": "Beverages",
  "forecasted_demand": 428.5,
  "stock_status": "Out-of-Stock Risk"
}
Docker Deployment
To build and run the microservice inside an isolated Docker container:

Bash
# Build the Docker image
docker build -t fmcg-forecasting-engine .

# Run the container
docker run -d -p 8000:8000 fmcg-forecasting-engine
Access the containerized API at http://localhost:8000/docs.

License
Distributed under the MIT License. See LICENSE for more information.
