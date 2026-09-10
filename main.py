from datetime import datetime
from pathlib import Path
import pickle

from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).with_name("model.pkl")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train_and_predict.py first.")

with MODEL_PATH.open("rb") as f:
    artifact = pickle.load(f)

models = artifact["models"]
feature_columns = artifact["feature_columns"]
sku_encoder = artifact["sku_encoder"]
category_encoder = artifact["category_encoder"]
region_encoder = artifact["region_encoder"]
sku_metadata = artifact["sku_metadata"]
valid_regions = artifact["valid_regions"]

app = FastAPI(
    title="FMCG Demand Forecasting & Inventory Risk API",
    version="2.0.0",
    description="Real-time multi-SKU demand prediction and stockout risk evaluation engine.",
)


class PredictionRequest(BaseModel):
    sku_id: str = Field(..., example="SKU_001")
    region: str = Field("North", example="North")
    date: str = Field(..., example="2026-09-09")
    lag_7d: float = Field(..., ge=0, example=450.0)
    lag_30d: float = Field(..., ge=0, example=420.0)
    promo_flag: int = Field(0, ge=0, le=1, example=1)
    holiday_flag: int = Field(0, ge=0, le=1, example=0)
    current_stock: int = Field(..., ge=0, example=300)


class PredictionResponse(BaseModel):
    sku_id: str
    region: str
    category: str
    target_date: str
    forecasted_demand: float
    current_stock: int
    stock_status: str


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "available_skus": list(models.keys()),
        "available_regions": valid_regions,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_demand(payload: PredictionRequest):
    if payload.sku_id not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown SKU '{payload.sku_id}'. Available SKUs: {list(models.keys())}",
        )

    if payload.region not in valid_regions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid region '{payload.region}'. Supported regions: {valid_regions}",
        )

    try:
        parsed_date = datetime.strptime(payload.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    category = sku_metadata[payload.sku_id]

    sku_enc = sku_encoder.transform([payload.sku_id])[0]
    cat_enc = category_encoder.transform([category])[0]
    reg_enc = region_encoder.transform([payload.region])[0]

    input_df = pd.DataFrame(
        [
            {
                "lag_7d": payload.lag_7d,
                "lag_30d": payload.lag_30d,
                "week_of_year": parsed_date.isocalendar().week,
                "month": parsed_date.month,
                "quarter": (parsed_date.month - 1) // 3 + 1,
                "year": parsed_date.year,
                "promo_flag": payload.promo_flag,
                "holiday_flag": payload.holiday_flag,
                "sku_enc": sku_enc,
                "cat_enc": cat_enc,
                "region_enc": reg_enc,
            }
        ]
    )[feature_columns]

    sku_model = models[payload.sku_id]
    forecast = float(sku_model.predict(input_df)[0])
    forecast = max(0.0, round(forecast, 2))

    stock_status = "Out-of-Stock Risk" if payload.current_stock < forecast else "Optimal"

    return PredictionResponse(
        sku_id=payload.sku_id,
        region=payload.region,
        category=category,
        target_date=payload.date,
        forecasted_demand=forecast,
        current_stock=payload.current_stock,
        stock_status=stock_status,
    )