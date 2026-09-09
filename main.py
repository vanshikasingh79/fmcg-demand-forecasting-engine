from datetime import date
from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).with_name("model.pkl")

with MODEL_PATH.open("rb") as model_file:
	MODEL_ARTIFACT = pickle.load(model_file)

MODELS = MODEL_ARTIFACT["models"]
FEATURE_COLUMNS = MODEL_ARTIFACT["feature_columns"]
SKU_ENCODER = MODEL_ARTIFACT["sku_encoder"]
CATEGORY_ENCODER = MODEL_ARTIFACT["category_encoder"]

SKU_CATEGORIES = {
	"SKU_001": "Beverages",
	"SKU_002": "Beverages",
	"SKU_003": "Snacks",
	"SKU_004": "Snacks",
	"SKU_005": "Personal Care",
	"SKU_006": "Personal Care",
	"SKU_007": "Household",
	"SKU_008": "Household",
}

app = FastAPI(title="Demand Forecasting API", version="1.0.0")


class PredictionRequest(BaseModel):
	sku_id: str = Field(..., description="SKU identifier, for example SKU_001")
	regional_sales_history: list[float] | dict[str, list[float]] = Field(
		...,
		description=(
			"Chronological weekly demand history, or a mapping of region names to "
			"chronological weekly histories"
		),
	)
	stock_level: float = Field(..., ge=0, description="Current inventory level in units")


class PredictionResponse(BaseModel):
	sku_id: str
	forecasted_demand: float
	stock_alert_status: str
	stock_level: float


@app.get("/health")
def health_check():
	return {"status": "ok", "models_loaded": len(MODELS)}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
	if request.sku_id not in MODELS:
		raise HTTPException(
			status_code=404,
			detail=f"Unknown SKU '{request.sku_id}'. Available SKUs: {sorted(MODELS)}",
		)

	category = SKU_CATEGORIES.get(request.sku_id)
	if category is None:
		raise HTTPException(status_code=500, detail="Category mapping is missing for this SKU")

	try:
		sku_encoded = int(SKU_ENCODER.transform([request.sku_id])[0])
		category_encoded = int(CATEGORY_ENCODER.transform([category])[0])
	except ValueError as error:
		raise HTTPException(status_code=500, detail="Saved encoders cannot encode this SKU") from error

	if isinstance(request.regional_sales_history, dict):
		regional_histories = list(request.regional_sales_history.values())
		if not regional_histories or any(len(history) < 4 for history in regional_histories):
			raise HTTPException(
				status_code=422,
				detail="Each regional sales history must contain at least 4 weekly values",
			)
		history = [
			sum(values[-offset] for values in regional_histories) / len(regional_histories)
			for offset in range(4, 0, -1)
		]
	else:
		if len(request.regional_sales_history) < 4:
			raise HTTPException(
				status_code=422,
				detail="regional_sales_history must contain at least 4 weekly values",
			)
		history = request.regional_sales_history

	today = date.today()
	features = pd.DataFrame(
		[
			{
				"lag_7d": history[-1],
				"lag_30d": history[-4],
				"week_of_year": today.isocalendar().week,
				"month": today.month,
				"quarter": (today.month - 1) // 3 + 1,
				"year": today.year,
				"promo_flag": 0,
				"holiday_flag": 0,
				"sku_enc": sku_encoded,
				"cat_enc": category_encoded,
			}
		],
		columns=FEATURE_COLUMNS,
	)

	forecast = max(0.0, float(MODELS[request.sku_id].predict(features)[0]))
	alert_status = "LOW_STOCK" if request.stock_level < forecast else "OK"
	return PredictionResponse(
		sku_id=request.sku_id,
		forecasted_demand=round(forecast, 2),
		stock_alert_status=alert_status,
		stock_level=request.stock_level,
	)
