from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


RANDOM_STATE = 42
TRAIN_WEEKS = 130
MODEL_PATH = Path(__file__).with_name("model.pkl")


def generate_data():
	"""Generate the weekly synthetic demand data used by the notebook."""
	rng = np.random.RandomState(RANDOM_STATE)
	skus = {
		"SKU_001": {"category": "Beverages", "base": 420, "trend": 0.08, "noise": 0.12, "promo_lift": 0.35},
		"SKU_002": {"category": "Beverages", "base": 280, "trend": 0.04, "noise": 0.15, "promo_lift": 0.25},
		"SKU_003": {"category": "Snacks", "base": 650, "trend": 0.12, "noise": 0.10, "promo_lift": 0.40},
		"SKU_004": {"category": "Snacks", "base": 310, "trend": -0.02, "noise": 0.18, "promo_lift": 0.30},
		"SKU_005": {"category": "Personal Care", "base": 190, "trend": 0.06, "noise": 0.08, "promo_lift": 0.20},
		"SKU_006": {"category": "Personal Care", "base": 240, "trend": 0.10, "noise": 0.11, "promo_lift": 0.28},
		"SKU_007": {"category": "Household", "base": 380, "trend": 0.03, "noise": 0.14, "promo_lift": 0.32},
		"SKU_008": {"category": "Household", "base": 155, "trend": 0.05, "noise": 0.20, "promo_lift": 0.22},
	}
	dates = pd.date_range("2021-01-04", periods=156, freq="W-MON")
	rows = []

	for sku_id, config in skus.items():
		time_index = np.arange(len(dates))
		trend = config["base"] * (1 + config["trend"] * time_index / 52)
		season = (
			0.20 * np.sin(2 * np.pi * time_index / 52)
			+ 0.08 * np.sin(4 * np.pi * time_index / 52 + 0.5)
		)
		promo_weeks = np.zeros(len(dates))
		for year in range(3):
			promo_indices = rng.choice(
				range(year * 52, (year + 1) * 52), size=8, replace=False
			)
			promo_weeks[promo_indices] = 1

		holiday_spike = np.zeros(len(dates))
		for year in range(3):
			for holiday_week in (47, 51):
				index = year * 52 + holiday_week
				if index < len(dates):
					holiday_spike[index] = 0.30

		noise = rng.normal(0, config["noise"], len(dates))
		demand = trend * (
			1 + season + promo_weeks * config["promo_lift"] + holiday_spike + noise
		)
		demand = np.maximum(demand, 0).round().astype(int)

		for index, date in enumerate(dates):
			rows.append(
				{
					"date": date,
					"sku_id": sku_id,
					"category": config["category"],
					"sales_units": demand[index],
					"promo_flag": int(promo_weeks[index]),
					"holiday_flag": int(holiday_spike[index] > 0),
				}
			)

	data = pd.DataFrame(rows).sort_values(["sku_id", "date"]).reset_index(drop=True)
	return data


def build_features(data):
	"""Add calendar, event, encoded categorical, and demand-lag features."""
	data = data.sort_values(["sku_id", "date"]).copy()
	data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
	data["month"] = data["date"].dt.month
	data["quarter"] = data["date"].dt.quarter
	data["year"] = data["date"].dt.year

	for _, sku_data in data.groupby("sku_id"):
		index = sku_data.index
		# The source data has one observation per week: 7 days = 1 row,
		# while the nearest weekly equivalent of 30 days is 4 rows.
		data.loc[index, "lag_7d"] = sku_data["sales_units"].shift(1).values
		data.loc[index, "lag_30d"] = sku_data["sales_units"].shift(4).values

	sku_encoder = LabelEncoder()
	category_encoder = LabelEncoder()
	data["sku_enc"] = sku_encoder.fit_transform(data["sku_id"])
	data["cat_enc"] = category_encoder.fit_transform(data["category"])

	feature_columns = [
		"lag_7d",
		"lag_30d",
		"week_of_year",
		"month",
		"quarter",
		"year",
		"promo_flag",
		"holiday_flag",
		"sku_enc",
		"cat_enc",
	]
	return data, feature_columns, sku_encoder, category_encoder


def train_models(data, feature_columns):
	"""Train one XGBoost regressor for each SKU using a time-based split."""
	cutoff_date = data["date"].sort_values().unique()[TRAIN_WEEKS]
	models = {}

	for sku_id in sorted(data["sku_id"].unique()):
		sku_data = data[(data["sku_id"] == sku_id) & data[feature_columns].notna().all(axis=1)]
		train_data = sku_data[sku_data["date"] < cutoff_date]
		test_data = sku_data[sku_data["date"] >= cutoff_date]

		model = XGBRegressor(
			n_estimators=300,
			max_depth=4,
			learning_rate=0.05,
			subsample=0.8,
			colsample_bytree=0.8,
			reg_alpha=0.1,
			reg_lambda=1.0,
			random_state=RANDOM_STATE,
			objective="reg:squarederror",
			verbosity=0,
		)
		model.fit(
			train_data[feature_columns],
			train_data["sales_units"],
			eval_set=[(test_data[feature_columns], test_data["sales_units"])],
			verbose=False,
		)
		models[sku_id] = model
		print(f"{sku_id}: trained on {len(train_data)} rows; tested on {len(test_data)} rows")

	return models


def main():
	data = generate_data()
	featured_data, feature_columns, sku_encoder, category_encoder = build_features(data)
	models = train_models(featured_data, feature_columns)

	artifact = {
		"models": models,
		"feature_columns": feature_columns,
		"sku_encoder": sku_encoder,
		"category_encoder": category_encoder,
	}
	with MODEL_PATH.open("wb") as model_file:
		pickle.dump(artifact, model_file)
	print(f"Saved {len(models)} trained models to {MODEL_PATH}")


if __name__ == "__main__":
	main()
