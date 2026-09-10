from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

RANDOM_STATE = 42
# Extended to 2026: 290 total weeks (2021-01-04 through mid-2026)
TOTAL_WEEKS = 290
TRAIN_WEEKS = 230
MODEL_PATH = Path(__file__).with_name("model.pkl")

REGIONS = ["North", "South", "East", "West"]

SKU_CONFIG = {
    "SKU_001": {"category": "Beverages", "base": 420, "trend": 0.08, "noise": 0.12, "promo_lift": 0.35},
    "SKU_002": {"category": "Beverages", "base": 280, "trend": 0.04, "noise": 0.15, "promo_lift": 0.25},
    "SKU_003": {"category": "Snacks", "base": 650, "trend": 0.12, "noise": 0.10, "promo_lift": 0.40},
    "SKU_004": {"category": "Snacks", "base": 310, "trend": -0.02, "noise": 0.18, "promo_lift": 0.30},
    "SKU_005": {"category": "Personal Care", "base": 190, "trend": 0.06, "noise": 0.08, "promo_lift": 0.20},
    "SKU_006": {"category": "Personal Care", "base": 240, "trend": 0.10, "noise": 0.11, "promo_lift": 0.28},
    "SKU_007": {"category": "Household", "base": 380, "trend": 0.03, "noise": 0.14, "promo_lift": 0.32},
    "SKU_008": {"category": "Household", "base": 155, "trend": 0.05, "noise": 0.20, "promo_lift": 0.22},
}


def generate_data():
    """Generate weekly demand data covering 2021 through 2026 across regions."""
    rng = np.random.RandomState(RANDOM_STATE)
    dates = pd.date_range("2021-01-04", periods=TOTAL_WEEKS, freq="W-MON")
    rows = []

    for sku_id, config in SKU_CONFIG.items():
        for region in REGIONS:
            region_multiplier = 0.85 + (REGIONS.index(region) * 0.1)
            time_index = np.arange(len(dates))
            trend = (config["base"] * region_multiplier) * (1 + config["trend"] * time_index / 52)
            season = 0.20 * np.sin(2 * np.pi * time_index / 52) + 0.08 * np.sin(4 * np.pi * time_index / 52 + 0.5)

            promo_weeks = np.zeros(len(dates))
            for year_idx in range(6):
                start = year_idx * 52
                end = min((year_idx + 1) * 52, len(dates))
                if start < len(dates):
                    promo_indices = rng.choice(range(start, end), size=min(8, end - start), replace=False)
                    promo_weeks[promo_indices] = 1

            holiday_spike = np.zeros(len(dates))
            for year_idx in range(6):
                for holiday_week in (47, 51):
                    idx = year_idx * 52 + holiday_week
                    if idx < len(dates):
                        holiday_spike[idx] = 0.30

            noise = rng.normal(0, config["noise"], len(dates))
            demand = trend * (1 + season + promo_weeks * config["promo_lift"] + holiday_spike + noise)
            demand = np.maximum(demand, 0).round().astype(int)

            for idx, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "sku_id": sku_id,
                        "region": region,
                        "category": config["category"],
                        "sales_units": demand[idx],
                        "promo_flag": int(promo_weeks[idx]),
                        "holiday_flag": int(holiday_spike[idx] > 0),
                    }
                )

    return pd.DataFrame(rows).sort_values(["sku_id", "region", "date"]).reset_index(drop=True)


def build_features(data):
    """Engineer temporal, regional, categorical, and lag features."""
    data = data.sort_values(["sku_id", "region", "date"]).copy()
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
    data["month"] = data["date"].dt.month
    data["quarter"] = data["date"].dt.quarter
    data["year"] = data["date"].dt.year

    for _, group in data.groupby(["sku_id", "region"]):
        idx = group.index
        data.loc[idx, "lag_7d"] = group["sales_units"].shift(1).values
        data.loc[idx, "lag_30d"] = group["sales_units"].shift(4).values

    sku_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    region_encoder = LabelEncoder()

    data["sku_enc"] = sku_encoder.fit_transform(data["sku_id"])
    data["cat_enc"] = category_encoder.fit_transform(data["category"])
    data["region_enc"] = region_encoder.fit_transform(data["region"])

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
        "region_enc",
    ]
    return data, feature_columns, sku_encoder, category_encoder, region_encoder


def train_models(data, feature_columns):
    """Train XGBoost regressors and log MAPE, MAE, and RMSE metrics."""
    cutoff_date = data["date"].sort_values().unique()[TRAIN_WEEKS]
    models = {}
    mapes, maes, rmses = [], [], []

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

        preds = model.predict(test_data[feature_columns])
        mape = mean_absolute_percentage_error(test_data["sales_units"], preds) * 100
        mae = mean_absolute_error(test_data["sales_units"], preds)
        rmse = root_mean_squared_error(test_data["sales_units"], preds)

        mapes.append(mape)
        maes.append(mae)
        rmses.append(rmse)
        models[sku_id] = model

        print(f"{sku_id} -> MAPE: {mape:.2f}% | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

    print("\n--- Portfolio Baseline Metrics ---")
    print(f"Overall Mean MAPE: {np.mean(mapes):.2f}%")
    print(f"Overall Mean MAE:  {np.mean(maes):.2f} units")
    print(f"Overall Mean RMSE: {np.mean(rmses):.2f} units")

    return models


def main():
    data = generate_data()
    featured_data, feature_columns, sku_encoder, category_encoder, region_encoder = build_features(data)
    models = train_models(featured_data, feature_columns)

    # Extract metadata map to remove duplication in API layer
    sku_metadata = {sku: config["category"] for sku, config in SKU_CONFIG.items()}

    artifact = {
        "models": models,
        "feature_columns": feature_columns,
        "sku_encoder": sku_encoder,
        "category_encoder": category_encoder,
        "region_encoder": region_encoder,
        "sku_metadata": sku_metadata,
        "valid_regions": REGIONS,
    }

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(artifact, model_file)
    print(f"\nSuccessfully serialized artifact to {MODEL_PATH}")


if __name__ == "__main__":
    main()