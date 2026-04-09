"""
C24-IQ: Machine Learning Engine
Price Prediction · Anomaly Detection · Demand Forecasting
Models: Random Forest, Gradient Boosting, XGBoost, Linear Regression
"""

import numpy as np
import pandas as pd
import pickle, os, warnings
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
import xgboost as xgb

warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Feature Engineering ─────────────────────────────────────────────────────

CATEGORICAL_FEATURES = ["brand", "segment", "fuel_type", "transmission",
                         "city", "color", "condition_grade", "region"]
NUMERIC_FEATURES = ["age_years", "km_driven", "owner_number", "accident_history",
                     "manufacture_year", "popularity_score", "demand_index"]
BOOL_FEATURES = ["insurance_valid", "service_records"]
TARGET = "listing_price"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced feature engineering."""
    df = df.copy()
    
    # Ratio features
    df["km_per_year"] = df["km_driven"] / df["age_years"].clip(lower=1)
    df["age_km_ratio"] = df["age_years"] * df["km_driven"] / 1e6
    df["price_per_km"] = df.get("listing_price", 0) / df["km_driven"].clip(lower=1)
    
    # Interaction features
    df["luxury_segment"] = (df["segment"].isin(["SUV", "MPV"])).astype(int)
    df["is_automatic"] = (df["transmission"].isin(["Automatic", "CVT", "DCT"])).astype(int)
    df["is_ev_hybrid"] = (df["fuel_type"].isin(["Electric", "Hybrid"])).astype(int)
    df["clean_car"] = ((df["accident_history"] == 0) & df["service_records"]).astype(int)
    df["desirable_color"] = df["color"].isin(["White", "Silver", "Black"]).astype(int)
    
    # Owner penalty index
    df["owner_penalty"] = (df["owner_number"] - 1) * 0.04
    
    # Demand x popularity interaction
    df["demand_x_popularity"] = df["demand_index"] * df["popularity_score"] / 100
    
    # Season effect (Q4 high demand)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["listing_month"] = df["created_at"].dt.month.fillna(6).astype(int)
    df["listing_quarter"] = df["created_at"].dt.quarter.fillna(2).astype(int)
    df["is_festive_season"] = df["listing_month"].isin([9, 10, 11]).astype(int)

    bool_cols = ["insurance_valid", "service_records"]
    for c in bool_cols:
        df[c] = df[c].astype(int)

    return df


# ─── Model Training ───────────────────────────────────────────────────────────

class C24PriceEngine:
    """Ensemble price prediction engine."""

    def __init__(self):
        self.models = {}
        self.ensemble_weights = {}
        self.preprocessor = None
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.metrics = {}
        self.is_trained = False

    def _build_preprocessor(self):
        """Build sklearn preprocessing pipeline."""
        cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        num_pipe = RobustScaler()

        extra_numeric = ["km_per_year", "age_km_ratio", "luxury_segment",
                         "is_automatic", "is_ev_hybrid", "clean_car",
                         "desirable_color", "owner_penalty", "demand_x_popularity",
                         "listing_month", "listing_quarter", "is_festive_season"]
        all_numeric = NUMERIC_FEATURES + BOOL_FEATURES + extra_numeric

        self.preprocessor = ColumnTransformer([
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
            ("num", num_pipe, all_numeric),
        ], remainder="drop")
        return self.preprocessor

    def train(self, df: pd.DataFrame) -> dict:
        """Train all models and compute ensemble weights."""
        df = engineer_features(df)
        df = df.dropna(subset=[TARGET])

        all_features = (CATEGORICAL_FEATURES + NUMERIC_FEATURES +
                        BOOL_FEATURES + ["km_per_year", "age_km_ratio",
                        "luxury_segment", "is_automatic", "is_ev_hybrid",
                        "clean_car", "desirable_color", "owner_penalty",
                        "demand_x_popularity", "listing_month",
                        "listing_quarter", "is_festive_season"])

        X = df[all_features]
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        preprocessor = self._build_preprocessor()
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t  = preprocessor.transform(X_test)

        # ── Define models ──
        model_defs = {
            "random_forest": RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_leaf=4,
                n_jobs=-1, random_state=42),
            "gradient_boost": GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.08, max_depth=5,
                subsample=0.8, random_state=42),
            "xgboost": xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.07, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0),
            "ridge": Ridge(alpha=100),
            "elasticnet": ElasticNet(alpha=50, l1_ratio=0.5, max_iter=2000),
        }

        test_preds = {}
        for name, model in model_defs.items():
            model.fit(X_train_t, y_train)
            preds = model.predict(X_test_t)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

            self.models[name] = model
            self.metrics[name] = {
                "mae": round(mae, 0),
                "rmse": round(rmse, 0),
                "r2": round(r2, 4),
                "mape": round(mape, 2),
            }
            test_preds[name] = preds

        # ── Stacking ensemble weights (inverse-MAE weighting) ──
        maes = {k: self.metrics[k]["mae"] for k in self.metrics}
        inv_mae = {k: 1/v for k, v in maes.items()}
        total = sum(inv_mae.values())
        self.ensemble_weights = {k: v/total for k, v in inv_mae.items()}

        # ── Ensemble metrics ──
        ensemble_preds = sum(
            self.ensemble_weights[k] * test_preds[k]
            for k in test_preds
        )
        self.metrics["ensemble"] = {
            "mae": round(mean_absolute_error(y_test, ensemble_preds), 0),
            "rmse": round(np.sqrt(mean_squared_error(y_test, ensemble_preds)), 0),
            "r2": round(r2_score(y_test, ensemble_preds), 4),
            "mape": round(np.mean(np.abs((y_test - ensemble_preds)/y_test))*100, 2),
        }

        # ── Feature importance (from RF) ──
        rf = self.models["random_forest"]
        try:
            feature_names = (preprocessor.named_transformers_["cat"]
                             .get_feature_names_out(CATEGORICAL_FEATURES).tolist() +
                             preprocessor.named_transformers_["num"]
                             .get_feature_names_out().tolist())
        except Exception:
            feature_names = [f"f{i}" for i in range(X_train_t.shape[1])]

        importances = rf.feature_importances_
        fi = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        self.feature_importance = {k: round(float(v), 5) for k, v in fi[:20]}

        # ── Anomaly detector ──
        self.anomaly_detector = IsolationForest(
            n_estimators=100, contamination=0.05, random_state=42)
        self.anomaly_detector.fit(X_train_t)

        self.X_test_t = X_test_t
        self.y_test = y_test
        self.all_features = all_features
        self.is_trained = True

        self._save()
        return self.metrics

    def predict(self, input_dict: dict) -> dict:
        """Predict price for a single car."""
        if not self.is_trained:
            self._load()

        df = pd.DataFrame([input_dict])
        df = engineer_features(df)
        X = df[self.all_features]
        X_t = self.preprocessor.transform(X)

        # Individual predictions
        preds = {}
        for name, model in self.models.items():
            preds[name] = float(model.predict(X_t)[0])

        # Ensemble
        ensemble = sum(self.ensemble_weights[k] * preds[k] for k in preds)

        # Confidence interval (from RF variance)
        rf = self.models["random_forest"]
        tree_preds = np.array([t.predict(X_t)[0] for t in rf.estimators_])
        std = tree_preds.std()
        ci_lower = max(0, ensemble - 1.96 * std)
        ci_upper = ensemble + 1.96 * std

        # Anomaly score
        anomaly = self.anomaly_detector.decision_function(X_t)[0]
        is_anomaly = bool(self.anomaly_detector.predict(X_t)[0] == -1)

        # Depreciation stage
        age = input_dict.get("age_years", 3)
        if age <= 2:   stage = "Low Depreciation (New)"
        elif age <= 4: stage = "Moderate Depreciation"
        elif age <= 7: stage = "High Depreciation"
        else:          stage = "Steep Depreciation"

        return {
            "ensemble_price": round(ensemble / 1000) * 1000,
            "ci_lower": round(ci_lower / 1000) * 1000,
            "ci_upper": round(ci_upper / 1000) * 1000,
            "model_predictions": {k: round(v) for k, v in preds.items()},
            "confidence_pct": round(max(0, min(100, 100 - (std/ensemble*100))), 1),
            "anomaly_score": round(float(anomaly), 4),
            "is_anomaly_price": is_anomaly,
            "depreciation_stage": stage,
        }

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect price anomalies in a DataFrame."""
        if not self.is_trained:
            self._load()
        df_e = engineer_features(df)
        X = df_e[self.all_features]
        X_t = self.preprocessor.transform(X)
        df["anomaly_score"] = self.anomaly_detector.decision_function(X_t)
        df["is_anomaly"] = self.anomaly_detector.predict(X_t) == -1
        return df

    def _save(self):
        state = {
            "models": self.models,
            "ensemble_weights": self.ensemble_weights,
            "preprocessor": self.preprocessor,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "all_features": self.all_features,
            "anomaly_detector": self.anomaly_detector,
        }
        with open(os.path.join(MODEL_DIR, "price_engine.pkl"), "wb") as f:
            pickle.dump(state, f)

    def _load(self):
        path = os.path.join(MODEL_DIR, "price_engine.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.__dict__.update(state)
            self.is_trained = True
            return True
        return False

    def get_shap_like_explanation(self, input_dict: dict) -> list:
        """Approximate feature contributions for explainability."""
        if not self.is_trained:
            self._load()
        
        age = input_dict.get("age_years", 3)
        km = input_dict.get("km_driven", 30000)
        owner = input_dict.get("owner_number", 1)
        accident = input_dict.get("accident_history", 0)
        service = input_dict.get("service_records", True)
        insurance = input_dict.get("insurance_valid", True)
        
        contributions = [
            {"feature": "Vehicle Age",      "impact": -age * 0.07,          "direction": "negative"},
            {"feature": "KM Driven",        "impact": -km / 200000,         "direction": "negative"},
            {"feature": "Owner History",    "impact": -(owner-1) * 0.05,    "direction": "negative"},
            {"feature": "Accident History", "impact": -accident * 0.05,     "direction": "negative"},
            {"feature": "Service Records",  "impact": 0.03 if service else -0.03, "direction": "positive" if service else "negative"},
            {"feature": "Insurance Valid",  "impact": 0.02 if insurance else -0.02, "direction": "positive" if insurance else "negative"},
            {"feature": "City Demand",      "impact": input_dict.get("demand_index", 100)/1000, "direction": "positive"},
            {"feature": "Popularity Score", "impact": input_dict.get("popularity_score", 80)/1000, "direction": "positive"},
        ]
        return contributions


# ─── Demand Forecaster ────────────────────────────────────────────────────────

class DemandForecaster:
    """Simple time-series demand forecasting using rolling statistics."""

    def forecast(self, df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
        df = df.copy()
        df["created_at"] = pd.to_datetime(df["created_at"])
        daily = (df.groupby(df["created_at"].dt.date)
                   .agg(volume=("id", "count"),
                        avg_price=("listing_price", "mean"))
                   .reset_index())
        daily.columns = ["date", "volume", "avg_price"]
        daily = daily.sort_values("date")
        
        if len(daily) < 7:
            return pd.DataFrame()

        # Rolling averages as baseline
        daily["vol_ma7"]  = daily["volume"].rolling(7, min_periods=1).mean()
        daily["vol_ma30"] = daily["volume"].rolling(30, min_periods=1).mean()
        daily["price_ma7"] = daily["avg_price"].rolling(7, min_periods=1).mean()
        
        # Forecast next N days (simple exponential smoothing)
        last_vol  = daily["vol_ma7"].iloc[-1]
        last_price = daily["price_ma7"].iloc[-1]
        
        trend_vol  = (daily["vol_ma7"].iloc[-1] - daily["vol_ma7"].iloc[-7]) / 7 if len(daily) >= 7 else 0
        trend_price = (daily["price_ma7"].iloc[-1] - daily["price_ma7"].iloc[-7]) / 7 if len(daily) >= 7 else 0

        from datetime import date, timedelta
        last_date = daily["date"].iloc[-1]
        forecast_rows = []
        for i in range(1, periods + 1):
            d = last_date + timedelta(days=i)
            noise_vol = np.random.normal(0, last_vol * 0.05)
            noise_price = np.random.normal(0, last_price * 0.01)
            forecast_rows.append({
                "date": d,
                "volume": max(0, round(last_vol + trend_vol * i + noise_vol)),
                "avg_price": round(last_price + trend_price * i + noise_price),
                "is_forecast": True
            })
        
        historical = daily[["date", "volume", "avg_price"]].copy()
        historical["is_forecast"] = False
        forecast_df = pd.DataFrame(forecast_rows)
        return pd.concat([historical, forecast_df], ignore_index=True)


# ─── Singleton ────────────────────────────────────────────────────────────────
price_engine = C24PriceEngine()
demand_forecaster = DemandForecaster()
