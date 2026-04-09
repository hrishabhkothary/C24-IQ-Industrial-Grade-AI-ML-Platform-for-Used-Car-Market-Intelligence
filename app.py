"""
C24-IQ: Cars24 Intelligence Quotient Platform
Flask Web Application — Main Entry Point
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import numpy as np
import json
import time
import threading
import random
from datetime import datetime, timedelta

# Internal modules
from data.data_generator import (
    generate_dataset, generate_loan_applications,
    generate_stream_record, CAR_CATALOGUE, CITIES, FUEL_TYPES
)
from database.db_manager import DatabaseManager
from ml.ml_engine import C24PriceEngine, DemandForecaster, price_engine, demand_forecaster
from rag.rag_engine import car_assistant

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "c24iq-secret-2025"

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "c24iq.db")
db = DatabaseManager(DB_PATH)

# ─── Bootstrap (run once on startup) ─────────────────────────────────────────
_bootstrapped = False
_stream_counter = 10000
_stream_lock = threading.Lock()

def bootstrap():
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True
    print("⚙  Bootstrapping C24-IQ ...")
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)

    db.initialize()

    # Check if data already loaded
    try:
        existing = db.query("SELECT COUNT(*) as n FROM car_listings").iloc[0]["n"]
    except Exception:
        existing = 0

    if existing < 100:
        print("  Generating dataset (5000 records)...")
        df = generate_dataset(5000)
        db.bulk_insert_listings(df)
        loans_df = generate_loan_applications(df, 1000)
        db.bulk_insert_loans(loans_df)
        db.compute_market_stats()
        print(f"  Dataset loaded: {len(df)} records")
    else:
        df = db.query("SELECT * FROM car_listings")
        print(f"  Dataset exists: {existing} records")

    # Train ML engine
    model_path = os.path.join(os.path.dirname(__file__), "models", "price_engine.pkl")
    if not os.path.exists(model_path) or existing < 100:
        print("  Training ML models...")
        if existing >= 100:
            df = db.query("SELECT * FROM car_listings")
        metrics = price_engine.train(df)
        print(f"  Ensemble R²={metrics['ensemble']['r2']} MAE=₹{metrics['ensemble']['mae']:,.0f}")
    else:
        price_engine._load()
        print("  ML models loaded from cache")

    print("✅ Bootstrap complete")

bootstrap()

# ─── Helper ──────────────────────────────────────────────────────────────────

def fmt_inr(n):
    if n >= 1e7: return f"₹{n/1e7:.2f} Cr"
    if n >= 1e5: return f"₹{n/1e5:.2f} L"
    return f"₹{n:,.0f}"


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    kpis = db.get_kpi_summary()
    return render_template("index.html", kpis=kpis,
                           fmt_inr=fmt_inr,
                           brands=list(CAR_CATALOGUE.keys()),
                           cities=list(CITIES.keys()))


@app.route("/dashboard")
def dashboard():
    kpis = db.get_kpi_summary()
    return render_template("dashboard.html", kpis=kpis, fmt_inr=fmt_inr)


@app.route("/predict")
def predict_page():
    brands = list(CAR_CATALOGUE.keys())
    cities = list(CITIES.keys())
    segments = ["Hatchback", "Sedan", "SUV", "MPV", "Electric"]
    return render_template("predict.html",
                           brands=brands, cities=cities,
                           segments=segments, fuel_types=FUEL_TYPES)


@app.route("/market")
def market_page():
    return render_template("market.html")


@app.route("/assistant")
def assistant_page():
    return render_template("assistant.html")


@app.route("/analytics")
def analytics_page():
    return render_template("analytics.html")


@app.route("/listings")
def listings_page():
    brands = list(CAR_CATALOGUE.keys())
    cities = list(CITIES.keys())
    return render_template("listings.html", brands=brands, cities=cities)


# ─── API: Dashboard KPIs ──────────────────────────────────────────────────────

@app.route("/api/kpis")
def api_kpis():
    kpis = db.get_kpi_summary()
    return jsonify({
        "total_listings": int(kpis.get("total", 0)),
        "avg_price": round(float(kpis.get("avg_price", 0))),
        "avg_days": round(float(kpis.get("avg_days", 0)), 1),
        "total_predictions": int(kpis.get("total_preds", 0)),
        "total_rag_queries": int(kpis.get("total_queries", 0)),
        "loan_approval_rate": round(float(kpis.get("approval_rate", 0)), 1),
    })


@app.route("/api/brand_distribution")
def api_brand_distribution():
    df = db.get_brand_distribution()
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/city_heatmap")
def api_city_heatmap():
    df = db.get_city_heatmap()
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/price_trend")
def api_price_trend():
    df = db.get_price_trend()
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/segment_analysis")
def api_segment_analysis():
    df = db.get_segment_analysis()
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/model_metrics")
def api_model_metrics():
    if not price_engine.is_trained:
        return jsonify({"error": "Model not trained"}), 503
    return jsonify({
        "metrics": price_engine.metrics,
        "feature_importance": price_engine.feature_importance,
        "ensemble_weights": price_engine.ensemble_weights,
    })


@app.route("/api/demand_forecast")
def api_demand_forecast():
    df = db.query("SELECT * FROM car_listings")
    forecast = demand_forecaster.forecast(df, periods=30)
    if forecast.empty:
        return jsonify([])
    forecast["date"] = forecast["date"].astype(str)
    return jsonify(forecast.to_dict(orient="records"))


# ─── API: Price Prediction ────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input"}), 400

    try:
        input_dict = {
            "brand": data.get("brand", "Maruti Suzuki"),
            "model": data.get("model", "Swift"),
            "segment": data.get("segment", "Hatchback"),
            "fuel_type": data.get("fuel_type", "Petrol"),
            "transmission": data.get("transmission", "Manual"),
            "city": data.get("city", "Delhi NCR"),
            "color": data.get("color", "White"),
            "region": CITIES.get(data.get("city", "Delhi NCR"), {}).get("region", "North"),
            "condition_grade": data.get("condition_grade", "Good"),
            "age_years": int(data.get("age_years", 3)),
            "manufacture_year": 2025 - int(data.get("age_years", 3)),
            "km_driven": int(data.get("km_driven", 36000)),
            "owner_number": int(data.get("owner_number", 1)),
            "accident_history": int(data.get("accident_history", 0)),
            "insurance_valid": bool(data.get("insurance_valid", True)),
            "service_records": bool(data.get("service_records", True)),
            "popularity_score": float(data.get("popularity_score", 80)),
            "demand_index": float(CITIES.get(data.get("city","Delhi NCR"),{}).get("demand_multiplier",1.0) * 100),
            "created_at": datetime.now().isoformat(),
        }

        result = price_engine.predict(input_dict)
        explanation = price_engine.get_shap_like_explanation(input_dict)

        # Store prediction
        db.insert_prediction(
            car_id=None,
            predicted=result["ensemble_price"],
            lower=result["ci_lower"],
            upper=result["ci_upper"],
            model_name="ensemble",
            features=input_dict,
        )

        return jsonify({
            **result,
            "explanation": explanation,
            "input": input_dict,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── API: RAG Assistant ───────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    query = data.get("message", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Inject live market context
    kpis = db.get_kpi_summary()
    brand_df = db.get_brand_distribution()
    city_df = db.get_city_heatmap()
    
    market_ctx = {
        "total": int(kpis.get("total", 0)),
        "avg_price": float(kpis.get("avg_price", 0)),
        "avg_days": float(kpis.get("avg_days", 0)),
        "top_brand": brand_df.iloc[0]["brand"] if len(brand_df) > 0 else "N/A",
        "top_city": city_df.iloc[0]["city"] if len(city_df) > 0 else "N/A",
    }

    result = car_assistant.query(query, market_context=market_ctx)
    
    db.insert_rag_query(
        query_text=query,
        response_text=result["answer"],
        sources=result["sources"],
        response_ms=result["response_time_ms"],
    )

    return jsonify(result)


@app.route("/api/chat/reset", methods=["POST"])
def api_chat_reset():
    car_assistant.reset()
    return jsonify({"status": "reset"})


# ─── API: Listings Search ─────────────────────────────────────────────────────

@app.route("/api/listings")
def api_listings():
    args = request.args
    df = db.search_listings(
        brand=args.get("brand") or None,
        city=args.get("city") or None,
        segment=args.get("segment") or None,
        fuel=args.get("fuel") or None,
        min_price=int(args.get("min_price", 0)) or None,
        max_price=int(args.get("max_price", 0)) or None,
        max_km=int(args.get("max_km", 0)) or None,
        limit=int(args.get("limit", 50)),
    )
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/listings/stats")
def api_listings_stats():
    df = db.query("""
        SELECT
            COUNT(*) as total,
            AVG(listing_price) as avg_price,
            MIN(listing_price) as min_price,
            MAX(listing_price) as max_price,
            AVG(km_driven) as avg_km,
            AVG(age_years) as avg_age,
            COUNT(DISTINCT brand) as brands,
            COUNT(DISTINCT city) as cities
        FROM car_listings
    """)
    return jsonify(df.to_dict(orient="records")[0])


# ─── API: Anomaly Detection ───────────────────────────────────────────────────

@app.route("/api/anomalies")
def api_anomalies():
    df = db.query("SELECT * FROM car_listings LIMIT 500")
    if not price_engine.is_trained:
        return jsonify([])
    result = price_engine.detect_anomalies(df)
    anomalies = result[result["is_anomaly"] == True][
        ["id", "brand", "model", "city", "listing_price",
         "km_driven", "age_years", "anomaly_score"]
    ].head(30)
    return jsonify(anomalies.to_dict(orient="records"))


# ─── API: Real-Time Stream ────────────────────────────────────────────────────

@app.route("/api/stream/ingest", methods=["POST"])
def api_stream_ingest():
    """Simulate real-time data ingestion."""
    global _stream_counter
    batch_size = int(request.get_json(force=True).get("batch_size", 1))
    ingested = []

    with _stream_lock:
        for _ in range(batch_size):
            _stream_counter += 1
            record = generate_stream_record(_stream_counter)
            db.insert_stream_event("new_listing", record)
            ingested.append(record)

    return jsonify({"ingested": len(ingested), "records": ingested})


@app.route("/api/stream/events")
def api_stream_events():
    df = db.get_recent_stream_events(limit=20)
    records = []
    for _, row in df.iterrows():
        try:
            payload = json.loads(row["payload_json"])
        except Exception:
            payload = {}
        records.append({
            "id": row["id"],
            "event_type": row["event_type"],
            "created_at": row["created_at"],
            "brand": payload.get("brand", ""),
            "model": payload.get("model", ""),
            "city": payload.get("city", ""),
            "price": payload.get("listing_price", 0),
            "km": payload.get("km_driven", 0),
        })
    return jsonify(records)


@app.route("/api/stream/sse")
def api_stream_sse():
    """Server-Sent Events endpoint for real-time feed."""
    def generate():
        for _ in range(50):
            global _stream_counter
            with _stream_lock:
                _stream_counter += 1
                record = generate_stream_record(_stream_counter)
            data = json.dumps({
                "brand": record["brand"],
                "model": record["model"],
                "city": record["city"],
                "price": record["listing_price"],
                "km": record["km_driven"],
                "age": record["age_years"],
                "fuel": record["fuel_type"],
                "ts": datetime.now().strftime("%H:%M:%S"),
            })
            yield f"data: {data}\n\n"
            time.sleep(1.5)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ─── API: Finance ─────────────────────────────────────────────────────────────

@app.route("/api/emi_calculate", methods=["POST"])
def api_emi():
    data = request.get_json()
    principal = float(data.get("principal", 500000))
    rate = float(data.get("rate", 12)) / 1200  # monthly
    tenure = int(data.get("tenure", 36))

    if rate == 0:
        emi = principal / tenure
    else:
        emi = principal * rate * (1 + rate)**tenure / ((1 + rate)**tenure - 1)

    total_payment = emi * tenure
    total_interest = total_payment - principal

    schedule = []
    balance = principal
    for month in range(1, tenure + 1):
        interest_comp = balance * rate
        principal_comp = emi - interest_comp
        balance -= principal_comp
        schedule.append({
            "month": month,
            "emi": round(emi, 2),
            "principal": round(principal_comp, 2),
            "interest": round(interest_comp, 2),
            "balance": round(max(0, balance), 2),
        })

    return jsonify({
        "emi": round(emi, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2),
        "schedule": schedule[:12],  # first 12 months
    })


# ─── API: Model Info ──────────────────────────────────────────────────────────

@app.route("/api/models/info")
def api_models_info():
    models_meta = [
        {"name": "Random Forest", "key": "random_forest", "type": "Ensemble",
         "description": "200 decision trees with max_depth=12. Best for non-linear relationships."},
        {"name": "Gradient Boosting", "key": "gradient_boost", "type": "Boosting",
         "description": "Sequential trees with learning_rate=0.08. Excellent for structured data."},
        {"name": "XGBoost", "key": "xgboost", "type": "Boosting",
         "description": "300 estimators with column sampling. Industry standard for tabular ML."},
        {"name": "Ridge Regression", "key": "ridge", "type": "Linear",
         "description": "L2 regularized linear model. Fast baseline with good generalization."},
        {"name": "ElasticNet", "key": "elasticnet", "type": "Linear",
         "description": "L1+L2 regularization. Feature selection + regularization combined."},
        {"name": "Ensemble (Stacked)", "key": "ensemble", "type": "Meta-Learner",
         "description": "Inverse-MAE weighted average of all models. Best overall performance."},
    ]
    metrics = price_engine.metrics if price_engine.is_trained else {}
    for m in models_meta:
        if m["key"] in metrics:
            m["metrics"] = metrics[m["key"]]
    return jsonify(models_meta)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
