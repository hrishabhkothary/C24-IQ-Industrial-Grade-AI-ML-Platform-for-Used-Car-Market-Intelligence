# C24-IQ: Cars24 Intelligence Quotient Platform
### Industrial-Grade AI/ML Platform for Used Car Market Intelligence

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Industry Value & Use Cases](#3-industry-value--use-cases)
4. [Architecture](#4-architecture)
5. [Technology Stack — Significance of Each](#5-technology-stack--significance-of-each)
6. [Feature Modules](#6-feature-modules)
7. [Machine Learning Pipeline](#7-machine-learning-pipeline)
8. [RAG / GenAI Pipeline](#8-rag--genai-pipeline)
9. [Database Schema (PostgreSQL-Compatible)](#9-database-schema-postgresql-compatible)
10. [How to Run — Step-by-Step](#10-how-to-run--step-by-step)
11. [Batch Demo Guide](#11-batch-demo-guide)
12. [Real-Time Data Ingestion Demo](#12-real-time-data-ingestion-demo)
13. [API Reference](#13-api-reference)
14. [Project Structure](#14-project-structure)

---

## 1. Project Overview

**C24-IQ** (Cars24 Intelligence Quotient) is an industrial-grade, full-stack AI/ML platform built specifically for the **Cars24** used car marketplace ecosystem. It transforms raw car listing data into actionable market intelligence using a sophisticated stack of classical machine learning, ensemble methods, and Generative AI with Retrieval-Augmented Generation (RAG).

The platform operates as a **dual-mode system**:
- **Batch Mode**: Processes historical datasets for trend analysis, model training, and strategic reporting
- **Real-Time Mode**: Ingests live car listing events via Server-Sent Events (SSE) for instant pricing, anomaly alerts, and demand signals

### Key Capabilities at a Glance

| Capability | Technology | Output |
|---|---|---|
| Price Prediction | XGBoost + RF + GBM Ensemble | Instant fair market value with CI |
| Anomaly Detection | Isolation Forest | Suspicious listing flags |
| Demand Forecasting | Rolling Time Series | 30-day volume & price outlook |
| Market Analytics | Pandas + Plotly + Chart.js | Interactive dashboards |
| AI Q&A | Claude LLM + RAG | Contextual car market advice |
| Loan Analysis | NumPy finance | EMI schedules, approval rates |
| Real-Time Feed | Flask SSE | Live listing stream |
| Data Storage | PostgreSQL / SQLite | Structured, indexed persistence |

---

## 2. Problem Statement

### The Used Car Market Challenge

India's used car market transacts **4.4 million vehicles annually** worth ~₹1.2 lakh crore. Yet it is plagued by:

#### 2.1 Information Asymmetry
Sellers don't know their car's true market value. Buyers can't determine if a listed price is fair. Dealers exploit this gap by 15–25%. **C24-IQ eliminates this gap** with ML-powered fair market pricing.

#### 2.2 Pricing Inconsistency
The same car listed in Delhi NCR vs Jaipur differs by 15–20%, but no platform contextualises this for buyers/sellers intelligently. **C24-IQ's city-aware model captures geographic demand signals.**

#### 2.3 Fraud & Anomalous Listings
~8% of listings have odometer tampering, undisclosed accidents, or fraudulent pricing. Manual review is impossible at scale. **C24-IQ's Isolation Forest flags these automatically.**

#### 2.4 Slow Market Intelligence
Merchandising teams lack real-time insight into demand shifts. By the time monthly reports arrive, the market has moved. **C24-IQ's SSE stream and 30-day forecast solve this.**

#### 2.5 Scalability of Expert Knowledge
Cars24 has domain experts, but they can't answer every buyer/seller query. **C24-IQ's RAG assistant scales domain expertise 24/7** without hallucination, using a curated knowledge base.

---

## 3. Industry Value & Use Cases

### 3.1 For Cars24 Operations Teams
- **Procurement Intelligence**: When buying cars from sellers, instantly know if the offered price is above/below market
- **Inventory Optimisation**: Forecast which segments will sell fastest in which cities next month
- **Risk Flagging**: Auto-detect overpriced or suspiciously cheap listings before publishing

### 3.2 For Cars24 Product Teams
- **Instant Valuation Widget**: Embed the price prediction API in the seller flow
- **Search Personalisation**: Use demand_index signals to rank listings in search results
- **Finance Pre-approval**: Use credit score + car value to pre-approve loans in 30s

### 3.3 For Business Analysts
- **Market Trend Reports**: Automated brand × city × segment dashboards
- **Competitive Intelligence**: Monitor segment-wise price movements over time
- **KPI Monitoring**: Real-time listing volume, conversion (inquiries/views), DOT

### 3.4 For Data Science Teams
- **Model Monitoring**: Compare R², MAE, MAPE across 5 ML models simultaneously
- **Feature Research**: Understand which car attributes drive price most (SHAP-like)
- **A/B Testing Foundation**: Ensemble architecture supports model swapping

### 3.5 For End Customers
- **Price Confidence**: Know the fair price before negotiating — seller and buyer
- **Loan Planning**: Calculate exact EMI, interest, and repayment schedule
- **AI Guidance**: Ask natural language questions about buying/selling decisions

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        C24-IQ Platform                          │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────┐ │
│  │   Data Layer  │   │  ML Engine   │   │    RAG / GenAI      │ │
│  │              │   │              │   │                     │ │
│  │ data_        │   │ Random Forest│   │ Knowledge Base      │ │
│  │ generator.py │──▶│ Gradient     │   │ (9 categories)      │ │
│  │              │   │ Boosting     │   │         │           │ │
│  │ PostgreSQL/  │   │ XGBoost      │   │ Keyword Retriever   │ │
│  │ SQLite DB    │◀──│ Ridge        │   │         │           │ │
│  │              │   │ ElasticNet   │   │ Claude API (LLM)    │ │
│  └──────────────┘   │              │   │         │           │ │
│         │           │ Isolation    │   │ Response + Sources  │ │
│         │           │ Forest       │   └─────────────────────┘ │
│         │           └──────────────┘           │               │
│         │                  │                   │               │
│         ▼                  ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Flask REST API                         │   │
│  │  /api/predict  /api/chat  /api/listings  /api/stream/*  │   │
│  │  /api/kpis  /api/anomalies  /api/demand_forecast        │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │                  Jinja2 + Chart.js UI                    │   │
│  │  Overview · Dashboard · Predictor · Assistant · Listings │   │
│  │  Market Intel · Analytics (EMI, Depreciation, Features) │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Real-Time Ingestion Layer (SSE)             │   │
│  │  Client ◀── EventSource ◀── /api/stream/sse ◀── Thread  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Technology Stack — Significance of Each

### 5.1 Python 3.11+
**Role**: Core language for all data science, ML, API, and orchestration.
**Why**: Unmatched ecosystem for ML (sklearn, pandas, numpy). Fast enough for API serving when combined with Flask + threading. Type-hinting improves code maintainability.

### 5.2 Pandas + NumPy
**Role**: Data ingestion, cleaning, feature engineering, aggregation.
**Significance**:
- `pandas.DataFrame` is used for all ETL from SQLite into model training
- `numpy` powers depreciation curve math, confidence interval computation, and random sampling
- Together they handle the full 5,000-record dataset in memory in < 200ms
- Key operations: `groupby`, `rolling`, `pivot_table`, `apply`, `pd.to_datetime`

### 5.3 Scikit-learn
**Role**: ML pipelines, preprocessing, baseline models, evaluation.
**Significance**:
- `ColumnTransformer` + `OneHotEncoder` + `RobustScaler` build the preprocessing pipeline
- `RandomForestRegressor` (200 trees) serves as the primary ensemble member
- `GradientBoostingRegressor` captures sequential error correction patterns
- `IsolationForest` powers the anomaly detection module (5% contamination)
- `train_test_split`, `cross_val_score`, `KFold` ensure robust evaluation
- `mean_absolute_error`, `r2_score`, `mean_squared_error` standardise metrics

### 5.4 XGBoost
**Role**: Primary price prediction engine and best single-model performer.
**Significance**:
- Industry standard for structured/tabular data — wins most Kaggle competitions in this domain
- Uses `colsample_bytree=0.8` and `subsample=0.8` to prevent overfitting on our synthetic data
- 300 estimators with `learning_rate=0.07` balances bias-variance tradeoff
- Typically achieves R² > 0.93 on the car pricing task

### 5.5 LightGBM (available)
**Role**: Alternative to XGBoost for faster training on larger datasets.
**Significance**: Leaf-wise tree growth vs level-wise (XGBoost), making it 3–5× faster on datasets > 100K records — essential when Cars24's production dataset (millions of records) is connected.

### 5.6 Flask
**Role**: REST API server + Jinja2 HTML rendering + SSE streaming.
**Significance**:
- Serves 15+ API endpoints covering predictions, analytics, RAG, stream ingestion
- `Response(generate(), mimetype='text/event-stream')` enables real-time SSE without WebSocket overhead
- `threading=True` allows concurrent requests during live demo
- Lightweight and fast — no Django overhead for a data platform

### 5.7 SQLite (Production: PostgreSQL)
**Role**: Persistent structured storage for all entities.
**Significance**:
- Schema is **100% PostgreSQL-compatible** — swap `sqlite3` for `psycopg2` + SQLAlchemy to deploy to production
- Indexed on `brand`, `city`, `segment`, `listing_price` for fast analytics queries
- `car_listings`, `price_predictions`, `market_stats`, `loan_applications`, `rag_queries`, `stream_events` tables cover the full platform lifecycle
- `market_stats` pre-aggregation reduces dashboard query time from O(n) to O(1)

### 5.8 PostgreSQL (Production Target)
**Role**: Replace SQLite for production-grade ACID compliance, concurrent writes, and advanced queries.
**Significance**:
- Supports `JSONB` for `features_json` and `payload_json` columns (native JSON querying)
- `pg_partman` can partition `car_listings` by `created_at` month for multi-million-row performance
- `EXPLAIN ANALYZE` on indexed columns gives sub-5ms query latency on 10M rows
- Connection pooling via `pgBouncer` supports 1000+ concurrent API users

### 5.9 Claude API (Anthropic) — RAG/GenAI
**Role**: Large Language Model powering the intelligent assistant.
**Significance**:
- `claude-sonnet-4-20250514` provides best-in-class reasoning for financial/automotive domain Q&A
- RAG architecture prevents hallucination: the model is grounded in our 9-category knowledge base
- Multi-turn conversation history (last 6 turns) maintained server-side for context continuity
- Live market context injected per query: total listings, avg price, top brand/city from live DB
- All queries and responses logged to `rag_queries` table for audit and analytics

### 5.10 Retrieval-Augmented Generation (RAG)
**Role**: Grounds LLM responses in verified domain knowledge.
**Significance**:
- Prevents hallucination on specific Indian market facts (interest rates, depreciation %, FAME-II subsidies)
- Knowledge base covers 9 categories: Depreciation, Pricing, Market Trends, Buying, Selling, Finance, EV, Cars24, Valuation
- Keyword-overlap retrieval (TF-IDF-style) retrieves top-4 most relevant chunks per query
- Production upgrade: swap to `sentence-transformers` + `FAISS` vector store for semantic retrieval

### 5.11 Chart.js 4.x
**Role**: All interactive browser-side visualisations.
**Significance**:
- Zero-server rendering: charts compute client-side, reducing server load
- Supports bar, line, scatter, doughnut, stacked charts — all used in the dashboard
- Animated transitions (duration 800ms) give professional feel
- Responsive by default — works on tablet/mobile

### 5.12 Plotly (available for notebooks)
**Role**: Advanced statistical visualisations for the data science team.
**Significance**:
- Box plots, violin plots, heatmaps, 3D scatter for EDA
- `plotly.express` enables one-liner visualisations during model development
- Output as standalone HTML for sharing in reports

### 5.13 SciPy + StatsModels
**Role**: Statistical testing and time series components.
**Significance**:
- `scipy.stats` for confidence interval computation in price predictions
- `statsmodels.tsa` available for ARIMA-based demand forecasting (upgrade from rolling average)
- Correlation analysis for feature selection validation

### 5.14 Server-Sent Events (SSE)
**Role**: Real-time unidirectional data streaming from server to browser.
**Significance**:
- Lighter than WebSockets for one-way push (car listings flowing in)
- Browser's `EventSource` API handles reconnection automatically
- `/api/stream/sse` yields new car records every 1.5s, simulating a Kafka consumer in production
- No external message broker needed for demo; production would use Kafka → Flask consumer → SSE

---

## 6. Feature Modules

### Module 1: AI Price Predictor
- **Input**: 15+ car parameters (brand, age, KM, city, fuel, condition, etc.)
- **Output**: Ensemble price, confidence interval (95%), per-model breakdown, SHAP-like explanation, anomaly flag, depreciation stage
- **Models**: Random Forest (200 trees) + Gradient Boosting + XGBoost (300 est.) + Ridge + ElasticNet
- **Ensemble**: Inverse-MAE weighted stacking

### Module 2: Market Dashboard
- Brand distribution bar chart (listings + avg price)
- City demand heatmap (demand index, inventory levels)
- Price trend line chart by segment (12 months)
- Fuel type doughnut chart
- ML model performance comparison bars
- 30-day demand forecast (historical + projection)

### Module 3: GenAI RAG Assistant
- Multi-turn conversation with Claude
- 9-category knowledge base with keyword retrieval
- Live market context injection per query
- Source attribution (which KB chunks were used)
- Session statistics (msg count, avg response time, chunks used)
- Reset conversation support

### Module 4: Listings Explorer
- Full-text search + 6 filters (brand, city, segment, fuel, price range, KM)
- Results table with colour-coded fuel tags and condition grades
- Real-time stats bar (count, avg/min/max price)
- Anomaly-flagged rows

### Module 5: Market Intelligence
- Isolation Forest anomaly listing table (top 30)
- Price vs KM scatter by segment
- Segment × Fuel analysis matrix table
- Price distribution histogram
- Top segment performance bar chart

### Module 6: Advanced Analytics
- EMI Calculator with stacked principal+interest bar chart
- Feature Importance ranking (from Random Forest)
- Depreciation curves (5 segments × 10 years)
- ML Model Comparison table (R², MAE, RMSE, MAPE, ensemble weight)

### Module 7: Real-Time Stream
- SSE-based live listing feed (1.5s interval)
- Event counter + stream status indicator
- Last 15 events displayed in reverse-chronological feed
- Stream events stored in `stream_events` DB table

---

## 7. Machine Learning Pipeline

```
Raw Data (5000 records)
        │
        ▼
Feature Engineering (engineer_features)
  ├── km_per_year = km_driven / age_years
  ├── age_km_ratio = age * km / 1e6
  ├── luxury_segment = segment in [SUV, MPV]
  ├── is_automatic = transmission in [Auto, CVT, DCT]
  ├── is_ev_hybrid = fuel in [Electric, Hybrid]
  ├── clean_car = no_accident AND service_records
  ├── desirable_color = White/Silver/Black
  ├── owner_penalty = (owner_number-1) * 0.04
  ├── demand_x_popularity = demand_index * popularity_score
  ├── listing_month, listing_quarter (seasonality)
  └── is_festive_season = month in [9, 10, 11]
        │
        ▼
Preprocessing (ColumnTransformer)
  ├── Categorical → OneHotEncoder (handle_unknown='ignore')
  └── Numeric → RobustScaler (outlier-robust)
        │
        ▼
Train/Test Split (80/20, random_state=42)
        │
        ▼
Model Training (parallel)
  ├── RandomForestRegressor(n_estimators=200, max_depth=12)
  ├── GradientBoostingRegressor(n_estimators=200, lr=0.08)
  ├── XGBRegressor(n_estimators=300, lr=0.07)
  ├── Ridge(alpha=100)
  └── ElasticNet(alpha=50, l1_ratio=0.5)
        │
        ▼
Evaluation (per model)
  ├── MAE, RMSE, R², MAPE
  └── Test-set predictions stored
        │
        ▼
Stacking Ensemble
  └── Weight_i = (1/MAE_i) / Σ(1/MAE_j)
        │
        ▼
Anomaly Detector
  └── IsolationForest(contamination=0.05).fit(X_train)
        │
        ▼
Model Persistence (pickle)
  └── models/price_engine.pkl
        │
        ▼
Inference (predict)
  ├── Ensemble price = Σ(weight_i × model_i.predict)
  ├── CI = ensemble ± 1.96 × std(RF tree predictions)
  └── Anomaly = IsolationForest.predict(X)
```

### Model Performance (Typical Results)

| Model | R² | MAE | MAPE |
|---|---|---|---|
| Ensemble | **0.93+** | **~₹22K** | **~7%** |
| XGBoost | 0.91 | ~₹26K | ~8% |
| Random Forest | 0.90 | ~₹28K | ~9% |
| Gradient Boost | 0.89 | ~₹30K | ~9.5% |
| Ridge | 0.78 | ~₹45K | ~14% |
| ElasticNet | 0.76 | ~₹47K | ~15% |

---

## 8. RAG / GenAI Pipeline

```
User Query
    │
    ▼
Keyword Extraction
    │
    ▼
Knowledge Base Retrieval (Top-4 chunks)
    │  9 categories × structured text chunks
    │  Scored by word-overlap (TF-IDF approximation)
    │
    ▼
Live Market Context Injection
    │  total_listings, avg_price, top_brand, top_city from DB
    │
    ▼
System Prompt Construction
    │  KB chunks + market context + persona instructions
    │
    ▼
Claude API Call (claude-sonnet-4-20250514)
    │  messages = [conversation_history[-6:] + current_query]
    │
    ▼
Response + Source Attribution
    │
    ▼
Log to rag_queries table
    │
    ▼
Return to UI (answer + sources + response_time_ms)
```

### Production Upgrade Path
```
Current:  Keyword matching → Claude API
Upgrade:  sentence-transformers (all-MiniLM-L6-v2)
          → FAISS vector index
          → Semantic top-K retrieval
          → Claude API
```

---

## 9. Database Schema (PostgreSQL-Compatible)

```sql
-- Core listing data
car_listings (id, brand, model, variant, manufacture_year, age_years,
              km_driven, fuel_type, transmission, owner_number, owner_type,
              city, region, color, segment, condition_grade, insurance_valid,
              accident_history, service_records, listing_price,
              estimated_market_price, days_on_market, views, inquiries,
              popularity_score, demand_index, created_at)

-- ML prediction audit log
price_predictions (id, car_listing_id, predicted_price, confidence_lower,
                   confidence_upper, model_name, model_version,
                   features_json, created_at)

-- Pre-aggregated market intelligence
market_stats (id, stat_date, city, segment, avg_price, median_price,
              volume, avg_days, demand_index, created_at)

-- Finance data
loan_applications (id, car_id, loan_amount, tenure_months, interest_rate,
                   monthly_emi, credit_score, approved, city, segment, created_at)

-- RAG query audit
rag_queries (id, query_text, response_text, sources_json,
             response_time_ms, created_at)

-- Real-time event log
stream_events (id, event_type, payload_json, processed, created_at)
```

---

## 10. How to Run — Step-by-Step

### Prerequisites
```
Python 3.10+
pip
Git (optional)
4GB RAM recommended
```

### Step 1: Clone / Download
```bash
# If using git:
git clone <repo-url> c24iq
cd c24iq

# Or navigate to the project folder:
cd /path/to/c24iq
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
```
flask, pandas, numpy, scikit-learn, xgboost,
lightgbm, plotly, scipy, statsmodels, faker,
psycopg2-binary, sqlalchemy, flask-cors
```

### Step 4: (Optional) Set Anthropic API Key
For the RAG AI Assistant to work:
```bash
# macOS/Linux:
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows:
set ANTHROPIC_API_KEY=sk-ant-...
```
Without an API key, the assistant returns a fallback KB-based answer.

### Step 5: (Optional) Configure PostgreSQL
By default, the project uses **SQLite** (zero config). To use PostgreSQL:

Edit `database/db_manager.py` — replace the engine:
```python
# SQLite (default):
conn = sqlite3.connect(self.db_path)

# PostgreSQL (production):
from sqlalchemy import create_engine
engine = create_engine("postgresql://user:pass@localhost:5432/c24iq")
```

### Step 6: Run the Application
```bash
# From the c24iq/ directory:
python app.py
```

You should see:
```
⚙  Bootstrapping C24-IQ ...
  Generating dataset (5000 records)...
  Dataset loaded: 5000 records
  Training ML models...
  Ensemble R²=0.93 MAE=₹22,000
✅ Bootstrap complete
 * Running on http://0.0.0.0:5000
```

### Step 7: Open in Browser
```
http://localhost:5000
```

**First startup takes ~60–90 seconds** (data generation + model training).
Subsequent startups take ~5 seconds (loads from cache).

---

## 11. Batch Demo Guide

### Demo Flow (Recommended 20-min walkthrough)

#### Stop 1: Overview Page (`/`)
1. Open `http://localhost:5000`
2. Show KPI strip: **5,000 listings**, avg price, days on market, loan approval rate
3. Point out the **Technology Stack** card — explain each layer
4. Show **ML Model Metrics** mini-card: R²=0.93, MAE, MAPE
5. Explain the system boots with synthetic but realistic Cars24-style data

#### Stop 2: Dashboard (`/dashboard`)
1. **Brand Distribution** chart — Maruti Suzuki dominates (realistic for India)
2. **City Demand Heatmap** — Mumbai/Gurgaon highest; explain demand_multiplier
3. **Price Trend** — switch segment filter: show how SUV prices are higher and more stable
4. **Fuel Type Mix** — 60% Petrol, 30% Diesel (realistic India split)
5. **ML Model Comparison** — show ensemble outperforms all individual models
6. **30-Day Demand Forecast** — historical (teal) + projected (orange dashed)

#### Stop 3: Price Predictor (`/predict`)
1. Set: **Maruti Swift**, Age=3yr, KM=45,000, Delhi NCR, Good condition
2. Click **Predict Price** → show ~₹5.5–6.5L range
3. Change to **Thar** + Mumbai + Excellent → ~₹18–22L
4. Add **2 accidents** → watch price drop
5. Show **Feature Impact** bars — age and KM are biggest drivers
6. Show **Model Breakdown** — all 5 models, ensemble wins

#### Stop 4: Analytics (`/analytics`)
1. **EMI Calculator**: ₹8L car, 20% down, 11.5%, 36 months → show EMI schedule chart
2. **Feature Importance**: explain top features from Random Forest
3. **Depreciation Curves**: show how SUV holds value better than Hatchback
4. **ML Model Table**: R², MAE, MAPE, ensemble weights side by side

#### Stop 5: Market Intelligence (`/market`)
1. **Anomaly Listings**: "These are cars our Isolation Forest flagged as suspiciously priced"
2. **Price vs KM Scatter**: show segment clustering
3. **Segment Matrix**: Diesel SUVs command highest prices; Electric has fastest DOT growth

#### Stop 6: Listings Explorer (`/listings`)
1. Search: Brand=Hyundai, City=Bangalore, Segment=SUV
2. Show results with colour-coded fuel tags, condition grading
3. Filter by price: ₹5L–₹15L

#### Stop 7: AI Assistant (`/assistant`)
1. Ask: *"What factors most affect my Maruti Swift's resale value?"*
2. Ask: *"I want to sell my 4-year-old diesel Creta in Mumbai — what price should I expect?"*
3. Ask: *"Explain the depreciation cliff at 5 years"*
4. Show sources panel — RAG retrieved chunks from KB

---

## 12. Real-Time Data Ingestion Demo

### What it simulates:
In production, Cars24 receives thousands of new car listing submissions per day. This demo simulates that stream using SSE.

### Demo Steps:

#### Via UI (Recommended)
1. Go to `http://localhost:5000` (Overview page)
2. Scroll to **Real-Time Stream** card
3. Click **Start** button
4. Watch new car listings populate in real-time (1 per 1.5 seconds)
5. Each entry shows: Brand, Model, City, Price, KM, Timestamp
6. Event counter increments in real time
7. Click **Stop** to pause the feed

#### Via API (Technical Demo)
```bash
# Ingest 1 record manually:
curl -X POST http://localhost:5000/api/stream/ingest \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 1}'

# Ingest batch of 10:
curl -X POST http://localhost:5000/api/stream/ingest \
  -d '{"batch_size": 10}'

# View recent stream events:
curl http://localhost:5000/api/stream/events

# Connect to SSE directly (raw feed):
curl -N http://localhost:5000/api/stream/sse
```

#### Python Client (Advanced)
```python
import requests, json

# Batch ingest
res = requests.post('http://localhost:5000/api/stream/ingest',
                    json={'batch_size': 5})
print(json.dumps(res.json(), indent=2))

# SSE consumer (simulates Kafka consumer)
import sseclient
response = requests.get('http://localhost:5000/api/stream/sse', stream=True)
client = sseclient.SSEClient(response)
for event in client.events():
    data = json.loads(event.data)
    print(f"{data['brand']} {data['model']} | {data['city']} | ₹{data['price']:,}")
```

### Production Real-Time Architecture
```
New Listing Submitted (Cars24 App)
         │
         ▼
    Kafka Topic: car-listings-raw
         │
         ▼
    Kafka Consumer (Python)
         │
         ├──▶ Validate & Enrich
         ├──▶ ML Price Check (flag if anomaly)
         ├──▶ Insert to PostgreSQL
         └──▶ SSE broadcast to dashboard
```

---

## 13. API Reference

### Price Prediction
```
POST /api/predict
Body: {
  "brand": "Maruti Suzuki",
  "model": "Swift",
  "segment": "Hatchback",
  "fuel_type": "Petrol",
  "transmission": "Manual",
  "city": "Delhi NCR",
  "color": "White",
  "condition_grade": "Good",
  "age_years": 3,
  "km_driven": 36000,
  "owner_number": 1,
  "accident_history": 0,
  "insurance_valid": true,
  "service_records": true
}
Response: {
  "ensemble_price": 580000,
  "ci_lower": 545000,
  "ci_upper": 615000,
  "confidence_pct": 87.3,
  "model_predictions": { "random_forest": 572000, "xgboost": 584000, ... },
  "explanation": [ { "feature": "Vehicle Age", "impact": -0.21, "direction": "negative" }, ... ],
  "is_anomaly_price": false,
  "depreciation_stage": "Moderate Depreciation"
}
```

### AI Chat
```
POST /api/chat
Body: { "message": "What is the best time to sell my car?" }
Response: {
  "answer": "The optimal time to sell...",
  "sources": [ { "id": "kb_005", "title": "How to Get Best Price", "category": "Selling Tips" } ],
  "response_time_ms": 1240
}
```

### Analytics
```
GET /api/kpis                → Platform KPIs
GET /api/brand_distribution  → Brand × listings × avg_price
GET /api/city_heatmap        → City × demand_index × listings
GET /api/price_trend         → Month × segment × avg_price
GET /api/segment_analysis    → Segment × fuel × metrics
GET /api/model_metrics       → R², MAE, MAPE per model
GET /api/demand_forecast     → 30-day volume forecast
GET /api/anomalies           → Isolation Forest flagged listings
```

### Finance
```
POST /api/emi_calculate
Body: { "principal": 700000, "rate": 11.5, "tenure": 36 }
Response: { "emi": 22984, "total_payment": 827424, "total_interest": 127424, "schedule": [...] }
```

### Stream
```
POST /api/stream/ingest      → Inject synthetic records
GET  /api/stream/events      → Last 20 stream events
GET  /api/stream/sse         → SSE feed (EventSource)
```

---

## 14. Project Structure

```
c24iq/
├── app.py                   # Flask application — routes & startup
├── requirements.txt         # Python dependencies
├── README.md                # This documentation
│
├── data/
│   ├── data_generator.py    # Synthetic data generation (Cars24-style)
│   ├── cars_dataset.csv     # Generated dataset (auto-created)
│   └── c24iq.db             # SQLite database (auto-created)
│
├── database/
│   └── db_manager.py        # All DB operations, schema, queries
│
├── ml/
│   └── ml_engine.py         # Price engine, anomaly detector, forecaster
│
├── rag/
│   └── rag_engine.py        # Knowledge base, retrieval, Claude API
│
├── models/
│   └── price_engine.pkl     # Trained model cache (auto-created)
│
└── templates/
    ├── base.html             # Shared layout, sidebar, topbar, styles
    ├── index.html            # Overview page
    ├── dashboard.html        # Analytics dashboard
    ├── predict.html          # AI price predictor
    ├── market.html           # Market intelligence
    ├── analytics.html        # EMI, features, depreciation
    ├── listings.html         # Car listings explorer
    └── assistant.html        # RAG AI chatbot
```

---

## Extending to Production

| Component | Current | Production |
|---|---|---|
| Database | SQLite | PostgreSQL 15 + pgBouncer |
| ML Retrieval | Keyword overlap | FAISS + sentence-transformers |
| Streaming | SSE (simulated) | Apache Kafka + SSE consumer |
| Auth | None | JWT + OAuth2 |
| Deployment | Flask dev server | Gunicorn + Nginx + Docker |
| Monitoring | Console logs | Prometheus + Grafana |
| Model Registry | Pickle file | MLflow + S3 |
| CI/CD | Manual | GitHub Actions + ArgoCD |

---

*C24-IQ © 2025 · Built for Cars24 Intelligence Platform Demo*
*Tech: Python · Pandas · NumPy · Scikit-learn · XGBoost · Flask · SQLite/PostgreSQL · Claude API · Chart.js · SSE*
