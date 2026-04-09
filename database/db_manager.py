"""
C24-IQ: Database Manager
SQLite implementation with PostgreSQL-compatible schema.
For production: swap engine to psycopg2/SQLAlchemy PostgreSQL.
"""

import sqlite3
import pandas as pd
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "c24iq.db")

SCHEMA = """
-- ══════════════════════════════════════════════════
--  C24-IQ PostgreSQL-Compatible Schema
-- ══════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS car_listings (
    id               INTEGER PRIMARY KEY,
    brand            TEXT NOT NULL,
    model            TEXT NOT NULL,
    variant          TEXT,
    manufacture_year INTEGER,
    age_years        INTEGER,
    km_driven        INTEGER,
    fuel_type        TEXT,
    transmission     TEXT,
    owner_number     INTEGER,
    owner_type       TEXT,
    city             TEXT,
    region           TEXT,
    color            TEXT,
    segment          TEXT,
    condition_grade  TEXT,
    insurance_valid  INTEGER,
    accident_history INTEGER,
    service_records  INTEGER,
    listing_price    INTEGER,
    estimated_market_price INTEGER,
    days_on_market   INTEGER,
    views            INTEGER,
    inquiries        INTEGER,
    popularity_score REAL,
    demand_index     REAL,
    created_at       TEXT
);

CREATE TABLE IF NOT EXISTS price_predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    car_listing_id   INTEGER,
    predicted_price  REAL,
    confidence_lower REAL,
    confidence_upper REAL,
    model_name       TEXT,
    model_version    TEXT,
    features_json    TEXT,
    created_at       TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (car_listing_id) REFERENCES car_listings(id)
);

CREATE TABLE IF NOT EXISTS market_stats (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    stat_date        TEXT,
    city             TEXT,
    segment          TEXT,
    avg_price        REAL,
    median_price     REAL,
    volume           INTEGER,
    avg_days         REAL,
    demand_index     REAL,
    created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS loan_applications (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    car_id           INTEGER,
    loan_amount      REAL,
    tenure_months    INTEGER,
    interest_rate    REAL,
    monthly_emi      REAL,
    credit_score     INTEGER,
    approved         INTEGER,
    city             TEXT,
    segment          TEXT,
    created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rag_queries (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text       TEXT,
    response_text    TEXT,
    sources_json     TEXT,
    response_time_ms INTEGER,
    created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stream_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type       TEXT,
    payload_json     TEXT,
    processed        INTEGER DEFAULT 0,
    created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_listings_brand   ON car_listings(brand);
CREATE INDEX IF NOT EXISTS idx_listings_city    ON car_listings(city);
CREATE INDEX IF NOT EXISTS idx_listings_segment ON car_listings(segment);
CREATE INDEX IF NOT EXISTS idx_listings_price   ON car_listings(listing_price);
CREATE INDEX IF NOT EXISTS idx_market_date_city ON market_stats(stat_date, city);
"""


class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self):
        """Create schema."""
        conn = self.get_connection()
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()

    def bulk_insert_listings(self, df: pd.DataFrame):
        """Load car listings from DataFrame."""
        conn = self.get_connection()
        df_clean = df.copy()
        df_clean["insurance_valid"] = df_clean["insurance_valid"].astype(int)
        df_clean["service_records"] = df_clean["service_records"].astype(int)
        df_clean.to_sql("car_listings", conn, if_exists="replace", index=False)
        conn.close()

    def bulk_insert_loans(self, df: pd.DataFrame):
        conn = self.get_connection()
        df.to_sql("loan_applications", conn, if_exists="replace", index=False)
        conn.close()

    def compute_market_stats(self):
        """Aggregate market statistics."""
        conn = self.get_connection()
        query = """
            INSERT OR REPLACE INTO market_stats
                (stat_date, city, segment, avg_price, median_price, volume, avg_days, demand_index)
            SELECT
                DATE(created_at) as stat_date,
                city,
                segment,
                AVG(listing_price)    as avg_price,
                AVG(listing_price)    as median_price,
                COUNT(*)              as volume,
                AVG(days_on_market)   as avg_days,
                AVG(demand_index)     as demand_index
            FROM car_listings
            GROUP BY DATE(created_at), city, segment
        """
        conn.execute(query)
        conn.commit()
        conn.close()

    def query(self, sql: str, params=()) -> pd.DataFrame:
        conn = self.get_connection()
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df

    def insert_prediction(self, car_id, predicted, lower, upper, model_name, features):
        conn = self.get_connection()
        conn.execute("""
            INSERT INTO price_predictions
                (car_listing_id, predicted_price, confidence_lower, confidence_upper,
                 model_name, model_version, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (car_id, predicted, lower, upper, model_name, "1.0.0", json.dumps(features)))
        conn.commit()
        conn.close()

    def insert_rag_query(self, query_text, response_text, sources, response_ms):
        conn = self.get_connection()
        conn.execute("""
            INSERT INTO rag_queries (query_text, response_text, sources_json, response_time_ms)
            VALUES (?, ?, ?, ?)
        """, (query_text, response_text, json.dumps(sources), response_ms))
        conn.commit()
        conn.close()

    def insert_stream_event(self, event_type, payload):
        conn = self.get_connection()
        conn.execute("""
            INSERT INTO stream_events (event_type, payload_json)
            VALUES (?, ?)
        """, (event_type, json.dumps(payload)))
        conn.commit()
        conn.close()

    def get_recent_stream_events(self, limit=20):
        return self.query(
            "SELECT * FROM stream_events ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )

    def get_kpi_summary(self) -> dict:
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as total, AVG(listing_price) as avg_price, "
                    "AVG(days_on_market) as avg_days FROM car_listings")
        row = dict(cur.fetchone())
        cur.execute("SELECT COUNT(*) as total_preds FROM price_predictions")
        preds = dict(cur.fetchone())
        cur.execute("SELECT COUNT(*) as total_queries FROM rag_queries")
        rag = dict(cur.fetchone())
        cur.execute("SELECT AVG(CAST(approved as REAL))*100 as approval_rate FROM loan_applications")
        loans = dict(cur.fetchone())
        conn.close()
        return {**row, **preds, **rag, **loans}

    def get_brand_distribution(self):
        return self.query("""
            SELECT brand, COUNT(*) as count, AVG(listing_price) as avg_price,
                   AVG(days_on_market) as avg_days
            FROM car_listings
            GROUP BY brand ORDER BY count DESC
        """)

    def get_city_heatmap(self):
        return self.query("""
            SELECT city, region,
                   COUNT(*) as listings,
                   AVG(listing_price) as avg_price,
                   AVG(demand_index) as demand_index
            FROM car_listings
            GROUP BY city ORDER BY listings DESC
        """)

    def get_price_trend(self):
        return self.query("""
            SELECT strftime('%Y-%m', created_at) as month,
                   segment,
                   AVG(listing_price) as avg_price,
                   COUNT(*) as volume
            FROM car_listings
            GROUP BY month, segment
            ORDER BY month
        """)

    def get_segment_analysis(self):
        return self.query("""
            SELECT segment, fuel_type,
                   COUNT(*) as count,
                   AVG(listing_price) as avg_price,
                   AVG(age_years) as avg_age,
                   AVG(km_driven) as avg_km,
                   AVG(days_on_market) as avg_days
            FROM car_listings
            GROUP BY segment, fuel_type
            ORDER BY count DESC
        """)

    def search_listings(self, brand=None, city=None, segment=None,
                        min_price=None, max_price=None, fuel=None,
                        max_km=None, limit=50):
        clauses = ["1=1"]
        params = []
        if brand:  clauses.append("brand=?");   params.append(brand)
        if city:   clauses.append("city=?");    params.append(city)
        if segment: clauses.append("segment=?"); params.append(segment)
        if fuel:   clauses.append("fuel_type=?"); params.append(fuel)
        if min_price: clauses.append("listing_price>=?"); params.append(min_price)
        if max_price: clauses.append("listing_price<=?"); params.append(max_price)
        if max_km: clauses.append("km_driven<=?"); params.append(max_km)
        where = " AND ".join(clauses)
        return self.query(f"SELECT * FROM car_listings WHERE {where} LIMIT ?",
                          params + [limit])


db = DatabaseManager()
