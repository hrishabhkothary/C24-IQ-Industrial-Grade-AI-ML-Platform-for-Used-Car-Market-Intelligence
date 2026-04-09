"""
C24-IQ: Synthetic Data Generator
Generates realistic Cars24-style used car market data for India
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json

np.random.seed(42)
random.seed(42)

# ─── Car Catalogue ────────────────────────────────────────────────────────────
CAR_CATALOGUE = {
    "Maruti Suzuki": {
        "Swift":      {"base": 400000, "segment": "Hatchback", "popularity": 0.95},
        "Dzire":      {"base": 550000, "segment": "Sedan",     "popularity": 0.88},
        "Alto":       {"base": 280000, "segment": "Hatchback", "popularity": 0.80},
        "Baleno":     {"base": 620000, "segment": "Hatchback", "popularity": 0.85},
        "Ertiga":     {"base": 820000, "segment": "MPV",       "popularity": 0.78},
        "Vitara Brezza": {"base": 900000, "segment": "SUV",    "popularity": 0.82},
    },
    "Hyundai": {
        "i20":        {"base": 600000, "segment": "Hatchback", "popularity": 0.87},
        "Creta":      {"base": 1200000, "segment": "SUV",      "popularity": 0.93},
        "Verna":      {"base": 900000, "segment": "Sedan",     "popularity": 0.80},
        "Grand i10":  {"base": 450000, "segment": "Hatchback", "popularity": 0.75},
        "Venue":      {"base": 950000, "segment": "SUV",       "popularity": 0.84},
    },
    "Honda": {
        "City":       {"base": 1000000, "segment": "Sedan",    "popularity": 0.85},
        "Amaze":      {"base": 700000,  "segment": "Sedan",    "popularity": 0.70},
        "Jazz":       {"base": 750000,  "segment": "Hatchback","popularity": 0.65},
        "WR-V":       {"base": 950000,  "segment": "SUV",      "popularity": 0.68},
    },
    "Tata": {
        "Nexon":      {"base": 1100000, "segment": "SUV",      "popularity": 0.88},
        "Punch":      {"base": 750000,  "segment": "SUV",      "popularity": 0.82},
        "Altroz":     {"base": 680000,  "segment": "Hatchback","popularity": 0.75},
        "Harrier":    {"base": 1800000, "segment": "SUV",      "popularity": 0.78},
        "Safari":     {"base": 2200000, "segment": "SUV",      "popularity": 0.72},
    },
    "Mahindra": {
        "Thar":       {"base": 1600000, "segment": "SUV",      "popularity": 0.90},
        "XUV700":     {"base": 2000000, "segment": "SUV",      "popularity": 0.85},
        "Scorpio":    {"base": 1400000, "segment": "SUV",      "popularity": 0.80},
        "Bolero":     {"base": 900000,  "segment": "SUV",      "popularity": 0.72},
        "XUV300":     {"base": 1100000, "segment": "SUV",      "popularity": 0.76},
    },
    "Toyota": {
        "Innova":     {"base": 1800000, "segment": "MPV",      "popularity": 0.88},
        "Fortuner":   {"base": 3500000, "segment": "SUV",      "popularity": 0.82},
        "Glanza":     {"base": 650000,  "segment": "Hatchback","popularity": 0.72},
        "Urban Cruiser": {"base": 1000000,"segment": "SUV",    "popularity": 0.74},
    },
    "Kia": {
        "Seltos":     {"base": 1300000, "segment": "SUV",      "popularity": 0.86},
        "Sonet":      {"base": 1000000, "segment": "SUV",      "popularity": 0.80},
        "Carens":     {"base": 1100000, "segment": "MPV",      "popularity": 0.75},
    },
    "MG": {
        "Hector":     {"base": 1600000, "segment": "SUV",      "popularity": 0.78},
        "Astor":      {"base": 1200000, "segment": "SUV",      "popularity": 0.72},
        "ZS EV":      {"base": 2200000, "segment": "Electric", "popularity": 0.70},
    },
    "Volkswagen": {
        "Polo":       {"base": 700000,  "segment": "Hatchback","popularity": 0.70},
        "Vento":      {"base": 950000,  "segment": "Sedan",    "popularity": 0.68},
        "Taigun":     {"base": 1400000, "segment": "SUV",      "popularity": 0.74},
    },
    "Skoda": {
        "Kushaq":     {"base": 1400000, "segment": "SUV",      "popularity": 0.72},
        "Slavia":     {"base": 1200000, "segment": "Sedan",    "popularity": 0.68},
        "Octavia":    {"base": 2500000, "segment": "Sedan",    "popularity": 0.60},
    },
}

CITIES = {
    "Delhi NCR":   {"demand_multiplier": 1.15, "region": "North"},
    "Mumbai":      {"demand_multiplier": 1.20, "region": "West"},
    "Bangalore":   {"demand_multiplier": 1.18, "region": "South"},
    "Hyderabad":   {"demand_multiplier": 1.10, "region": "South"},
    "Chennai":     {"demand_multiplier": 1.08, "region": "South"},
    "Pune":        {"demand_multiplier": 1.12, "region": "West"},
    "Kolkata":     {"demand_multiplier": 1.05, "region": "East"},
    "Ahmedabad":   {"demand_multiplier": 1.08, "region": "West"},
    "Jaipur":      {"demand_multiplier": 0.98, "region": "North"},
    "Lucknow":     {"demand_multiplier": 0.95, "region": "North"},
    "Noida":       {"demand_multiplier": 1.10, "region": "North"},
    "Gurgaon":     {"demand_multiplier": 1.18, "region": "North"},
}

FUEL_TYPES = ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
TRANSMISSION = ["Manual", "Automatic", "AMT", "CVT", "DCT"]
COLORS = ["White", "Silver", "Black", "Grey", "Red", "Blue", "Brown", "Orange", "Green"]
OWNERS = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]
CONDITION_GRADES = ["Excellent", "Good", "Fair", "Poor"]


def depreciation_curve(age_years: float, segment: str) -> float:
    """Realistic depreciation based on age and segment."""
    curves = {
        "Hatchback": [1.0, 0.82, 0.70, 0.60, 0.52, 0.46, 0.40, 0.35, 0.31, 0.28, 0.25],
        "Sedan":     [1.0, 0.80, 0.67, 0.57, 0.49, 0.43, 0.37, 0.32, 0.28, 0.25, 0.22],
        "SUV":       [1.0, 0.84, 0.72, 0.63, 0.55, 0.49, 0.43, 0.38, 0.34, 0.30, 0.27],
        "MPV":       [1.0, 0.81, 0.68, 0.58, 0.50, 0.44, 0.38, 0.33, 0.29, 0.26, 0.23],
        "Electric":  [1.0, 0.78, 0.65, 0.55, 0.48, 0.42, 0.37, 0.33, 0.30, 0.27, 0.25],
    }
    curve = curves.get(segment, curves["Sedan"])
    idx = min(int(age_years), len(curve) - 1)
    return curve[idx]


def generate_car_record(record_id: int, timestamp: datetime = None) -> dict:
    """Generate a single realistic car listing."""
    brand = random.choices(list(CAR_CATALOGUE.keys()),
                           weights=[len(v) for v in CAR_CATALOGUE.values()])[0]
    model = random.choice(list(CAR_CATALOGUE[brand].keys()))
    car_info = CAR_CATALOGUE[brand][model]
    base_price = car_info["base"]

    # Age and year
    age = random.choices(range(1, 11), weights=[12, 18, 16, 14, 12, 10, 7, 5, 4, 2])[0]
    manufacture_year = 2025 - age
    
    # KMs driven (correlated with age)
    km_per_year = np.random.normal(12000, 3000)
    km_driven = max(1000, int(age * km_per_year + np.random.normal(0, 5000)))
    
    # Fuel type (segment-aware)
    if car_info["segment"] == "Electric":
        fuel = "Electric"
    else:
        fuel = random.choices(
            ["Petrol", "Diesel", "CNG"],
            weights=[0.60, 0.30, 0.10]
        )[0]

    # Transmission (correlated with fuel and segment)
    if fuel == "Electric":
        transmission = "Automatic"
    elif car_info["segment"] in ["SUV", "MPV"] and fuel == "Diesel":
        transmission = random.choices(["Manual", "Automatic", "AMT"], weights=[0.4, 0.4, 0.2])[0]
    else:
        transmission = random.choices(["Manual", "Automatic", "AMT", "CVT"],
                                      weights=[0.5, 0.25, 0.15, 0.10])[0]

    # Owner history
    owner = random.choices(OWNERS, weights=[0.50, 0.30, 0.15, 0.05])[0]
    owner_num = OWNERS.index(owner) + 1

    # City
    city = random.choice(list(CITIES.keys()))
    city_multiplier = CITIES[city]["demand_multiplier"]

    # Condition
    if age <= 2:
        condition = random.choices(CONDITION_GRADES, weights=[0.55, 0.35, 0.09, 0.01])[0]
    elif age <= 5:
        condition = random.choices(CONDITION_GRADES, weights=[0.30, 0.45, 0.20, 0.05])[0]
    else:
        condition = random.choices(CONDITION_GRADES, weights=[0.15, 0.35, 0.35, 0.15])[0]

    condition_factor = {"Excellent": 1.08, "Good": 1.00, "Fair": 0.90, "Poor": 0.78}[condition]

    # Insurance validity
    insurance_valid = random.choices([True, False], weights=[0.75, 0.25])[0]

    # Accidents
    accident_history = random.choices([0, 1, 2], weights=[0.70, 0.22, 0.08])[0]
    accident_factor = 1.0 - (accident_history * 0.05)

    # Service records
    service_records = age <= 5 or random.random() < 0.4
    service_factor = 1.03 if service_records else 0.97

    # Color premium
    color = random.choice(COLORS)
    color_premium = {"White": 0.02, "Silver": 0.01, "Black": 0.01}.get(color, -0.01)

    # Fuel efficiency variation
    fuel_multiplier = {"Diesel": 1.05, "Electric": 1.10, "Hybrid": 1.08,
                       "Petrol": 1.00, "CNG": 0.95}.get(fuel, 1.0)

    # Compute price
    depr = depreciation_curve(age, car_info["segment"])
    km_penalty = max(0, (km_driven - (age * 12000)) / 100000) * 0.03
    owner_penalty = (owner_num - 1) * 0.04

    price = (base_price * depr
             * city_multiplier
             * condition_factor
             * accident_factor
             * service_factor
             * fuel_multiplier
             * (1 + color_premium)
             * (1 - km_penalty)
             * (1 - owner_penalty))
    
    # Add noise ±3%
    price *= np.random.uniform(0.97, 1.03)
    price = round(price / 1000) * 1000  # round to nearest 1000

    # Days on market (popular cars sell faster)
    days_on_market = int(np.random.exponential(
        scale=15 / car_info["popularity"]
    ))
    days_on_market = max(1, min(days_on_market, 90))

    # Views and inquiries
    views = int(np.random.poisson(30 * car_info["popularity"])) + days_on_market * 3
    inquiries = int(views * np.random.uniform(0.05, 0.20))
    
    if timestamp is None:
        timestamp = datetime.now() - timedelta(days=random.randint(0, 365))

    return {
        "id": record_id,
        "brand": brand,
        "model": model,
        "variant": f"V{random.randint(1,3)}",
        "manufacture_year": manufacture_year,
        "age_years": age,
        "km_driven": km_driven,
        "fuel_type": fuel,
        "transmission": transmission,
        "owner_number": owner_num,
        "owner_type": owner,
        "city": city,
        "region": CITIES[city]["region"],
        "color": color,
        "segment": car_info["segment"],
        "condition_grade": condition,
        "insurance_valid": insurance_valid,
        "accident_history": accident_history,
        "service_records": service_records,
        "listing_price": int(price),
        "estimated_market_price": int(price * np.random.uniform(0.95, 1.05)),
        "days_on_market": days_on_market,
        "views": views,
        "inquiries": inquiries,
        "popularity_score": round(car_info["popularity"] * 100, 1),
        "demand_index": round(city_multiplier * car_info["popularity"] * 100, 1),
        "created_at": timestamp.isoformat(),
    }


def generate_dataset(n: int = 5000, include_stream: bool = False) -> pd.DataFrame:
    """Generate the full training/analytics dataset."""
    records = []
    start = datetime(2023, 1, 1)
    span = (datetime.now() - start).days
    
    for i in range(n):
        ts = start + timedelta(days=random.randint(0, span),
                               hours=random.randint(0, 23),
                               minutes=random.randint(0, 59))
        records.append(generate_car_record(i + 1, ts))
    
    df = pd.DataFrame(records)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def generate_stream_record(record_id: int) -> dict:
    """Generate a single real-time stream record (last 24h)."""
    ts = datetime.now() - timedelta(minutes=random.randint(0, 1440))
    return generate_car_record(record_id, ts)


def generate_loan_applications(cars_df: pd.DataFrame, n: int = 1000) -> pd.DataFrame:
    """Generate loan application records linked to car listings."""
    records = []
    sample = cars_df.sample(min(n, len(cars_df)), replace=True)
    
    for _, car in sample.iterrows():
        loan_amount = car["listing_price"] * np.random.uniform(0.6, 0.85)
        tenure_months = random.choice([12, 24, 36, 48, 60, 72])
        interest_rate = np.random.uniform(8.5, 16.0)
        
        monthly_emi = (loan_amount * (interest_rate/1200) *
                       (1 + interest_rate/1200)**tenure_months /
                       ((1 + interest_rate/1200)**tenure_months - 1))
        
        credit_score = int(np.random.normal(700, 80))
        credit_score = max(300, min(900, credit_score))
        
        approved = (credit_score > 650 and
                    monthly_emi < np.random.uniform(20000, 80000))
        
        records.append({
            "car_id": car["id"],
            "loan_amount": round(loan_amount, 2),
            "tenure_months": tenure_months,
            "interest_rate": round(interest_rate, 2),
            "monthly_emi": round(monthly_emi, 2),
            "credit_score": credit_score,
            "approved": approved,
            "city": car["city"],
            "segment": car["segment"],
        })
    
    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_dataset(5000)
    df.to_csv("/home/claude/c24iq/data/cars_dataset.csv", index=False)
    print(f"Generated {len(df)} records")
    print(df.describe())
