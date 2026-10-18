"""Script to generate a structured synthetic demonstration dataset for air quality analysis."""
import os
import numpy as np
import pandas as pd

np.random.seed(42)

dates = pd.date_range(start="2026-01-01", periods=60, freq="D")
locations = ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Chennai", "Hyderabad"]

# Profile characteristics for each city to simulate realistic regional air quality variations
city_profiles = {
    "Delhi": {"base_aqi": 280, "aqi_std": 60, "pm25_ratio": 0.58, "pm10_ratio": 0.88, "no2_base": 48, "so2_base": 18, "co_base": 1.9, "o3_base": 35},
    "Mumbai": {"base_aqi": 140, "aqi_std": 35, "pm25_ratio": 0.50, "pm10_ratio": 0.80, "no2_base": 32, "so2_base": 15, "co_base": 1.2, "o3_base": 28},
    "Bengaluru": {"base_aqi": 85, "aqi_std": 22, "pm25_ratio": 0.45, "pm10_ratio": 0.70, "no2_base": 22, "so2_base": 10, "co_base": 0.8, "o3_base": 25},
    "Kolkata": {"base_aqi": 190, "aqi_std": 45, "pm25_ratio": 0.55, "pm10_ratio": 0.82, "no2_base": 38, "so2_base": 14, "co_base": 1.5, "o3_base": 30},
    "Chennai": {"base_aqi": 95, "aqi_std": 25, "pm25_ratio": 0.46, "pm10_ratio": 0.72, "no2_base": 20, "so2_base": 11, "co_base": 0.9, "o3_base": 27},
    "Hyderabad": {"base_aqi": 125, "aqi_std": 30, "pm25_ratio": 0.48, "pm10_ratio": 0.75, "no2_base": 26, "so2_base": 12, "co_base": 1.1, "o3_base": 32},
}

records = []

for dt in dates:
    # Introduce seasonal / weekly wave factor
    day_factor = 1.0 + 0.15 * np.sin(dt.dayofyear / 7.0)
    for city, prof in city_profiles.items():
        raw_aqi = np.random.normal(prof["base_aqi"] * day_factor, prof["aqi_std"])
        aqi = int(np.clip(raw_aqi, 20, 480))
        
        # Calculate consistent realistic pollutant concentrations
        pm25 = round(max(5.0, aqi * prof["pm25_ratio"] * np.random.uniform(0.85, 1.15) * 0.4), 1)
        pm10 = round(max(pm25 * 1.2, aqi * prof["pm10_ratio"] * np.random.uniform(0.85, 1.15) * 0.55), 1)
        no2 = round(max(2.0, prof["no2_base"] * np.random.uniform(0.75, 1.35)), 1)
        so2 = round(max(1.0, prof["so2_base"] * np.random.uniform(0.7, 1.3)), 1)
        co = round(max(0.1, prof["co_base"] * np.random.uniform(0.8, 1.25)), 2)
        o3 = round(max(3.0, prof["o3_base"] * np.random.uniform(0.7, 1.3)), 1)
        
        records.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "Location": city,
            "AQI": aqi,
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3
        })

df = pd.DataFrame(records)

# Add a couple of realistic anomalies/edge-cases to demonstrate data validation:
# 1) One row with a missing pollutant value (O3 is NaN)
df.loc[12, "O3"] = np.nan
# 2) One duplicate row appended at the end to demonstrate duplicate tracking
df = pd.concat([df, df.iloc[[5]]], ignore_index=True)

out_dir = r"D:\PIMV Site\Air Quality Analysis System\data"
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "sample_air_quality.csv")
df.to_csv(csv_path, index=False)
print(f"Generated {len(df)} rows in {csv_path}")
