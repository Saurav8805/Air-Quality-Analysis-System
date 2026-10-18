"""Statistical and descriptive analysis module for air quality data.

Implements reusable, mathematically rigorous functions for univariate statistics,
time-series aggregation, location grouping, and AQI classification standards.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# AQI Category Definitions for Explicit Standards
# ---------------------------------------------------------------------------

# India CPCB National Air Quality Index (NAQI) Standard
INDIA_CPCB_BREAKPOINTS = [
    (0, 50, "Good", "#00b050", "Minimal health impact"),
    (51, 100, "Satisfactory", "#92d050", "Minor breathing discomfort to sensitive people"),
    (101, 200, "Moderate", "#ffff00", "Breathing discomfort to people with lung disease/asthma/heart disease"),
    (201, 300, "Poor", "#ff9900", "Breathing discomfort to most people on prolonged exposure"),
    (301, 400, "Very Poor", "#ff0000", "Respiratory illness on prolonged exposure"),
    (401, 500, "Severe", "#7030a0", "Affects healthy people and seriously impacts vulnerable groups"),
]

# US EPA Air Quality Index Standard
US_EPA_BREAKPOINTS = [
    (0, 50, "Good", "#00e400", "Air quality is satisfactory and poses little or no risk"),
    (51, 100, "Moderate", "#ffff00", "Acceptable; moderate health concern for a very small number of individuals"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00", "General public not likely affected; sensitive groups at risk"),
    (151, 200, "Unhealthy", "#ff0000", "Everyone may begin to experience health effects; sensitive groups more serious"),
    (201, 300, "Very Unhealthy", "#8f3f97", "Health alert: everyone may experience more serious health effects"),
    (301, 500, "Hazardous", "#7e0023", "Health warnings of emergency conditions; entire population affected"),
]


@dataclass
class SummaryStatistics:
    """Statistical summary container for numeric columns."""

    count: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    val_range: float
    q1: float
    q3: float
    iqr: float
    missing_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Valid Count (n)": self.count,
            "Mean": round(self.mean, 2),
            "Median": round(self.median, 2),
            "Std Deviation": round(self.std, 2),
            "Minimum": round(self.minimum, 2),
            "Maximum": round(self.maximum, 2),
            "Range": round(self.val_range, 2),
            "Q1 (25th %)": round(self.q1, 2),
            "Q3 (75th %)": round(self.q3, 2),
            "IQR": round(self.iqr, 2),
            "Missing Count": self.missing_count,
        }


def calculate_descriptive_stats(series: pd.Series) -> Optional[SummaryStatistics]:
    """Calculate complete descriptive statistics for a numeric series.

    Handles edge cases like empty series, single observation, or all NaNs.
    """
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    total_missing = int(series.isna().sum())

    if clean_series.empty:
        return None

    n = len(clean_series)
    val_mean = float(clean_series.mean())
    val_median = float(clean_series.median())
    val_min = float(clean_series.min())
    val_max = float(clean_series.max())
    val_range = val_max - val_min
    val_std = float(clean_series.std(ddof=1)) if n > 1 else 0.0
    q1 = float(clean_series.quantile(0.25))
    q3 = float(clean_series.quantile(0.75))
    iqr = q3 - q1

    return SummaryStatistics(
        count=n,
        mean=val_mean,
        median=val_median,
        std=val_std,
        minimum=val_min,
        maximum=val_max,
        val_range=val_range,
        q1=q1,
        q3=q3,
        iqr=iqr,
        missing_count=total_missing,
    )


def classify_aqi(
    aqi_value: float,
    standard: str = "India NAQI (CPCB)",
) -> Tuple[str, str, str]:
    """Classify an individual AQI value according to the specified standard.

    Returns
    -------
    Tuple[str, str, str]
        (Category Name, Hex Color, Advisory / Description)
    """
    if pd.isna(aqi_value) or aqi_value < 0:
        return "Unknown", "#808080", "No valid AQI recorded"

    if standard == "India NAQI (CPCB)":
        for low, high, cat, color, desc in INDIA_CPCB_BREAKPOINTS:
            if low <= aqi_value <= high:
                return cat, color, desc
        if aqi_value > 500:
            return "Severe+ (>500)", "#4a154b", "Extremely hazardous condition"
    elif standard == "US EPA AQI":
        for low, high, cat, color, desc in US_EPA_BREAKPOINTS:
            if low <= aqi_value <= high:
                return cat, color, desc
        if aqi_value > 500:
            return "Beyond Index (>500)", "#4a154b", "Emergency health hazard"
    
    return "Uncategorized", "#4a5568", "Raw numeric index"


def get_aqi_category_distribution(
    df: pd.DataFrame,
    standard: str = "India NAQI (CPCB)",
) -> pd.DataFrame:
    """Calculate frequency and percentage breakdown across AQI categories."""
    if df.empty or "aqi" not in df.columns:
        return pd.DataFrame()

    categories = []
    for val in df["aqi"]:
        cat, color, _ = classify_aqi(val, standard=standard)
        categories.append(cat)

    cat_series = pd.Series(categories, name="Category")
    counts = cat_series.value_counts()
    percentages = (counts / len(df) * 100).round(1)

    # Establish orderly list based on standard
    if standard == "India NAQI (CPCB)":
        ordered_cats = [b[2] for b in INDIA_CPCB_BREAKPOINTS]
    elif standard == "US EPA AQI":
        ordered_cats = [b[2] for b in US_EPA_BREAKPOINTS]
    else:
        ordered_cats = list(counts.index)

    # Compile result preserving canonical order
    rows = []
    for cat in ordered_cats:
        c = int(counts.get(cat, 0))
        p = float(percentages.get(cat, 0.0))
        rows.append({"Category": cat, "Count": c, "Percentage (%)": p})

    # Include any overflow category (e.g. Severe+)
    for cat in counts.index:
        if cat not in ordered_cats:
            rows.append({"Category": cat, "Count": int(counts[cat]), "Percentage (%)": float(percentages[cat])})

    return pd.DataFrame(rows)


def group_by_location(df: pd.DataFrame) -> pd.DataFrame:
    """Compute location-wise AQI descriptive aggregates with sample count."""
    if df.empty or "location" not in df.columns or "aqi" not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby("location")["aqi"]
        .agg(
            sample_count="count",
            mean_aqi="mean",
            median_aqi="median",
            min_aqi="min",
            max_aqi="max",
            std_aqi=lambda s: s.std(ddof=1) if len(s) > 1 else 0.0,
        )
        .reset_index()
    )

    grouped["mean_aqi"] = grouped["mean_aqi"].round(1)
    grouped["median_aqi"] = grouped["median_aqi"].round(1)
    grouped["std_aqi"] = grouped["std_aqi"].round(1)

    # Sort by descending mean AQI for clear ranking
    return grouped.sort_values(by="mean_aqi", ascending=False).reset_index(drop=True)


def aggregate_by_time(
    df: pd.DataFrame,
    freq: str = "D",
) -> pd.DataFrame:
    """Aggregate AQI and available pollutants over date intervals (Daily 'D', Weekly 'W', Monthly 'M').

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parsed 'date' column and 'aqi'.
    freq : str, default 'D'
        Pandas frequency alias ('D' for Daily, 'W' for Weekly, 'M' for Monthly).
    """
    if df.empty or "date" not in df.columns or "aqi" not in df.columns:
        return pd.DataFrame()

    valid_dates = df.dropna(subset=["date"]).copy()
    if valid_dates.empty:
        return pd.DataFrame()

    # Identify numeric columns to aggregate
    numeric_cols = ["aqi"]
    for col in ["pm2.5", "pm10", "no2", "so2", "co", "o3"]:
        if col in valid_dates.columns:
            numeric_cols.append(col)

    grouped = (
        valid_dates.set_index("date")
        .resample(freq)[numeric_cols]
        .mean()
        .reset_index()
    )

    # Count observations per period to communicate coverage
    counts = (
        valid_dates.set_index("date")
        .resample(freq)["aqi"]
        .count()
        .reset_index()
        .rename(columns={"aqi": "sample_count"})
    )

    merged = pd.merge(grouped, counts, on="date")
    # Drop empty resampled bins with 0 samples
    merged = merged[merged["sample_count"] > 0].reset_index(drop=True)
    # Round only numeric columns to avoid datetime dtype warning
    num_cols = merged.select_dtypes(include=[np.number]).columns
    merged[num_cols] = merged[num_cols].round(2)
    return merged


def get_all_pollutants_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a consolidated comparative statistics table for all detected pollutants."""
    if df.empty:
        return pd.DataFrame()

    pollutant_cols = [p for p in ["pm2.5", "pm10", "no2", "so2", "co", "o3"] if p in df.columns]
    if not pollutant_cols:
        return pd.DataFrame()

    rows = []
    for p in pollutant_cols:
        stats = calculate_descriptive_stats(df[p])
        if stats:
            d = stats.to_dict()
            d["Pollutant"] = p.upper()
            rows.append(d)

    if not rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows)
    # Reorder so Pollutant name is the first column
    cols = ["Pollutant"] + [c for c in summary_df.columns if c != "Pollutant"]
    return summary_df[cols]


def compute_pollutant_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix between AQI and available pollutants."""
    cols_to_check = ["aqi", "pm2.5", "pm10", "no2", "so2", "co", "o3"]
    active_cols = [c for c in cols_to_check if c in df.columns]

    if len(active_cols) < 2 or df.empty:
        return pd.DataFrame()

    numeric_subset = df[active_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric_subset.corr(method="pearson").round(3)
    # Rename columns to uppercase for clean display
    corr.columns = [c.upper() for c in corr.columns]
    corr.index = [i.upper() for i in corr.index]
    return corr
