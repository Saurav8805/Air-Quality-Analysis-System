"""Automated test suite for analysis, statistics, and classification logic."""

import numpy as np
import pandas as pd
import pytest
from src.analysis import (
    calculate_descriptive_stats,
    classify_aqi,
    get_aqi_category_distribution,
    group_by_location,
    aggregate_by_time,
    get_all_pollutants_summary,
    compute_pollutant_correlation,
)
from src.pdf_exporter import generate_pdf_report
from src.ml_predictor import train_aqi_estimation_model


def test_descriptive_stats_known_dataset():
    # Known dataset: [10, 20, 30, 40, 50]
    # Mean: 30.0, Median: 30.0, Min: 10, Max: 50, Range: 40
    # Q1: 20.0, Q3: 40.0, IQR: 20.0
    # Sample std (ddof=1) of [10, 20, 30, 40, 50] = sqrt(250) = 15.811
    series = pd.Series([10, 20, 30, 40, 50])
    stats = calculate_descriptive_stats(series)

    assert stats is not None
    assert stats.count == 5
    assert stats.mean == 30.0
    assert stats.median == 30.0
    assert stats.minimum == 10.0
    assert stats.maximum == 50.0
    assert stats.val_range == 40.0
    assert stats.q1 == 20.0
    assert stats.q3 == 40.0
    assert stats.iqr == 20.0
    assert round(stats.std, 2) == 15.81


def test_descriptive_stats_empty_and_nan():
    empty_series = pd.Series([], dtype=float)
    assert calculate_descriptive_stats(empty_series) is None

    nan_series = pd.Series([np.nan, np.nan])
    assert calculate_descriptive_stats(nan_series) is None


def test_cpcb_aqi_classification():
    # India CPCB Standard breakpoints:
    # 0-50 Good, 51-100 Satisfactory, 101-200 Moderate, 201-300 Poor, 301-400 Very Poor, 401-500 Severe
    assert classify_aqi(25, standard="India NAQI (CPCB)")[0] == "Good"
    assert classify_aqi(75, standard="India NAQI (CPCB)")[0] == "Satisfactory"
    assert classify_aqi(150, standard="India NAQI (CPCB)")[0] == "Moderate"
    assert classify_aqi(250, standard="India NAQI (CPCB)")[0] == "Poor"
    assert classify_aqi(350, standard="India NAQI (CPCB)")[0] == "Very Poor"
    assert classify_aqi(450, standard="India NAQI (CPCB)")[0] == "Severe"
    assert classify_aqi(520, standard="India NAQI (CPCB)")[0] == "Severe+ (>500)"


def test_us_epa_aqi_classification():
    assert classify_aqi(40, standard="US EPA AQI")[0] == "Good"
    assert classify_aqi(80, standard="US EPA AQI")[0] == "Moderate"
    assert classify_aqi(120, standard="US EPA AQI")[0] == "Unhealthy for Sensitive Groups"
    assert classify_aqi(180, standard="US EPA AQI")[0] == "Unhealthy"
    assert classify_aqi(250, standard="US EPA AQI")[0] == "Very Unhealthy"
    assert classify_aqi(350, standard="US EPA AQI")[0] == "Hazardous"


def test_category_distribution_frequency():
    df = pd.DataFrame({"aqi": [20, 30, 75, 150, 450]})
    cat_dist = get_aqi_category_distribution(df, standard="India NAQI (CPCB)")
    assert not cat_dist.empty
    good_row = cat_dist[cat_dist["Category"] == "Good"]
    assert good_row["Count"].values[0] == 2
    assert good_row["Percentage (%)"].values[0] == 40.0


def test_group_by_location():
    df = pd.DataFrame({
        "location": ["Delhi", "Delhi", "Mumbai", "Mumbai"],
        "aqi": [200, 300, 80, 100]
    })
    grouped = group_by_location(df)
    assert len(grouped) == 2
    delhi_row = grouped[grouped["location"] == "Delhi"].iloc[0]
    assert delhi_row["mean_aqi"] == 250.0
    assert delhi_row["sample_count"] == 2
    assert delhi_row["min_aqi"] == 200
    assert delhi_row["max_aqi"] == 300


def test_aggregate_by_time():
    dates = pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"])
    df = pd.DataFrame({
        "date": dates,
        "aqi": [100, 120, 200],
        "pm2.5": [40, 50, 90]
    })
    time_agg = aggregate_by_time(df, freq="D")
    assert len(time_agg) == 2
    # Jan 1 mean AQI = (100 + 120) / 2 = 110.0
    assert time_agg.loc[0, "aqi"] == 110.0
    assert time_agg.loc[0, "sample_count"] == 2
    assert time_agg.loc[1, "aqi"] == 200.0


def test_pollutants_summary():
    df = pd.DataFrame({
        "aqi": [100, 150],
        "pm2.5": [30, 40],
        "pm10": [60, 80]
    })
    summary = get_all_pollutants_summary(df)
    assert len(summary) == 2
    assert set(summary["Pollutant"]) == {"PM2.5", "PM10"}


def test_correlation_matrix():
    df = pd.DataFrame({
        "aqi": [50, 100, 150, 200],
        "pm2.5": [20, 40, 60, 80]
    })
    corr = compute_pollutant_correlation(df)
    assert not corr.empty
    assert round(corr.loc["AQI", "PM2.5"], 2) == 1.0


def test_pdf_report_generation():
    stats = calculate_descriptive_stats(pd.Series([100, 150, 200]))
    assert stats is not None
    pdf_bytes = generate_pdf_report(
        summary_stats=stats,
        data_mode="Demo Dataset",
        selected_locations=["Delhi"],
        date_filter_str="All Dates",
        quality_notes=["Test note"],
        standard_name="India NAQI (CPCB)",
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500  # Valid PDF binary


def test_ml_predictor():
    # Build 30 synthetic samples
    np.random.seed(42)
    pm25 = np.linspace(10, 150, 30)
    aqi = pm25 * 2.2 + np.random.normal(0, 5, 30)
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    df = pd.DataFrame({"date": dates, "pm2.5": pm25, "aqi": aqi})

    eval_result = train_aqi_estimation_model(df, test_size=0.2, model_type="Linear Regression")
    assert eval_result.is_suitable is True
    assert eval_result.test_r2 > 0.8  # Strong fit
    assert eval_result.test_mae < eval_result.baseline_mae  # Outperforms mean baseline


def test_comparison_charts():
    from src.charts import (
        plot_dual_location_trend,
        plot_location_pollutants_comparison,
        plot_pollutant_scatter,
    )
    import matplotlib.pyplot as plt

    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": list(dates) + list(dates),
        "location": ["Delhi"] * 10 + ["Mumbai"] * 10,
        "aqi": [200 + i * 2 for i in range(10)] + [100 + i for i in range(10)],
        "pm2.5": [80 + i for i in range(10)] + [40 + i for i in range(10)],
        "pm10": [140 + i for i in range(10)] + [70 + i for i in range(10)],
    })

    fig1 = plot_dual_location_trend(df, "Delhi", "Mumbai")
    assert fig1 is not None
    plt.close(fig1)

    fig2 = plot_location_pollutants_comparison(df, "Delhi", "Mumbai", ["pm2.5", "pm10"])
    assert fig2 is not None
    plt.close(fig2)

    fig3 = plot_pollutant_scatter(df, "pm2.5", "aqi")
    assert fig3 is not None
    plt.close(fig3)

