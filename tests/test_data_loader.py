"""Automated test suite for data_loader module."""

import io
import pytest
import pandas as pd
from src.data_loader import load_and_validate_csv, normalize_column_name


def test_normalize_column_name():
    assert normalize_column_name("AQI") == "aqi"
    assert normalize_column_name("  Air Quality Index  ") == "aqi"
    assert normalize_column_name("City") == "location"
    assert normalize_column_name("DateTime") == "date"
    assert normalize_column_name("PM2.5") == "pm2.5"
    assert normalize_column_name("pm_10") == "pm10"


def test_load_valid_csv():
    csv_data = """Date,Location,AQI,PM2.5
2026-01-01,Delhi,150,65.2
2026-01-02,Delhi,160,70.1
2026-01-01,Mumbai,85,32.0
"""
    df, report = load_and_validate_csv(io.StringIO(csv_data))
    assert report.is_valid is True
    assert df is not None
    assert len(df) == 3
    assert report.total_raw_rows == 3
    assert report.valid_rows == 3
    assert report.has_date is True
    assert report.has_location is True
    assert "pm2.5" in report.detected_pollutants


def test_missing_required_aqi_column():
    csv_data = """Date,Location,Temperature,Humidity
2026-01-01,Delhi,22.5,50
2026-01-02,Delhi,23.0,52
"""
    df, report = load_and_validate_csv(io.StringIO(csv_data))
    assert report.is_valid is False
    assert df is None
    assert "Required column missing" in report.error_message


def test_non_numeric_and_missing_aqi():
    csv_data = """Location,AQI
CityA,100
CityB,INVALID_TEXT
CityC,
CityD,200
"""
    df, report = load_and_validate_csv(io.StringIO(csv_data))
    assert report.is_valid is True
    assert df is not None
    assert len(df) == 2  # Only CityA and CityD are valid
    assert report.missing_aqi_count == 1
    assert report.non_numeric_aqi_count == 1
    assert list(df["aqi"]) == [100.0, 200.0]


def test_negative_aqi_values():
    csv_data = """Location,AQI
CityA,150
CityB,-99
CityC,0
CityD,-1
"""
    df, report = load_and_validate_csv(io.StringIO(csv_data))
    assert report.is_valid is True
    assert df is not None
    assert len(df) == 2  # CityA (150) and CityC (0) are valid non-negative
    assert report.negative_aqi_count == 2
    assert list(df["aqi"]) == [150.0, 0.0]


def test_empty_dataset_handling():
    empty_csv = ""
    df, report = load_and_validate_csv(io.StringIO(empty_csv))
    assert report.is_valid is False
    assert df is None
    assert "empty" in report.error_message.lower()


def test_all_invalid_rows():
    csv_data = """AQI
-50
abc
NaN
"""
    df, report = load_and_validate_csv(io.StringIO(csv_data))
    assert report.is_valid is False
    assert df is None
    assert "No valid records remaining" in report.error_message


def test_missing_optional_columns():
    csv_data = """AQI
120
140
160
"""
    df, report = load_and_validate_csv(io.StringIO(csv_data))
    assert report.is_valid is True
    assert df is not None
    assert len(df) == 3
    assert report.has_date is False
    assert report.has_location is False
    assert len(report.detected_pollutants) == 0


def test_duplicate_row_detection():
    csv_data = """Location,AQI
Delhi,150
Delhi,150
Mumbai,80
"""
    # Test preserve duplicates
    df_preserve, report_preserve = load_and_validate_csv(io.StringIO(csv_data), drop_duplicates_choice=False)
    assert report_preserve.duplicate_rows_count == 1
    assert len(df_preserve) == 3

    # Test drop duplicates
    df_drop, report_drop = load_and_validate_csv(io.StringIO(csv_data), drop_duplicates_choice=True)
    assert report_drop.duplicate_rows_count == 1
    assert len(df_drop) == 2


def test_unparseable_dates():
    csv_data = """Date,AQI
2026-01-01,100
not-a-date,110
2026-01-03,120
"""
    df, report = load_and_validate_csv(io.StringIO(csv_data))
    assert report.is_valid is True
    assert df is not None
    assert report.unparseable_dates_count == 1
    assert pd.isna(df.loc[1, "date"])
