"""Data loading, normalization, validation, and quality assessment module.

This module provides reusable, UI-independent functions to ingest, validate,
clean, and inspect air quality CSV datasets according to strict data integrity standards.
"""

from __future__ import annotations
import io
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


# Column alias dictionary for safe normalization
COLUMN_ALIASES: Dict[str, str] = {
    # Required target
    "aqi": "aqi",
    "air quality index": "aqi",
    "air_quality_index": "aqi",
    "airqualityindex": "aqi",
    "overall aqi": "aqi",
    "overall_aqi": "aqi",
    "index_value": "aqi",
    # Date target
    "date": "date",
    "datetime": "date",
    "date_time": "date",
    "timestamp": "date",
    "time": "date",
    "recorded_at": "date",
    "observation_date": "date",
    # Location target
    "location": "location",
    "city": "location",
    "station": "location",
    "site": "location",
    "station_name": "location",
    "monitoring_station": "location",
    "place": "location",
    "area": "location",
    # Pollutants
    "pm2.5": "pm2.5",
    "pm25": "pm2.5",
    "pm_2.5": "pm2.5",
    "pm_25": "pm2.5",
    "pm2_5": "pm2.5",
    "pm10": "pm10",
    "pm_10": "pm10",
    "no2": "no2",
    "nitrogen_dioxide": "no2",
    "so2": "so2",
    "sulfur_dioxide": "so2",
    "co": "co",
    "carbon_monoxide": "co",
    "o3": "o3",
    "ozone": "o3",
}

KNOWN_POLLUTANTS: Set[str] = {"pm2.5", "pm10", "no2", "so2", "co", "o3"}


@dataclass
class ValidationReport:
    """Detailed metadata and diagnostic statistics from dataset validation."""

    is_valid: bool = False
    error_message: Optional[str] = None
    total_raw_rows: int = 0
    valid_rows: int = 0
    missing_aqi_count: int = 0
    non_numeric_aqi_count: int = 0
    negative_aqi_count: int = 0
    duplicate_rows_count: int = 0
    unparseable_dates_count: int = 0
    raw_columns: List[str] = field(default_factory=list)
    normalized_columns: List[str] = field(default_factory=list)
    detected_columns: Dict[str, str] = field(default_factory=dict)
    detected_pollutants: List[str] = field(default_factory=list)
    has_date: bool = False
    has_location: bool = False
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    cleaning_notes: List[str] = field(default_factory=list)


def normalize_column_name(col: str) -> str:
    """Normalize a raw column header by stripping whitespace, lowercasing, and matching aliases."""
    clean = str(col).strip().lower().replace("-", "_")
    return COLUMN_ALIASES.get(clean, clean)


def load_and_validate_csv(
    source: Union[str, BinaryIO, io.StringIO],
    drop_duplicates_choice: bool = False,
) -> Tuple[Optional[pd.DataFrame], ValidationReport]:
    """Ingest, validate, and clean an air quality CSV file.

    Parameters
    ----------
    source : Union[str, BinaryIO, io.StringIO]
        Filepath or file-like buffer of the CSV dataset.
    drop_duplicates_choice : bool, default False
        Whether to drop identified duplicate rows. Never done silently.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], ValidationReport]
        Cleaned pandas DataFrame (or None if invalid) and detailed ValidationReport.
    """
    report = ValidationReport()

    # 1. Read CSV safely
    try:
        df_raw = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        report.is_valid = False
        report.error_message = "The provided CSV file is completely empty."
        return None, report
    except Exception as exc:
        report.is_valid = False
        report.error_message = f"Failed to read CSV file: {str(exc)}"
        return None, report

    report.total_raw_rows = len(df_raw)
    report.raw_columns = [str(c) for c in df_raw.columns]

    if df_raw.empty:
        report.is_valid = False
        report.error_message = "The dataset contains header columns but has zero rows of data."
        return None, report

    # 2. Map and normalize column headers without mutating original
    rename_map: Dict[str, str] = {}
    detected_targets: Dict[str, str] = {}
    normalized_cols: List[str] = []

    for orig_col in df_raw.columns:
        norm = normalize_column_name(orig_col)
        normalized_cols.append(norm)
        # Prevent collisions if two raw headers map to the same normalized alias
        if norm in detected_targets and detected_targets[norm] != orig_col:
            # Keep original name if conflict
            rename_map[orig_col] = orig_col
        else:
            detected_targets[norm] = orig_col
            rename_map[orig_col] = norm

    df = df_raw.rename(columns=rename_map).copy()
    report.normalized_columns = list(df.columns)
    report.detected_columns = detected_targets

    # 3. Check required 'aqi' column
    if "aqi" not in df.columns:
        report.is_valid = False
        report.error_message = (
            "Required column missing: Could not identify an 'AQI' or 'Air Quality Index' column. "
            f"Detected columns: {report.raw_columns}. Please ensure an AQI column is included."
        )
        return None, report

    # 4. Detect optional features
    report.has_date = "date" in df.columns
    report.has_location = "location" in df.columns
    report.detected_pollutants = [p for p in df.columns if p in KNOWN_POLLUTANTS]

    # 5. Check duplicate rows before dropping
    duplicate_mask = df.duplicated(keep="first")
    report.duplicate_rows_count = int(duplicate_mask.sum())

    if report.duplicate_rows_count > 0:
        if drop_duplicates_choice:
            df = df[~duplicate_mask].copy()
            report.cleaning_notes.append(
                f"Removed {report.duplicate_rows_count} exact duplicate row(s) per user request."
            )
        else:
            report.cleaning_notes.append(
                f"Detected {report.duplicate_rows_count} exact duplicate row(s). Preserved in dataset."
            )

    # 6. Validate AQI column numeric integrity
    raw_aqi_series = df["aqi"]
    numeric_aqi_series = pd.to_numeric(raw_aqi_series, errors="coerce")

    missing_or_non_numeric = numeric_aqi_series.isna()
    report.missing_aqi_count = int(raw_aqi_series.isna().sum())
    report.non_numeric_aqi_count = int(missing_or_non_numeric.sum() - report.missing_aqi_count)

    # Filter out missing or non-numeric AQI
    valid_numeric_mask = ~missing_or_non_numeric
    df_valid = df[valid_numeric_mask].copy()
    df_valid["aqi"] = numeric_aqi_series[valid_numeric_mask]

    # Check for negative AQI values
    negative_mask = df_valid["aqi"] < 0
    report.negative_aqi_count = int(negative_mask.sum())
    if report.negative_aqi_count > 0:
        report.cleaning_notes.append(
            f"Filtered out {report.negative_aqi_count} row(s) with invalid negative AQI (< 0)."
        )
        df_valid = df_valid[~negative_mask].copy()

    if report.missing_aqi_count > 0 or report.non_numeric_aqi_count > 0:
        report.cleaning_notes.append(
            f"Excluded {report.missing_aqi_count} row(s) with missing AQI and "
            f"{report.non_numeric_aqi_count} row(s) with unparseable non-numeric AQI."
        )

    # 7. Safe date parsing if date column exists
    if report.has_date:
        parsed_dates = pd.to_datetime(df_valid["date"], errors="coerce")
        unparseable_dates = parsed_dates.isna() & df_valid["date"].notna()
        report.unparseable_dates_count = int(unparseable_dates.sum())
        df_valid["date"] = parsed_dates
        
        valid_dates = parsed_dates.dropna()
        if not valid_dates.empty:
            report.date_range = (valid_dates.min(), valid_dates.max())
        if report.unparseable_dates_count > 0:
            report.cleaning_notes.append(
                f"Identified {report.unparseable_dates_count} unparseable date value(s) set to NaT."
            )

    # 8. Safe numeric conversion for pollutants
    for poll in report.detected_pollutants:
        df_valid[poll] = pd.to_numeric(df_valid[poll], errors="coerce")

    # 9. Final validity check
    report.valid_rows = len(df_valid)
    if report.valid_rows == 0:
        report.is_valid = False
        report.error_message = (
            "No valid records remaining after validation. All rows had invalid, missing, "
            "or negative AQI values."
        )
        return None, report

    report.is_valid = True
    return df_valid, report
