"""Visualization module for the Air Quality Analysis System.

Creates sleek, glassmorphism-harmonized Matplotlib figures formatted with
a dark slate/cyan aesthetic, glowing accents, clear legends, and responsive annotations.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import INDIA_CPCB_BREAKPOINTS, US_EPA_BREAKPOINTS, classify_aqi

# Sleek Light Frosted Glass Theme Palette (Optimized for pure white / glassmorphism)
BG_FIGURE = "#ffffff"       # Clean figure backdrop
BG_AXES = "#f8fafc"         # Subtle frosted slate tint
COLOR_TEXT_MAIN = "#0f172a" # Deep navy slate for high contrast
COLOR_TEXT_MUTED = "#64748b"# Slate 500 for secondary metrics
COLOR_GRID = "#e2e8f0"      # Light slate gridlines
COLOR_BORDER = "#cbd5e1"    # Clean border stroke

# Vibrant data colors for high contrast on light backgrounds
COLOR_CYAN = "#0284c7"      # Vibrant Sky/Cyan
COLOR_SKY = "#0369a1"       # Deep Sky
COLOR_EMERALD = "#059669"   # Rich Emerald Green
COLOR_AMBER = "#d97706"     # Deep Amber
COLOR_ROSE = "#e11d48"      # Crimson Rose
COLOR_PURPLE = "#7c3aed"    # Royal Purple


def _apply_glass_theme(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Apply consistent glassmorphism dark aesthetic to figure and axes."""
    fig.patch.set_facecolor(BG_FIGURE)
    fig.patch.set_alpha(0.0)  # Transparent figure background for glass container blend
    ax.set_facecolor(BG_AXES)
    
    ax.set_title(title, fontsize=12, weight="bold", color=COLOR_TEXT_MAIN, pad=12)
    ax.set_xlabel(xlabel, fontsize=9.5, weight="semibold", color=COLOR_TEXT_MUTED, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=9.5, weight="semibold", color=COLOR_TEXT_MUTED, labelpad=8)
    
    ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID, zorder=0)
    ax.set_axisbelow(True)
    
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLOR_BORDER)
        ax.spines[spine].set_linewidth(1.0)


def plot_aqi_distribution(
    df: pd.DataFrame,
    bins: int = 25,
) -> plt.Figure:
    """Plot an annotated histogram of AQI values with mean and median markers."""
    fig, ax = plt.subplots(figsize=(8, 4.3), dpi=100)

    if df.empty or "aqi" not in df.columns:
        ax.text(0.5, 0.5, "No AQI records available", ha="center", va="center", color=COLOR_TEXT_MUTED)
        _apply_glass_theme(fig, ax, "AQI Distribution", "AQI Value", "Frequency")
        plt.tight_layout()
        return fig

    aqi_data = df["aqi"].dropna()
    mean_val = float(aqi_data.mean())
    median_val = float(aqi_data.median())

    # Draw histogram
    n, bins_edges, patches = ax.hist(
        aqi_data,
        bins=bins,
        color=COLOR_CYAN,
        edgecolor=BG_FIGURE,
        alpha=0.85,
        zorder=3,
    )

    # Reference indicator lines
    ax.axvline(
        mean_val,
        color=COLOR_ROSE,
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.1f}",
        zorder=4,
    )
    ax.axvline(
        median_val,
        color=COLOR_AMBER,
        linestyle="-.",
        linewidth=2,
        label=f"Median: {median_val:.1f}",
        zorder=4,
    )

    legend = ax.legend(frameon=True, facecolor=BG_FIGURE, edgecolor=COLOR_BORDER, fontsize=8.5)
    plt.setp(legend.get_texts(), color=COLOR_TEXT_MAIN)
    _apply_glass_theme(fig, ax, "Air Quality Index (AQI) Frequency Distribution", "AQI Value", "Number of Observations")
    plt.tight_layout()
    return fig


def plot_aqi_by_location(location_summary_df: pd.DataFrame) -> plt.Figure:
    """Plot a bar chart comparing mean AQI across locations with sample counts."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=100)

    if location_summary_df.empty or "location" not in location_summary_df.columns:
        ax.text(0.5, 0.5, "No location data available to compare", ha="center", va="center", color=COLOR_TEXT_MUTED)
        _apply_glass_theme(fig, ax, "Location Comparison", "Location", "Mean AQI")
        plt.tight_layout()
        return fig

    locations = location_summary_df["location"]
    mean_values = location_summary_df["mean_aqi"]
    sample_counts = location_summary_df["sample_count"]

    # Color gradient based on AQI severity
    colors = []
    for val in mean_values:
        if val <= 100:
            colors.append(COLOR_EMERALD)  # Good / Satisfactory green
        elif val <= 200:
            colors.append(COLOR_AMBER)    # Moderate amber
        elif val <= 300:
            colors.append("#fb923c")      # Poor orange
        else:
            colors.append(COLOR_ROSE)     # Very Poor / Severe red

    bars = ax.bar(locations, mean_values, color=colors, edgecolor=BG_FIGURE, width=0.52, zorder=3)

    # Annotate bars
    for bar, val, count in zip(bars, mean_values, sample_counts):
        height = bar.get_height()
        ax.annotate(
            f"{val:.1f}\n(n={count})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            weight="bold",
            color=COLOR_TEXT_MAIN,
        )

    if len(mean_values) > 0:
        ax.set_ylim(0, max(mean_values) * 1.25)

    plt.xticks(rotation=15, ha="right")
    _apply_glass_theme(fig, ax, "Mean AQI by Monitoring Location (with Sample Counts)", "Monitoring Station / City", "Mean AQI")
    plt.tight_layout()
    return fig


def plot_time_series_trend(
    time_df: pd.DataFrame,
    show_rolling: bool = True,
) -> plt.Figure:
    """Plot a time-series line chart of aggregated AQI with rolling mean."""
    fig, ax = plt.subplots(figsize=(9, 4.3), dpi=100)

    if time_df.empty or "date" not in time_df.columns or "aqi" not in time_df.columns:
        ax.text(0.5, 0.5, "No temporal data found", ha="center", va="center", color=COLOR_TEXT_MUTED)
        _apply_glass_theme(fig, ax, "AQI Trend Over Time", "Date", "Average AQI")
        plt.tight_layout()
        return fig

    sorted_df = time_df.sort_values(by="date").dropna(subset=["date", "aqi"]).copy()
    dates = sorted_df["date"]
    aqi_values = sorted_df["aqi"]

    # Main trend line
    ax.plot(
        dates,
        aqi_values,
        marker="o",
        markersize=4,
        color=COLOR_CYAN,
        linewidth=2,
        label="Observed Average AQI",
        zorder=3,
    )
    ax.fill_between(dates, aqi_values, color=COLOR_CYAN, alpha=0.15, zorder=2)

    # 7-point rolling average
    if show_rolling and len(sorted_df) >= 7:
        rolling_series = aqi_values.rolling(window=7, min_periods=3).mean()
        ax.plot(
            dates,
            rolling_series,
            color=COLOR_ROSE,
            linewidth=2.2,
            linestyle="-",
            label="7-Day Rolling Trend",
            zorder=4,
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=20, ha="right")

    legend = ax.legend(frameon=True, facecolor=BG_FIGURE, edgecolor=COLOR_BORDER, fontsize=8.5)
    plt.setp(legend.get_texts(), color=COLOR_TEXT_MAIN)
    _apply_glass_theme(fig, ax, "Temporal Air Quality Progression", "Observation Date", "Average AQI")
    plt.tight_layout()
    return fig


def plot_pollutant_distribution(
    df: pd.DataFrame,
    pollutant: str,
    bins: int = 20,
) -> plt.Figure:
    """Plot distribution histogram with stats for a single pollutant."""
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=100)

    p_col = pollutant.lower()
    if df.empty or p_col not in df.columns:
        ax.text(0.5, 0.5, f"Column '{pollutant}' not present", ha="center", va="center", color=COLOR_TEXT_MUTED)
        _apply_glass_theme(fig, ax, f"{pollutant.upper()} Distribution", f"{pollutant.upper()} Concentration", "Count")
        plt.tight_layout()
        return fig

    vals = pd.to_numeric(df[p_col], errors="coerce").dropna()
    if vals.empty:
        ax.text(0.5, 0.5, f"No non-null values for '{pollutant}'", ha="center", va="center", color=COLOR_TEXT_MUTED)
        _apply_glass_theme(fig, ax, f"{pollutant.upper()} Distribution", f"{pollutant.upper()} Concentration", "Count")
        plt.tight_layout()
        return fig

    mean_p = float(vals.mean())
    median_p = float(vals.median())

    ax.hist(vals, bins=bins, color=COLOR_EMERALD, edgecolor=BG_FIGURE, alpha=0.85, zorder=3)
    ax.axvline(mean_p, color=COLOR_ROSE, linestyle="--", linewidth=1.8, label=f"Mean: {mean_p:.1f}", zorder=4)
    ax.axvline(median_p, color=COLOR_AMBER, linestyle="-.", linewidth=1.8, label=f"Median: {median_p:.1f}", zorder=4)

    legend = ax.legend(frameon=True, facecolor=BG_FIGURE, edgecolor=COLOR_BORDER, fontsize=8.5)
    plt.setp(legend.get_texts(), color=COLOR_TEXT_MAIN)
    _apply_glass_theme(
        fig,
        ax,
        f"{pollutant.upper()} Concentration Distribution",
        f"{pollutant.upper()} Concentration",
        "Number of Records",
    )
    plt.tight_layout()
    return fig


def plot_category_breakdown(
    category_df: pd.DataFrame,
    standard: str = "India NAQI (CPCB)",
) -> plt.Figure:
    """Plot horizontal bar chart of AQI categories with official color mappings."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

    if category_df.empty or "Category" not in category_df.columns:
        ax.text(0.5, 0.5, "No categorical data available", ha="center", va="center", color=COLOR_TEXT_MUTED)
        _apply_glass_theme(fig, ax, f"AQI Category Breakdown ({standard})", "Percentage (%)", "Category")
        plt.tight_layout()
        return fig

    color_map = {}
    if standard == "India NAQI (CPCB)":
        for _, _, cat, color, _ in INDIA_CPCB_BREAKPOINTS:
            color_map[cat] = color
        color_map["Severe+ (>500)"] = "#581c87"
    elif standard == "US EPA AQI":
        for _, _, cat, color, _ in US_EPA_BREAKPOINTS:
            color_map[cat] = color
        color_map["Beyond Index (>500)"] = "#581c87"

    categories = list(category_df["Category"])
    percentages = list(category_df["Percentage (%)"])
    counts = list(category_df["Count"])

    bar_colors = [color_map.get(cat, COLOR_CYAN) for cat in categories]
    y_pos = np.arange(len(categories))
    
    bars = ax.barh(y_pos, percentages, color=bar_colors, edgecolor=BG_FIGURE, height=0.55, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=9, weight="bold", color=COLOR_TEXT_MAIN)
    ax.invert_yaxis()

    for bar, pct, cnt in zip(bars, percentages, counts):
        width = bar.get_width()
        ax.annotate(
            f" {pct:.1f}% ({cnt})",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            weight="bold",
            color=COLOR_TEXT_MAIN,
        )

    ax.set_xlim(0, max(percentages, default=10) * 1.25 if max(percentages, default=10) > 0 else 100)
    _apply_glass_theme(fig, ax, f"AQI Breakdown ({standard})", "Share of Observations (%)", "AQI Category")
    plt.tight_layout()
    return fig


def plot_pollutant_correlation_matrix(corr_df: pd.DataFrame) -> plt.Figure:
    """Plot a correlation heatmap matrix between AQI and pollutants."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=100)

    if corr_df.empty:
        ax.text(0.5, 0.5, "Insufficient numeric columns for correlation", ha="center", va="center", color=COLOR_TEXT_MUTED)
        _apply_glass_theme(fig, ax, "Pollutant Correlation Matrix", "", "")
        plt.tight_layout()
        return fig

    fig.patch.set_facecolor(BG_FIGURE)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor(BG_AXES)

    cax = ax.matshow(corr_df.values, cmap="magma", vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=8)

    cols = list(corr_df.columns)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8.5, weight="bold", color=COLOR_TEXT_MAIN)
    ax.set_yticklabels(cols, fontsize=8.5, weight="bold", color=COLOR_TEXT_MAIN)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr_df.values[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.4 else "#cbd5e1"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8.5, weight="bold")

    ax.set_title("Pearson Correlation Between Indicators", fontsize=11, weight="bold", color=COLOR_TEXT_MAIN, pad=16)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# NEW COMPARISON & INTERACTIVE CHARTS
# ---------------------------------------------------------------------------

def plot_dual_location_trend(
    df: pd.DataFrame,
    loc_a: str,
    loc_b: str,
) -> plt.Figure:
    """Plot head-to-head dual time series trend comparing Location A and Location B."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=100)

    if df.empty or "date" not in df.columns or "location" not in df.columns:
        ax.text(0.5, 0.5, "Required columns missing for comparison", ha="center", va="center", color=COLOR_TEXT_MUTED)
        _apply_glass_theme(fig, ax, "Location Comparison", "Date", "AQI")
        plt.tight_layout()
        return fig

    df_a = df[df["location"] == loc_a].dropna(subset=["date", "aqi"]).sort_values(by="date")
    df_b = df[df["location"] == loc_b].dropna(subset=["date", "aqi"]).sort_values(by="date")

    if df_a.empty or df_b.empty:
        ax.text(0.5, 0.5, "Insufficient records for one or both selected locations", ha="center", va="center", color=COLOR_TEXT_MUTED)
        _apply_glass_theme(fig, ax, "Location Comparison", "Date", "AQI")
        plt.tight_layout()
        return fig

    # Daily aggregation for each city to align timestamps
    agg_a = df_a.groupby(df_a["date"].dt.date)["aqi"].mean().reset_index()
    agg_b = df_b.groupby(df_b["date"].dt.date)["aqi"].mean().reset_index()

    ax.plot(agg_a["date"], agg_a["aqi"], marker="o", markersize=3.5, color=COLOR_CYAN, linewidth=2.2, label=f"{loc_a} (x̄ = {agg_a['aqi'].mean():.1f})", zorder=3)
    ax.plot(agg_b["date"], agg_b["aqi"], marker="s", markersize=3.5, color=COLOR_ROSE, linewidth=2.2, label=f"{loc_b} (x̄ = {agg_b['aqi'].mean():.1f})", zorder=3)

    ax.fill_between(agg_a["date"], agg_a["aqi"], color=COLOR_CYAN, alpha=0.1, zorder=2)
    ax.fill_between(agg_b["date"], agg_b["aqi"], color=COLOR_ROSE, alpha=0.1, zorder=2)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=20, ha="right")

    legend = ax.legend(frameon=True, facecolor=BG_FIGURE, edgecolor=COLOR_BORDER, fontsize=8.5)
    plt.setp(legend.get_texts(), color=COLOR_TEXT_MAIN)
    _apply_glass_theme(fig, ax, f"Head-to-Head AQI Trajectory: {loc_a} vs {loc_b}", "Date", "Average Daily AQI")
    plt.tight_layout()
    return fig


def plot_location_pollutants_comparison(
    df: pd.DataFrame,
    loc_a: str,
    loc_b: str,
    pollutants: List[str],
) -> plt.Figure:
    """Grouped bar chart comparing multiple pollutant concentrations between two locations."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=100)

    if not pollutants or df.empty:
        ax.text(0.5, 0.5, "No pollutant indicators selected", ha="center", va="center", color=COLOR_TEXT_MUTED)
        _apply_glass_theme(fig, ax, "Pollutant Comparison", "Pollutant", "Mean Concentration")
        plt.tight_layout()
        return fig

    sub_a = df[df["location"] == loc_a]
    sub_b = df[df["location"] == loc_b]

    means_a = [float(pd.to_numeric(sub_a[p], errors="coerce").mean()) for p in pollutants]
    means_b = [float(pd.to_numeric(sub_b[p], errors="coerce").mean()) for p in pollutants]

    x = np.arange(len(pollutants))
    width = 0.35

    rects1 = ax.bar(x - width/2, means_a, width, label=loc_a, color=COLOR_CYAN, edgecolor=BG_FIGURE, zorder=3)
    rects2 = ax.bar(x + width/2, means_b, width, label=loc_b, color=COLOR_ROSE, edgecolor=BG_FIGURE, zorder=3)

    # Annotate values
    for rect in rects1:
        h = rect.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=COLOR_TEXT_MAIN, weight="bold")

    for rect in rects2:
        h = rect.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=COLOR_TEXT_MAIN, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in pollutants], fontsize=9, weight="bold", color=COLOR_TEXT_MAIN)

    max_val = max(max(means_a, default=1), max(means_b, default=1))
    ax.set_ylim(0, max_val * 1.25)

    legend = ax.legend(frameon=True, facecolor=BG_FIGURE, edgecolor=COLOR_BORDER, fontsize=8.5)
    plt.setp(legend.get_texts(), color=COLOR_TEXT_MAIN)
    _apply_glass_theme(fig, ax, f"Mean Pollutant Profile: {loc_a} vs {loc_b}", "Pollutant Indicator", "Mean Concentration")
    plt.tight_layout()
    return fig


def plot_pollutant_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> plt.Figure:
    """Interactive scatter plot of Indicator X vs Indicator Y with fitted trendline."""
    fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=100)

    clean_subset = df[[x_col, y_col]].dropna().apply(pd.to_numeric, errors="coerce").dropna()

    if len(clean_subset) < 3:
        ax.text(0.5, 0.5, "Insufficient paired observations for scatter plot", ha="center", va="center", color=COLOR_TEXT_MUTED)
        _apply_glass_theme(fig, ax, "Cross-Correlation Scatter", x_col.upper(), y_col.upper())
        plt.tight_layout()
        return fig

    x_vals = clean_subset[x_col].values
    y_vals = clean_subset[y_col].values

    ax.scatter(x_vals, y_vals, color=COLOR_CYAN, alpha=0.6, edgecolors=COLOR_BORDER, s=36, zorder=3)

    # Compute regression line
    m, b = np.polyfit(x_vals, y_vals, 1)
    corr = float(np.corrcoef(x_vals, y_vals)[0, 1])

    x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
    y_line = m * x_line + b
    ax.plot(x_line, y_line, color=COLOR_ROSE, linestyle="--", linewidth=2, label=f"Trendline (r = {corr:.2f})", zorder=4)

    legend = ax.legend(frameon=True, facecolor=BG_FIGURE, edgecolor=COLOR_BORDER, fontsize=8.5)
    plt.setp(legend.get_texts(), color=COLOR_TEXT_MAIN)
    _apply_glass_theme(fig, ax, f"Bivariate Relationship: {x_col.upper()} vs {y_col.upper()}", f"{x_col.upper()}", f"{y_col.upper()}")
    plt.tight_layout()
    return fig
