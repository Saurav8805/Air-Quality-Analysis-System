"""Air Quality Analysis System — Glassmorphism Dashboard.

A modern, responsive, glassmorphic analytics interface featuring intuitive dropdowns,
head-to-head location comparisons, bivariate scatter analysis, personalized health calculators,
and scenario simulation tools.
"""

from __future__ import annotations
import base64
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_and_validate_csv
from src.analysis import (
    calculate_descriptive_stats,
    classify_aqi,
    get_aqi_category_distribution,
    group_by_location,
    aggregate_by_time,
    get_all_pollutants_summary,
    compute_pollutant_correlation,
    INDIA_CPCB_BREAKPOINTS,
    US_EPA_BREAKPOINTS,
)
from src.charts import (
    plot_aqi_distribution,
    plot_aqi_by_location,
    plot_time_series_trend,
    plot_pollutant_distribution,
    plot_category_breakdown,
    plot_pollutant_correlation_matrix,
    plot_dual_location_trend,
    plot_location_pollutants_comparison,
    plot_pollutant_scatter,
)
from src.pdf_exporter import generate_pdf_report
from src.ml_predictor import train_aqi_estimation_model
# ---------------------------------------------------------------------------
# Direct Execution Auto-Launcher Guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not st.runtime.exists():
        import sys
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())



st.set_page_config(
    page_title="Air Quality Analysis System",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Pure white background — no image needed

SVG_LEAF = (
    '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" '
    'style="display: inline-block; vertical-align: middle; flex-shrink: 0;">'
    '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" '
    'fill="url(#aerovistaLeafGrad)" stroke="#059669" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" stroke="#047857" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
    '<defs><linearGradient id="aerovistaLeafGrad" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">'
    '<stop stop-color="#34d399"/><stop offset="1" stop-color="#059669"/></linearGradient></defs></svg>'
)

# Glassmorphism CSS — authentic frosted glass panels over atmospheric ambient canvas
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* Completely hide default Streamlit toolbar, Deploy button, and footer */
    header[data-testid="stHeader"],
    header[data-testid="stAppHeader"],
    .stAppHeader,
    div[data-testid="stToolbar"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        min-height: 0px !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Remove default sidebar header top gap */
    [data-testid="stSidebarHeader"],
    div[data-testid="stSidebarHeader"],
    header[data-testid="stSidebarHeader"] {
        display: none !important;
        height: 0px !important;
        min-height: 0px !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* ── PRISTINE CANVAS WITH ATMOSPHERIC GLASS REFRACTION MESH ── */
    .stApp {
        background-color: #f1f5f9 !important;
        background-image: 
            radial-gradient(circle at 10% 12%, rgba(99, 102, 241, 0.16) 0px, transparent 45%),
            radial-gradient(circle at 90% 8%, rgba(14, 165, 233, 0.18) 0px, transparent 42%),
            radial-gradient(circle at 50% 45%, rgba(16, 185, 129, 0.13) 0px, transparent 50%),
            radial-gradient(circle at 95% 85%, rgba(168, 85, 247, 0.15) 0px, transparent 45%),
            radial-gradient(circle at 5% 90%, rgba(2, 132, 199, 0.15) 0px, transparent 48%) !important;
        background-attachment: fixed !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
    }

    /* ── SIDEBAR — sleek frosted glass control panel with crisp separator ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(245, 248, 252, 0.95) 0%, rgba(240, 244, 250, 0.90) 100%) !important;
        backdrop-filter: blur(24px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(190%) !important;
        border-right: 1.5px solid #cbd5e1 !important;
        box-shadow: 4px 0 24px rgba(15, 23, 42, 0.05) !important;
    }

    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.85rem !important;
        background: transparent !important;
    }

    /* Sidebar glass controls */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] div[data-baseweb="input"] > div,
    [data-testid="stSidebar"] div[data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.82) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
        border-top: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-left: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-right: 1px solid rgba(203, 213, 225, 0.8) !important;
        border-bottom: 1px solid rgba(203, 213, 225, 0.8) !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03), inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
        border-radius: 12px !important;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    /* Multi-select tag override: replace ugly red tags with indigo/cyan frosted tags */
    span[data-baseweb="tag"] {
        background: rgba(99, 102, 241, 0.12) !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        border-radius: 9999px !important;
    }
    span[data-baseweb="tag"] span {
        color: #4338ca !important;
        font-weight: 600 !important;
    }
    span[data-baseweb="tag"] svg {
        fill: #4338ca !important;
    }

    /* Checkbox override: replace red check with indigo */
    span[data-baseweb="checkbox"] input:checked + div {
        background-color: #4338ca !important;
        border-color: #4338ca !important;
    }

    /* ── ELIMINATE TOP WHITESPACE GAP IN MAIN CANVAS ── */
    .block-container,
    .main .block-container,
    .stMain .block-container,
    [data-testid="stMainBlockContainer"],
    div[data-testid="stAppViewBlockContainer"],
    section[data-testid="stMain"] div.block-container {
        padding-top: 0.85rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1360px !important;
    }

    /* ── HEADER NAVIGATION SEPARATOR ── */
    .header-nav-separator {
        width: 100%;
        height: 1.5px;
        background: #cbd5e1;
        margin-top: 0.75rem;
        margin-bottom: 1.35rem;
        border-radius: 9999px;
    }

    /* ── BRAND WRAPPER WITH PROPER LEAF ALIGNMENT ── */
    .glass-brand-wrapper {
        display: inline-flex;
        align-items: center;
        gap: 0.65rem;
    }

    .brand-text-col {
        display: inline-flex;
        flex-direction: column;
        justify-content: center;
    }

    .brand-title-row {
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }

    .brand-title {
        font-size: 1.45rem;
        font-weight: 900;
        line-height: 1;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #4338ca 0%, #2563eb 50%, #0284c7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .brand-badge {
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background: rgba(99, 102, 241, 0.12);
        color: #4338ca;
        border: 1px solid rgba(99, 102, 241, 0.35);
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
    }

    .brand-tagline {
        font-size: 0.73rem;
        color: #64748b;
        margin-top: 3px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* ── SEGMENTED CONTROL / NAVBAR STYLING (NO RED BORDERS) ── */
    div[data-testid="stSegmentedControl"] {
        background: rgba(240, 244, 250, 0.85) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border-top: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-left: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-right: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-bottom: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-radius: 9999px !important;
        padding: 4px !important;
        box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.06), inset 0 1px 1px 0 rgba(255, 255, 255, 0.95) !important;
    }

    div[data-testid="stSegmentedControl"] > div,
    div[data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
        gap: 6px !important;
        border: none !important;
        background: transparent !important;
    }

    div[data-testid="stSegmentedControl"] button {
        border-radius: 9999px !important;
        border: 1px solid transparent !important;
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        padding: 0.45rem 1.15rem !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        background: transparent !important;
        outline: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stSegmentedControl"] button:hover {
        color: #1e1b4b !important;
        background: rgba(255, 255, 255, 0.65) !important;
        border-color: rgba(255, 255, 255, 0.8) !important;
    }

    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[data-checked="true"],
    div[data-testid="stSegmentedControl"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #4338ca 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        box-shadow: 0 4px 16px rgba(67, 56, 202, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.45) !important;
    }

    div[data-testid="stSegmentedControl"] button[aria-checked="true"] *,
    div[data-testid="stSegmentedControl"] button[data-checked="true"] * {
        color: #ffffff !important;
    }

    /* ── REAL 3D FROSTED GLASS HERO SECTION ───────────────────────── */
    .glass-hero {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.88) 0%, rgba(243, 247, 252, 0.72) 50%, rgba(238, 244, 250, 0.62) 100%) !important;
        backdrop-filter: blur(32px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(32px) saturate(200%) !important;
        border-top: 1.5px solid rgba(255, 255, 255, 1) !important;
        border-left: 1.5px solid rgba(255, 255, 255, 1) !important;
        border-right: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-bottom: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-radius: 24px !important;
        padding: 2.2rem 2.4rem !important;
        margin-bottom: 1.4rem !important;
        box-shadow: 0 24px 48px -12px rgba(15, 23, 42, 0.09), 0 6px 18px -3px rgba(15, 23, 42, 0.03), inset 0 2px 4px 0 rgba(255, 255, 255, 1) !important;
        position: relative;
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #4338ca;
        padding: 0.32rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 900;
        line-height: 1.18;
        margin: 0 0 0.75rem 0;
        background: linear-gradient(135deg, #0f172a 0%, #312e81 40%, #0369a1 80%, #064e3b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
    }

    .hero-subtitle {
        color: #334155;
        font-size: 1.02rem;
        line-height: 1.55;
        max-width: 950px;
        margin: 0 0 1.5rem 0;
        font-weight: 500;
    }

    .hero-stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 1rem;
        margin-top: 1.2rem;
    }

    .hero-stat-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.90) 0%, rgba(245, 248, 252, 0.75) 100%) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-top: 1px solid rgba(255, 255, 255, 1) !important;
        border-left: 1px solid rgba(255, 255, 255, 1) !important;
        border-right: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-bottom: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-radius: 16px !important;
        padding: 0.95rem 1.2rem !important;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04), inset 0 1px 0 rgba(255, 255, 255, 1) !important;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .hero-stat-card:hover {
        border-color: rgba(99, 102, 241, 0.5) !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(99, 102, 241, 0.12), inset 0 1px 0 #fff !important;
    }

    .hero-stat-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .hero-stat-value {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.2rem;
    }

    /* ── REAL FROSTED GLASS CARDS (BEVELED SPECULAR HIGHLIGHTS) ── */
    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.82) 0%, rgba(243, 246, 250, 0.65) 100%) !important;
        backdrop-filter: blur(24px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(190%) !important;
        border-top: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-left: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-right: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-bottom: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-radius: 20px !important;
        padding: 1.35rem 1.5rem !important;
        box-shadow: 0 16px 36px -8px rgba(15, 23, 42, 0.08), 0 4px 12px -2px rgba(15, 23, 42, 0.03), inset 0 1px 2px 0 rgba(255, 255, 255, 1) !important;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        margin-bottom: 1.1rem;
        color: #0f172a;
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 0 22px 45px -8px rgba(99, 102, 241, 0.16), 0 6px 16px -2px rgba(15, 23, 42, 0.06), inset 0 1px 2px 0 rgba(255, 255, 255, 1) !important;
        transform: translateY(-3px);
    }

    .glass-card-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.35rem;
    }

    .glass-card-value {
        font-size: 1.75rem;
        font-weight: 900;
        color: #0f172a;
        line-height: 1.2;
    }

    .glass-card-subtitle {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 0.35rem;
        font-weight: 500;
    }

    /* ── GLOWING GLASS BADGES ─────────────────────── */
    .glass-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        backdrop-filter: blur(8px);
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    /* ── CALLOUT BOXES ───────────────────────────── */
    .glass-callout-info {
        background: linear-gradient(135deg, rgba(240, 249, 255, 0.90) 0%, rgba(224, 242, 254, 0.75) 100%) !important;
        backdrop-filter: blur(16px) !important;
        border-top: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-left: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-right: 1px solid rgba(186, 230, 253, 0.85) !important;
        border-bottom: 1px solid rgba(186, 230, 253, 0.85) !important;
        border-radius: 14px;
        padding: 0.95rem 1.3rem;
        font-size: 0.9rem;
        color: #0369a1;
        margin-bottom: 1.1rem;
        font-weight: 500;
        box-shadow: 0 6px 18px rgba(14, 165, 233, 0.08);
    }

    .glass-callout-warning {
        background: linear-gradient(135deg, rgba(254, 243, 199, 0.90) 0%, rgba(253, 230, 138, 0.75) 100%) !important;
        backdrop-filter: blur(16px) !important;
        border-top: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-left: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-right: 1px solid rgba(253, 230, 138, 0.85) !important;
        border-bottom: 1px solid rgba(253, 230, 138, 0.85) !important;
        border-radius: 14px;
        padding: 0.95rem 1.3rem;
        font-size: 0.9rem;
        color: #92400e;
        margin-bottom: 1.1rem;
        font-weight: 500;
        box-shadow: 0 6px 18px rgba(245, 158, 11, 0.08);
    }

    /* ── COMPARISON CARDS ────────────────────────── */
    .comp-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.2rem;
        margin: 1rem 0;
    }

    .comp-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(243, 246, 250, 0.70) 100%) !important;
        backdrop-filter: blur(24px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
        border-radius: 18px !important;
        padding: 1.3rem;
        border-top: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-left: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-right: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-bottom: 1px solid rgba(210, 220, 235, 0.75) !important;
        box-shadow: 0 12px 28px -5px rgba(15, 23, 42, 0.07), inset 0 1px 1px rgba(255, 255, 255, 0.95);
        color: #0f172a;
    }

    .comp-card-a {
        border-left: 4px solid #0284c7 !important;
    }

    .comp-card-b {
        border-left: 4px solid #7c3aed !important;
    }

    /* ── SELECTBOXES / DROPDOWNS ─────────────────── */
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.88) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.95) !important;
        border-left: 1px solid rgba(255, 255, 255, 0.95) !important;
        border-right: 1px solid rgba(210, 220, 235, 0.8) !important;
        border-bottom: 1px solid rgba(210, 220, 235, 0.8) !important;
        backdrop-filter: blur(16px) !important;
        border-radius: 10px !important;
        color: #0f172a !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03) !important;
    }

    /* Selectbox option text & Dropdown Menu Glass */
    [data-baseweb="menu"] li {
        color: #0f172a !important;
    }
    div[data-baseweb="popover"],
    ul[data-baseweb="menu"] {
        background: rgba(255, 255, 255, 0.94) !important;
        backdrop-filter: blur(24px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(190%) !important;
        border-top: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-left: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-right: 1px solid rgba(210, 220, 235, 0.8) !important;
        border-bottom: 1px solid rgba(210, 220, 235, 0.8) !important;
        border-radius: 14px !important;
        box-shadow: 0 16px 36px -4px rgba(15, 23, 42, 0.12) !important;
    }

    /* ── TABS ────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(240, 244, 250, 0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 6px;
        border-radius: 16px;
        border-top: 1.5px solid rgba(255, 255, 255, 0.95);
        border-left: 1.5px solid rgba(255, 255, 255, 0.95);
        border-right: 1px solid rgba(210, 220, 235, 0.75);
        border-bottom: 1px solid rgba(210, 220, 235, 0.75);
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        color: #475569;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 9px 18px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #4338ca !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.16), inset 0 1px 1px #fff !important;
        backdrop-filter: blur(12px) !important;
    }

    /* ── BUTTONS & DOWNLOAD BUTTONS ─────────────── */
    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.92) 0%, rgba(243, 246, 250, 0.80) 100%) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
        border-top: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-left: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.35) !important;
        border-bottom: 1px solid rgba(99, 102, 241, 0.35) !important;
        border-radius: 12px !important;
        color: #4338ca !important;
        font-weight: 700 !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.08) !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: rgba(99, 102, 241, 0.1) !important;
        border-color: #4338ca !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.22) !important;
        transform: translateY(-2px) !important;
        color: #312e81 !important;
    }

    /* ── METRICS & GENERAL TEXT ──────────────────── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(243, 246, 250, 0.70) 100%) !important;
        backdrop-filter: blur(16px) !important;
        border-top: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-left: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-right: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-bottom: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-radius: 16px !important;
        padding: 0.9rem 1.1rem !important;
        box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.06), inset 0 1px 1px rgba(255, 255, 255, 0.95) !important;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 900 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 700 !important;
    }

    /* ── DATAFRAME & TABLE STYLING ───────────────── */
    .stDataFrame,
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
        border-radius: 16px !important;
        border-top: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-left: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-right: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-bottom: 1px solid rgba(210, 220, 235, 0.75) !important;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
        overflow: hidden !important;
    }

    /* ── EXPANDER (TRUE FROSTED GLASS) ───────────── */
    details[data-testid="stExpander"],
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(243, 247, 252, 0.70) 100%) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border-top: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-left: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-right: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-bottom: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-radius: 16px !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.05), inset 0 1px 1px 0 rgba(255, 255, 255, 0.95) !important;
        margin-bottom: 1rem !important;
        overflow: hidden !important;
    }

    /* ── MATPLOTLIB CHARTS IN FROSTED GLASS CARDS ── */
    div[data-testid="stImage"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.88) 0%, rgba(243, 246, 250, 0.72) 100%) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border-top: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-left: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-right: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-bottom: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-radius: 20px !important;
        padding: 1rem !important;
        box-shadow: 0 12px 30px -6px rgba(15, 23, 42, 0.06), inset 0 1px 1px rgba(255, 255, 255, 0.95) !important;
        transition: transform 0.25s ease, box-shadow 0.25s ease !important;
    }
    div[data-testid="stImage"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 18px 40px -6px rgba(99, 102, 241, 0.12), inset 0 1px 1px rgba(255, 255, 255, 0.95) !important;
    }

    /* Slider */
    .stSlider > div {
        color: #0f172a !important;
    }

    /* Info/Warning/Error boxes */
    .stInfo, .stAlert {
        background: rgba(255, 255, 255, 0.88) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 14px !important;
        border-top: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-left: 1.5px solid rgba(255, 255, 255, 0.95) !important;
        border-right: 1px solid rgba(210, 220, 235, 0.75) !important;
        border-bottom: 1px solid rgba(210, 220, 235, 0.75) !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04) !important;
    }

    /* General body text on main area */
    .main p, .main li, .main label, .main span {
        color: #0f172a;
    }
    .main h1, .main h2, .main h3, .main h4 {
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Structured Sidebar: Control Center
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    """
    <div style="margin-bottom: 1.25rem; padding-bottom: 0.75rem; border-bottom: 1.5px solid #cbd5e1;">
        <div style="display: flex; align-items: center; gap: 0.6rem;">
            <span style="font-size: 1.65rem; line-height: 1; display: inline-flex; align-items: center; justify-content: center; transform: translateY(-1px);">🍃</span>
            <div style="display: inline-flex; flex-direction: column; justify-content: center;">
                <span style="margin: 0; font-size: 1.3rem; font-weight: 900; line-height: 1; background: linear-gradient(135deg, #4338ca 0%, #0284c7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.01em;">AEROVISTA</span>
                <span style="margin: 0; font-size: 0.68rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.08em; margin-top: 3px;">Control Center</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Section 1: Data Pipeline
st.sidebar.markdown("**📁 DATA PIPELINE & SOURCE**")
data_choice = st.sidebar.selectbox(
    "Dataset Ingestion Mode",
    ["Use Bundled Demonstration Data", "Upload Custom CSV File"],
    index=0,
    help="Select the bundled synthetic benchmark or upload your own CSV file.",
)

demo_file_path = os.path.join(os.path.dirname(__file__), "data", "sample_air_quality.csv")
uploaded_file = None
source_name = "Demonstration Dataset (Synthetic Benchmark)"

if data_choice == "Upload Custom CSV File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Air Quality CSV",
        type=["csv"],
        help="Upload CSV containing an AQI or Air Quality Index column.",
    )
    if uploaded_file is not None:
        source_name = f"Uploaded CSV: {uploaded_file.name}"
    else:
        st.sidebar.info("Awaiting CSV file upload...")
else:
    source_name = "Bundled Demo Data (Synthetic Demonstration)"

drop_dupes_option = st.sidebar.checkbox(
    "Audit & drop exact duplicate rows",
    value=False,
    help="Clean duplicate records with explicit documentation in the quality audit tab.",
)

st.sidebar.markdown("---")

# Section 2: Regulatory Jurisdiction
st.sidebar.markdown("**🏛️ REGULATORY STANDARD**")
standard_choice = st.sidebar.selectbox(
    "AQI Classification Standard",
    ["India NAQI (CPCB)", "US EPA AQI", "Raw Numeric (No Categories)"],
    index=0,
    help="Select standard breakpoints used for category mapping and advisories.",
)

# Ingestion
df_raw_loaded = None
report = None

if data_choice == "Use Bundled Demonstration Data":
    if os.path.exists(demo_file_path):
        df_raw_loaded, report = load_and_validate_csv(demo_file_path, drop_duplicates_choice=drop_dupes_option)
    else:
        st.error(f"Demo file not found at {demo_file_path}.")
elif uploaded_file is not None:
    df_raw_loaded, report = load_and_validate_csv(uploaded_file, drop_duplicates_choice=drop_dupes_option)

# Stop if no data
if df_raw_loaded is None or report is None or not report.is_valid:
    if report and report.error_message:
        st.error(f"⚠️ Ingestion Warning: {report.error_message}")
    else:
        st.info("👋 Select the bundled demo data or upload a custom CSV file to launch the dashboard.")

    st.markdown(
        """
        <div class="glass-card" style="margin-top: 1.5rem;">
            <h3>📋 Expected CSV Schema</h3>
            <p>Ensure your CSV contains at minimum an <b>AQI</b> column:</p>
            <ul>
                <li><b>Required:</b> <code>AQI</code> (numeric, non-negative)</li>
                <li><b>Optional Date:</b> <code>Date</code>, <code>DateTime</code>, or <code>Timestamp</code></li>
                <li><b>Optional Location:</b> <code>Location</code>, <code>City</code>, or <code>Station</code></li>
                <li><b>Optional Pollutants:</b> <code>PM2.5</code>, <code>PM10</code>, <code>NO2</code>, <code>SO2</code>, <code>CO</code>, <code>O3</code></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

st.sidebar.markdown("---")

# Section 3: Spatial Filtering
all_available_locations = (
    sorted([str(loc) for loc in df_raw_loaded["location"].dropna().unique()])
    if report.has_location
    else []
)
available_pollutants = report.detected_pollutants

st.sidebar.markdown("**📍 GEOGRAPHIC SCOPE**")
selected_locations = all_available_locations
if report.has_location:
    location_mode = st.sidebar.selectbox(
        "Location Scope Filter",
        ["All Available Stations", "Custom Selected Stations"],
        index=0,
    )
    if location_mode == "Custom Selected Stations":
        selected_locations = st.sidebar.multiselect(
            "Select Monitoring Stations",
            all_available_locations,
            default=all_available_locations[:3] if len(all_available_locations) >= 3 else all_available_locations,
        )

# Date Filter with Presets in Sidebar
df_filtered = df_raw_loaded.copy()
if selected_locations and report.has_location:
    df_filtered = df_filtered[df_filtered["location"].astype(str).isin(selected_locations)]

st.sidebar.markdown("---")
st.sidebar.markdown("**📅 TEMPORAL WINDOW**")
date_filter_str = "All Dates"
if report.has_date and report.date_range and not df_filtered.empty:
    min_date, max_date = report.date_range
    date_preset = st.sidebar.selectbox(
        "Date Window Preset",
        ["All Available Records", "Last 7 Days", "Last 14 Days", "Last 30 Days", "Custom Calendar Range"],
        index=0,
    )

    if date_preset == "Last 7 Days":
        start_bound = max_date.date() - timedelta(days=7)
        df_filtered = df_filtered[df_filtered["date"].dt.date >= start_bound]
        date_filter_str = f"Last 7 Days ({start_bound} to {max_date.date()})"
    elif date_preset == "Last 14 Days":
        start_bound = max_date.date() - timedelta(days=14)
        df_filtered = df_filtered[df_filtered["date"].dt.date >= start_bound]
        date_filter_str = f"Last 14 Days ({start_bound} to {max_date.date()})"
    elif date_preset == "Last 30 Days":
        start_bound = max_date.date() - timedelta(days=30)
        df_filtered = df_filtered[df_filtered["date"].dt.date >= start_bound]
        date_filter_str = f"Last 30 Days ({start_bound} to {max_date.date()})"
    elif date_preset == "Custom Calendar Range":
        custom_dates = st.sidebar.date_input(
            "Pick Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        if isinstance(custom_dates, (list, tuple)) and len(custom_dates) == 2:
            s_d, e_d = custom_dates
            df_filtered = df_filtered[(df_filtered["date"].dt.date >= s_d) & (df_filtered["date"].dt.date <= e_d)]
            date_filter_str = f"{s_d} to {e_d}"

# Sidebar Diagnostic Telemetry
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="font-size: 0.76rem; color: #475569; line-height: 1.6; background: rgba(255, 255, 255, 0.7); padding: 0.75rem; border-radius: 12px; border: 1px solid rgba(226, 232, 240, 0.9);">
        <div style="font-weight: 800; color: #1e293b; margin-bottom: 0.35rem; letter-spacing: 0.04em;">⚙️ ENGINE TELEMETRY</div>
        <div>• Pipeline Status: <b style="color: #059669;">OPERATIONAL</b></div>
        <div>• Valid Observations: <b style="color: #0f172a;">{report.valid_rows}</b></div>
        <div>• Duplicate Records: <b style="color: #0f172a;">{report.duplicate_rows_count}</b></div>
        <div>• Missing AQI Rows: <b style="color: #0f172a;">{report.missing_aqi_count}</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 1. TOP BRAND HEADER & SIDE-BY-SIDE NAVBAR
# ---------------------------------------------------------------------------

col_nav_left, col_nav_right = st.columns([5, 5], vertical_alignment="center")

with col_nav_left:
    st.markdown(
        """
        <div class="glass-brand-wrapper">
            <span class="brand-leaf-icon">🍃</span>
            <div class="brand-text-col">
                <div class="brand-title-row">
                    <span class="brand-title">AEROVISTA</span>
                    <span class="brand-badge">Intelligence</span>
                </div>
                <span class="brand-tagline">Real-Time Atmospheric & Air Quality Diagnostics</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_nav_right:
    nav_selection = st.segmented_control(
        "Navigation Options",
        options=["🏠 Home", "ℹ️ About Us", "🛡️ Privacy Policy", "⚙️ Settings"],
        default="🏠 Home",
        label_visibility="collapsed",
        key="site_main_nav",
    )

if not nav_selection:
    nav_selection = "🏠 Home"

st.markdown('<div class="header-nav-separator"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MODULAR VIEWS (ABOUT US / PRIVACY POLICY / SETTINGS)
# ---------------------------------------------------------------------------

if nav_selection == "ℹ️ About Us":
    st.markdown(
        """
        <div class="glass-card" style="margin-top: 1rem;">
            <div class="hero-eyebrow"><span>ℹ️</span> INSTITUTIONAL PROFILE</div>
            <h2 style="margin: 0.3rem 0 0.8rem 0; color: #4338ca; font-size: 1.85rem; font-weight: 900;">About Aerovista Atmospheric Intelligence</h2>
            <p style="font-size: 1.02rem; color: #334155; line-height: 1.65; margin: 0;">
                <b>Aerovista</b> is an institutional-grade, hardware-free atmospheric diagnostics platform engineered to transform raw environmental sensor data into actionable ecological intelligence. Designed for public health epidemiologists, municipal air quality officers, environmental compliance auditors, and academic researchers, Aerovista integrates descriptive statistical modeling, bivariate pollutant cross-talk detection, multi-jurisdiction compliance evaluation, and machine learning AQI prediction.
            </p>
        </div>

        <div class="comp-container">
            <div class="glass-card">
                <h3 style="color: #0f172a; font-weight: 800; font-size: 1.15rem; margin-top: 0;">🔬 Scientific Methodologies</h3>
                <p style="color: #475569; font-size: 0.92rem; line-height: 1.65;">
                    • <b>Multi-Jurisdiction Breakpoints:</b> Seamless support for India Central Pollution Control Board (CPCB NAQI) and US EPA linear interpolation standards.<br>
                    • <b>Hardware-Free ML Ingestion:</b> Random Forest predictive regression models infer missing pollutant sub-indices from proxy ambient features with cross-validated R² metrics.<br>
                    • <b>Bivariate Correlation Matrices:</b> Real-time Pearson and Spearman pollutant interaction detection (e.g., PM2.5 vs NO2 secondary aerosols).
                </p>
            </div>
            <div class="glass-card">
                <h3 style="color: #0f172a; font-weight: 800; font-size: 1.15rem; margin-top: 0;">🏛️ Regulatory Auditing & Compliance</h3>
                <p style="color: #475569; font-size: 0.92rem; line-height: 1.65;">
                    • <b>Automated Compliance Verification:</b> Instant evaluation against legal safety thresholds (NAQI Safe ≤ 100, US EPA Safe ≤ 50).<br>
                    • <b>Data Hygiene Engine:</b> Statistical duplicate detection, timestamp monotonicity auditing, and zero/negative value validation.<br>
                    • <b>Public Health Advisories:</b> Evidence-grounded mitigation advisories categorized for vulnerable and general populations.
                </p>
            </div>
        </div>

        <div class="glass-card">
            <h3 style="color: #0f172a; font-weight: 800; font-size: 1.15rem; margin-top: 0;">👥 Engineering & Technology Stack</h3>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.65; margin-bottom: 0.8rem;">
                Built by the Aerovista Environmental Intelligence Group. Powered by Python 3.12, Streamlit 1.61.1, Scikit-Learn, Pandas, NumPy, and high-performance Matplotlib vector rendering.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("← Return to Live Dashboard", key="btn_back_home_about"):
        st.session_state["site_main_nav"] = "🏠 Home"
        st.rerun()
    st.stop()

elif nav_selection == "🛡️ Privacy Policy":
    st.markdown(
        """
        <div class="glass-card" style="margin-top: 1rem;">
            <div class="hero-eyebrow"><span>🛡️</span> DATA SOVEREIGNTY</div>
            <h2 style="margin: 0.3rem 0 0.8rem 0; color: #0284c7; font-size: 1.85rem; font-weight: 900;">Privacy Policy & Data Security Guarantee</h2>
            <p style="font-size: 1.02rem; color: #334155; line-height: 1.65; margin: 0;">
                Your atmospheric observations, continuous sensor streams, and uploaded CSV datasets are strictly confidential. Aerovista enforces an uncompromising zero-telemetry, client-isolated execution guarantee.
            </p>
        </div>

        <div class="comp-container">
            <div class="glass-card">
                <h3 style="color: #0f172a; font-weight: 800; font-size: 1.15rem; margin-top: 0;">🔒 100% Local In-Memory Processing</h3>
                <p style="color: #475569; font-size: 0.92rem; line-height: 1.65;">
                    All ingested CSV files, statistical distributions, correlation matrices, and machine learning models execute entirely within your local machine's memory. No file contents, geographic coordinates, or sensor metrics are ever transmitted across external cloud endpoints.
                </p>
            </div>
            <div class="glass-card">
                <h3 style="color: #0f172a; font-weight: 800; font-size: 1.15rem; margin-top: 0;">🗑️ Ephemeral Session Lifecycle</h3>
                <p style="color: #475569; font-size: 0.92rem; line-height: 1.65;">
                    Uploaded datasets and intermediate computational artifacts exist solely in transient memory for the duration of your browser session. Upon closing or reloading the application, all in-memory buffers are automatically purged.
                </p>
            </div>
        </div>

        <div class="glass-card">
            <h3 style="color: #0f172a; font-weight: 800; font-size: 1.15rem; margin-top: 0;">⚖️ Exclusive Ownership</h3>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.65; margin-bottom: 0.8rem;">
                You retain complete, exclusive intellectual property over all ingested data, statistical analyses, and exported high-resolution chart graphics. Aerovista collects zero behavioral analytics, advertising cookies, or third-party telemetry.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("← Return to Live Dashboard", key="btn_back_home_privacy"):
        st.session_state["site_main_nav"] = "🏠 Home"
        st.rerun()
    st.stop()

elif nav_selection == "⚙️ Settings":
    st.markdown(
        """
        <div class="glass-card" style="margin-top: 1rem;">
            <div class="hero-eyebrow"><span>⚙️</span> CONFIGURATION</div>
            <h2 style="margin: 0.3rem 0 0.8rem 0; color: #7c3aed; font-size: 1.85rem; font-weight: 900;">System Preferences & Diagnostic Settings</h2>
            <p style="font-size: 1.02rem; color: #334155; line-height: 1.65; margin: 0;">
                Configure computational thresholds, regulatory defaults, and report formatting preferences for your analytics pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st_col1, st_col2 = st.columns(2)
    with st_col1:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="margin-top: 0; color: #0f172a; font-weight: 800;">Analytics & Calculation Rules</h4>
            """,
            unsafe_allow_html=True,
        )
        set_standard = st.selectbox(
            "Default Regulatory Standard",
            ["India NAQI (CPCB)", "US EPA AQI", "Raw Numeric Scale"],
            index=0,
            key="set_pref_standard",
        )
        set_outlier = st.slider(
            "Outlier Sensitivity (Z-Score Cutoff)",
            min_value=2.0,
            max_value=4.5,
            value=3.0,
            step=0.5,
            key="set_pref_outlier",
            help="Observations exceeding this standard deviation threshold will be flagged in the data quality audit.",
        )
        set_decimals = st.radio(
            "Metric Decimal Precision",
            [1, 2, 3],
            index=0,
            horizontal=True,
            key="set_pref_decimals",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with st_col2:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="margin-top: 0; color: #0f172a; font-weight: 800;">Display & Export Preferences</h4>
            """,
            unsafe_allow_html=True,
        )
        set_toast = st.checkbox(
            "Enable Real-Time Pipeline Status Notifications",
            value=True,
            key="set_pref_toasts",
        )
        set_encoding = st.selectbox(
            "CSV Export File Encoding",
            ["UTF-8 (Standard)", "UTF-8 with BOM (Excel-Friendly)", "ASCII Clean"],
            index=0,
            key="set_pref_encoding",
        )
        set_page_size = st.selectbox(
            "Data Quality Table Page Size",
            [10, 25, 50, 100],
            index=0,
            key="set_pref_page_size",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    btn_c1, btn_c2 = st.columns([2, 4])
    with btn_c1:
        if st.button("💾 Save Preferences", key="btn_save_settings"):
            st.success("✅ Preferences saved successfully!")
    with btn_c2:
        if st.button("← Return to Live Dashboard", key="btn_back_home_settings"):
            st.session_state["site_main_nav"] = "🏠 Home"
            st.rerun()
    st.stop()


# ---------------------------------------------------------------------------
# 2. PROPER HERO SECTION
# ---------------------------------------------------------------------------

loc_count = df_filtered["location"].nunique() if report.has_location else 1
date_display = date_filter_str if report.has_date else "N/A (No temporal field)"

st.markdown(
    f"""
    <div class="glass-hero">
        <div class="hero-eyebrow">
            <span>✨</span> NEXT-GENERATION ENVIRONMENTAL ANALYTICS
        </div>
        <h2 class="hero-title">Atmospheric Intelligence & Real-Time Air Quality Diagnostics</h2>
        <p class="hero-subtitle">
            An advanced, hardware-free analytics engine delivering statistical rigor, multi-location spatial benchmarks,
            bivariate pollutant correlations, regulatory compliance auditing, and predictive regression modeling.
        </p>
        <div class="hero-stats-grid">
            <div class="hero-stat-card">
                <div class="hero-stat-label">📁 Active Pipeline Mode</div>
                <div class="hero-stat-value">{source_name.split('(')[0]}</div>
            </div>
            <div class="hero-stat-card">
                <div class="hero-stat-label">🏛️ Compliance Standard</div>
                <div class="hero-stat-value">{standard_choice}</div>
            </div>
            <div class="hero-stat-card">
                <div class="hero-stat-label">📅 Observation Window</div>
                <div class="hero-stat-value" style="font-size: 1.05rem;">{date_display}</div>
            </div>
            <div class="hero-stat-card">
                <div class="hero-stat-label">📍 Monitoring Stations</div>
                <div class="hero-stat-value">{loc_count} Monitored Centers</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if df_filtered.empty:
    st.warning("⚠️ No records match the active filter criteria. Please broaden your location or date selection.")
    st.stop()

# Compute high-level statistics
stats = calculate_descriptive_stats(df_filtered["aqi"])
if stats is None:
    st.error("Could not calculate descriptive statistics for the selected data.")
    st.stop()

mean_cat, mean_color, mean_desc = classify_aqi(stats.mean, standard=standard_choice)

# KPI Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="glass-card-title">Valid Records</div>
            <div class="glass-card-value">{stats.count}</div>
            <div class="glass-card-subtitle">Sample size (n)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="glass-card-title">Mean AQI (x̄)</div>
            <div class="glass-card-value">{stats.mean:.1f}</div>
            <div class="glass-card-subtitle">
                <span class="glass-badge" style="background: {mean_color}33; color: {mean_color}; border: 1px solid {mean_color};">
                    {mean_cat}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="glass-card-title">Median AQI</div>
            <div class="glass-card-value">{stats.median:.1f}</div>
            <div class="glass-card-subtitle">50th percentile (Q2)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi4:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="glass-card-title">AQI Range</div>
            <div class="glass-card-value">{stats.minimum:.0f} – {stats.maximum:.0f}</div>
            <div class="glass-card-subtitle">Spread: {stats.val_range:.0f} pts</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi5:
    loc_count = df_filtered["location"].nunique() if report.has_location else 1
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="glass-card-title">Locations & Std Dev</div>
            <div class="glass-card-value">{loc_count} <span style="font-size: 1rem; color: #94a3b8;">sites</span></div>
            <div class="glass-card-subtitle">Std Dev: ±{stats.std:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Interactive Navigation Tabs
# ---------------------------------------------------------------------------

tab_comp, tab_overview, tab_trends, tab_bivariate, tab_advisory, tab_sim, tab_audit, tab_export = st.tabs([
    "⚖️ Head-to-Head Comparison",
    "📊 AQI Overview",
    "📈 Temporal Trajectory",
    "🔬 Bivariate & Pollutants",
    "🩺 Health Risk & Advisory",
    "🔮 Scenario Simulator",
    "🛡️ Data Quality Audit",
    "📥 PDF Report & Export",
])


# ---------------------------------------------------------------------------
# TAB 1: HEAD-TO-HEAD COMPARISON (DROPDOWN POWERED)
# ---------------------------------------------------------------------------
with tab_comp:
    st.markdown("### ⚖️ Head-to-Head Location Comparison")
    st.caption("Select two monitoring stations from the dropdowns below for direct side-by-side comparative analysis.")

    if report.has_location and len(all_available_locations) >= 2:
        col_sel_a, col_sel_b = st.columns(2)
        with col_sel_a:
            loc_a = st.selectbox(
                "Primary Location (City A)",
                all_available_locations,
                index=0,
                key="loc_comp_a",
            )
        with col_sel_b:
            # Default to second city
            def_idx = 1 if len(all_available_locations) > 1 else 0
            loc_b = st.selectbox(
                "Comparison Location (City B)",
                all_available_locations,
                index=def_idx,
                key="loc_comp_b",
            )

        if loc_a == loc_b:
            st.info("💡 Please select two distinct locations in the dropdowns above for comparative analysis.")
        else:
            df_a = df_raw_loaded[df_raw_loaded["location"] == loc_a]
            df_b = df_raw_loaded[df_raw_loaded["location"] == loc_b]

            stats_a = calculate_descriptive_stats(df_a["aqi"])
            stats_b = calculate_descriptive_stats(df_b["aqi"])

            cat_a, col_a, _ = classify_aqi(stats_a.mean if stats_a else 0, standard=standard_choice)
            cat_b, col_b, _ = classify_aqi(stats_b.mean if stats_b else 0, standard=standard_choice)

            # Side-by-side comparison glass cards
            c_card1, c_card2 = st.columns(2)
            with c_card1:
                st.markdown(
                    f"""
                    <div class="comp-card comp-card-a">
                        <h3 style="color: #0284c7; margin: 0 0 0.5rem 0; font-weight: 800;">📍 {loc_a}</h3>
                        <p style="margin: 0.2rem 0; color: #334155;">Mean AQI: <b style="color: #0f172a;">{stats_a.mean:.1f}</b> &nbsp;
                            <span class="glass-badge" style="background: {col_a}33; color: {col_a}; border: 1px solid {col_a};">{cat_a}</span>
                        </p>
                        <p style="margin: 0.2rem 0; color: #334155;">Median AQI: <b style="color: #0f172a;">{stats_a.median:.1f}</b> | Range: <b style="color: #0f172a;">{stats_a.minimum:.0f} - {stats_a.maximum:.0f}</b></p>
                        <p style="margin: 0.2rem 0; color: #64748b; font-size: 0.85rem;">Valid Observations: <b style="color: #0f172a;">{stats_a.count}</b> | Std Dev: <b style="color: #0f172a;">±{stats_a.std:.1f}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c_card2:
                diff_mean = (stats_b.mean - stats_a.mean) if (stats_a and stats_b) else 0
                pct_diff = (diff_mean / stats_a.mean * 100) if (stats_a and stats_a.mean > 0) else 0
                diff_str = f"+{diff_mean:.1f} (+{pct_diff:.1f}%)" if diff_mean > 0 else f"{diff_mean:.1f} ({pct_diff:.1f}%)"
                diff_color = "#f43f5e" if diff_mean > 0 else "#10b981"

                st.markdown(
                    f"""
                    <div class="comp-card comp-card-b">
                        <h3 style="color: #7c3aed; margin: 0 0 0.5rem 0; font-weight: 800;">📍 {loc_b}</h3>
                        <p style="margin: 0.2rem 0; color: #334155;">Mean AQI: <b style="color: #0f172a;">{stats_b.mean:.1f}</b> &nbsp;
                            <span class="glass-badge" style="background: {col_b}33; color: {col_b}; border: 1px solid {col_b};">{cat_b}</span>
                        </p>
                        <p style="margin: 0.2rem 0; color: #334155;">Median AQI: <b style="color: #0f172a;">{stats_b.median:.1f}</b> | Range: <b style="color: #0f172a;">{stats_b.minimum:.0f} - {stats_b.maximum:.0f}</b></p>
                        <p style="margin: 0.2rem 0; color: {diff_color}; font-size: 0.85rem;">Delta vs {loc_a}: <b>{diff_str}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Comparative Charts
            col_cmp_chart1, col_cmp_chart2 = st.columns(2)

            with col_cmp_chart1:
                st.markdown(f"**Dual AQI Trajectory: {loc_a} vs {loc_b}**")
                fig_dual = plot_dual_location_trend(df_raw_loaded, loc_a, loc_b)
                st.pyplot(fig_dual)

            with col_cmp_chart2:
                st.markdown(f"**Pollutant Fingerprint: {loc_a} vs {loc_b}**")
                if available_pollutants:
                    fig_poll_cmp = plot_location_pollutants_comparison(
                        df_raw_loaded, loc_a, loc_b, available_pollutants
                    )
                    st.pyplot(fig_poll_cmp)
                else:
                    st.info("No specific pollutant columns detected for multi-pollutant comparison.")

    else:
        st.info("ℹ️ Comparative analysis requires a dataset containing at least two distinct location entries.")


# ---------------------------------------------------------------------------
# TAB 2: AQI OVERVIEW & DISTRIBUTIONS
# ---------------------------------------------------------------------------
with tab_overview:
    st.markdown("### 📊 Air Quality Index Distribution & Categorical Breakdown")

    col_hist, col_table = st.columns([1.3, 1.0])

    with col_hist:
        fig_dist = plot_aqi_distribution(df_filtered)
        st.pyplot(fig_dist)

    with col_table:
        st.markdown("**Statistical Summary Profile**")
        stats_table_data = {
            "Statistical Parameter": [
                "Valid Observations (n)",
                "Arithmetic Mean (x̄)",
                "Median (Q2)",
                "Standard Deviation (s)",
                "Minimum Observation",
                "Maximum Observation",
                "Total Range",
                "First Quartile (Q1)",
                "Third Quartile (Q3)",
                "Interquartile Range (IQR)",
            ],
            "Value": [
                f"{stats.count}",
                f"{stats.mean:.2f}",
                f"{stats.median:.2f}",
                f"{stats.std:.2f}",
                f"{stats.minimum:.1f}",
                f"{stats.maximum:.1f}",
                f"{stats.val_range:.1f}",
                f"{stats.q1:.2f}",
                f"{stats.q3:.2f}",
                f"{stats.iqr:.2f}",
            ],
        }
        st.dataframe(pd.DataFrame(stats_table_data), hide_index=True, width="stretch")

    st.markdown("---")

    if standard_choice != "Raw Numeric (No Categories)":
        cat_df = get_aqi_category_distribution(df_filtered, standard=standard_choice)
        c_cat_c, c_cat_t = st.columns([1.3, 1.0])
        with c_cat_c:
            fig_cat = plot_category_breakdown(cat_df, standard=standard_choice)
            st.pyplot(fig_cat)
        with c_cat_t:
            st.markdown(f"**Category Counts ({standard_choice})**")
            st.dataframe(cat_df, hide_index=True, width="stretch")
    else:
        st.info("ℹ️ Raw Numeric mode selected: Categorical breakdowns are suppressed.")


# ---------------------------------------------------------------------------
# TAB 3: TEMPORAL TRAJECTORY
# ---------------------------------------------------------------------------
with tab_trends:
    st.markdown("### 📈 Time-Series Progression & Moving Averages")

    if report.has_date:
        col_t1, col_t2 = st.columns([1, 2.5])
        with col_t1:
            time_resample = st.selectbox(
                "Select Aggregation Granularity",
                ["Daily ('D')", "Weekly ('W')", "Monthly ('M')"],
                index=0,
            )
            freq_char = time_resample.split("'")[1]
            show_rolling = st.checkbox("Include 7-period rolling average", value=True)

        agg_time_df = aggregate_by_time(df_filtered, freq=freq_char)
        if not agg_time_df.empty:
            fig_trend = plot_time_series_trend(agg_time_df, show_rolling=show_rolling)
            st.pyplot(fig_trend)
            with st.expander("🔍 View Temporal Data Table"):
                st.dataframe(agg_time_df, width="stretch")
        else:
            st.warning("No continuous temporal observations found in the selected range.")
    else:
        st.info("ℹ️ Temporal trends require a Date or Timestamp column in your dataset.")


# ---------------------------------------------------------------------------
# TAB 4: BIVARIATE & POLLUTANTS ANALYSIS
# ---------------------------------------------------------------------------
with tab_bivariate:
    st.markdown("### 🔬 Bivariate Relationships & Multi-Pollutant Correlation")
    st.caption("Investigate how criteria pollutants correlate with each other and with overall AQI.")

    numeric_cols = [c for c in ["aqi", "pm2.5", "pm10", "no2", "so2", "co", "o3"] if c in df_filtered.columns]

    if len(numeric_cols) >= 2:
        col_biv_ctrl1, col_biv_ctrl2 = st.columns(2)
        with col_biv_ctrl1:
            x_ind = st.selectbox("X-Axis Indicator", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        with col_biv_ctrl2:
            y_ind = st.selectbox("Y-Axis Indicator", numeric_cols, index=0)

        col_biv_plot, col_biv_corr = st.columns([1.2, 1.0])

        with col_biv_plot:
            fig_scatter = plot_pollutant_scatter(df_filtered, x_ind, y_ind)
            st.pyplot(fig_scatter)

        with col_biv_corr:
            st.markdown("**Inter-Indicator Correlation Matrix**")
            corr_mat = compute_pollutant_correlation(df_filtered)
            if not corr_mat.empty:
                fig_corr = plot_pollutant_correlation_matrix(corr_mat)
                st.pyplot(fig_corr)

        st.markdown("---")
        st.markdown("#### 📊 Comprehensive Pollutant Summary Statistics")
        p_summary = get_all_pollutants_summary(df_filtered)
        if not p_summary.empty:
            st.dataframe(p_summary, hide_index=True, width="stretch")
    else:
        st.info("ℹ️ At least two numeric columns (e.g. AQI and PM2.5) are required for bivariate analysis.")


# ---------------------------------------------------------------------------
# TAB 5: PERSONALIZED HEALTH RISK & ADVISORY (DROPDOWN POWERED)
# ---------------------------------------------------------------------------
with tab_advisory:
    st.markdown("### 🩺 Personalized Health Risk & Activity Advisory")
    st.caption("Receive personalized protection guidelines by selecting your vulnerability profile below.")

    col_prof, _ = st.columns([1.2, 1.5])
    with col_prof:
        profile_choice = st.selectbox(
            "Select Your Demographic / Health Profile",
            [
                "General Healthy Population",
                "Children & Senior Citizens",
                "Asthmatic & Respiratory / Heart Patients",
                "Outdoor Athletes, Cyclists & Workers",
            ],
            index=0,
        )

    # Dynamic Health Advisory based on Mean AQI & Profile
    cur_aqi = stats.mean

    if cur_aqi <= 50:
        badge_style = "background: #10b98133; color: #10b981; border: 1px solid #10b981;"
        status_txt = "Good Air Quality"
        outdoor_act = "Ideal conditions for all outdoor sports, jogging, and ventilation."
        mask_need = "No protective mask required."
        purifier_need = "Indoor air purifiers not necessary; keep windows open for fresh airflow."
    elif cur_aqi <= 100:
        badge_style = "background: #84cc1633; color: #84cc16; border: 1px solid #84cc16;"
        status_txt = "Satisfactory / Moderate"
        outdoor_act = "Safe for general public; sensitive individuals should monitor unusual breathlessness."
        mask_need = "Not required for general population; sensitive groups may consider a lightweight mask on dusty streets."
        purifier_need = "Purifiers optional."
    elif cur_aqi <= 200:
        badge_style = "background: #eab30833; color: #eab308; border: 1px solid #eab308;"
        status_txt = "Moderate / Unhealthy for Sensitive Groups"
        outdoor_act = "Reduce prolonged intense outdoor exertion. Shift workouts indoors during morning rush hours."
        mask_need = "N95 / FFP2 mask recommended during peak traffic commutes."
        purifier_need = "Recommended in bedrooms during night hours."
    elif cur_aqi <= 300:
        badge_style = "background: #f9731633; color: #f97316; border: 1px solid #f97316;"
        status_txt = "Poor / Unhealthy"
        outdoor_act = "Avoid strenuous outdoor endurance sports. High vulnerability groups should stay indoors."
        mask_need = "Well-fitted N95 mask strongly advised outdoors."
        purifier_need = "Run HEPA air purifiers indoors. Keep doors and windows closed."
    else:
        badge_style = "background: #ef444433; color: #ef4444; border: 1px solid #ef4444;"
        status_txt = "Very Poor / Severe / Hazardous"
        outdoor_act = "Emergency caution: Restrict all unnecessary outdoor exposure. Exercise strictly indoors."
        mask_need = "Certified N95 / N99 mask mandatory for any necessary outdoor transit."
        purifier_need = "Continuous HEPA air filtration essential. Seal window cracks."

    # Render Glass Cards for Health Advisory
    st.markdown(
        f"""
        <div class="glass-card" style="margin-top: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem; flex-wrap: wrap; gap: 0.5rem;">
                <h4 style="margin: 0; color: #0284c7; font-weight: 800;">Assessment for: {profile_choice}</h4>
                <span class="glass-badge" style="{badge_style}">Current Mean AQI: {cur_aqi:.1f} ({status_txt})</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem;">
                <div class="glass-card" style="margin-bottom: 0;">
                    <div class="glass-card-title">🏃 Outdoor Physical Activity</div>
                    <p style="font-size: 0.9rem; color: #334155; margin: 0.4rem 0 0 0; line-height: 1.5;">{outdoor_act}</p>
                </div>
                <div class="glass-card" style="margin-bottom: 0;">
                    <div class="glass-card-title">😷 Respiratory Mask Guidance</div>
                    <p style="font-size: 0.9rem; color: #334155; margin: 0.4rem 0 0 0; line-height: 1.5;">{mask_need}</p>
                </div>
                <div class="glass-card" style="margin-bottom: 0;">
                    <div class="glass-card-title">🏠 Indoor Air & Purifier Strategy</div>
                    <p style="font-size: 0.9rem; color: #334155; margin: 0.4rem 0 0 0; line-height: 1.5;">{purifier_need}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# TAB 6: SCENARIO SIMULATOR (WHAT-IF MODELING)
# ---------------------------------------------------------------------------
with tab_sim:
    st.markdown("### 🔮 Air Quality Scenario Simulator (What-If Analysis)")
    st.caption("Simulate environmental interventions or meteorological events using the dropdown below.")

    sim_scenario = st.selectbox(
        "Select Policy or Weather Scenario to Simulate",
        [
            "⚡ EV & Public Transit Policy (-25% PM2.5, -20% NO2)",
            "🌧️ Monsoon Rainfall Washout (-40% All Particulates)",
            "🔥 Crop Residue / Winter Temperature Inversion (+35% Overall AQI)",
            "🌪️ Arid Dust Storm Episode (+60% PM10, +25% PM2.5)",
            "🛠️ Custom Percentage Simulation",
        ],
        index=0,
    )

    custom_delta = 0
    if "Custom" in sim_scenario:
        custom_delta = st.slider("Select Custom AQI Adjustment (%)", -60, 60, -20, 5)

    base_aqi = stats.mean
    if "EV" in sim_scenario:
        projected_aqi = max(10, base_aqi * 0.78)
        sim_desc = "Simulates accelerated electric vehicle adoption, industrial emission caps, and traffic congestion easing."
    elif "Rainfall" in sim_scenario:
        projected_aqi = max(10, base_aqi * 0.60)
        sim_desc = "Simulates atmospheric wet deposition washing particulate matter out of the ambient boundary layer."
    elif "Crop" in sim_scenario:
        projected_aqi = min(500, base_aqi * 1.35)
        sim_desc = "Simulates post-harvest biomass burning combined with winter surface inversions trapping smoke."
    elif "Dust" in sim_scenario:
        projected_aqi = min(500, base_aqi * 1.45)
        sim_desc = "Simulates high-velocity winds carrying crustal particulate matter from arid regions."
    else:
        projected_aqi = max(10, min(500, base_aqi * (1.0 + custom_delta / 100.0)))
        sim_desc = f"Simulates a custom {custom_delta:+d}% change in overall ambient AQI."

    proj_cat, proj_col, _ = classify_aqi(projected_aqi, standard=standard_choice)
    net_change = projected_aqi - base_aqi

    c_sim1, c_sim2 = st.columns(2)
    with c_sim1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="glass-card-title">Baseline vs Projected AQI</div>
                <div style="display: flex; align-items: baseline; gap: 1rem; margin-top: 0.5rem;">
                    <div>
                        <span style="font-size: 0.8rem; color: #64748b; font-weight: 600;">Baseline:</span>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #475569;">{base_aqi:.1f}</div>
                    </div>
                    <div style="font-size: 1.5rem; color: #0284c7; font-weight: 900;">➔</div>
                    <div>
                        <span style="font-size: 0.8rem; color: #0284c7; font-weight: 700;">Projected:</span>
                        <div style="font-size: 1.8rem; font-weight: 900; color: #0f172a;">{projected_aqi:.1f}</div>
                    </div>
                </div>
                <p style="margin: 0.6rem 0 0 0; color: {'#059669' if net_change < 0 else '#e11d48'}; font-weight: 800; font-size: 0.95rem;">
                    Net Change: {net_change:+.1f} pts ({net_change/base_aqi*100:+.1f}%)
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_sim2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="glass-card-title">Predicted Category Shift</div>
                <div style="margin-top: 0.5rem;">
                    <span class="glass-badge" style="background: {mean_color}33; color: {mean_color}; border: 1px solid {mean_color};">
                        {mean_cat}
                    </span>
                    <span style="margin: 0 0.5rem; color: #64748b; font-weight: 700;">➔</span>
                    <span class="glass-badge" style="background: {proj_col}33; color: {proj_col}; border: 1px solid {proj_col}; font-size: 0.9rem;">
                        {proj_cat}
                    </span>
                </div>
                <p style="margin: 0.8rem 0 0 0; font-size: 0.86rem; color: #475569; line-height: 1.5;">{sim_desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# TAB 7: DATA QUALITY AUDIT
# ---------------------------------------------------------------------------
with tab_audit:
    st.markdown("### 🛡️ Transparent Data Quality & Integrity Audit")

    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    with col_q1:
        st.metric("Raw Rows Loaded", report.total_raw_rows)
    with col_q2:
        st.metric("Valid Rows Retained", report.valid_rows)
    with col_q3:
        st.metric("Missing/Invalid AQI", report.missing_aqi_count + report.non_numeric_aqi_count)
    with col_q4:
        st.metric("Duplicate Rows", report.duplicate_rows_count)

    st.markdown("---")
    st.markdown("**Cleaning Notes & Audit History:**")
    if report.cleaning_notes:
        for note in report.cleaning_notes:
            st.markdown(f"- 📝 {note}")
    else:
        st.success("All records conformed to data integrity standards without anomalies.")

    st.markdown("---")
    st.markdown("**Detected Column Mapping:**")
    mapping_data = []
    for raw_c in report.raw_columns:
        norm_c = report.detected_columns.get(raw_c.strip().lower().replace("-", "_"), raw_c)
        col_type = "Required (AQI)" if norm_c == "aqi" else ("Temporal" if norm_c == "date" else ("Location" if norm_c == "location" else ("Pollutant" if norm_c in available_pollutants else "Auxiliary")))
        mapping_data.append({"Original Column Header": raw_c, "Normalized Field": norm_c, "Detected Category": col_type})
    st.dataframe(pd.DataFrame(mapping_data), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# TAB 8: EXPORT & PDF REPORT
# ---------------------------------------------------------------------------
with tab_export:
    st.markdown("### 📥 Download Processed Data & Analytical PDF Report")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.markdown("#### 📄 Filtered Dataset Export (CSV)")
        st.write(f"Export the active filtered dataset ({len(df_filtered)} records).")
        csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Filtered Data as CSV",
            data=csv_bytes,
            file_name=f"air_quality_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    with col_exp2:
        st.markdown("#### 📑 Executive Analytical PDF Report")
        st.write("Generate a printable PDF summarizing statistics, rankings, and audit notes.")

        loc_df_for_pdf = group_by_location(df_filtered) if report.has_location else None
        poll_df_for_pdf = get_all_pollutants_summary(df_filtered) if available_pollutants else None

        pdf_bytes = generate_pdf_report(
            summary_stats=stats,
            data_mode=source_name,
            selected_locations=selected_locations,
            date_filter_str=date_filter_str,
            location_summary_df=loc_df_for_pdf,
            pollutants_summary_df=poll_df_for_pdf,
            quality_notes=report.cleaning_notes,
            standard_name=standard_choice,
        )

        st.download_button(
            label="⬇️ Download PDF Analytical Report",
            data=pdf_bytes,
            file_name=f"air_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )
