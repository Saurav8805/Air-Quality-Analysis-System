"""PDF report generation module using ReportLab.

Compiles an executive summary report containing dataset metadata, data quality audit notes,
univariate descriptive statistics, location rankings, and scientific limitations.
"""

from __future__ import annotations
import io
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.analysis import SummaryStatistics


def generate_pdf_report(
    summary_stats: SummaryStatistics,
    data_mode: str,
    selected_locations: List[str],
    date_filter_str: str,
    location_summary_df: Optional[pd.DataFrame] = None,
    pollutants_summary_df: Optional[pd.DataFrame] = None,
    quality_notes: Optional[List[str]] = None,
    standard_name: str = "India NAQI (CPCB)",
) -> bytes:
    """Generate a structured, professional PDF report document.

    Returns
    -------
    bytes
        Binary PDF payload for download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0e7490"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
    )
    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )

    story = []

    # 1. Header
    story.append(Paragraph("Air Quality Analysis System — Analytical Report", title_style))
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(
        Paragraph(
            f"Generated: <b>{now_str}</b> | Active Standard: <b>{standard_name}</b> | Data Mode: <b>{data_mode}</b>",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0e7490"), spaceAfter=10))

    # 2. Scope & Filter Context
    locs_text = ", ".join(selected_locations) if selected_locations else "All Monitoring Stations"
    filter_data = [
        [Paragraph("<b>Scope / Filters Applied:</b>", body_style), Paragraph(f"Locations: {locs_text}", body_style)],
        [Paragraph("<b>Date Window:</b>", body_style), Paragraph(date_filter_str, body_style)],
        [Paragraph("<b>Valid Observations:</b>", body_style), Paragraph(f"{summary_stats.count} records", body_style)],
    ]
    t_filter = Table(filter_data, colWidths=[150, 390])
    t_filter.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(t_filter)
    story.append(Spacer(1, 10))

    # 3. Descriptive AQI Statistics Table
    story.append(Paragraph("1. Overall Air Quality Index (AQI) Statistical Profile", h2_style))
    stats_rows = [
        ["Metric", "Value", "Metric", "Value"],
        ["Valid Count (n)", str(summary_stats.count), "Minimum", f"{summary_stats.minimum:.1f}"],
        ["Mean (x̄)", f"{summary_stats.mean:.1f}", "Maximum", f"{summary_stats.maximum:.1f}"],
        ["Median", f"{summary_stats.median:.1f}", "Range (Max - Min)", f"{summary_stats.val_range:.1f}"],
        ["Standard Deviation", f"{summary_stats.std:.1f}", "Q1 (25th percentile)", f"{summary_stats.q1:.1f}"],
        ["Interquartile Range (IQR)", f"{summary_stats.iqr:.1f}", "Q3 (75th percentile)", f"{summary_stats.q3:.1f}"],
    ]
    t_stats = Table(stats_rows, colWidths=[135, 135, 135, 135])
    t_stats.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0e7490")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ])
    )
    story.append(t_stats)
    story.append(Spacer(1, 10))

    # 4. Location Rankings if available
    if location_summary_df is not None and not location_summary_df.empty:
        story.append(Paragraph("2. Location-wise Mean AQI and Sampling Density", h2_style))
        loc_table_data = [["Location", "Mean AQI", "Median", "Min", "Max", "Samples (n)"]]
        for _, r in location_summary_df.head(10).iterrows():
            loc_table_data.append([
                str(r.get("location", "")),
                f"{r.get('mean_aqi', 0):.1f}",
                f"{r.get('median_aqi', 0):.1f}",
                f"{r.get('min_aqi', 0):.1f}",
                f"{r.get('max_aqi', 0):.1f}",
                str(int(r.get("sample_count", 0))),
            ])
        t_loc = Table(loc_table_data, colWidths=[140, 80, 80, 80, 80, 80])
        t_loc.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(t_loc)
        story.append(Spacer(1, 10))

    # 5. Pollutants Summary if available
    if pollutants_summary_df is not None and not pollutants_summary_df.empty:
        story.append(Paragraph("3. Detected Pollutant Concentrations Overview", h2_style))
        poll_table_data = [["Pollutant", "Mean", "Median", "Min", "Max", "Std Dev", "Valid (n)"]]
        for _, r in pollutants_summary_df.iterrows():
            poll_table_data.append([
                str(r.get("Pollutant", "")),
                f"{r.get('Mean', 0):.1f}",
                f"{r.get('Median', 0):.1f}",
                f"{r.get('Minimum', 0):.1f}",
                f"{r.get('Maximum', 0):.1f}",
                f"{r.get('Std Deviation', 0):.1f}",
                str(int(r.get("Valid Count (n)", 0))),
            ])
        t_poll = Table(poll_table_data, colWidths=[100, 70, 70, 70, 70, 80, 80])
        t_poll.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#059669")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(t_poll)
        story.append(Spacer(1, 10))

    # 6. Data Quality & Limitations Audit
    story.append(Paragraph("4. Data Quality Audit & Methodological Limitations", h2_style))
    notes = quality_notes if quality_notes else ["Dataset passed standard integrity checks without warnings."]
    for note in notes:
        story.append(Paragraph(f"• {note}", body_style))
    
    limitations_text = (
        "<b>Notice of Scientific & Descriptive Limitations:</b> All calculations are descriptive and summarize "
        "only the provided dataset observations. These findings do not imply causal relationships. Pollutant "
        "concentrations are reported as recorded in the file and should only be interpreted against official "
        "standards when calibrated sensors and certified units are confirmed."
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph(limitations_text, callout_style))

    # Build document
    doc.build(story)
    return buffer.getvalue()
