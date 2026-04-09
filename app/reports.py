"""
reports.py — PDF report generation using ReportLab.
"""

import io
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_report(
    username: str,
    full_name: str = "",
    region: str = "All",
    crop: str = "All",
    include_predictions: bool = True,
    user_id: int = None,
    is_admin: bool = False,
) -> str | None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table,
            TableStyle, HRFlowable, PageBreak,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        logger.error("ReportLab not installed.")
        return None

    # ── Data ─────────────────────────────────────────────────
    try:
        from app.database import get_crop_records_df
        df = get_crop_records_df()
    except Exception:
        df = None

    if df is None or df.empty:
        logger.warning("No data for report.")

    df_f = df.copy() if df is not None and not df.empty else None
    if df_f is not None:
        if region != "All":
            df_f = df_f[df_f["Region"] == region]
        if crop != "All":
            df_f = df_f[df_f["Crop"] == crop]

    # ── Path ─────────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"crop_yield_report_{ts}.pdf"
    filepath = os.path.join("/tmp", filename)

    # ── Colors & Styles ───────────────────────────────────────
    GREEN       = colors.HexColor("#1a6b3c")
    LIGHT_GREEN = colors.HexColor("#e8f5e9")
    AMBER       = colors.HexColor("#e8a020")
    DARK        = colors.HexColor("#1e2d1f")
    GREY        = colors.HexColor("#666666")
    WHITE       = colors.white

    styles     = getSampleStyleSheet()
    title_s    = ParagraphStyle("T", parent=styles["Title"], fontSize=26,
                                 textColor=GREEN, alignment=TA_CENTER,
                                 fontName="Helvetica-Bold", spaceAfter=4)
    sub_s      = ParagraphStyle("S", parent=styles["Normal"], fontSize=11,
                                 textColor=GREY, alignment=TA_CENTER, spaceAfter=20)
    section_s  = ParagraphStyle("H", parent=styles["Heading1"], fontSize=14,
                                 textColor=GREEN, spaceBefore=18, spaceAfter=8,
                                 fontName="Helvetica-Bold")
    body_s     = ParagraphStyle("B", parent=styles["Normal"], fontSize=10,
                                 textColor=DARK, leading=17, spaceAfter=8)
    caption_s  = ParagraphStyle("C", parent=styles["Normal"], fontSize=8,
                                 textColor=GREY, alignment=TA_CENTER)

    def _tbl_style(header_color):
        return TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), header_color),
            ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,0), 10),
            ("FONTSIZE",      (0,1),(-1,-1), 9),
            ("ALIGN",         (0,0),(-1,-1), "LEFT"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [LIGHT_GREEN, WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#c8e6c9")),
            ("ROWHEIGHT",     (0,0),(-1,-1), 22),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ])

    # ── Build story ───────────────────────────────────────────
    doc   = SimpleDocTemplate(filepath, pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2.5*cm, bottomMargin=2*cm)
    story = []

    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("🌾 Kenya Crop Yield", title_s))
    story.append(Paragraph("Prediction &amp; Analysis Report", title_s))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"Generated: <b>{datetime.now().strftime('%d %B %Y, %H:%M')}</b> &nbsp;·&nbsp; "
        f"Prepared by: <b>{full_name or username}</b> &nbsp;·&nbsp; "
        f"Filter: <b>{region}</b> / <b>{crop}</b>",
        sub_s,
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=GREEN, spaceAfter=16))

    # Executive Summary
    story.append(Paragraph("Executive Summary", section_s))
    if df_f is not None and not df_f.empty:
        avg_y  = df_f["Past_Yield_tons_acre"].mean()
        std_y  = df_f["Past_Yield_tons_acre"].std()
        n      = len(df_f)
        t_reg  = df_f.groupby("Region")["Past_Yield_tons_acre"].mean().idxmax()
        t_crop = df_f.groupby("Crop")["Past_Yield_tons_acre"].mean().idxmax()
        story.append(Paragraph(
            f"This report analyses <b>{n:,}</b> crop yield records from Kenya. "
            f"The dataset covers {'all regions' if region=='All' else region} and "
            f"{'all crop types' if crop=='All' else crop}. "
            f"Mean yield is <b>{avg_y:.3f} t/acre</b> (σ = {std_y:.3f}), indicating "
            f"{'moderate' if std_y < 0.3 else 'significant'} variability. "
            f"Top-performing region: <b>{t_reg}</b>. Highest-yielding crop: <b>{t_crop}</b>.",
            body_s,
        ))
    else:
        story.append(Paragraph("No dataset loaded. Upload a CSV via the Data Upload tab.", body_s))

    story.append(Spacer(1, 0.4*cm))

    # KPI Table
    if df_f is not None and not df_f.empty:
        story.append(Paragraph("Key Performance Indicators", section_s))
        kpi_data = [
            ["Metric", "Value"],
            ["Total Records",       f"{len(df_f):,}"],
            ["Average Yield",       f"{df_f['Past_Yield_tons_acre'].mean():.3f} t/acre"],
            ["Maximum Yield",       f"{df_f['Past_Yield_tons_acre'].max():.3f} t/acre"],
            ["Minimum Yield",       f"{df_f['Past_Yield_tons_acre'].min():.3f} t/acre"],
            ["Std Deviation",       f"{df_f['Past_Yield_tons_acre'].std():.3f} t/acre"],
            ["Average Rainfall",    f"{df_f['Rainfall_mm'].mean():.1f} mm"],
            ["Average Temperature", f"{df_f['Temperature_C'].mean():.1f} °C"],
            ["Average Humidity",    f"{df_f['Humidity_pct'].mean():.1f}%"],
            ["Average Soil pH",     f"{df_f['Soil_pH'].mean():.2f}"],
        ]
        t = Table(kpi_data, colWidths=[9*cm, 8*cm])
        t.setStyle(_tbl_style(GREEN))
        story.append(t)
        story.append(Spacer(1, 0.4*cm))

        # Region-Crop breakdown
        story.append(Paragraph("Yield by Region and Crop", section_s))
        pivot = (
            df_f.groupby(["Region","Crop"])["Past_Yield_tons_acre"]
            .mean().round(3).reset_index()
        )
        pivot.columns = ["Region", "Crop", "Avg Yield (t/acre)"]
        tdata = [pivot.columns.tolist()] + pivot.values.tolist()
        t2 = Table(tdata, colWidths=[6*cm, 5.5*cm, 5.5*cm])
        t2.setStyle(_tbl_style(GREEN))
        story.append(t2)

    # Predictions
    if include_predictions:
        story.append(PageBreak())
        story.append(Paragraph("Recent LSTM Predictions", section_s))
        try:
            if is_admin:
                from app.database import get_all_predictions
                preds = get_all_predictions(limit=30)
            else:
                from app.database import get_predictions_for_user
                preds = get_predictions_for_user(user_id, limit=30)
        except Exception:
            preds = []

        if preds:
            pd_data = [["Date", "Region", "Crop", "Soil pH", "Rainfall", "Yield", "Category"]]
            for p in preds:
                pd_data.append([
                    p.created_at.strftime("%d/%m/%Y %H:%M"),
                    p.region or "—",
                    p.crop   or "—",
                    f"{p.soil_ph:.2f}"         if p.soil_ph        else "—",
                    f"{p.rainfall_mm:.0f} mm"  if p.rainfall_mm    else "—",
                    f"{p.predicted_yield:.3f}" if p.predicted_yield else "—",
                    p.yield_category           or "—",
                ])
            pt = Table(pd_data, colWidths=[3.5*cm,3*cm,2.8*cm,2.2*cm,2.5*cm,2.5*cm,2.5*cm])
            pt.setStyle(_tbl_style(AMBER))
            story.append(pt)
        else:
            story.append(Paragraph("No predictions found.", body_s))

    # Model notes
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c8e6c9")))
    story.append(Paragraph("Model Architecture", section_s))
    story.append(Paragraph(
        "<b>Architecture:</b> Bidirectional LSTM with dual-input design. "
        "Temporal climate features (14 variables, 12-month lookback) "
        "feed into stacked BiLSTM layers. Static categorical embeddings "
        "(Region, Crop, Soil Texture) injected at Dense layer. "
        "<b>Target:</b> Past_Yield_tons_acre (t/acre, monthly).",
        body_s,
    ))

    # Footer
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        f"<font color='#aaaaaa'>Kenya Crop Yield Intelligence System · "
        f"{datetime.now().strftime('%d %B %Y')} · For research and planning purposes only.</font>",
        caption_s,
    ))

    doc.build(story)
    logger.info(f"Report saved: {filepath}")
    return filepath
