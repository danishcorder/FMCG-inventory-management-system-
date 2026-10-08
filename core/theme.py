from __future__ import annotations

import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go

UNICORN_BG = "#08111F"
UNICORN_PANEL = "#0F1B2D"
UNICORN_PANEL_2 = "#132238"
UNICORN_TEXT = "#E8EEF8"
UNICORN_MUTED = "#94A3B8"
UNICORN_PURPLE = "#7C3AED"
UNICORN_CYAN = "#22D3EE"
UNICORN_GOLD = "#F59E0B"
UNICORN_GREEN = "#10B981"
UNICORN_RED = "#EF4444"
UNICORN_GRID = "rgba(148,163,184,0.12)"

def configure_plotly_theme():
    pio.templates["unicorn_enterprise"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=UNICORN_PANEL,
            plot_bgcolor=UNICORN_PANEL,
            font=dict(color=UNICORN_TEXT, family="Inter, Segoe UI, Arial"),
            title=dict(font=dict(size=18, color=UNICORN_TEXT)),
            xaxis=dict(
                gridcolor=UNICORN_GRID,
                zerolinecolor=UNICORN_GRID,
                linecolor="rgba(148,163,184,0.20)",
                tickfont=dict(color=UNICORN_MUTED),
                title_font=dict(color=UNICORN_MUTED),
            ),
            yaxis=dict(
                gridcolor=UNICORN_GRID,
                zerolinecolor=UNICORN_GRID,
                linecolor="rgba(148,163,184,0.20)",
                tickfont=dict(color=UNICORN_MUTED),
                title_font=dict(color=UNICORN_MUTED),
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=UNICORN_MUTED),
            ),
            colorway=[UNICORN_PURPLE, UNICORN_CYAN, UNICORN_GOLD, UNICORN_GREEN, UNICORN_RED, "#60A5FA", "#A78BFA"],
            margin=dict(l=30, r=24, t=56, b=30),
            hoverlabel=dict(
                bgcolor=UNICORN_PANEL_2,
                font_color=UNICORN_TEXT,
                bordercolor=UNICORN_PURPLE,
            ),
        )
    )
    pio.templates.default = "unicorn_enterprise"

def apply_unicorn_enterprise_theme():
    configure_plotly_theme()
    st.markdown(
        f"""
        <style>
        :root {{
            --u-bg: {UNICORN_BG};
            --u-panel: {UNICORN_PANEL};
            --u-panel2: {UNICORN_PANEL_2};
            --u-text: {UNICORN_TEXT};
            --u-muted: {UNICORN_MUTED};
            --u-purple: {UNICORN_PURPLE};
            --u-cyan: {UNICORN_CYAN};
            --u-gold: {UNICORN_GOLD};
            --u-green: {UNICORN_GREEN};
            --u-red: {UNICORN_RED};
        }}

        .stApp {{
            background:
              radial-gradient(circle at 10% 0%, rgba(124,58,237,.12), transparent 28%),
              radial-gradient(circle at 90% 0%, rgba(34,211,238,.09), transparent 25%),
              var(--u-bg);
            color: var(--u-text);
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0A1527 0%, #0B172A 100%);
            border-right: 1px solid rgba(148,163,184,.12);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--u-text);
        }}

        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1600px;
        }}

        h1, h2, h3, h4 {{
            letter-spacing: -0.02em;
            color: var(--u-text) !important;
        }}

        p, label, .stCaption {{
            color: var(--u-muted) !important;
        }}

        .unicorn-page-header {{
            padding: 1.05rem 1.15rem;
            border: 1px solid rgba(124,58,237,.28);
            border-radius: 18px;
            background:
                linear-gradient(120deg, rgba(124,58,237,.16), rgba(34,211,238,.06)),
                rgba(15,27,45,.92);
            box-shadow: 0 12px 36px rgba(0,0,0,.20);
            margin-bottom: 1rem;
        }}
        .unicorn-page-header .eyebrow {{
            color: var(--u-cyan);
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
            margin-bottom: .25rem;
        }}
        .unicorn-page-header .headline {{
            color: var(--u-text);
            font-weight: 760;
            font-size: 1.7rem;
            line-height: 1.15;
            margin: 0;
        }}
        .unicorn-page-header .subline {{
            color: var(--u-muted);
            font-size: .94rem;
            margin-top: .4rem;
        }}

        [data-testid="stMetric"] {{
            background: linear-gradient(180deg, rgba(19,34,56,.98), rgba(15,27,45,.98));
            border: 1px solid rgba(148,163,184,.14);
            border-radius: 16px;
            padding: .85rem .9rem;
            box-shadow: 0 10px 30px rgba(0,0,0,.16);
            min-height: 104px;
        }}

        [data-testid="stMetricLabel"] {{
            color: var(--u-muted) !important;
            font-weight: 600;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--u-text) !important;
            font-weight: 750;
        }}

        [data-testid="stSelectbox"] > div > div,
        [data-testid="stMultiSelect"] > div > div,
        [data-testid="stDateInput"] > div > div,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input {{
            background: var(--u-panel2);
            border-color: rgba(148,163,184,.18);
            border-radius: 10px;
            color: var(--u-text);
        }}

        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button {{
            border-radius: 10px;
            border: 1px solid rgba(124,58,237,.45);
            background: linear-gradient(100deg, var(--u-purple), #5B6CF9);
            color: white;
            font-weight: 700;
            box-shadow: 0 8px 20px rgba(124,58,237,.18);
        }}
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {{
            border-color: var(--u-cyan);
            transform: translateY(-1px);
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid rgba(148,163,184,.13);
            border-radius: 14px;
            overflow: hidden;
            background: var(--u-panel);
        }}

        [data-testid="stPlotlyChart"] {{
            background: var(--u-panel);
            border: 1px solid rgba(148,163,184,.12);
            border-radius: 16px;
            padding: .35rem;
            box-shadow: 0 10px 28px rgba(0,0,0,.13);
        }}

        .unicorn-filter-caption {{
            padding: .55rem .65rem;
            border-radius: 10px;
            background: rgba(34,211,238,.06);
            border: 1px solid rgba(34,211,238,.14);
            color: var(--u-muted);
            font-size: .82rem;
            margin-bottom: .65rem;
        }}

        .unicorn-badge {{
            display: inline-block;
            padding: .22rem .55rem;
            border-radius: 999px;
            border: 1px solid rgba(124,58,237,.35);
            background: rgba(124,58,237,.11);
            color: #C4B5FD;
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .08em;
        }}

        hr {{
            border-color: rgba(148,163,184,.10) !important;
        }}

        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{background: transparent;}}
        </style>
        """,
        unsafe_allow_html=True
    )
