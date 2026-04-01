"""
美股行业拥挤度分析 Dashboard v3.0
玻璃箱评分 · 六层框架 · 拥挤+出清 · 赔率视角
让每一个分数都可被理解、可被验证、可被质疑。
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from config import (
    SECTOR_ETFS, DIMENSION_WEIGHTS, CROWDING_LEVELS,
    DIMENSION_META, INDICATOR_META, INDICATOR_QUALITY, CATEGORY_ORDER,
)
from data_fetcher import fetch_price_volume, fetch_etf_info, fetch_pcr, fetch_news_count
from factor_engine import (
    compute_trading, compute_positioning,
    compute_valuation, compute_narrative,
    compute_breadth, compute_clearance,
    build_scorecard, compute_completeness,
)
from scoring import aggregate, commentary, get_level
from history import compute_score_history, get_trend, get_trend_series, trend_arrow

# ─── 页面配置 ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="行业拥挤度分析",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root {
  --bg-base: #060a14;
  --bg-raised: #0c1121;
  --bg-card: #0f1629;
  --bg-card-hover: #131b33;
  --border: #162040;
  --border-light: #1e2d52;
  --accent: #00e5b8;
  --accent-glow: rgba(0,229,184,0.08);
  --text-1: #e8ecf4;
  --text-2: #94a3b8;
  --text-3: #536580;
  --text-4: #2d3f5a;
  --score-ex: #ef4444;
  --score-hi: #f97316;
  --score-md: #eab308;
  --score-lo: #22c55e;
  --font-display: 'Outfit', -apple-system, 'PingFang SC', sans-serif;
  --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
  --font-body: 'Plus Jakarta Sans', -apple-system, 'PingFang SC', sans-serif;
}

/* ── 全局覆写 ────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: var(--bg-base); color: var(--text-2);
    font-family: var(--font-body);
}
[data-testid="stSidebar"] {
    background: var(--bg-raised);
    border-right: 1px solid var(--border);
    font-family: var(--font-body);
}
[data-testid="stSidebar"] label {
    color: var(--text-3) !important; font-size: 11.5px;
    font-family: var(--font-body);
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 18px;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: var(--border-light); }
[data-testid="metric-container"] > div {
    color: var(--text-3) !important; font-size: 10px;
    font-family: var(--font-body); text-transform: uppercase;
    letter-spacing: 0.08em; font-weight: 500;
}
[data-testid="metric-container"] > div > div {
    color: var(--text-1) !important; font-size: 22px; font-weight: 700;
    font-family: var(--font-mono); letter-spacing: -0.02em;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-raised);
    border-bottom: 1px solid var(--border); gap: 0;
    padding: 0 4px;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-3); font-size: 12px;
    letter-spacing: 0.05em; padding: 10px 24px;
    font-family: var(--font-display); font-weight: 500;
    transition: color 0.2s;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-2); }
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent);
    background: transparent;
}

/* ── 通用 ── */
hr { border-color: var(--border) !important; opacity: 0.6; }
h1,h2,h3 {
    color: var(--text-1) !important; font-weight: 400 !important;
    font-family: var(--font-display) !important;
    letter-spacing: -0.01em;
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
}
[data-testid="stExpander"] {
    border: 1px solid var(--border); border-radius: 8px;
    background: var(--bg-raised);
}
[data-testid="stExpander"] summary {
    font-family: var(--font-body); font-weight: 500;
}

/* ── 文字样式 ── */
.note {
    color: var(--text-3); font-size: 11px; font-style: italic;
    font-family: var(--font-body);
}

/* ── 徽章 ── */
.bx {
    padding: 3px 10px; border-radius: 20px; font-size: 10px;
    font-weight: 600; letter-spacing: 0.04em;
    font-family: var(--font-display);
    display: inline-block;
}
.bx-ex { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.25); }
.bx-hi { background: rgba(249,115,22,0.15); color: #fdba74; border: 1px solid rgba(249,115,22,0.25); }
.bx-md { background: rgba(234,179,8,0.15); color: #fde047; border: 1px solid rgba(234,179,8,0.25); }
.bx-lo { background: rgba(34,197,94,0.15); color: #86efac; border: 1px solid rgba(34,197,94,0.25); }

/* ── 卡片 ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    line-height: 1.75;
    transition: border-color 0.2s, transform 0.15s;
}
.card:hover { border-color: var(--border-light); transform: translateY(-1px); }
.card-warn {
    border-left: 3px solid var(--score-hi);
    background: linear-gradient(135deg, #140c04 0%, var(--bg-card) 100%);
}
.card-info {
    border-left: 3px solid var(--accent);
    background: linear-gradient(135deg, #041410 0%, var(--bg-card) 100%);
}
.card-note {
    border-left: 3px solid #6366f1;
    background: linear-gradient(135deg, #080820 0%, var(--bg-card) 100%);
}
.placeholder {
    color: var(--text-4); font-size: 10px;
    background: #060a10; border: 1px dashed var(--border);
    border-radius: 4px; padding: 2px 8px;
}

/* ── 分数区间图例 */
.legend-bar {
    display: flex; gap: 0; border-radius: 6px; overflow: hidden;
    height: 28px; margin: 10px 0; font-size: 11px;
    font-family: var(--font-display);
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.3);
}
.legend-seg {
    display: flex; align-items: center; justify-content: center;
    flex: 1; color: rgba(255,255,255,0.9); font-weight: 600;
    letter-spacing: 0.02em;
}

/* ── Tooltip */
.tt { position: relative; display: inline; cursor: help; }
.tt .tip {
    visibility: hidden; opacity: 0;
    position: absolute; z-index: 9999;
    bottom: 130%; left: 0;
    width: 290px; max-width: 85vw;
    background: #111d38; border: 1px solid #243860;
    border-radius: 8px; padding: 12px 14px;
    font-size: 11px; line-height: 1.7;
    font-family: var(--font-body);
    color: #c0d0e0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(0,229,184,0.05);
    white-space: normal; pointer-events: none;
    transition: opacity 0.15s, transform 0.15s;
    transform: translateY(4px);
}
.tt:hover .tip { visibility: visible; opacity: 1; transform: translateY(0); }
.tt-q {
    color: var(--accent); font-size: 8px; vertical-align: super;
    margin-left: 2px; opacity: 0.6;
}

/* ── 分数分解表 */
.bk-table {
    width: 100%; border-collapse: collapse;
    font-size: 12px; margin-top: 8px;
    font-family: var(--font-body);
}
.bk-table th {
    color: var(--text-3); font-size: 10px; font-weight: 500;
    text-align: left; padding: 6px 10px;
    border-bottom: 1px solid var(--border);
    text-transform: uppercase; letter-spacing: 0.06em;
}
.bk-table td {
    padding: 7px 10px; border-bottom: 1px solid rgba(22,32,64,0.5);
    font-family: var(--font-mono); font-size: 11px;
}
.bk-table td:first-child { font-family: var(--font-body); font-size: 12px; }
.bk-total td {
    border-top: 2px solid var(--border-light) !important;
    border-bottom: none !important;
}

/* ── 子指标行 */
.sub-row {
    display: grid; grid-template-columns: 140px 80px 1fr 50px 50px;
    align-items: center; gap: 8px; padding: 6px 0;
    border-bottom: 1px solid rgba(22,32,64,0.4); font-size: 12px;
}
.sub-bar-bg {
    background: var(--bg-base); border-radius: 4px;
    height: 6px; width: 100%;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.3);
}
.sub-bar-fg { height: 6px; border-radius: 4px; }

/* ── 维度解读 */
.dim-interp {
    margin-top: 12px; padding: 10px 14px;
    background: linear-gradient(135deg, #060e1e 0%, #0a1428 100%);
    border-left: 2px solid #3b82f6;
    border-radius: 6px;
    font-size: 12px; color: var(--text-2); line-height: 1.75;
    font-family: var(--font-body);
}
.dim-meta {
    font-size: 11px; color: var(--text-3); margin-bottom: 10px; line-height: 1.65;
    padding: 8px 12px; background: var(--bg-base);
    border-radius: 6px; border: 1px solid rgba(22,32,64,0.3);
}

/* ── 方法说明页 */
.method-section {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: 10px; padding: 20px 24px; margin-bottom: 16px;
}
.method-h {
    color: var(--accent); font-size: 14px; font-weight: 600;
    font-family: var(--font-display);
    letter-spacing: 0.02em; margin-bottom: 10px;
}
.method-body {
    color: var(--text-2); font-size: 12.5px; line-height: 1.9;
    font-family: var(--font-body);
}
.method-body b { color: var(--text-1); font-weight: 600; }
.data-status {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 10px; font-weight: 600; margin-right: 6px;
    font-family: var(--font-display);
}
.ds-live { background: rgba(34,197,94,0.12); color: #86efac; border: 1px solid rgba(34,197,94,0.2); }
.ds-proxy { background: rgba(234,179,8,0.12); color: #fde047; border: 1px solid rgba(234,179,8,0.2); }
.ds-placeholder { background: rgba(100,116,139,0.08); color: #64748b; border: 1px dashed rgba(100,116,139,0.2); }

/* ── 分数说明横幅 */
.score-banner {
    padding: 12px 18px; border-radius: 8px; font-size: 12px;
    margin-bottom: 14px; border-left: 3px solid; line-height: 1.65;
    font-family: var(--font-body);
}

/* ── 按钮 */
[data-testid="stSidebar"] button[kind="secondary"] {
    border-radius: 8px; font-family: var(--font-display);
    font-weight: 500; letter-spacing: 0.03em;
    transition: all 0.2s;
}

/* ── 数字强调 */
.mono-num {
    font-family: var(--font-mono); font-weight: 600;
    letter-spacing: -0.02em;
}
</style>
""", unsafe_allow_html=True)

# ─── 常量 / 工具 ──────────────────────────────────────────────────────────────
PT = dict(paper_bgcolor="#060a14", plot_bgcolor="#0c1121",
          font=dict(color="#94a3b8", size=11, family="Outfit, Plus Jakarta Sans, sans-serif"))
DIMS = ["叙事拥挤", "持仓拥挤", "交易拥挤", "估值拥挤", "广度与领导权"]

LEVEL_DESC = {
    "极度拥挤": "市场预期极度饱和，继续做多的期望回报已显著下降。向上需要持续不断的超预期信息；一旦边际走弱，回撤容易被放大。",
    "高拥挤":   "多个维度出现明显拥挤信号，赔率已不占优。该行业仍可能继续上涨，但继续加仓需要有更明确的基本面支撑。",
    "中等拥挤": "市场结构尚稳，各维度无极端信号。当前配置赔率相对合理，但需跟踪是否出现快速积累。",
    "低拥挤":   "资金和叙事尚未集中，赔率相对较好。主要跟踪方向是各维度是否开始同步上行。",
}


def badge(level: str) -> str:
    m = {"极度拥挤": "bx-ex", "高拥挤": "bx-hi", "中等拥挤": "bx-md", "低拥挤": "bx-lo"}
    return f'<span class="bx {m.get(level,"bx-lo")}">{level}</span>'


def tt(label: str, key: str) -> str:
    """返回带 tooltip 的 HTML span"""
    meta = INDICATOR_META.get(key) or DIMENSION_META.get(key, {})
    if not meta:
        return label
    lines = []
    if meta.get("what"):  lines.append(f"<b>是什么：</b>{meta['what']}")
    if meta.get("high"):  lines.append(f"<span style='color:#c07a6a'><b>高分：</b></span>{meta['high']}")
    if meta.get("low"):   lines.append(f"<span style='color:#6aaa7a'><b>低分：</b></span>{meta['low']}")
    if meta.get("limit"): lines.append(f"<span style='color:#6a7aaa'><b>局限：</b></span>{meta['limit']}")
    tip = "<br>".join(lines)
    return (f'<span class="tt">{label}'
            f'<sup class="tt-q">?</sup>'
            f'<span class="tip">{tip}</span></span>')


def score_color(v: float) -> str:
    if v >= 70: return "#ef4444"
    if v >= 58: return "#f97316"
    if v >= 42: return "#eab308"
    return "#22c55e"


def gauge_fig(value: float, title: str) -> go.Figure:
    _, color = get_level(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(font=dict(size=38, color=color, family="JetBrains Mono, monospace")),
        title=dict(text=title, font=dict(size=12, color="#536580",
                   family="Outfit, Plus Jakarta Sans, sans-serif")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#162040",
                      tickfont=dict(size=9, color="#475569")),
            bar=dict(color=color, thickness=0.22),
            bgcolor="#0a0f1c",
            steps=[
                {"range": [0,  35], "color": "rgba(34,197,94,0.06)"},
                {"range": [35, 60], "color": "rgba(234,179,8,0.06)"},
                {"range": [60, 80], "color": "rgba(249,115,22,0.06)"},
                {"range": [80,100], "color": "rgba(239,68,68,0.06)"},
            ],
            threshold=dict(line=dict(color=color, width=2),
                           thickness=0.75, value=value),
        ),
    ))
    fig.update_layout(height=190, margin=dict(l=15, r=15, t=42, b=10), **PT)
    return fig


def radar_fig(dim_scores: dict, label: str, color: str = "#00e5b8") -> go.Figure:
    cats = DIMS + [DIMS[0]]
    vals = [dim_scores.get(d, 50) for d in DIMS] + [dim_scores.get(DIMS[0], 50)]
    fig  = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        line=dict(color=color, width=2.5),
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
        name=label,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#080e1c",
            radialaxis=dict(range=[0,100], visible=True,
                            tickfont=dict(size=8, color="#334155",
                                          family="JetBrains Mono, monospace"),
                            gridcolor="#162040", linecolor="#162040"),
            angularaxis=dict(tickfont=dict(size=10, color="#94a3b8",
                                           family="Plus Jakarta Sans, sans-serif"),
                             gridcolor="#162040", linecolor="#162040"),
        ),
        paper_bgcolor="#060a14", height=280,
        showlegend=False,
        margin=dict(l=48, r=48, t=20, b=20),
    )
    return fig


def score_range_legend() -> str:
    """返回分数区间图例 HTML"""
    segs = [
        ("rgba(34,197,94,0.15)", "0–35", "低拥挤"),
        ("rgba(234,179,8,0.15)", "35–60", "中等"),
        ("rgba(249,115,22,0.15)", "60–80", "高拥挤"),
        ("rgba(239,68,68,0.15)", "80–100", "极度"),
    ]
    inner = "".join(
        f'<div class="legend-seg" style="background:{c}">'
        f'<span>{label}</span>&nbsp;<span style="font-weight:300;font-size:9px">{lev}</span></div>'
        for c, label, lev in segs
    )
    return f'<div class="legend-bar">{inner}</div>'


def score_breakdown_html(row: pd.Series, weights: dict) -> str:
    """生成总分拆解表的 HTML"""
    dims_sorted = sorted(DIMS, key=lambda d: -float(row.get(d, 50)))
    tbody = ""
    total_contrib = 0.0
    for d in dims_sorted:
        score = float(row.get(d, 50))
        w = weights.get(d, DIMENSION_WEIGHTS.get(d, 0.2))
        contrib = score * w
        total_contrib += contrib
        sc = score_color(score)
        dim_tt = tt(d, d)
        tbody += (
            f"<tr>"
            f"<td style='color:#94a3b8;font-family:Plus Jakarta Sans,sans-serif'>{dim_tt}</td>"
            f"<td style='color:{sc};font-weight:600;text-align:center'>{score:.1f}</td>"
            f"<td style='color:#536580;text-align:center'>{w*100:.0f}%</td>"
            f"<td style='color:#a3e635;text-align:right;font-weight:600'>{contrib:.1f}</td>"
            f"</tr>"
        )
    tbody += (
        f"<tr class='bk-total'>"
        f"<td style='color:#e8ecf4;font-weight:600;font-family:Plus Jakarta Sans,sans-serif'>合计</td>"
        f"<td></td><td></td>"
        f"<td style='color:#00e5b8;font-weight:700;font-size:15px;text-align:right;font-family:JetBrains Mono,monospace;letter-spacing:-0.02em'>{total_contrib:.1f}</td>"
        f"</tr>"
    )
    return (
        f"<table class='bk-table'>"
        f"<thead><tr>"
        f"<th>维度</th><th style='text-align:center'>得分</th>"
        f"<th style='text-align:center'>权重</th><th style='text-align:right'>贡献</th>"
        f"</tr></thead>"
        f"<tbody>{tbody}</tbody></table>"
    )


TIER_BADGE = {
    "real":    '<span style="background:rgba(34,197,94,0.12);color:#86efac;'
               'border:1px solid rgba(34,197,94,0.2);border-radius:20px;'
               'padding:2px 7px;font-size:9px;font-weight:600;font-family:Outfit,sans-serif">真实数据</span>',
    "proxy":   '<span style="background:rgba(234,179,8,0.12);color:#fde047;'
               'border:1px solid rgba(234,179,8,0.2);border-radius:20px;'
               'padding:2px 7px;font-size:9px;font-weight:600;font-family:Outfit,sans-serif">代理变量</span>',
    "missing": '<span style="background:rgba(100,116,139,0.08);color:#64748b;'
               'border:1px dashed rgba(100,116,139,0.2);border-radius:20px;'
               'padding:2px 7px;font-size:9px;font-weight:600;font-family:Outfit,sans-serif">暂未接入</span>',
    "data_missing": '<span style="background:rgba(239,68,68,0.1);color:#fca5a5;'
                    'border:1px dashed rgba(239,68,68,0.2);border-radius:20px;'
                    'padding:2px 7px;font-size:9px;font-weight:600;font-family:Outfit,sans-serif">数据缺失</span>',
}

CONFIDENCE_STYLE = {
    "高": ("color:#22c55e", "高"),
    "中": ("color:#eab308", "中"),
    "低": ("color:#ef4444", "低"),
}


def completeness_banner(comp: dict) -> str:
    pct = comp["completeness_pct"]
    conf, conf_label = CONFIDENCE_STYLE[comp["confidence"]][:1], comp["confidence"]
    conf_color = CONFIDENCE_STYLE[comp["confidence"]][0]
    bar_color = "#22c55e" if pct >= 78 else "#eab308" if pct >= 60 else "#ef4444"

    dim_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:3px 0;'
        f'font-size:11px;border-bottom:1px solid rgba(22,32,64,0.3);'
        f'font-family:Plus Jakarta Sans,sans-serif">'
        f'<span style="color:#64748b">{d}</span>'
        f'<span style="color:{bar_color};font-family:JetBrains Mono,monospace;'
        f'font-weight:500">{v:.0f}%</span></div>'
        for d, v in comp["dim_completeness"].items()
    )
    return (
        f'<div style="background:#080e1e;border:1px solid #162040;border-radius:8px;'
        f'padding:12px 16px;margin-bottom:12px">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">'
        f'<span style="color:#536580;font-size:10px;font-family:Outfit,sans-serif;'
        f'font-weight:600;letter-spacing:0.1em;text-transform:uppercase">DATA COMPLETENESS</span>'
        f'<span style="font-size:20px;font-weight:700;color:{bar_color};'
        f'font-family:JetBrains Mono,monospace;letter-spacing:-0.02em">{pct:.0f}%</span>'
        f'</div>'
        f'<div style="background:#060a14;border-radius:4px;height:6px;margin-bottom:10px;'
        f'box-shadow:inset 0 1px 2px rgba(0,0,0,0.3)">'
        f'<div style="width:{pct}%;background:{bar_color};height:6px;border-radius:4px;'
        f'transition:width 0.3s"></div></div>'
        f'{dim_rows}'
        f'<div style="margin-top:10px;display:flex;gap:8px;align-items:center">'
        f'<span style="color:#475569;font-size:10px;font-family:Outfit,sans-serif;'
        f'letter-spacing:0.06em">CONFIDENCE:</span>'
        f'<span style="font-size:12px;font-weight:600;font-family:Outfit,sans-serif;'
        f'{conf_color}">{comp["confidence"]}</span>'
        f'</div></div>'
    )


def dim_interpretation(dim: str, score: float, sub: dict) -> str:
    """为每个维度自动生成一句解读"""
    if dim == "交易拥挤":
        rsi = sub.get("RSI(14)", 50)
        vol = sub.get("成交量Surge", 50)
        mom = sub.get("1M动量", 50)
        if score >= 70:
            drivers = []
            if rsi >= 65: drivers.append("短期价格强势（RSI偏高）")
            if vol >= 65: drivers.append("成交量放大")
            if mom >= 65: drivers.append("1M涨幅偏强")
            d = "、".join(drivers) if drivers else "多项指标共振"
            return f"交易拥挤较高，主要来自{d}，快钱参与度上升。行业仍可能继续涨，但继续上涨需要更强新信息驱动；若催化缺席，短期有降温压力。"
        elif score >= 50:
            return "交易热度中等偏高，有一定动量，但尚未到极端程度。可持有，不建议此时追高重仓。"
        else:
            return "交易热度偏低，短期动量不强，市场对该行业交易兴趣一般，无明显过热风险。"
    elif dim == "持仓拥挤":
        if score >= 70:
            return "资金持续高强度流入，持仓集中度处于历史高位。这类「慢拥挤」比价格动量拥挤更危险——方向无争议，但一旦出现踩踏，没有对手盘接。"
        elif score >= 50:
            return "资金流入趋势明显，中期成交量高于历史均值，持仓集中度在上升通道中。景气度不等于拥挤度，需持续跟踪。"
        else:
            return "资金流入温和，持仓集中度处于历史低位，未见大规模一致性建仓迹象。"
    elif dim == "估值拥挤":
        if score >= 70:
            return "价格已偏离历史均值较远，市场对远期乐观预期定价相对充分。基本面好 ≠ 估值不贵；若盈利不及预期，估值回归空间大。"
        elif score >= 50:
            return "估值分位在历史中等偏高区间，定价尚未极端，但安全边际已有所收窄。"
        else:
            return "估值处于历史低分位，市场对该行业预期相对悲观，当前定价具备一定安全边际。"
    elif dim == "叙事拥挤":
        if score >= 70:
            return "市场叙事已高度集中，「人尽皆知」程度较高，期权市场也偏向乐观。预期一旦落空，回调没有缓冲。属于「叙事先行」型拥挤。"
        elif score >= 50:
            return "叙事层面有升温迹象，动量加速且期权市场偏向看涨，但尚未极端，值得持续跟踪是否进一步累积。"
        else:
            return "市场对该行业预期尚未形成集中叙事，属于被忽视或预期分散阶段，潜在赔率相对合理。"
    elif dim == "广度与领导权":
        if score >= 70:
            return "广度已出现明显恶化：ETF距近期高点有明显回撤，趋势均线一致性下降，短期动量弱于中期。这是拥挤松动的早期信号，需警惕结构进一步分裂。"
        elif score >= 50:
            return "广度开始出现轻微恶化迹象，ETF内部趋势一致性下降但尚未极端，值得跟踪是否继续恶化。"
        else:
            return "广度良好，ETF仍在近期高位附近，趋势均线稳定，内部结构未出现明显松动。"
    return ""


# ─── 数据加载 ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="计算六层拥挤度…")
def load_scores(w_tuple: tuple) -> tuple:
    weights  = dict(w_tuple)
    prices, volumes = fetch_price_volume()
    info_d   = fetch_etf_info()
    pcr_d    = fetch_pcr()
    news_d   = fetch_news_count()

    t_df  = compute_trading(prices, volumes)
    p_df  = compute_positioning(prices, volumes)
    v_df  = compute_valuation(prices, info_d)
    n_df  = compute_narrative(prices, news_d, pcr_d)
    br_df = compute_breadth(prices)
    cl_df = compute_clearance(prices)

    scores = aggregate(t_df, p_df, v_df, n_df, br_df, cl_df, weights)

    # 计算历史拥挤度时序（60个交易日）
    history = compute_score_history(prices, volumes, lookback=60, weights=weights)

    detail = dict(trading=t_df, positioning=p_df, valuation=v_df,
                  narrative=n_df, breadth=br_df, clearance=cl_df,
                  prices=prices, volumes=volumes, pcr=pcr_d,
                  history=history)
    return scores, detail


# ─── 侧边栏 ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="padding:4px 0 16px 0">'
            '<div style="font-family:Outfit,sans-serif;font-size:11px;font-weight:600;'
            'letter-spacing:0.18em;text-transform:uppercase;color:#536580;margin-bottom:4px">SECTOR</div>'
            '<div style="font-family:Outfit,sans-serif;font-size:22px;font-weight:700;'
            'letter-spacing:-0.02em;color:#00e5b8;line-height:1.1">Crowding<br>'
            '<span style="color:#e8ecf4">Monitor</span></div>'
            '<div style="font-family:Plus Jakarta Sans,sans-serif;color:#475569;'
            'font-size:10.5px;margin-top:8px;letter-spacing:0.03em">'
            '六层框架 · 出清状态机 · 赔率视角 v3.0</div></div>',
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown(
            '<div style="font-family:Outfit,sans-serif;color:#536580;font-size:10px;'
            'font-weight:600;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px">'
            'DIMENSION WEIGHTS</div>',
            unsafe_allow_html=True)
        dw = {}
        for dim, default in [("叙事拥挤",20),("持仓拥挤",18),("交易拥挤",22),
                              ("估值拥挤",20),("广度与领导权",20)]:
            dw[dim] = st.slider(dim, 0, 40, default, 5) / 100
        total_w = sum(dw.values())
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"权重合计 {total_w:.0%}，标准为 100%")
        else:
            st.markdown(
                '<div style="color:#22c55e;font-size:10px;font-family:Outfit,sans-serif;'
                'font-weight:500;letter-spacing:0.04em">'
                '&#10003; Normalized to 100%</div>',
                unsafe_allow_html=True)
        st.markdown("---")
        if st.button("刷新数据", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.markdown(
            f'<div style="color:#475569;font-size:10px;font-family:JetBrains Mono,monospace;'
            f'letter-spacing:0.04em">'
            f'UPDATED {datetime.now().strftime("%m/%d %H:%M")}</div>',
            unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:11px;line-height:2.2;margin-top:10px;font-family:'Plus Jakarta Sans',sans-serif">
<span style="color:#22c55e">&#9679;</span> <span style="color:#64748b">价格/成交量 · RSI · 动量 · 波动率</span><br>
<span style="color:#22c55e">&#9679;</span> <span style="color:#64748b">期权 P/C Ratio</span><br>
<span style="color:#eab308">&#9679;</span> <span style="color:#536580">Beta扩张 · 媒体热度 (代理)</span><br>
<span style="color:#334155">&#9675;</span> <span style="color:#334155">基金持仓 (暂未接入)</span>
</div>""", unsafe_allow_html=True)
    return dw


# ─── Tab 1: 总览 ─────────────────────────────────────────────────────────────
def tab_overview(scores: pd.DataFrame, prices: pd.DataFrame, history: pd.DataFrame = None):
    # 「如何理解这个分数」横幅
    st.markdown("""
<div style="background:linear-gradient(135deg,#080e1e 0%,#0c1428 100%);
     border:1px solid #162040;border-radius:10px;
     padding:16px 22px;margin-bottom:16px">
  <div style="font-family:Outfit,sans-serif;color:#00e5b8;font-size:12px;
       font-weight:600;margin-bottom:8px;letter-spacing:0.04em">
    HOW TO READ THE SCORE</div>
  <div style="display:flex;gap:2px;border-radius:6px;overflow:hidden;height:24px;margin-bottom:10px;
       box-shadow:inset 0 1px 2px rgba(0,0,0,0.3)">
    <div style="flex:35;background:rgba(34,197,94,0.12);display:flex;align-items:center;justify-content:center;
         font-size:10px;color:#86efac;font-weight:600;font-family:Outfit,sans-serif">0–35 低拥挤</div>
    <div style="flex:25;background:rgba(234,179,8,0.12);display:flex;align-items:center;justify-content:center;
         font-size:10px;color:#fde047;font-weight:600;font-family:Outfit,sans-serif">35–60 中等</div>
    <div style="flex:20;background:rgba(249,115,22,0.12);display:flex;align-items:center;justify-content:center;
         font-size:10px;color:#fdba74;font-weight:600;font-family:Outfit,sans-serif">60–80 高拥挤</div>
    <div style="flex:20;background:rgba(239,68,68,0.12);display:flex;align-items:center;justify-content:center;
         font-size:10px;color:#fca5a5;font-weight:600;font-family:Outfit,sans-serif">80–100 极度</div>
  </div>
  <div style="font-family:'Plus Jakarta Sans',sans-serif;color:#64748b;font-size:11px;line-height:1.75">
    <b style="color:#94a3b8">高分不代表马上跌</b>，而代表继续上涨所需的新信息门槛在升高、赔率在下降。
    <b style="color:#94a3b8">低分不代表马上涨</b>，而代表当前不拥挤、潜在赔率相对更好。
    这个工具判断的是「行业被市场挤得有多满」，而不是预测涨跌。
  </div>
</div>
""", unsafe_allow_html=True)

    avg = scores["总拥挤度"].mean()
    top3 = scores.head(3)
    bot3 = scores.tail(3)
    _, avg_color = get_level(avg)

    col_g, col_r = st.columns([1, 2])
    with col_g:
        st.plotly_chart(gauge_fig(avg, "MARKET CROWDING TEMPERATURE"), use_container_width=True)
        level_now, _ = get_level(avg)
        # 市场平均趋势
        if history is not None:
            tickers = [t for t in scores.index if t in SECTOR_ETFS]
            total_cols = [(t, "总拥挤度") for t in tickers
                          if (t, "总拥挤度") in history.columns]
            if total_cols:
                mkt_avg_hist = history[total_cols].mean(axis=1).dropna()
                if len(mkt_avg_hist) >= 6:
                    mkt_7d = float(mkt_avg_hist.iloc[-1] - mkt_avg_hist.iloc[-6])
                    mkt_30d = float(mkt_avg_hist.iloc[-1] - mkt_avg_hist.iloc[0]) if len(mkt_avg_hist) >= 23 else None
                    trend_html = (
                        f'<div style="text-align:center;margin-top:4px;font-size:11px">'
                        f'<span style="color:#6a7a9a">7D:</span> {trend_arrow(mkt_7d)}'
                        f'&nbsp;&nbsp;<span style="color:#6a7a9a">30D:</span> {trend_arrow(mkt_30d)}'
                        f'</div>'
                    )
                else:
                    trend_html = ""
            else:
                trend_html = ""
        else:
            trend_html = ""
        st.markdown(
            f'<div style="text-align:center;margin-top:-4px">{badge(level_now)}</div>'
            f'{trend_html}'
            f'<div style="text-align:center;font-size:11px;color:#536580;margin-top:8px;'
            f'font-family:Plus Jakarta Sans,sans-serif;line-height:1.5">'
            f'{LEVEL_DESC.get(level_now,"")[:50]}…</div>',
            unsafe_allow_html=True
        )

    with col_r:
        if "SPY" in prices.columns:
            spy = prices["SPY"].dropna()
            def ret(n): return f"{(spy.iloc[-1]/spy.iloc[-n]-1)*100:+.1f}%" if len(spy)>n else "N/A"
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("SPY 1M",  ret(22))
            c2.metric("SPY 3M",  ret(63))
            c3.metric("SPY 6M",  ret(126))
            c4.metric("行业分化",
                      f"{scores['总拥挤度'].max()-scores['总拥挤度'].min():.0f}pts",
                      help="最高拥挤 - 最低拥挤，越大说明行业间分化越明显")

        st.markdown("---")
        sorted_s = scores.sort_values("总拥挤度", ascending=False)
        cols_strip = st.columns(len(sorted_s))
        for i, (tk, row) in enumerate(sorted_s.iterrows()):
            s = float(row["总拥挤度"])
            _, c = get_level(s)
            cols_strip[i].markdown(
                f'<div style="background:rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.15);'
                f'border:1px solid rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.3);'
                f'border-radius:6px;padding:4px 2px;'
                f'text-align:center;font-size:9px;color:{c};line-height:1.4;'
                f'font-family:Outfit,sans-serif">'
                f'<b>{tk}</b><br>'
                f'<span style="font-family:JetBrains Mono,monospace;font-weight:700;font-size:10px">{s:.0f}</span></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    col_hot, col_cold = st.columns(2)

    with col_hot:
        st.markdown(
            '<div style="font-family:Outfit,sans-serif;color:#ef4444;font-weight:600;'
            'font-size:13px;letter-spacing:0.04em;margin-bottom:8px">'
            'MOST CROWDED</div>',
            unsafe_allow_html=True)
        for tk, row in top3.iterrows():
            s = float(row["总拥挤度"])
            _, c = get_level(s)
            primary = max(DIMS, key=lambda d: float(row.get(d, 0)))
            cmt = commentary(row)
            level_lbl  = row["拥挤等级"]
            state_icon = row.get("状态图标", "")
            state_name = row.get("状态", "")
            state_color= row.get("状态色", c)
            state_expl = row.get("状态说明", "")
            t_data = get_trend(history, tk) if history is not None else {}
            t7 = trend_arrow(t_data.get("change_7d"))
            t30 = trend_arrow(t_data.get("change_30d"))
            st.markdown(f"""
<div class="card" style="border-left:3px solid {c}">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="font-family:'Plus Jakarta Sans',sans-serif;color:#e8ecf4;font-weight:600">
      {row['行业']} <span style="color:#475569;font-weight:400">({tk})</span></span>
    <span style="color:{c};font-size:24px;font-weight:700;font-family:'JetBrains Mono',monospace;
           letter-spacing:-0.03em">{s:.0f}</span>
  </div>
  <div style="color:#64748b;font-size:11px;margin-top:5px;font-family:'Plus Jakarta Sans',sans-serif">
    主导维度: <span style="color:#94a3b8">{primary}</span>&nbsp;&nbsp;{badge(level_lbl)}
    &nbsp;·&nbsp; <span style="color:#475569">7D</span>{t7}
    &nbsp;<span style="color:#475569">30D</span>{t30}
  </div>
  <div style="margin-top:6px;padding:5px 10px;border-radius:6px;
       background:{row.get('状态背景','#0f1629')};display:inline-block">
    <span style="color:{state_color};font-size:11px;font-weight:600;font-family:Outfit,sans-serif">
      {state_icon} {state_name}</span>
    <span style="color:#475569;font-size:10px;margin-left:10px">{row.get('操作偏向','')}</span>
  </div>
  <div style="color:#64748b;font-size:10px;margin-top:5px;line-height:1.55;
       font-family:'Plus Jakarta Sans',sans-serif">{state_expl}</div>
</div>""", unsafe_allow_html=True)

    with col_cold:
        st.markdown(
            '<div style="font-family:Outfit,sans-serif;color:#22c55e;font-weight:600;'
            'font-size:13px;letter-spacing:0.04em;margin-bottom:8px">'
            'LEAST CROWDED</div>',
            unsafe_allow_html=True)
        for tk, row in bot3.iterrows():
            s = float(row["总拥挤度"])
            _, c = get_level(s)
            narrative_s = float(row.get("叙事拥挤", 50))
            heating = "叙事维度开始升温，值得跟踪" if narrative_s > 50 else "各维度均处于历史低分位"
            t_data = get_trend(history, tk) if history is not None else {}
            t7 = trend_arrow(t_data.get("change_7d"))
            t30 = trend_arrow(t_data.get("change_30d"))
            st.markdown(f"""
<div class="card" style="border-left:3px solid #22c55e">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="font-family:'Plus Jakarta Sans',sans-serif;color:#e8ecf4;font-weight:600">
      {row['行业']} <span style="color:#475569;font-weight:400">({tk})</span></span>
    <span style="color:{c};font-size:24px;font-weight:700;font-family:'JetBrains Mono',monospace;
           letter-spacing:-0.03em">{s:.0f}</span>
  </div>
  <div style="color:#64748b;font-size:11px;margin-top:4px;font-family:'Plus Jakarta Sans',sans-serif">
    <span style="color:#475569">7D</span>{t7}
    &nbsp;<span style="color:#475569">30D</span>{t30}
  </div>
  <div style="color:#64748b;font-size:11px;margin-top:4px;font-family:'Plus Jakarta Sans',sans-serif">
    {heating}</div>
  <div style="color:#536580;font-size:11px;margin-top:4px">低拥挤阶段，做多赔率相对占优。</div>
</div>""", unsafe_allow_html=True)


# ─── Tab 2: 排名 ─────────────────────────────────────────────────────────────
def tab_ranking(scores: pd.DataFrame, history: pd.DataFrame = None):
    # 分数区间说明条
    st.markdown(
        '<div style="font-size:10px;color:#536580;margin-bottom:5px;font-family:Outfit,sans-serif;'
        'text-transform:uppercase;letter-spacing:0.08em;font-weight:500">Score Range</div>'
        + score_range_legend()
        + '<div style="font-size:10px;color:#475569;margin-bottom:12px;font-family:Plus Jakarta Sans,sans-serif">'
          '分数越高 = 该行业被市场「挤」得越满，并非预测涨跌，而是反映赔率结构</div>',
        unsafe_allow_html=True
    )

    rc1, rc2 = st.columns([1, 3])
    with rc1:
        sort_by = st.selectbox("排序维度", ["总拥挤度"] + DIMS, label_visibility="collapsed")
    with rc2:
        cats_available = [c for c in CATEGORY_ORDER if any(
            SECTOR_ETFS[t]["category"] == c for t in scores.index
        )]
        sel_cats = st.multiselect(
            "筛选类别", cats_available, default=cats_available, key="rank_cats",
            label_visibility="collapsed",
        )
    if sel_cats:
        mask = scores.index.map(lambda t: SECTOR_ETFS[t]["category"]).isin(sel_cats)
        disp = scores[mask].sort_values(sort_by, ascending=False)
    else:
        disp = scores.sort_values(sort_by, ascending=False)

    n_rows = max(len(disp), 1)
    chart_h = max(380, n_rows * 34)

    labels_y = [f"{SECTOR_ETFS[t]['name']}({t})" for t in disp.index]
    z = disp[DIMS].values

    fig = go.Figure(go.Heatmap(
        z=z, x=DIMS, y=labels_y,
        text=z.round(0).astype(int).astype(str),
        texttemplate="%{text}",
        colorscale=[
            [0.00, "#061210"], [0.30, "#0c2618"],
            [0.50, "#1a2a0c"], [0.65, "#3a2808"],
            [0.80, "#4a1808"], [1.00, "#7a1010"],
        ],
        zmin=0, zmax=100,
        showscale=True,
        colorbar=dict(title=dict(text="得分", font=dict(size=10, color="#536580")),
                      thickness=10, tickfont=dict(size=9, color="#536580")),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(
        height=chart_h, margin=dict(l=170, r=60, t=30, b=30),
        xaxis=dict(side="top", tickfont=dict(size=11, color="#94a3b8",
                   family="Plus Jakarta Sans, sans-serif")),
        yaxis=dict(tickfont=dict(size=10, color="#e8ecf4",
                   family="Plus Jakarta Sans, sans-serif")),
        **PT,
    )
    st.plotly_chart(fig, use_container_width=True)

    sd = disp.sort_values("总拥挤度")
    bar_labels = [f"{SECTOR_ETFS[t]['name']}({t})" for t in sd.index]
    bar_colors = [get_level(v)[1] for v in sd["总拥挤度"]]
    bar_text   = [f"{v:.0f}  {sd.loc[t,'拥挤等级']}" for t, v in zip(sd.index, sd["总拥挤度"])]

    fig2 = go.Figure(go.Bar(
        x=sd["总拥挤度"], y=bar_labels, orientation="h",
        marker_color=bar_colors,
        text=bar_text, textposition="outside",
        textfont=dict(size=10, color="#64748b", family="JetBrains Mono, monospace"),
        hovertemplate="<b>%{y}</b><br>总拥挤度: %{x:.1f}<extra></extra>",
    ))
    fig2.update_layout(
        xaxis=dict(title="总拥挤度", range=[0, 115],
                   tickfont=dict(size=10, color="#536580",
                   family="JetBrains Mono, monospace")),
        height=chart_h, margin=dict(l=170, r=100, t=20, b=40), **PT,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**快速信号表**")
    rows_out = []
    for tk, row in disp.iterrows():
        cmt = commentary(row)
        s = float(row["总拥挤度"])
        level_lbl = row["拥挤等级"]
        t_data = get_trend(history, tk) if history is not None else {}
        rows_out.append({
            "行业(ETF)":    f"{SECTOR_ETFS[tk]['name']}({tk})",
            "总分":          f"{s:.0f}",
            "Δ7D":           f"{t_data['change_7d']:+.1f}" if t_data.get("change_7d") is not None else "—",
            "Δ30D":          f"{t_data['change_30d']:+.1f}" if t_data.get("change_30d") is not None else "—",
            "等级":          level_lbl,
            "主导维度":      max(DIMS, key=lambda d: float(row.get(d, 0))),
            "操作信号":      cmt["action"],
            "核心判断":      cmt["summary"][:45] + "…",
        })
    st.dataframe(pd.DataFrame(rows_out), use_container_width=True,
                 hide_index=True, height=420)


# ─── Tab 3: 详情（玻璃箱核心页）─────────────────────────────────────────────
def tab_detail(scores: pd.DataFrame, detail: dict, weights: dict,
               history: pd.DataFrame = None):
    cats_available = [c for c in CATEGORY_ORDER if any(
        SECTOR_ETFS[t]["category"] == c for t in SECTOR_ETFS
    )]
    dc1, dc2 = st.columns([1, 2])
    with dc1:
        cat_sel = st.selectbox("类别", cats_available, key="detail_cat")
    with dc2:
        cat_tickers = [t for t in SECTOR_ETFS if SECTOR_ETFS[t]["category"] == cat_sel]
        sel = st.selectbox(
            "选择行业/主题", cat_tickers,
            format_func=lambda t: f"{SECTOR_ETFS[t]['name']} ({t}) — {SECTOR_ETFS[t]['desc']}",
            key="detail_etf",
        )

    if sel not in scores.index:
        st.error("数据暂不可用")
        return

    row   = scores.loc[sel]
    cmt   = commentary(row)
    total = float(row["总拥挤度"])
    level_lbl, color = get_level(total)
    level_desc = LEVEL_DESC.get(level_lbl, "")

    # ── 分数说明横幅（含趋势）
    banner_bg = {"极度拥挤": "#180606", "高拥挤": "#160c04",
                 "中等拥挤": "#100e04", "低拥挤": "#061008"}.get(level_lbl, "#0c1121")
    t_data = get_trend(history, sel) if history is not None else {}
    t7_html = trend_arrow(t_data.get("change_7d"))
    t30_html = trend_arrow(t_data.get("change_30d"))
    trend_span = (
        f'&nbsp;&nbsp;<span style="font-size:11px">'
        f'<span style="color:#475569">7D</span>{t7_html}'
        f'&nbsp;<span style="color:#475569">30D</span>{t30_html}'
        f'</span>'
    ) if history is not None else ""
    st.markdown(
        f'<div class="score-banner" style="background:{banner_bg};border-color:{color}">'
        f'<span style="color:{color};font-size:26px;font-weight:800;'
        f'font-family:JetBrains Mono,monospace;letter-spacing:-0.03em">{total:.1f}</span>'
        f'&nbsp;&nbsp;{badge(level_lbl)}{trend_span}&nbsp;&nbsp;&nbsp;'
        f'<span style="color:#64748b;font-size:12px;font-family:Plus Jakarta Sans,sans-serif">'
        f'{level_desc}</span></div>',
        unsafe_allow_html=True
    )

    # ── 顶部三列：Gauge / 总分拆解 / Radar
    cg, cb, cr = st.columns([1, 1.1, 1])
    with cg:
        st.plotly_chart(gauge_fig(total, f"{SECTOR_ETFS[sel]['name']} 总拥挤度"),
                        use_container_width=True)
        st.markdown(
            f'<div style="text-align:center">{badge(level_lbl)}</div>'
            f'<div style="text-align:center;margin-top:8px;font-size:13px;color:{color}">'
            f'{cmt["action"]}</div>',
            unsafe_allow_html=True
        )

    with cb:
        st.markdown(
            '<div style="font-family:Outfit,sans-serif;color:#94a3b8;font-size:11px;'
            'font-weight:600;margin-bottom:6px;letter-spacing:0.04em">'
            'SCORE BREAKDOWN</div>',
            unsafe_allow_html=True
        )
        st.markdown(score_breakdown_html(row, weights), unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#475569;font-size:10px;margin-top:8px;'
            'font-family:Plus Jakarta Sans,sans-serif">'
            '总分 = 各维度得分 × 对应权重之和。'
            '悬停维度名查看说明。</div>',
            unsafe_allow_html=True
        )

    with cr:
        st.plotly_chart(
            radar_fig({d: float(row.get(d, 50)) for d in DIMS},
                      SECTOR_ETFS[sel]["name"], color),
            use_container_width=True
        )

    # ── 五维可展开卡（按得分降序）
    st.markdown("---")
    st.markdown(
        '<div style="font-family:Outfit,sans-serif;color:#94a3b8;font-size:12px;'
        'font-weight:600;margin-bottom:10px;letter-spacing:0.03em">'
        'DIMENSION DETAILS'
        '<span style="color:#475569;font-weight:400;font-size:11px;margin-left:10px;'
        'font-family:Plus Jakarta Sans,sans-serif">点击展开查看子指标明细</span></div>',
        unsafe_allow_html=True
    )

    t_df  = detail["trading"]
    p_df  = detail["positioning"]
    v_df  = detail["valuation"]
    n_df  = detail["narrative"]
    br_df = detail["breadth"]
    cl_df = detail["clearance"]
    pcr_d = detail.get("pcr", {})
    scorecard = build_scorecard(sel, t_df, p_df, v_df, n_df, br_df, cl_df)

    # ── 状态机卡片
    state_name  = row.get("状态", "—")
    state_icon  = row.get("状态图标", "")
    state_color = row.get("状态色", "#8899bb")
    state_bg    = row.get("状态背景", "#111827")
    state_act   = row.get("操作偏向", "")
    state_expl  = row.get("状态说明", "")
    cl_score    = float(row.get("出清状态", 50))
    br_score    = float(row.get("广度与领导权", 50))
    st.markdown(
        f'<div style="background:{state_bg};border:1px solid {state_color}30;'
        f'border-left:3px solid {state_color};border-radius:8px;'
        f'padding:14px 18px;margin-bottom:12px">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<span style="color:{state_color};font-size:14px;font-weight:700;'
        f'font-family:Outfit,sans-serif">{state_icon} {state_name}</span>'
        f'<span style="color:#536580;font-size:11px;font-family:Outfit,sans-serif">{state_act}</span>'
        f'</div>'
        f'<div style="color:#94a3b8;font-size:11px;margin-top:6px;line-height:1.65;'
        f'font-family:Plus Jakarta Sans,sans-serif">{state_expl}</div>'
        f'<div style="color:#475569;font-size:10px;margin-top:6px;'
        f'font-family:JetBrains Mono,monospace">'
        f'BREADTH {br_score:.0f} · CLEARANCE {cl_score:.0f}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── 数据完整度面板（全宽，在雷达图后）
    comp = compute_completeness(sel, t_df, p_df, v_df, n_df, br_df, cl_df, pcr_d)
    st.markdown(completeness_banner(comp), unsafe_allow_html=True)

    dims_sorted = sorted(DIMS, key=lambda d: -float(row.get(d, 50)))
    for dim in dims_sorted:
        dim_score = float(row.get(dim, 50))
        dim_w = weights.get(dim, DIMENSION_WEIGHTS.get(dim, 0.2))
        dim_contrib = dim_score * dim_w
        _, dim_color = get_level(dim_score)
        dim_records = [r for r in scorecard if r["维度"] == dim]
        # 过滤掉 status=missing 的指标，避免 NaN 影响解读
        sub_scores = {r["子指标"]: r["历史分位"] for r in dim_records
                      if r.get("status") != "missing"}
        interp = dim_interpretation(dim, dim_score, sub_scores)
        dim_meta = DIMENSION_META.get(dim, {})
        dim_comp_pct = comp["dim_completeness"].get(dim, 100)

        expander_label = (
            f"{dim}  ·  {dim_score:.1f} 分  ·  "
            f"权重 {dim_w*100:.0f}%  ·  "
            f"贡献总分 {dim_contrib:.1f} 分  ·  "
            f"数据 {dim_comp_pct:.0f}%"
        )
        with st.expander(expander_label, expanded=(dim == dims_sorted[0])):
            # 维度说明
            if dim_meta.get("what"):
                st.markdown(
                    f'<div class="dim-meta">'
                    f'<b style="color:#7a9abb">{dim}</b>：{dim_meta["what"]}'
                    + (f'<br><span style="color:#5a6a4a">⚠ 局限：{dim_meta["limit"]}</span>'
                       if dim_meta.get("limit") else "")
                    + '</div>',
                    unsafe_allow_html=True
                )

            # 叙事拥挤：媒体热度或PCR缺失时显示重归一化提示
            if dim == "叙事拥挤" and sel in n_df.index:
                nr = n_df.loc[sel]
                if bool(nr.get("_renorm", False)):
                    aw = float(nr.get("accel_eff_w", 0.545)) * 100
                    sw = float(nr.get("skew_eff_w", 0.455)) * 100
                    nw = float(nr.get("news_eff_w", 0.0)) * 100
                    pw = float(nr.get("pcr_eff_w", 0.0)) * 100
                    missing_parts = []
                    if bool(nr.get("news_missing", True)): missing_parts.append("媒体热度")
                    if bool(nr.get("pcr_missing",  True)): missing_parts.append("PCR")
                    st.markdown(
                        f'<div style="color:#ddaa44;font-size:11px;margin-bottom:6px;'
                        f'padding:5px 10px;background:#181208;border-radius:3px;'
                        f'border-left:2px solid #8a7020">'
                        f'⚠️ {" / ".join(missing_parts)}数据缺失 — '
                        f'本维度已按有效指标重新归一化权重：'
                        f'动量加速度 {aw:.1f}% · 收益偏度 {sw:.1f}%'
                        + (f' · 媒体热度 {nw:.1f}%' if nw > 0 else '')
                        + (f' · PCR {pw:.1f}%' if pw > 0 else '')
                        + '</div>',
                        unsafe_allow_html=True
                    )

            # 子指标表格
            rows_html = ""
            for rec in dim_records:
                ind_name    = rec["子指标"]
                ind_raw     = rec["原始值"]
                ind_w_str   = rec["维度内权重"]
                ind_contrib = rec["子项贡献"]
                status      = rec.get("status", "ok")
                name_tt     = tt(ind_name, ind_name)

                if status == "missing":
                    rows_html += (
                        f"<tr style='opacity:0.6'>"
                        f"<td style='color:#8899bb;width:130px'>"
                        f"  {name_tt}&nbsp;{TIER_BADGE['data_missing']}</td>"
                        f"<td style='color:#ee6644;font-size:10px;width:60px'>N/A</td>"
                        f"<td style='width:100px'><div class='sub-bar-bg'></div></td>"
                        f"<td style='color:#4a5a7a;text-align:center;width:45px;font-size:10px'>—</td>"
                        f"<td style='color:#4a5a7a;text-align:center;width:40px'>{ind_w_str}</td>"
                        f"<td style='color:#4a5a7a;text-align:right;width:45px'>—</td>"
                        f"<td style='color:#ee6644;font-size:10px;padding-left:10px'>{rec['说明']}</td>"
                        f"</tr>"
                    )
                    continue

                ind_score = rec["历史分位"]
                ic    = score_color(ind_score)
                bar_w = max(3, int(ind_score))
                ind_q = INDICATOR_QUALITY.get(ind_name, {})
                tier  = ind_q.get("tier", "real")
                proxy_note = ind_q.get("proxy_note", "")
                if tier == "proxy" and proxy_note:
                    tier_html = (
                        '<span class="tt" style="display:inline">'
                        + TIER_BADGE["proxy"]
                        + f'<span class="tip" style="width:300px">{proxy_note}</span>'
                        + '</span>'
                    )
                else:
                    tier_html = TIER_BADGE.get(tier, "")
                _pct_ctx   = rec.get("pct_context", "")
                _low_badge = rec.get("low_score_note", "")
                _low_html  = (
                    f'&nbsp;<span style="background:#0d2010;color:#55cc77;font-size:9px;'
                    f'padding:1px 4px;border-radius:3px;border:1px solid #2a7040">'
                    f'{_low_badge}</span>'
                ) if _low_badge else ""
                _score_cell = (
                    f"<td style='color:{ic};font-weight:600;text-align:center;width:65px;"
                    f"font-size:10px;line-height:1.3'>{_pct_ctx}</td>"
                ) if _pct_ctx else (
                    f"<td style='color:{ic};font-weight:600;text-align:center;width:45px'>"
                    f"{ind_score:.0f}</td>"
                )
                rows_html += (
                    f"<tr>"
                    f"<td style='color:#8899bb;width:130px'>{name_tt}&nbsp;{tier_html}{_low_html}</td>"
                    f"<td style='color:#5a6a7a;font-size:10px;width:60px'>{ind_raw}</td>"
                    f"<td style='width:100px'>"
                    f"  <div class='sub-bar-bg'>"
                    f"    <div class='sub-bar-fg' style='width:{bar_w}%;background:{ic}'></div>"
                    f"  </div>"
                    f"</td>"
                    + _score_cell +
                    f"<td style='color:#4a5a7a;text-align:center;width:40px'>{ind_w_str}</td>"
                    f"<td style='color:#aabb88;text-align:right;width:45px'>{ind_contrib:.1f}</td>"
                    f"<td style='color:#4a5a6a;font-size:10px;padding-left:10px'>{rec['说明']}</td>"
                    f"</tr>"
                )
            st.markdown(
                f"<table style='width:100%;border-collapse:collapse;font-size:12px'>"
                f"<thead><tr style='color:#3a4a6a;font-size:10px;border-bottom:1px solid #1a2540'>"
                f"<th style='text-align:left;padding:3px 0'>指标</th>"
                f"<th style='text-align:left'>原始值</th>"
                f"<th>分位分布</th>"
                f"<th style='text-align:center'>得分</th>"
                f"<th style='text-align:center'>权重</th>"
                f"<th style='text-align:right'>贡献</th>"
                f"<th style='text-align:left;padding-left:10px'>说明</th>"
                f"</tr></thead>"
                f"<tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True
            )

            # 解读句
            if interp:
                st.markdown(
                    f'<div class="dim-interp">💬 {interp}</div>',
                    unsafe_allow_html=True
                )

    # ── 完整打分卡（折叠）
    st.markdown("---")
    with st.expander("📋 完整打分卡明细（可人工复核所有子指标）"):
        st.markdown(
            '<div class="note" style="margin-bottom:8px">'
            '所有子指标统一归一化为 0-100 历史分位数分。'
            '「子项贡献」= 历史分位 × 该指标在本维度内的权重，反映其对维度总分的贡献。</div>',
            unsafe_allow_html=True
        )
        sc_df = pd.DataFrame(scorecard)

        def color_pct(val):
            try:
                v = float(val)
                if v >= 70: return "color:#c0392b;font-weight:600"
                if v >= 58: return "color:#d35400"
                if v >= 42: return "color:#b7950b"
                return "color:#1e8449"
            except: return "color:#4a5a6a"

        st.dataframe(
            sc_df.style.map(color_pct, subset=["历史分位", "子项贡献"]),
            use_container_width=True, hide_index=True, height=500,
        )

    # ── 综合研判
    st.markdown("---")
    st.markdown("**综合研判**")
    rc1, rc2 = st.columns([3, 2])
    with rc1:
        st.markdown(
            f'<div class="card card-info" style="color:#b8c8d8;font-size:13px">'
            f'{cmt["summary"]}</div>',
            unsafe_allow_html=True
        )
        if cmt["structure"]:
            st.markdown(
                f'<div class="card card-note" style="color:#8899aa;font-size:12px">'
                f'📐 结构解读<br>{cmt["structure"]}</div>',
                unsafe_allow_html=True
            )
    with rc2:
        st.markdown(
            f'<div class="card" style="color:#7a9aaa;font-size:12px">'
            f'📊 赔率判断<br>{cmt["odds"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="card card-warn" style="color:#c07a40;font-size:12px">'
            f'⚡ 风险提示<br>{cmt["risk"]}</div>',
            unsafe_allow_html=True
        )

    # ── 价格走势
    st.markdown("---")
    st.markdown("**价格走势（过去12个月，归一化=100）**")
    prices = detail["prices"]
    if sel in prices.columns:
        p = prices[sel].dropna().iloc[-252:]
        spy_p = prices["SPY"].dropna().iloc[-252:] if "SPY" in prices.columns else None
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=p.index, y=(p/p.iloc[0]*100),
            name=f"{SECTOR_ETFS[sel]['name']}({sel})",
            line=dict(color=color, width=2.5),
        ))
        if spy_p is not None:
            fig.add_trace(go.Scatter(
                x=spy_p.index, y=(spy_p/spy_p.iloc[0]*100),
                name="SPY", line=dict(color="#6366f1", width=1.5, dash="dot"),
            ))
        fig.add_hline(y=100, line_dash="dash", line_color="#1e2d52", opacity=0.5)
        fig.update_layout(
            height=280, hovermode="x unified",
            legend=dict(orientation="h", y=-0.2,
                        font=dict(family="Outfit, sans-serif", size=11)),
            margin=dict(l=40, r=20, t=20, b=60),
            yaxis_title="相对表现", **PT,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── 拥挤度走势（60个交易日）
    if history is not None and (sel, "总拥挤度") in history.columns:
        st.markdown("---")
        st.markdown("**拥挤度走势（过去60个交易日）**")
        total_s = get_trend_series(history, sel, "总拥挤度")
        if len(total_s) > 5:
            fig_h = go.Figure()
            # 背景色带
            fig_h.add_hrect(y0=80, y1=100, fillcolor="#8b1010", opacity=0.08,
                            line_width=0)
            fig_h.add_hrect(y0=60, y1=80, fillcolor="#d35400", opacity=0.06,
                            line_width=0)
            fig_h.add_hrect(y0=35, y1=60, fillcolor="#b7950b", opacity=0.04,
                            line_width=0)
            fig_h.add_hrect(y0=0, y1=35, fillcolor="#1e8449", opacity=0.04,
                            line_width=0)
            # 总拥挤度
            fig_h.add_trace(go.Scatter(
                x=total_s.index, y=total_s.values,
                name="总拥挤度", line=dict(color="#00d4aa", width=2.5),
                fill="tozeroy", fillcolor="rgba(0,212,170,0.05)",
            ))
            # 各维度
            dim_colors = {"交易拥挤": "#c0392b", "持仓拥挤": "#d4a017",
                          "估值拥挤": "#8e44ad", "叙事拥挤": "#2980b9",
                          "广度与领导权": "#27ae60"}
            for dim in DIMS:
                ds = get_trend_series(history, sel, dim)
                if len(ds) > 5:
                    fig_h.add_trace(go.Scatter(
                        x=ds.index, y=ds.values,
                        name=dim, line=dict(color=dim_colors.get(dim, "#666"),
                                            width=1, dash="dot"),
                        visible="legendonly",
                    ))
            fig_h.update_layout(
                height=260, hovermode="x unified",
                yaxis=dict(range=[0, 100], title="拥挤度得分"),
                legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
                margin=dict(l=40, r=20, t=20, b=60), **PT,
            )
            st.plotly_chart(fig_h, use_container_width=True)
            st.markdown(
                '<div class="note">总拥挤度为实线，各维度可在图例中切换显示。'
                '叙事维度历史值仅含动量加速度+收益偏度（无快照数据），绝对值可能与当前分数有偏差，趋势方向准确。</div>',
                unsafe_allow_html=True
            )


# ─── Tab 4: 信号监控 ─────────────────────────────────────────────────────────
def tab_signals(scores: pd.DataFrame):
    sig_col, tog_col = st.columns([3, 1])
    with sig_col:
        st.markdown(
            '<div class="note">信号基于当前截面快照。拥挤度时序趋势（7D/30D变化）已在排名表和详情页中显示。</div>',
            unsafe_allow_html=True
        )
    with tog_col:
        show_all = st.toggle("显示全部行业", value=False, key="sig_show_all")

    if show_all:
        # ── 全量展示模式：按总拥挤度降序，分四个等级显示
        for level_label, lo, hi, color in [
            ("极度拥挤", 80, 101, "#c0392b"),
            ("高拥挤",   60,  80, "#d35400"),
            ("中等拥挤", 35,  60, "#b7950b"),
            ("低拥挤",    0,  35, "#1e8449"),
        ]:
            subset = scores[(scores["总拥挤度"] >= lo) & (scores["总拥挤度"] < hi)].sort_values("总拥挤度", ascending=False)
            if subset.empty:
                continue
            st.markdown(
                f'<div style="color:{color};font-size:13px;font-weight:600;margin:14px 0 6px">'
                f'{level_label}（{len(subset)} 个）</div>',
                unsafe_allow_html=True
            )
            cols = st.columns(2)
            for i, (tk, row) in enumerate(subset.iterrows()):
                s = float(row["总拥挤度"])
                _, c = get_level(s)
                top_dim = max(DIMS, key=lambda d: float(row.get(d, 50)))
                cat = SECTOR_ETFS[tk]["category"]
                with cols[i % 2]:
                    st.markdown(f"""
<div class="card" style="border-left:3px solid {color}">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span>
      <span style="color:#c8d8e8;font-weight:600">{SECTOR_ETFS[tk]['name']} ({tk})</span>
      <span style="color:#3a4a6a;font-size:10px;margin-left:6px">{cat}</span>
    </span>
    <span style="color:{c};font-size:20px;font-weight:700">{s:.0f}</span>
  </div>
  <div style="color:#6a7a8a;font-size:11px;margin-top:5px">
    主导维度：{top_dim[:2] if len(top_dim)>2 else top_dim} {float(row.get(top_dim,50)):.0f}分
  </div>
</div>""", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            '<div class="note">叙事拥挤维度「媒体热度代理」因子基于 yfinance 新闻条目计数的横截面排名（非历史分位），'
            '衡量相对媒体关注度。持仓拥挤各指标为ETF成交量/Beta代理，非直接基金持仓数据。'
            '广度与领导权为ETF级代理，非板块内个股广度数据。</div>',
            unsafe_allow_html=True
        )
        return

    def signal_card(tk, row, note, border_color="#1e2a3a"):
        s = float(row["总拥挤度"])
        _, c = get_level(s)
        st.markdown(f"""
<div class="card" style="border-left:3px solid {border_color}">
  <div style="display:flex;justify-content:space-between">
    <span style="color:#c8d8e8;font-weight:600">{SECTOR_ETFS[tk]['name']} ({tk})</span>
    <span style="color:{c};font-size:20px;font-weight:700">{s:.0f}</span>
  </div>
  <div style="color:#6a7a8a;font-size:11px;margin-top:5px;line-height:1.7">{note}</div>
</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div style="color:#c0392b;font-size:13px;font-weight:600;margin-bottom:6px">🔴 全面拥挤（多维度共振）</div>',
                    unsafe_allow_html=True)
        found = False
        for tk, row in scores.iterrows():
            n_high = sum(1 for d in DIMS if float(row.get(d,50)) >= 60)
            if n_high >= 3 and float(row["总拥挤度"]) >= 60:
                found = True
                high_dims = [d for d in DIMS if float(row.get(d,50)) >= 60]
                signal_card(tk, row,
                    f"共振维度: {' · '.join(d[:2] for d in high_dims)} | "
                    f"当前最大风险：预期过满后的边际降温",
                    border_color="#7b1e1e")
        if not found:
            st.markdown('<div class="note" style="padding:8px">暂无行业达到全面拥挤标准</div>',
                        unsafe_allow_html=True)

        st.markdown('<div style="color:#d35400;font-size:13px;font-weight:600;margin:14px 0 6px">⚠️ 估值透支 + 动量仍强</div>',
                    unsafe_allow_html=True)
        found2 = False
        for tk, row in scores.iterrows():
            if float(row.get("估值拥挤",50)) >= 65 and float(row.get("交易拥挤",50)) >= 60:
                found2 = True
                signal_card(tk, row,
                    f"估值 {row.get('估值拥挤','—'):.0f} · 交易 {row.get('交易拥挤','—'):.0f} | "
                    f"基本面好 ≠ 赔率好，动量可能掩盖透支风险",
                    border_color="#7b3b1e")
        if not found2:
            st.markdown('<div class="note" style="padding:8px">暂无</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div style="color:#b7950b;font-size:13px;font-weight:600;margin-bottom:6px">⚡ 叙事先行（叙事 > 价格）</div>',
                    unsafe_allow_html=True)
        found3 = False
        for tk, row in scores.iterrows():
            if (float(row.get("叙事拥挤",50)) >= 55 and
                float(row.get("交易拥挤",50)) < 50 and
                float(row["总拥挤度"]) < 62):
                found3 = True
                signal_card(tk, row,
                    f"叙事 {row.get('叙事拥挤','—'):.0f} > 交易 {row.get('交易拥挤','—'):.0f} | "
                    f"叙事已集中但价格未跟进，属于潜在拥挤积聚早期",
                    border_color="#7b6a10")
        if not found3:
            st.markdown('<div class="note" style="padding:8px">暂无</div>', unsafe_allow_html=True)

        st.markdown('<div style="color:#1e8449;font-size:13px;font-weight:600;margin:14px 0 6px">🟢 低拥挤 · 赔率相对占优</div>',
                    unsafe_allow_html=True)
        low_df = scores[scores["总拥挤度"] < 38].head(4)
        if low_df.empty:
            st.markdown('<div class="note" style="padding:8px">暂无行业低于阈值</div>',
                        unsafe_allow_html=True)
        for tk, row in low_df.iterrows():
            signal_card(tk, row,
                "各维度均处于历史低分位 | 资金和叙事尚未集中，赔率相对合理",
                border_color="#1e4a2a")

    st.markdown("---")
    st.markdown(
        '<div class="note">叙事拥挤维度「媒体热度代理」因子基于 yfinance 新闻条目计数的横截面排名（非历史分位），'
        '衡量相对媒体关注度。持仓拥挤各指标为ETF成交量/Beta代理，非直接基金持仓数据。'
        '广度与领导权为ETF级代理，非板块内个股广度数据。</div>',
        unsafe_allow_html=True
    )


# ─── Tab 5: 方法说明 ─────────────────────────────────────────────────────────
def tab_method():
    st.markdown(
        '<h3 style="font-family:Outfit,sans-serif !important;color:#e8ecf4 !important;'
        'font-weight:600 !important;margin-bottom:4px !important">方法说明</h3>'
        '<div style="color:#536580;font-size:11px;margin-bottom:22px;'
        'font-family:Outfit,sans-serif;letter-spacing:0.08em;text-transform:uppercase">'
        'How This Tool Works — Scoring Framework, Data Sources & Limitations</div>',
        unsafe_allow_html=True
    )

    # ── 核心理念
    st.markdown("""
<div class="method-section">
  <div class="method-h">这个工具在做什么？</div>
  <div class="method-body">
    <b>核心问题：这个行业被市场「挤」得有多满？</b><br><br>
    我们把「拥挤」定义为：资金、预期、情绪、定价在一个方向上过度集中的状态。
    这种状态下，继续上涨所需的新信息门槛在升高，一旦出现哪怕轻微的不及预期，回撤容易被放大。<br><br>
    <b>高分不代表马上跌</b>——它代表继续做多的期望回报在下降，赔率在变差。<br>
    <b>低分不代表马上涨</b>——它代表当前不拥挤，潜在赔率相对更好，但需要等待催化剂。<br><br>
    这个工具的目标是帮助你判断「现在进入这个行业，赔率结构是否合理」，而不是预测涨跌。
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 六层框架
    st.markdown("""
<div class="method-section">
  <div class="method-h">六层框架：五维拥挤分 + 出清状态</div>
  <div class="method-body">
    我们把「拥挤」分解为<b>五个独立维度</b>（计入总拥挤度），并引入<b>出清状态</b>（仅进状态机，不影响总分）：
  </div>
</div>
""", unsafe_allow_html=True)

    dim_details = [
        ("叙事拥挤", "20%", "#2471a3",
         "衡量市场叙事/共识是否已高度集中，「人尽皆知」的程度。"
         "包含：动量加速度（1M vs 3M月均）、收益率偏度（近60日）、媒体热度代理（yfinance新闻计数）、PCR情绪（Put/Call Ratio）。",
         "媒体热度为代理变量，PCR基于近2个到期日期权链，部分行业流动性不足时可能缺失。"),
        ("持仓拥挤", "18%", "#d35400",
         "衡量资金是否持续、集中流入该行业，形成一致性持仓。"
         "包含：成交量中期趋势（63D/252D）、Beta扩张（30D/90D）、相对SPY资金流分位。",
         "使用成交量趋势和Beta作为持仓集中度的间接代理，无法直接获取基金实际持仓（如13F）。"),
        ("交易拥挤", "22%", "#c0392b",
         "衡量近期价格动量、成交量和波动率是否出现短期过热。"
         "包含：RSI历史分位、1M动量分位、成交量Surge、波动率扩张、价格/50MA、价格/200MA、上涨日比例。",
         "交易热 ≠ 全面拥挤。如果持仓未极端、估值未透支，交易维度高分只代表短期超热。"),
        ("估值拥挤", "20%", "#b7950b",
         "衡量市场定价是否已透支远期预期，相对历史偏贵。"
         "包含：52W Z-Score（价格偏离年度均值）、3M相对SPY超额收益分位、PE/PB横截面代理。",
         "PE/PB 来自 yfinance 点值，做全量ETF横截面分位排名，非历史时间序列分位，精度有限。"),
        ("广度与领导权", "20%", "#6a3a9a",
         "衡量拥挤结构是否正在内部松动：ETF距近期高点回撤、趋势均线一致性、短长期动量背离。高分=广度恶化。",
         "无法直接获取板块内个股数据，使用ETF级别的价格结构作为广度代理，精确度有限。"),
        ("出清状态", "仅进状态机", "#4a6a4a",
         "衡量市场是否正在主动出清拥挤：距252日高点回撤深度、波动率突刺（10日/90日）、正负收益不对称（坏消息主导）。",
         "出清状态不影响总拥挤度，仅用于状态机分类（踩踏风险区/拥挤松动中/接近出清等7种状态）。"),
    ]

    for dim, weight, color, desc, limit in dim_details:
        st.markdown(f"""
<div style="background:#0a1020;border:1px solid #1a2540;border-left:3px solid {color};
     border-radius:5px;padding:12px 16px;margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
    <span style="color:#c8d8e8;font-weight:600">{dim}</span>
    <span style="color:{color};font-size:13px;font-weight:700">权重 {weight}</span>
  </div>
  <div style="color:#8899aa;font-size:12px;line-height:1.7;margin-bottom:6px">{desc}</div>
  <div style="color:#4a5a6a;font-size:11px">⚠ 局限：{limit}</div>
</div>""", unsafe_allow_html=True)

    # ── 归一化方式
    st.markdown("""
<div class="method-section">
  <div class="method-h">分数是怎么计算的？</div>
  <div class="method-body">
    <b>第一步：子指标归一化</b><br>
    我们把每个子指标（如RSI、成交量、波动率等）统一换算成 <b>0–100 历史分位数分</b>。
    分数越高，代表该指标当前处于历史越高的位置（即越拥挤）。
    这样可以把不同量纲的数据（价格、成交量、RSI等）放到同一框架下比较。<br><br>
    例如：RSI 当前值为 72，在过去252个交易日中处于第 88 百分位 → RSI 得分 = 88。<br><br>
    <b>第二步：维度加权</b><br>
    每个子指标在其维度内有权重，加权汇总得到维度得分（0-100）。<br><br>
    <b>第三步：五维加权（六层版本）</b><br>
    五个计入总分的维度按设定权重（叙事20% / 持仓18% / 交易22% / 估值20% / 广度20%）加权汇总，
    得到总拥挤度（0-100）。出清状态不进总分，仅用于状态机。权重可在侧边栏自定义。
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 分数解读
    st.markdown("""
<div class="method-section">
  <div class="method-h">分数区间对应什么？</div>
  <div class="method-body">
  </div>
</div>
""", unsafe_allow_html=True)

    level_rows = [
        (0, 35,   "#1e8449", "低拥挤",   "✅ 赔率较好 / 可积极关注",
         "资金和叙事尚未集中，各维度均处于历史低分位。做多赔率相对占优，主要跟踪是否开始同步上行。"),
        (35, 60,  "#b7950b", "中等拥挤", "👀 正常配置 / 持续跟踪",
         "市场结构尚稳，各维度无极端信号。当前配置赔率相对合理，但需跟踪是否出现快速积累。"),
        (60, 80,  "#d35400", "高拥挤",   "⚠️ 控制仓位 / 等待拥挤释放",
         "多个维度出现明显拥挤信号，赔率已不占优。仍可能继续上涨，但继续加仓需更明确的基本面支撑。"),
        (80, 100, "#c0392b", "极度拥挤", "⛔ 回避追高 / 考虑减仓",
         "市场预期极度饱和，继续做多期望回报已显著下降。当前最大风险不是基本面证伪，而是预期过满后的边际降温。"),
    ]
    for lo, hi, c, lev, action, desc in level_rows:
        st.markdown(f"""
<div style="display:flex;gap:12px;align-items:flex-start;
     background:#0a0e1a;border:1px solid #1a2540;border-radius:5px;
     padding:10px 14px;margin-bottom:8px">
  <div style="min-width:80px;text-align:center">
    <div style="color:{c};font-size:18px;font-weight:700">{lo}–{hi}</div>
    <div style="color:{c};font-size:11px;font-weight:600">{lev}</div>
  </div>
  <div>
    <div style="color:#c8d8e8;font-size:12px;margin-bottom:4px">{action}</div>
    <div style="color:#6a7a8a;font-size:11px;line-height:1.65">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── 数据状态
    st.markdown("""
<div class="method-section">
  <div class="method-h">数据来源与当前状态</div>
  <div class="method-body">
  </div>
</div>
""", unsafe_allow_html=True)

    data_items = [
        ("ds-live",        "✅ 真实数据",  "价格、成交量",         "通过 yfinance 实时抓取，日频更新"),
        ("ds-live",        "✅ 真实数据",  "RSI / 动量 / 波动率",  "基于价格数据实时计算"),
        ("ds-live",        "✅ 真实数据",  "期权 P/C Ratio",       "从期权链实时计算，部分行业流动性不足时可能缺失"),
        ("ds-proxy",       "⚠️ 代理指标", "Beta 扩张",            "用近期/中期 Beta 比值代理资金集中度，非直接持仓数据"),
        ("ds-proxy",       "⚠️ 代理指标", "估值拥挤各因子",       "Z-Score/超额收益为价格行为代理；PE/PB取自yfinance，做全量ETF横截面排名"),
        ("ds-proxy",       "⚠️ 代理指标", "媒体热度代理",         "yfinance新闻条目数量横截面排名（全量ETF互比），衡量相对媒体关注度"),
        ("ds-placeholder", "🔲 暂未接入", "基金持仓（13F）",      "待接入，接入后持仓拥挤维度将显著改善"),
    ]
    for css, status, name, note in data_items:
        st.markdown(
            f'<div style="padding:5px 0;border-bottom:1px solid #0d1520">'
            f'<span class="data-status {css}">{status}</span>'
            f'<span style="color:#8899bb;font-size:12px">{name}</span>'
            f' — <span style="color:#4a5a6a;font-size:11px">{note}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("")
    st.markdown(
        '<div class="note" style="margin-top:12px">'
        '本工具为研究辅助工具，不构成投资建议。所有指标均基于公开市场数据计算，'
        '历史分位数基于过去约252个交易日（约1年）的滚动窗口。</div>',
        unsafe_allow_html=True
    )


# ─── Tab 5: 状态机说明 ───────────────────────────────────────────────────────
def tab_state_machine(scores: pd.DataFrame):
    from config import STATE_CONFIG
    st.markdown(
        '<h3 style="font-family:Outfit,sans-serif !important;color:#e8ecf4 !important;'
        'font-weight:600 !important;margin-bottom:4px !important">出清状态机</h3>'
        '<div style="color:#536580;font-size:11px;margin-bottom:18px;'
        'font-family:Outfit,sans-serif;letter-spacing:0.06em">'
        'STATE MACHINE — 根据总拥挤度、广度恶化程度、出清信号强弱，自动归类7种市场状态</div>',
        unsafe_allow_html=True
    )

    # 状态分布（当前所有行业）
    st.markdown(
        '<div style="font-family:Outfit,sans-serif;color:#94a3b8;font-size:12px;'
        'font-weight:600;margin-bottom:10px;letter-spacing:0.04em">'
        'CURRENT DISTRIBUTION</div>',
        unsafe_allow_html=True
    )
    state_groups: dict = {}
    for tk, row in scores.iterrows():
        sname = row.get("状态", "低拥挤/早期升温")
        state_groups.setdefault(sname, []).append((tk, row))

    ncols = min(3, max(1, len(state_groups)))
    cols  = st.columns(ncols)
    col_i = 0
    for sn, entries in sorted(state_groups.items(),
                               key=lambda x: -float(scores.loc[x[1][0][0], "总拥挤度"])):
        cfg  = STATE_CONFIG.get(sn, {})
        sc   = cfg.get("color", "#8899bb")
        sb   = cfg.get("bg", "#111827")
        icon = cfg.get("icon", "")
        act  = cfg.get("action", "")
        with cols[col_i % ncols]:
            entries_html = "".join(
                f'<div style="color:#c8d8e8;font-size:12px;padding:3px 0;'
                f'border-bottom:1px solid #0d1520">'
                f'{SECTOR_ETFS[tk2]["name"]} ({tk2}) '
                f'<span style="color:{sc};font-weight:600">'
                f'{float(r2["总拥挤度"]):.0f}</span></div>'
                for tk2, r2 in entries
            )
            st.markdown(
                f'<div style="background:{sb};border:1px solid {sc}40;'
                f'border-top:3px solid {sc};border-radius:5px;padding:10px 14px;margin-bottom:10px">'
                f'<div style="color:{sc};font-weight:700;font-size:13px;margin-bottom:6px">'
                f'{icon} {sn}</div>'
                f'<div style="color:#4a5a6a;font-size:10px;margin-bottom:6px">{act}</div>'
                f'{entries_html}</div>',
                unsafe_allow_html=True
            )
        col_i += 1

    # 状态机规则说明
    st.markdown("---")
    st.markdown(
        '<div style="font-family:Outfit,sans-serif;color:#94a3b8;font-size:12px;'
        'font-weight:600;margin-bottom:10px;letter-spacing:0.04em">'
        'CLASSIFICATION RULES <span style="color:#475569;font-weight:400;font-size:10px">'
        '(priority descending, first match)</span></div>',
        unsafe_allow_html=True
    )
    rules = [
        ("踩踏风险区",     "总拥挤 ≥72 且 出清信号 ≤35",
         "极度拥挤且无出清迹象，是踩踏风险最高的状态"),
        ("拥挤松动中",     "总拥挤 ≥58 且 出清信号 ≥60",
         "拥挤偏高但出清信号已出现，市场正在主动释放压力"),
        ("高拥挤/赔率下降","总拥挤 ≥62",
         "高拥挤但出清信号尚弱，赔率显著收窄"),
        ("接近出清",       "总拥挤 ≥48 且 出清信号 ≥65",
         "中等拥挤叠加强出清信号，建议等待出清完成"),
        ("拥挤扩张中",     "总拥挤 ≥48 且 广度恶化 ≥55",
         "拥挤偏高且广度在恶化，拥挤格局仍在扩张"),
        ("反向观察区",     "总拥挤 <40 且 出清信号 ≥55",
         "低拥挤叠加出清信号，可能是反转积累阶段"),
        ("低拥挤/早期升温","默认（不符合以上任何条件）",
         "整体低拥挤，各维度无极端信号，早期升温或低关注阶段"),
    ]
    for i, (state_name, condition, desc) in enumerate(rules, 1):
        cfg = STATE_CONFIG.get(state_name, {})
        sc  = cfg.get("color", "#8899bb")
        sb  = cfg.get("bg", "#111827")
        icon= cfg.get("icon", "")
        act = cfg.get("action", "")
        st.markdown(
            f'<div style="display:flex;gap:12px;align-items:flex-start;'
            f'background:{sb};border:1px solid {sc}30;'
            f'border-left:3px solid {sc};border-radius:4px;'
            f'padding:8px 14px;margin-bottom:6px">'
            f'<div style="min-width:22px;color:#4a5a6a;font-size:10px;'
            f'font-weight:700;margin-top:2px">#{i}</div>'
            f'<div style="flex:1">'
            f'<div style="color:{sc};font-weight:600;font-size:12px">{icon} {state_name}</div>'
            f'<div style="color:#8899aa;font-size:11px;margin-top:2px">'
            f'条件：<code style="color:#aabb88;background:#0a1020;padding:1px 5px;'
            f'border-radius:2px">{condition}</code></div>'
            f'<div style="color:#6a7a8a;font-size:11px;margin-top:3px">{desc}</div>'
            f'<div style="color:#4a5a6a;font-size:10px;margin-top:3px">{act}</div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="note" style="margin-top:12px">'
        '状态机使用三个输入：总拥挤度（叙事+持仓+交易+估值+广度五维加权）、'
        '广度与领导权得分（ETF级代理）、出清状态得分（距高点回撤+波动率突刺+收益不对称）。'
        '出清状态得分不影响总拥挤度，仅用于状态分类。</div>',
        unsafe_allow_html=True
    )


# ─── 主入口 ───────────────────────────────────────────────────────────────────
def main():
    weights = sidebar()
    tw = sum(weights.values())
    if tw > 0:
        weights = {k: v/tw for k, v in weights.items()}

    with st.spinner("正在计算六层拥挤度…"):
        try:
            scores, detail = load_scores(tuple(sorted(weights.items())))
        except Exception as e:
            st.error(f"数据加载失败: {e}")
            st.stop()

    st.markdown(
        '<div style="margin-bottom:8px">'
        '<h2 style="font-family:Outfit,sans-serif !important;font-weight:700 !important;'
        'letter-spacing:-0.02em;margin-bottom:2px !important">'
        '美股行业拥挤度分析</h2>'
        '<div style="font-family:Outfit,sans-serif;color:#00e5b8;font-size:11px;'
        'font-weight:500;letter-spacing:0.12em;text-transform:uppercase">'
        'Glass-Box Scoring &middot; Six-Layer Framework &middot; '
        'State Machine &middot; Odds Perspective</div></div>',
        unsafe_allow_html=True
    )

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "  市场总览  ", "  行业排名  ",
        "  行业详情  ", "  信号监控  ",
        "  状态机说明  ", "  方法说明  "
    ])
    history = detail.get("history")
    with t1: tab_overview(scores, detail["prices"], history)
    with t2: tab_ranking(scores, history)
    with t3: tab_detail(scores, detail, weights, history)
    with t4: tab_signals(scores)
    with t5: tab_state_machine(scores)
    with t6: tab_method()


if __name__ == "__main__":
    main()
