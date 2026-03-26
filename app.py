"""
美股行业拥挤度分析 Dashboard v2.0
五层框架：预期 / 持仓 / 交易 / 估值 / 行为
赔率视角 · 买方研究风格
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from config import SECTOR_ETFS, DIMENSION_WEIGHTS, CROWDING_LEVELS
from data_fetcher import fetch_price_volume, fetch_etf_info, fetch_pcr
from factor_engine import (
    compute_trading, compute_positioning,
    compute_valuation, compute_narrative, compute_behavioral,
)
from scoring import aggregate, commentary, get_level

# ─── 页面配置 ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="行业拥挤度分析",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 专业暗色 CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 全局 */
[data-testid="stAppViewContainer"] { background:#0a0e1a; color:#d1d9e6; }
[data-testid="stSidebar"] { background:#0d1220; border-right:1px solid #1a2540; }
/* metric 卡片 */
[data-testid="metric-container"] {
    background:#111827; border:1px solid #1e3050;
    border-radius:5px; padding:10px 14px;
}
[data-testid="metric-container"] > div { color:#8899bb !important; font-size:11px; }
[data-testid="metric-container"] > div > div { color:#c8d8e8 !important; font-size:20px; font-weight:600; }
/* tabs */
.stTabs [data-baseweb="tab-list"] { background:#0d1220; border-bottom:1px solid #1a2540; gap:0; }
.stTabs [data-baseweb="tab"] { color:#6677aa; font-size:12px; letter-spacing:.04em; padding:8px 22px; }
.stTabs [aria-selected="true"] { color:#00d4aa; border-bottom:2px solid #00d4aa; background:transparent; }
/* 分隔线 */
hr { border-color:#1a2540 !important; }
h1,h2,h3 { color:#c8d8e8 !important; font-weight:300 !important; letter-spacing:.03em; }
/* sidebar label */
[data-testid="stSidebar"] label { color:#8899aa !important; font-size:12px; }
[data-testid="stSidebar"] [data-baseweb="slider"] { margin-bottom:4px; }
/* 数据框 */
[data-testid="stDataFrame"] { border:1px solid #1a2540; border-radius:4px; }
/* 小注释文字 */
.note { color:#4a5a7a; font-size:11px; font-style:italic; }
/* 拥挤徽章 */
.bx { padding:2px 9px; border-radius:3px; font-size:11px; font-weight:700; }
.bx-ex { background:#7b1e1e; color:#ffaaaa; }
.bx-hi { background:#7b3b1e; color:#ffccaa; }
.bx-md { background:#5a4a10; color:#ffe080; }
.bx-lo { background:#1a4a2a; color:#88ee99; }
/* info 卡 */
.card { background:#111827; border:1px solid #1e3050; border-radius:5px;
        padding:12px 16px; margin-bottom:8px; line-height:1.7; }
.card-warn { border-left:3px solid #d35400; background:#130f08; }
.card-info { border-left:3px solid #00d4aa; background:#08130f; }
.card-note { border-left:3px solid #5577aa; background:#080d18; }
/* 占位标记 */
.placeholder { color:#4a6060; font-size:10px; background:#0a1818;
               border:1px dashed #1a3a3a; border-radius:3px; padding:1px 6px; }
</style>
""", unsafe_allow_html=True)

# ─── 常量 / 工具 ──────────────────────────────────────────────────────────────
PT = dict(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
          font=dict(color="#99aabb", size=11))

DIMS = ["预期拥挤", "持仓拥挤", "交易拥挤", "估值拥挤", "行为拥挤"]


def badge(level: str) -> str:
    m = {"极度拥挤": "bx-ex", "高拥挤": "bx-hi", "中等拥挤": "bx-md", "低拥挤": "bx-lo"}
    return f'<span class="bx {m.get(level,"bx-lo")}">{level}</span>'


def gauge_fig(value: float, title: str) -> go.Figure:
    _, color = get_level(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(font=dict(size=34, color=color)),
        title=dict(text=title, font=dict(size=12, color="#6677aa")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#1a2540",
                      tickfont=dict(size=9, color="#4a5a7a")),
            bar=dict(color=color, thickness=0.22),
            bgcolor="#0d1220",
            steps=[
                {"range": [0,  35], "color": "#0a1a10"},
                {"range": [35, 60], "color": "#1a1a08"},
                {"range": [60, 80], "color": "#1a0d08"},
                {"range": [80,100], "color": "#1a0808"},
            ],
            threshold=dict(line=dict(color=color, width=2),
                           thickness=0.75, value=value),
        ),
    ))
    fig.update_layout(height=170, margin=dict(l=15, r=15, t=40, b=10), **PT)
    return fig


def radar_fig(dim_scores: dict, label: str, color: str = "#00d4aa") -> go.Figure:
    cats = DIMS + [DIMS[0]]
    vals = [dim_scores.get(d, 50) for d in DIMS] + [dim_scores.get(DIMS[0], 50)]
    fig  = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        line=dict(color=color, width=2),
        fillcolor=f"rgba(0,212,170,0.12)",
        name=label,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d1220",
            radialaxis=dict(range=[0,100], visible=True,
                            tickfont=dict(size=8, color="#3a4a5a"),
                            gridcolor="#1a2540", linecolor="#1a2540"),
            angularaxis=dict(tickfont=dict(size=10, color="#8899bb"),
                             gridcolor="#1a2540", linecolor="#1a2540"),
        ),
        paper_bgcolor="#0a0e1a", height=290,
        showlegend=False,
        margin=dict(l=45, r=45, t=30, b=30),
    )
    return fig


# ─── 数据加载 ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="计算五层拥挤度…")
def load_scores(w_tuple: tuple) -> tuple:
    """w_tuple 是 sorted weight items，方便 cache key 哈希"""
    weights = dict(w_tuple)
    prices, volumes = fetch_price_volume()
    info    = fetch_etf_info()
    pcr_d   = fetch_pcr()

    t_df = compute_trading(prices, volumes)
    p_df = compute_positioning(prices, volumes)
    v_df = compute_valuation(prices)
    n_df = compute_narrative(prices)
    b_df = compute_behavioral(prices, pcr_d)

    scores = aggregate(t_df, p_df, v_df, n_df, b_df, weights)
    detail = dict(trading=t_df, positioning=p_df, valuation=v_df,
                  narrative=n_df, behavioral=b_df,
                  prices=prices, volumes=volumes)
    return scores, detail


# ─── 侧边栏 ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="color:#00d4aa;font-size:15px;font-weight:600;letter-spacing:.08em">'
            '⚡ CROWDING MONITOR</div>'
            '<div style="color:#4a5a7a;font-size:11px;margin-bottom:12px">五层框架 · 赔率视角 v2.0</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown('<div style="color:#6677aa;font-size:11px;margin-bottom:6px">维度权重（合计需=100%）</div>',
                    unsafe_allow_html=True)

        dw = {}
        for dim, default in [("预期拥挤",20),("持仓拥挤",20),("交易拥挤",25),
                              ("估值拥挤",20),("行为拥挤",15)]:
            dw[dim] = st.slider(dim, 0, 40, default, 5) / 100

        total_w = sum(dw.values())
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"权重合计 {total_w:.0%}，标准为 100%")
        else:
            st.markdown('<div style="color:#1e8449;font-size:11px">✓ 权重已归一化</div>',
                        unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🔄 刷新数据", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown(f'<div class="note">更新: {datetime.now().strftime("%m/%d %H:%M")}</div>',
                    unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:11px;line-height:2;margin-top:8px">
<span style="color:#3a7a4a">✅</span> <span style="color:#5a7a5a">价格/成交量</span><br>
<span style="color:#3a7a4a">✅</span> <span style="color:#5a7a5a">RSI / 动量 / 波动率</span><br>
<span style="color:#3a7a4a">✅</span> <span style="color:#5a7a5a">期权 P/C Ratio</span><br>
<span style="color:#5a7a4a">⚠️</span> <span style="color:#4a6a4a">Beta扩张代理</span><br>
<span style="color:#3a4a5a">🔲</span> <span style="color:#3a4a5a">新闻热度 (占位)</span><br>
<span style="color:#3a4a5a">🔲</span> <span style="color:#3a4a5a">基金持仓 (占位)</span><br>
<span style="color:#3a4a5a">🔲</span> <span style="color:#3a4a5a">卖方覆盖 (占位)</span>
</div>""", unsafe_allow_html=True)
    return dw


# ─── Tab 1: 总览 ─────────────────────────────────────────────────────────────
def tab_overview(scores: pd.DataFrame, prices: pd.DataFrame):
    avg = scores["总拥挤度"].mean()
    top3 = scores.head(3)
    bot3 = scores.tail(3)
    _, avg_color = get_level(avg)

    col_g, col_r = st.columns([1, 2])
    with col_g:
        st.plotly_chart(gauge_fig(avg, "市场整体拥挤温度"), use_container_width=True)
        level_now, _ = get_level(avg)
        st.markdown(f'<div style="text-align:center;margin-top:-8px">{badge(level_now)}</div>',
                    unsafe_allow_html=True)

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
        # 快速热力条
        sorted_s = scores.sort_values("总拥挤度", ascending=False)
        cols_strip = st.columns(len(sorted_s))
        for i, (tk, row) in enumerate(sorted_s.iterrows()):
            s = float(row["总拥挤度"])
            _, c = get_level(s)
            cols_strip[i].markdown(
                f'<div style="background:{c};border-radius:3px;padding:3px 2px;'
                f'text-align:center;font-size:9px;color:white;line-height:1.4">'
                f'<b>{tk}</b><br>{s:.0f}</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    col_hot, col_cold = st.columns(2)

    with col_hot:
        st.markdown('<div style="color:#c0392b;font-weight:600;margin-bottom:6px">🔴 最拥挤行业</div>',
                    unsafe_allow_html=True)
        for tk, row in top3.iterrows():
            s = float(row["总拥挤度"])
            _, c = get_level(s)
            primary = max(DIMS, key=lambda d: float(row.get(d, 0)))
            cmt = commentary(row)
            st.markdown(f"""
<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="color:#c8d8e8;font-weight:600">{row['行业']} <span style="color:#4a5a7a">({tk})</span></span>
    <span style="color:{c};font-size:22px;font-weight:700">{s:.0f}</span>
  </div>
  <div style="color:#6677aa;font-size:11px;margin-top:4px">
    主导维度: <span style="color:#8899bb">{primary}</span> · {badge(row['拥挤等级'])}
  </div>
  <div style="color:#8a6a4a;font-size:11px;margin-top:6px;line-height:1.6">{cmt['action']}</div>
</div>""", unsafe_allow_html=True)

    with col_cold:
        st.markdown('<div style="color:#1e8449;font-weight:600;margin-bottom:6px">🟢 低拥挤 / 赔率相对合理</div>',
                    unsafe_allow_html=True)
        for tk, row in bot3.iterrows():
            s = float(row["总拥挤度"])
            _, c = get_level(s)
            narrative_s = float(row.get("预期拥挤", 50))
            heating = "⚡ 预期维度开始升温，值得跟踪" if narrative_s > 50 else "各维度均处于历史低分位"
            st.markdown(f"""
<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="color:#c8d8e8;font-weight:600">{row['行业']} <span style="color:#4a5a7a">({tk})</span></span>
    <span style="color:{c};font-size:22px;font-weight:700">{s:.0f}</span>
  </div>
  <div style="color:#4a7a5a;font-size:11px;margin-top:6px">{heating}</div>
</div>""", unsafe_allow_html=True)


# ─── Tab 2: 排名 ─────────────────────────────────────────────────────────────
def tab_ranking(scores: pd.DataFrame):
    sort_by = st.selectbox("排序维度",
                           ["总拥挤度"] + DIMS,
                           label_visibility="collapsed")
    disp = scores.sort_values(sort_by, ascending=False)

    # 热力图矩阵
    labels_y = [f"{SECTOR_ETFS[t]['name']}({t})" for t in disp.index]
    z = disp[DIMS].values

    fig = go.Figure(go.Heatmap(
        z=z, x=DIMS, y=labels_y,
        text=z.round(0).astype(int).astype(str),
        texttemplate="%{text}",
        colorscale=[
            [0.00, "#0a1810"], [0.35, "#152b15"],
            [0.60, "#3d3208"], [0.80, "#5a1808"],
            [1.00, "#8b1010"],
        ],
        zmin=0, zmax=100,
        showscale=True,
        colorbar=dict(title=dict(text="得分", font=dict(size=10, color="#6677aa")),
                      thickness=10,
                      tickfont=dict(size=9, color="#6677aa")),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(
        height=430, margin=dict(l=160, r=60, t=30, b=30),
        xaxis=dict(side="top", tickfont=dict(size=11, color="#8899aa")),
        yaxis=dict(tickfont=dict(size=10, color="#c8d8e8")),
        **PT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 总分横向柱状图
    sd = disp.sort_values("总拥挤度")
    bar_labels  = [f"{SECTOR_ETFS[t]['name']}({t})" for t in sd.index]
    bar_colors  = [get_level(v)[1] for v in sd["总拥挤度"]]
    bar_text    = [f"{v:.0f}  {sd.loc[t,'拥挤等级']}" for t, v in zip(sd.index, sd["总拥挤度"])]

    fig2 = go.Figure(go.Bar(
        x=sd["总拥挤度"], y=bar_labels, orientation="h",
        marker_color=bar_colors,
        text=bar_text, textposition="outside",
        textfont=dict(size=10, color="#6677aa"),
        hovertemplate="<b>%{y}</b><br>总拥挤度: %{x:.1f}<extra></extra>",
    ))
    fig2.update_layout(
        xaxis=dict(title="总拥挤度", range=[0, 110],
                   tickfont=dict(size=10, color="#6677aa")),
        height=430, margin=dict(l=160, r=100, t=20, b=40), **PT,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 信号摘要表（纯文字信息密度）
    st.markdown("**快速信号表**")
    rows_out = []
    for tk, row in disp.iterrows():
        cmt = commentary(row)
        rows_out.append({
            "行业(ETF)": f"{SECTOR_ETFS[tk]['name']}({tk})",
            "总分": f"{row['总拥挤度']:.0f}",
            "等级": row["拥挤等级"],
            "主导维度": max(DIMS, key=lambda d: float(row.get(d, 0))),
            "操作信号": cmt["action"],
        })
    st.dataframe(pd.DataFrame(rows_out), use_container_width=True,
                 hide_index=True, height=420)


# ─── Tab 3: 详情 ─────────────────────────────────────────────────────────────
def tab_detail(scores: pd.DataFrame, detail: dict):
    tickers = list(SECTOR_ETFS.keys())
    sel = st.selectbox("选择行业", tickers,
                       format_func=lambda t: f"{SECTOR_ETFS[t]['name']} ({t})")

    if sel not in scores.index:
        st.error("数据暂不可用")
        return

    row   = scores.loc[sel]
    cmt   = commentary(row)
    total = float(row["总拥挤度"])
    _, color = get_level(total)

    # ── 顶部三列
    cg, cr, ct = st.columns([1, 1.1, 2.1])
    with cg:
        st.plotly_chart(gauge_fig(total, f"{SECTOR_ETFS[sel]['name']} 总拥挤度"),
                        use_container_width=True)
        st.markdown(
            f'<div style="text-align:center">{badge(row["拥挤等级"])}</div>'
            f'<div style="text-align:center;margin-top:8px;font-size:13px;color:{color}">'
            f'{cmt["action"]}</div>',
            unsafe_allow_html=True
        )

    with cr:
        st.plotly_chart(
            radar_fig({d: float(row.get(d,50)) for d in DIMS},
                      SECTOR_ETFS[sel]["name"], color),
            use_container_width=True
        )

    with ct:
        st.markdown("**综合研判**")
        st.markdown(
            f'<div class="card card-info" style="color:#b8c8d8;font-size:13px">{cmt["summary"]}</div>',
            unsafe_allow_html=True
        )
        if cmt["structure"]:
            st.markdown(
                f'<div class="card card-note" style="color:#8899aa;font-size:12px">'
                f'📐 结构解读<br>{cmt["structure"]}</div>',
                unsafe_allow_html=True
            )
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

    st.markdown("---")
    # ── 因子明细
    st.markdown("**五维因子明细**")
    records = []

    t_df = detail["trading"]
    if sel in t_df.index:
        r = t_df.loc[sel]
        for factor, raw_key, score_key, desc in [
            ("RSI(14)",       "rsi_raw",      "rsi_score",          "RSI 历史分位——短期超买程度，非景气判断"),
            ("1M动量",         "mom_1m_raw",   "mom_1m_score",       "1M收益率历史分位——高=近期涨幅偏强"),
            ("成交量Surge",     None,           "volume_surge_score", "20日均量/252日均量，放量=交易拥挤信号"),
            ("波动率扩张",      None,           "vol_exp_score",      "30日/90日实现波动率比，波动扩张=交易紧张"),
            ("52W价格位置",     None,           "price_prox_score",   "当前价在52周高低点区间内的位置，100=贴近年高"),
        ]:
            raw_v = f"{r.get(raw_key,'—')}" if raw_key else "—"
            records.append({"维度":"交易拥挤","因子":factor,
                             "原始值":raw_v,"得分":r.get(score_key,"—"),"说明":desc})

    p_df = detail["positioning"]
    if sel in p_df.index:
        r = p_df.loc[sel]
        for factor, score_key, desc in [
            ("成交量中期趋势","vol_trend_score","63日均量/252日均量——持续放量=持仓进入信号"),
            ("Beta扩张",      "beta_exp_score", "近期Beta相对长期Beta，扩张=追涨资金增多"),
            ("相对SPY资金流", "rel_flow_score", "ETF成交额/SPY成交额历史分位——资金集中度代理"),
        ]:
            records.append({"维度":"持仓拥挤","因子":factor,
                             "原始值":"—","得分":r.get(score_key,"—"),"说明":desc})

    v_df = detail["valuation"]
    if sel in v_df.index:
        r = v_df.loc[sel]
        for factor, raw_key, score_key, desc in [
            ("52W Z-Score",   "zscore_52w_raw","zscore_52w_score","价格偏离52周均值标准差，高=估值透支"),
            ("价格/200日均线", None,            "pta_score",       "当前价相对200日均线分位——趋势偏离程度"),
            ("相对SPY超额",   None,            "exc_score",       "3M超额收益历史分位——相对估值溢价代理"),
        ]:
            raw_v = f"{r.get(raw_key,'—')}" if raw_key else "—"
            records.append({"维度":"估值拥挤","因子":factor,
                             "原始值":raw_v,"得分":r.get(score_key,"—"),"说明":desc})

    n_df = detail["narrative"]
    if sel in n_df.index:
        r = n_df.loc[sel]
        for factor, score_key, desc in [
            ("动量加速度","accel_score","1M收益vs3M月均——加速=叙事共振信号"),
            ("收益偏度",  "skew_score", "近60日收益右偏=市场单边乐观，预期集中信号"),
            ("新闻热度",  "news_score", "占位50 | 待接入新闻API/GTrends"),
        ]:
            score_v = r.get(score_key, "—")
            display = f"{score_v} ⚠️占位" if factor == "新闻热度" else score_v
            records.append({"维度":"预期拥挤","因子":factor,
                             "原始值":"—","得分":display,"说明":desc})

    b_df = detail["behavioral"]
    if sel in b_df.index:
        r = b_df.loc[sel]
        pcr_raw = r.get("pcr_raw")
        pcr_disp = f"{pcr_raw:.2f}" if pcr_raw else "无数据"
        for factor, raw_key, score_key, desc in [
            ("上涨日比例",     None,      "up_day_score", "近30日上涨日占比，>70%=情绪极端单边"),
            ("正负收益不对称", None,      "asym_score",   "正/负日均值比——高=坏消息被忽视，最危险信号"),
            ("P/C Ratio",      "pcr_raw", "pcr_score",    "看跌/看涨期权成交量比，低=市场过度乐观"),
        ]:
            raw_v = pcr_disp if raw_key == "pcr_raw" else "—"
            records.append({"维度":"行为拥挤","因子":factor,
                             "原始值":raw_v,"得分":r.get(score_key,"—"),"说明":desc})

    def color_score(val):
        try:
            v = float(str(val).replace("⚠️占位","").strip())
            if v >= 70: return "color:#c0392b;font-weight:600"
            if v >= 58: return "color:#d35400"
            if v >= 42: return "color:#b7950b"
            return "color:#1e8449"
        except: return "color:#4a5a6a"

    rec_df = pd.DataFrame(records)
    st.dataframe(
        rec_df.style.applymap(color_score, subset=["得分"]),
        use_container_width=True, hide_index=True, height=490,
    )

    st.markdown("---")
    # ── 价格走势
    st.markdown("**价格走势（过去 12 个月，归一化 = 100）**")
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
                name="SPY", line=dict(color="#4466aa", width=1.5, dash="dot"),
            ))
        fig.add_hline(y=100, line_dash="dash", line_color="#2a3a4a", opacity=0.5)
        fig.update_layout(
            height=280, hovermode="x unified",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=40, r=20, t=20, b=60),
            yaxis_title="相对表现", **PT,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── Tab 4: 信号监控 ─────────────────────────────────────────────────────────
def tab_signals(scores: pd.DataFrame):
    st.markdown(
        '<div class="note">信号基于当前截面快照。拥挤度时序跟踪（是否快速上升）需历史快照积累，后续版本实现。</div>',
        unsafe_allow_html=True
    )

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
        st.markdown('<div style="color:#b7950b;font-size:13px;font-weight:600;margin-bottom:6px">⚡ 预期先行（叙事 > 价格）</div>',
                    unsafe_allow_html=True)
        found3 = False
        for tk, row in scores.iterrows():
            if (float(row.get("预期拥挤",50)) >= 55 and
                float(row.get("交易拥挤",50)) < 50 and
                float(row["总拥挤度"]) < 62):
                found3 = True
                signal_card(tk, row,
                    f"预期 {row.get('预期拥挤','—'):.0f} > 交易 {row.get('交易拥挤','—'):.0f} | "
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
    st.markdown('<div class="note">⚠️ 占位提示：预期拥挤维度中「新闻热度」因子当前固定为 50（待接入新闻API或Google Trends）。接入后预期拥挤得分将更准确。</div>',
                unsafe_allow_html=True)


# ─── 主入口 ───────────────────────────────────────────────────────────────────
def main():
    weights = sidebar()

    # 归一化权重
    tw = sum(weights.values())
    if tw > 0:
        weights = {k: v/tw for k, v in weights.items()}

    with st.spinner("正在计算五层拥挤度…"):
        try:
            scores, detail = load_scores(tuple(sorted(weights.items())))
        except Exception as e:
            st.error(f"数据加载失败: {e}")
            st.stop()

    st.markdown(
        '<h2 style="font-weight:300;letter-spacing:.04em">'
        '美股行业拥挤度分析 '
        '<span style="color:#00d4aa;font-size:13px;font-weight:400">'
        '· 五层框架 · 赔率视角</span></h2>',
        unsafe_allow_html=True
    )

    t1, t2, t3, t4 = st.tabs(["  市场总览  ", "  行业排名  ", "  行业详情  ", "  信号监控  "])
    with t1: tab_overview(scores, detail["prices"])
    with t2: tab_ranking(scores)
    with t3: tab_detail(scores, detail)
    with t4: tab_signals(scores)


if __name__ == "__main__":
    main()
