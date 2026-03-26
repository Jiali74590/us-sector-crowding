"""
行业拥挤度分析 — 配置文件
五层拥挤框架: 预期 / 持仓 / 交易 / 估值 / 行为
"""

SECTOR_ETFS = {
    "XLK":  {"name": "科技",       "desc": "Technology"},
    "XLF":  {"name": "金融",       "desc": "Financials"},
    "XLE":  {"name": "能源",       "desc": "Energy"},
    "XLV":  {"name": "医疗",       "desc": "Health Care"},
    "XLY":  {"name": "非必需消费",  "desc": "Consumer Discretionary"},
    "XLP":  {"name": "必需消费",    "desc": "Consumer Staples"},
    "XLI":  {"name": "工业",       "desc": "Industrials"},
    "XLB":  {"name": "材料",       "desc": "Materials"},
    "XLU":  {"name": "公用事业",    "desc": "Utilities"},
    "XLRE": {"name": "房地产",     "desc": "Real Estate"},
    "XLC":  {"name": "通信服务",    "desc": "Comm. Services"},
}

# 五大维度默认权重（合计=1）
DIMENSION_WEIGHTS = {
    "预期拥挤": 0.20,
    "持仓拥挤": 0.20,
    "交易拥挤": 0.25,
    "估值拥挤": 0.20,
    "行为拥挤": 0.15,
}

# 拥挤等级阈值 (min, max, label, hex_color)
CROWDING_LEVELS = [
    (80, 101, "极度拥挤", "#c0392b"),
    (60,  80, "高拥挤",   "#d35400"),
    (35,  60, "中等拥挤", "#b7950b"),
    (0,   35, "低拥挤",   "#1e8449"),
]

# ── 子维度权重 ─────────────────────────────────────────────────────────────
TRADING_W = {
    "rsi":              0.25,
    "momentum_1m":      0.20,
    "volume_surge":     0.25,
    "vol_expansion":    0.15,
    "price_proximity":  0.15,
}

POSITIONING_W = {
    "volume_trend":   0.35,
    "beta_expansion": 0.30,
    "relative_flow":  0.35,
}

VALUATION_W = {
    "zscore_52w":      0.35,
    "price_to_ma200":  0.30,
    "excess_vs_spy":   0.35,
}

NARRATIVE_W = {
    "momentum_accel": 0.40,
    "return_skew":    0.30,
    "news_proxy":     0.30,   # 占位，暂返回 50
}

BEHAVIORAL_W = {
    "up_day_ratio":     0.35,
    "return_asymmetry": 0.35,
    "pcr_proxy":        0.30,   # P/C ratio，失败返回 50
}
