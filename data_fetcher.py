"""
数据获取层 — 价格/成交量、ETF信息、期权P/C
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple

from config import SECTOR_ETFS

_TICKERS = list(SECTOR_ETFS.keys()) + ["SPY"]


@st.cache_data(ttl=3600, show_spinner="获取行情数据…")
def fetch_price_volume(days: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (prices_close, volumes)，列名=ticker"""
    end = datetime.today()
    start = end - timedelta(days=days)
    raw = yf.download(_TICKERS, start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices  = raw["Close"].copy()
        volumes = raw["Volume"].copy()
    else:
        prices  = raw.copy()
        volumes = pd.DataFrame(index=raw.index)

    # 确保所有 ticker 列存在
    for t in _TICKERS:
        if t not in prices.columns:
            prices[t]  = np.nan
            volumes[t] = np.nan

    return prices.dropna(how="all"), volumes.dropna(how="all")


@st.cache_data(ttl=7200, show_spinner="获取ETF元数据…")
def fetch_etf_info() -> Dict[str, dict]:
    """获取 totalAssets / trailingPE / priceToBook / beta 等"""
    result = {}
    for t in _TICKERS:
        try:
            info = yf.Ticker(t).info
            result[t] = {
                "totalAssets":  info.get("totalAssets"),
                "trailingPE":   info.get("trailingPE"),
                "priceToBook":  info.get("priceToBook"),
                "beta":         info.get("beta"),
                "avgVolume":    info.get("averageVolume3Month") or info.get("averageVolume"),
            }
        except Exception:
            result[t] = {}
    return result


@st.cache_data(ttl=3600, show_spinner="获取期权数据…")
def fetch_pcr() -> Dict[str, float]:
    """获取近月Put/Call成交量比率，失败返回 None"""
    pcr = {}
    for t in SECTOR_ETFS:
        try:
            tk = yf.Ticker(t)
            exps = tk.options
            if not exps:
                pcr[t] = None
                continue
            total_c, total_p = 0, 0
            for exp in exps[:2]:
                chain  = tk.option_chain(exp)
                total_c += chain.calls["volume"].fillna(0).sum()
                total_p += chain.puts["volume"].fillna(0).sum()
            pcr[t] = round(total_p / (total_c + 1e-6), 3)
        except Exception:
            pcr[t] = None
    return pcr
