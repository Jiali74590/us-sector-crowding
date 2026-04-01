"""
拥挤度时序引擎 — 向量化计算历史拥挤度分数
通过 rolling rank / fixed-scale 在单次计算中生成 N 天的维度分数时序。
不需要数据库存储，直接从 500 天价格数据反推。

注意：叙事维度中媒体热度/PCR 为快照数据，历史计算仅用 accel + skew，
因此叙事维度的绝对值可能与当前截面分数有偏差，但趋势方向准确。
"""

import numpy as np
import pandas as pd
from config import (
    SECTOR_ETFS, DIMENSION_WEIGHTS,
    TRADING_W, POSITIONING_W, VALUATION_W, NARRATIVE_W,
    BREADTH_W, CLEARANCE_W,
)


def _rolling_pct_rank(s: pd.Series, window: int = 252) -> pd.Series:
    """向量化 rolling percentile rank → 0-100"""
    return s.rolling(window, min_periods=20).rank(pct=True) * 100


def _clip_scale(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """线性映射 [lo, hi] → [0, 100]，两端裁切"""
    return ((s - lo) / (hi - lo) * 100).clip(0, 100)


def compute_score_history(prices: pd.DataFrame,
                          volumes: pd.DataFrame,
                          lookback: int = 60,
                          weights: dict = None) -> pd.DataFrame:
    """
    计算过去 lookback 个交易日的六维拥挤度时序。

    Parameters:
        prices:   收盘价 DataFrame (index=dates, columns=tickers)
        volumes:  成交量 DataFrame
        lookback: 回看天数（默认60个交易日≈3个月）
        weights:  维度权重覆盖

    Returns:
        DataFrame, index=dates, columns=MultiIndex (ticker, dimension)
        dimension ∈ {交易拥挤, 持仓拥挤, 估值拥挤, 叙事拥挤, 广度与领导权, 出清状态, 总拥挤度}
    """
    w = weights or DIMENSION_WEIGHTS
    spy_p = prices["SPY"].dropna() if "SPY" in prices.columns else None
    spy_v = volumes["SPY"].dropna() if "SPY" in volumes.columns else None
    dates = prices.index[-lookback:]
    dims = ["交易拥挤", "持仓拥挤", "估值拥挤", "叙事拥挤",
            "广度与领导权", "出清状态", "总拥挤度"]

    results = {}

    for t in SECTOR_ETFS:
        p = prices[t].dropna() if t in prices.columns else pd.Series(dtype=float)
        v = volumes[t].dropna() if t in volumes.columns else pd.Series(dtype=float)

        if len(p) < 60:
            for dim in dims:
                results[(t, dim)] = pd.Series(50.0, index=dates)
            continue

        # ── 交易拥挤 ──────────────────────────────────────────────────
        # RSI(14) rolling percentile
        delta = p.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        rsi_score = _rolling_pct_rank(rsi)

        # 1M Momentum rolling percentile
        mom1m = (p / p.shift(21) - 1) * 100
        mom1m_score = _rolling_pct_rank(mom1m)

        # Volume Surge rolling percentile
        if len(v) > 60:
            surge = v.rolling(20).mean() / (v.rolling(252).mean() + 1e-6)
            surge_score = _rolling_pct_rank(surge)
        else:
            surge_score = pd.Series(50.0, index=p.index)

        # Volatility Expansion rolling percentile
        if len(p) > 90:
            rv30 = p.pct_change().rolling(30).std()
            rv90 = p.pct_change().rolling(90).std()
            vexp = rv30 / (rv90 + 1e-9)
            vexp_score = _rolling_pct_rank(vexp)
        else:
            vexp_score = pd.Series(50.0, index=p.index)

        # Price/50MA rolling percentile
        if len(p) >= 50:
            pp50 = p / p.rolling(50).mean()
            pp50_score = _rolling_pct_rank(pp50)
        else:
            pp50_score = pd.Series(50.0, index=p.index)

        # Price/200MA rolling percentile
        if len(p) >= 200:
            pp200 = p / p.rolling(200).mean()
            pp200_score = _rolling_pct_rank(pp200)
        else:
            pp200_score = pd.Series(50.0, index=p.index)

        # Up Day Ratio (fixed scale)
        up_days = (p.pct_change() > 0).astype(float).rolling(30).mean()
        up_score = _clip_scale(up_days, 0.45, 0.75)

        trading = (
            rsi_score.fillna(50)    * TRADING_W["rsi"] +
            mom1m_score.fillna(50)  * TRADING_W["momentum_1m"] +
            surge_score.fillna(50)  * TRADING_W["volume_surge"] +
            vexp_score.fillna(50)   * TRADING_W["vol_expansion"] +
            pp50_score.fillna(50)   * TRADING_W["price_prox_50"] +
            pp200_score.fillna(50)  * TRADING_W["price_prox_200"] +
            up_score.fillna(50)     * TRADING_W["up_day_ratio"]
        )

        # ── 持仓拥挤 ──────────────────────────────────────────────────
        # Volume Trend rolling percentile
        if len(v) > 63:
            vt = v.rolling(63).mean() / (v.rolling(252).mean() + 1e-6)
            vt_score = _rolling_pct_rank(vt)
        else:
            vt_score = pd.Series(50.0, index=p.index)

        # Beta Expansion (rolling cov/var)
        if spy_p is not None and len(p) > 90:
            r_etf = p.pct_change()
            r_spy = spy_p.pct_change().reindex(r_etf.index)
            beta30 = r_etf.rolling(30).cov(r_spy) / (r_spy.rolling(30).var() + 1e-12)
            beta90 = r_etf.rolling(90).cov(r_spy) / (r_spy.rolling(90).var() + 1e-12)
            beta_exp = beta30 / (beta90.abs() + 0.1)
            beta_score = _clip_scale(beta_exp, 0.5, 2.0)
        else:
            beta_score = pd.Series(50.0, index=p.index)

        # Relative SPY Flow rolling percentile
        if spy_p is not None and spy_v is not None and len(v) > 63:
            etf_flow = (p.reindex(v.index) * v).rolling(20).mean()
            spy_flow = (spy_p.reindex(spy_v.index) * spy_v).rolling(20).mean()
            rf = etf_flow / (spy_flow.reindex(etf_flow.index) + 1e-6)
            rf_score = _rolling_pct_rank(rf)
        else:
            rf_score = pd.Series(50.0, index=p.index)

        positioning = (
            vt_score.fillna(50)    * POSITIONING_W["volume_trend"] +
            beta_score.fillna(50)  * POSITIONING_W["beta_expansion"] +
            rf_score.fillna(50)    * POSITIONING_W["relative_flow"]
        )

        # ── 估值拥挤 ──────────────────────────────────────────────────
        # Z-Score (fixed scale)
        w_len = min(252, len(p))
        mu = p.rolling(w_len).mean()
        sig = p.rolling(w_len).std().replace(0, np.nan)
        z = (p - mu) / sig
        z_score = _clip_scale(z, -3, 3)

        # Excess vs SPY rolling percentile
        if spy_p is not None and len(p) > 63:
            exc = (p.pct_change(63) - spy_p.reindex(p.index).pct_change(63)) * 100
            exc_score = _rolling_pct_rank(exc)
        else:
            exc_score = pd.Series(50.0, index=p.index)

        # PE/PB 为快照数据，历史不可用，用 50 填充
        pe_score = pd.Series(50.0, index=p.index)

        valuation = (
            z_score.fillna(50)   * VALUATION_W["zscore_52w"] +
            exc_score.fillna(50) * VALUATION_W["excess_vs_spy"] +
            pe_score             * VALUATION_W["pe_proxy"]
        )

        # ── 叙事拥挤（仅 accel + skew，无快照数据）────────────────────
        if len(p) > 63:
            r1m = (p / p.shift(22) - 1) * 100
            r3m = (p / p.shift(64) - 1) * 100
            accel = r1m - r3m / 3
            accel_score = _clip_scale(accel, -5, 5)
        else:
            accel_score = pd.Series(50.0, index=p.index)

        if len(p) > 60:
            skew = p.pct_change().rolling(60).skew()
            skew_score = _clip_scale(skew, -2, 2)
        else:
            skew_score = pd.Series(50.0, index=p.index)

        narr_w_total = NARRATIVE_W["momentum_accel"] + NARRATIVE_W["return_skew"]
        narrative = (
            accel_score.fillna(50) * NARRATIVE_W["momentum_accel"] +
            skew_score.fillna(50)  * NARRATIVE_W["return_skew"]
        ) / narr_w_total

        # ── 广度与领导权 ──────────────────────────────────────────────
        # Drawdown from 60d high
        hi60 = p.rolling(min(60, len(p))).max()
        dd = 1.0 - p / (hi60 + 1e-9)
        dd_score = _clip_scale(dd, 0, 0.20)

        # Trend Consistency
        if len(p) >= 50:
            below_50 = (p < p.rolling(50).mean()).astype(float).rolling(20).mean()
            trend_score = (below_50 * 100).clip(0, 100)
        else:
            trend_score = pd.Series(50.0, index=p.index)

        # Momentum Divergence
        if spy_p is not None and len(p) > 63:
            exc_1m = (p.pct_change(21) - spy_p.reindex(p.index).pct_change(21)) * 100
            exc_3m = (p.pct_change(63) - spy_p.reindex(p.index).pct_change(63)) * 100
            div = exc_3m - exc_1m
            div_score = _clip_scale(div, -5, 15)
        else:
            div_score = pd.Series(50.0, index=p.index)

        breadth = (
            dd_score.fillna(0)      * BREADTH_W["drawdown_breadth"] +
            trend_score.fillna(50)  * BREADTH_W["trend_consistency"] +
            div_score.fillna(50)    * BREADTH_W["momentum_divergence"]
        )

        # ── 出清状态 ──────────────────────────────────────────────────
        hi252 = p.rolling(min(252, len(p))).max()
        dd252 = 1.0 - p / (hi252 + 1e-9)
        dd252_score = _clip_scale(dd252, 0, 0.30)

        if len(p) > 90:
            rv10 = p.pct_change().rolling(10).std()
            rv90 = p.pct_change().rolling(90).std()
            vspike = rv10 / (rv90 + 1e-9)
            vspike_score = _rolling_pct_rank(vspike)
        else:
            vspike_score = pd.Series(50.0, index=p.index)

        if len(p) > 60:
            rets = p.pct_change()
            pos = rets.clip(lower=0)
            neg = -rets.clip(upper=0)
            count_pos = (rets > 0).astype(float).rolling(60).sum()
            count_neg = (rets < 0).astype(float).rolling(60).sum()
            mean_pos = pos.rolling(60).sum() / (count_pos + 1e-9)
            mean_neg = neg.rolling(60).sum() / (count_neg + 1e-9)
            asym = mean_pos / (mean_neg + 1e-9)
            asym_score = _clip_scale(1.5 - asym, 0, 1.5)
        else:
            asym_score = pd.Series(50.0, index=p.index)

        clearance = (
            dd252_score.fillna(0)    * CLEARANCE_W["drawdown_depth"] +
            vspike_score.fillna(50)  * CLEARANCE_W["vol_spike"] +
            asym_score.fillna(50)    * CLEARANCE_W["return_asymmetry"]
        )

        # ── 总拥挤度 ──────────────────────────────────────────────────
        total = (
            trading     * w.get("交易拥挤",     0.22) +
            positioning * w.get("持仓拥挤",     0.18) +
            valuation   * w.get("估值拥挤",     0.20) +
            narrative   * w.get("叙事拥挤",     0.20) +
            breadth     * w.get("广度与领导权", 0.20)
        )

        for dim_name, dim_series in [
            ("交易拥挤",     trading),
            ("持仓拥挤",     positioning),
            ("估值拥挤",     valuation),
            ("叙事拥挤",     narrative),
            ("广度与领导权", breadth),
            ("出清状态",     clearance),
            ("总拥挤度",     total),
        ]:
            results[(t, dim_name)] = dim_series.reindex(dates).round(1)

    df = pd.DataFrame(results)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["ticker", "dimension"])
    return df


def get_trend(history: pd.DataFrame, ticker: str,
              dim: str = "总拥挤度") -> dict:
    """获取某 ticker 某维度的趋势数据（7D/30D变化）"""
    if (ticker, dim) not in history.columns:
        return {"current": None, "change_7d": None, "change_30d": None}

    s = history[(ticker, dim)].dropna()
    if len(s) == 0:
        return {"current": None, "change_7d": None, "change_30d": None}

    current = float(s.iloc[-1])
    # 5个交易日 ≈ 1周，22个交易日 ≈ 1月
    change_7d = float(current - s.iloc[-6]) if len(s) >= 6 else None
    change_30d = float(current - s.iloc[-23]) if len(s) >= 23 else None

    return {
        "current":    round(current, 1),
        "change_7d":  round(change_7d, 1) if change_7d is not None else None,
        "change_30d": round(change_30d, 1) if change_30d is not None else None,
    }


def get_trend_series(history: pd.DataFrame, ticker: str,
                     dim: str = "总拥挤度") -> pd.Series:
    """获取某 ticker 某维度的完整时序（用于画图）"""
    if (ticker, dim) not in history.columns:
        return pd.Series(dtype=float)
    return history[(ticker, dim)].dropna()


def trend_arrow(change: float) -> str:
    """变化值 → 趋势箭头 + 颜色 HTML"""
    if change is None:
        return '<span style="color:#4a5a7a">—</span>'
    if abs(change) < 1.0:
        return f'<span style="color:#4a5a7a">→ {change:+.1f}</span>'
    if change > 0:
        return f'<span style="color:#c0392b">↑ {change:+.1f}</span>'
    return f'<span style="color:#1e8449">↓ {change:+.1f}</span>'
