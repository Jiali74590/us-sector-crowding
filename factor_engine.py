"""
六层拥挤因子计算引擎
六层框架: 叙事拥挤 / 持仓拥挤 / 交易拥挤 / 估值拥挤 / 广度与领导权 / 出清状态
出清状态仅进 state machine，不影响总拥挤分。
"""

import numpy as np
import pandas as pd
from typing import Dict
from config import (
    SECTOR_ETFS,
    TRADING_W, POSITIONING_W, VALUATION_W, NARRATIVE_W,
    BREADTH_W, CLEARANCE_W, STATE_CONFIG,
    INDICATOR_QUALITY, DIMENSION_WEIGHTS, SPREAD_POWER,
)


# ── 工具 ──────────────────────────────────────────────────────────────────────

def hist_pct(series: pd.Series, value: float, window: int = 252) -> float:
    """value 在 series 最近 window 期中的历史分位数 → 0-100"""
    s = series.dropna()
    if len(s) < 10 or np.isnan(value):
        return 50.0
    hist = s.iloc[-window:] if len(s) > window else s
    return round(float((hist < value).sum() / len(hist) * 100), 1)


def _spread_stretch(s):
    """重新展开被加权平均压缩的维度分数，缓解方差塌缩。
    单调映射 [0,100]→[0,100]，保持 0→0, 50→50, 100→100。"""
    z = (s - 50) / 50
    return (50 + np.sign(z) * (np.abs(z) ** SPREAD_POWER) * 50).clip(0, 100).round(1)


def safe_float(val, default=50.0) -> float:
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except Exception:
        return default


# ── 1. 交易拥挤 ───────────────────────────────────────────────────────────────

def compute_trading(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    衡量短期交易热度（不等于景气度）:
    RSI / 1M动量 / 成交量Surge / 波动率扩张 /
    价格/50MA / 价格/200MA（从估值移入） / 上涨日比例（从行为移入）
    """
    rows = {}
    for t in SECTOR_ETFS:
        p = prices[t].dropna() if t in prices.columns else pd.Series(dtype=float)
        v = volumes[t].dropna() if t in volumes.columns else pd.Series(dtype=float)

        # RSI(14)
        delta = p.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi_s = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        rsi_v = safe_float(rsi_s.iloc[-1]) if len(rsi_s) > 14 else 50.0
        rsi_score = hist_pct(rsi_s, rsi_v)

        # 1M 动量
        mom1m_s = (p / p.shift(21) - 1) * 100
        mom1m_v = safe_float(mom1m_s.iloc[-1]) if len(p) > 21 else np.nan
        mom1m_score = hist_pct(mom1m_s, mom1m_v) if not np.isnan(mom1m_v) else 50.0

        # 成交量 surge
        if len(v) > 60:
            surge_s = v.rolling(20).mean() / (v.rolling(252).mean() + 1e-6)
            surge_v = safe_float(surge_s.iloc[-1])
            vol_surge_score = hist_pct(surge_s, surge_v)
        else:
            vol_surge_score = 50.0

        # 波动率扩张
        if len(p) > 90:
            rv30 = p.pct_change().rolling(30).std()
            rv90 = p.pct_change().rolling(90).std()
            vexp_s = rv30 / (rv90 + 1e-9)
            vexp_v = safe_float(vexp_s.iloc[-1])
            vol_exp_score = hist_pct(vexp_s.dropna(), vexp_v)
        else:
            vol_exp_score = 50.0

        # 价格/50日均线
        if len(p) >= 50:
            ma50       = p.rolling(50).mean()
            pp50_s     = p / ma50
            pp50_v     = safe_float(pp50_s.iloc[-1])
            pp50_score = hist_pct(pp50_s.dropna(), pp50_v)
            pp50_raw   = round(pp50_v, 3)
        else:
            pp50_score = 50.0
            pp50_raw   = float("nan")

        # 价格/200日均线（移入交易层）
        if len(p) >= 200:
            ma200      = p.rolling(200).mean()
            pp200_s    = p / ma200
            pp200_hist = pp200_s.dropna()
            pp200_v    = safe_float(pp200_s.iloc[-1])
            pp200_score = hist_pct(pp200_hist, pp200_v)
            pp200_raw  = round(pp200_v, 3)
            pp200_n    = min(len(pp200_hist), 252)
        else:
            pp200_score = 50.0
            pp200_raw  = float("nan")
            pp200_n    = 0

        # 上涨日比例（移入交易层）
        if len(p) > 30:
            rets    = p.pct_change().dropna().iloc[-30:]
            up_r    = float((rets > 0).sum()) / len(rets)
            up_score = min(100.0, max(0.0, (up_r - 0.45) / 0.30 * 100))
            up_raw  = round(up_r * 100, 1)
        else:
            up_score = 50.0
            up_raw   = float("nan")

        rows[t] = {
            "rsi_raw":            round(rsi_v, 1),
            "mom_1m_raw":         round(safe_float(mom1m_v, 0.0), 2),
            "rsi_score":          rsi_score,
            "mom_1m_score":       mom1m_score,
            "volume_surge_score": vol_surge_score,
            "vol_exp_score":      vol_exp_score,
            "pp50_raw":           pp50_raw,
            "pp50_score":         round(pp50_score, 1),
            "pp200_raw":          pp200_raw,
            "pp200_score":        round(pp200_score, 1),
            "pp200_n":            pp200_n,
            "up_day_raw":         up_raw,
            "up_day_score":       round(up_score, 1),
        }

    df = pd.DataFrame(rows).T
    for c in ["rsi_score", "mom_1m_score", "volume_surge_score", "vol_exp_score",
              "pp50_score", "pp200_score", "up_day_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    pp200_col = pd.to_numeric(df["pp200_score"], errors="coerce").fillna(50)

    df["交易拥挤"] = (
        df["rsi_score"]           * TRADING_W["rsi"]            +
        df["mom_1m_score"]        * TRADING_W["momentum_1m"]    +
        df["volume_surge_score"]  * TRADING_W["volume_surge"]   +
        df["vol_exp_score"]       * TRADING_W["vol_expansion"]  +
        df["pp50_score"].fillna(50) * TRADING_W["price_prox_50"] +
        pp200_col                 * TRADING_W["price_prox_200"] +
        df["up_day_score"].fillna(50) * TRADING_W["up_day_ratio"]
    ).round(1)
    df["交易拥挤"] = _spread_stretch(df["交易拥挤"])
    return df


# ── 2. 持仓拥挤 ───────────────────────────────────────────────────────────────

def compute_positioning(prices: pd.DataFrame, volumes: pd.DataFrame,
                        info: Dict = None) -> pd.DataFrame:
    """
    衡量资金是否持续、集中流入:
    成交量中期趋势 / Beta扩张 / 相对SPY资金流 / AUM资金沉淀（新）
    """
    spy_p = prices["SPY"].dropna() if "SPY" in prices.columns else None
    spy_v = volumes["SPY"].dropna() if "SPY" in volumes.columns else None
    rows  = {}

    for t in SECTOR_ETFS:
        p = prices[t].dropna() if t in prices.columns else pd.Series(dtype=float)
        v = volumes[t].dropna() if t in volumes.columns else pd.Series(dtype=float)

        # 成交量中期趋势
        if len(v) > 63:
            vt_s = v.rolling(63).mean() / (v.rolling(252).mean() + 1e-6)
            vt_v = safe_float(vt_s.iloc[-1])
            vol_trend_score = hist_pct(vt_s.dropna(), vt_v)
        else:
            vol_trend_score = 50.0

        # Beta 扩张
        if spy_p is not None and len(p) > 90:
            r   = p.pct_change().dropna()
            sr  = spy_p.pct_change().reindex(r.index).dropna()
            idx = r.index.intersection(sr.index)
            if len(idx) >= 30:
                r2, sr2 = r.loc[idx], sr.loc[idx]
                def beta(ret, mkt):
                    cov = np.cov(ret.values, mkt.values)
                    return cov[0][1] / (cov[1][1] + 1e-12)
                b30 = beta(r2.iloc[-30:], sr2.iloc[-30:])
                b90 = beta(r2.iloc[-90:], sr2.iloc[-90:])
                beta_exp = b30 / (abs(b90) + 0.1)
                beta_exp_score = min(100, max(0, (beta_exp - 0.5) / 1.5 * 100))
            else:
                beta_exp_score = 50.0
        else:
            beta_exp_score = 50.0

        # 相对 SPY 资金流
        if spy_p is not None and spy_v is not None and len(v) > 63:
            etf_flow = (p.reindex(v.index) * v).rolling(20).mean()
            spy_flow = (spy_p.reindex(spy_v.index) * spy_v).rolling(20).mean()
            rf_s = (etf_flow / (spy_flow.reindex(etf_flow.index) + 1e-6)).dropna()
            if len(rf_s) > 10:
                rf_v = safe_float(rf_s.iloc[-1])
                rel_flow_score = hist_pct(rf_s, rf_v)
            else:
                rel_flow_score = 50.0
        else:
            rel_flow_score = 50.0

        rows[t] = {
            "vol_trend_score": vol_trend_score,
            "beta_exp_score":  round(beta_exp_score, 1),
            "rel_flow_score":  rel_flow_score,
        }

    df = pd.DataFrame(rows).T.astype(float)

    # ── AUM 资金沉淀密度（横截面排名）──────────────────────────────
    # totalAssets / 20日均日成交额 → 资金沉淀越多、持仓越集中
    aum_density = {}
    for t in SECTOR_ETFS:
        p = prices[t].dropna() if t in prices.columns else pd.Series(dtype=float)
        v = volumes[t].dropna() if t in volumes.columns else pd.Series(dtype=float)
        aum = float((info or {}).get(t, {}).get("totalAssets") or 0)
        if aum > 0 and len(p) > 20 and len(v) > 20:
            avg_dvol = float((p.iloc[-20:] * v.iloc[-20:]).mean())
            aum_density[t] = aum / (avg_dvol + 1e-6)
        else:
            aum_density[t] = float("nan")

    aum_s = pd.Series(aum_density, dtype=float)
    n_aum = int(aum_s.notna().sum())
    has_fund_flow = n_aum >= 3
    for t in SECTOR_ETFS:
        val = aum_s.get(t, float("nan"))
        if has_fund_flow and not np.isnan(val):
            df.loc[t, "fund_flow_score"] = round(
                float((aum_s.dropna() < val).sum()) / max(n_aum, 1) * 100, 1)
            df.loc[t, "fund_flow_raw"] = round(val, 1)
        else:
            df.loc[t, "fund_flow_score"] = float("nan")
            df.loc[t, "fund_flow_raw"] = float("nan")

    # 动态权重归一化（AUM 数据可能缺失）
    w_vt = POSITIONING_W["volume_trend"]
    w_be = POSITIONING_W["beta_expansion"]
    w_rf = POSITIONING_W["relative_flow"]
    w_ff = POSITIONING_W["fund_flow"]

    if has_fund_flow:
        ff_filled = pd.to_numeric(df["fund_flow_score"], errors="coerce").fillna(50)
        total_w = w_vt + w_be + w_rf + w_ff
        df["持仓拥挤"] = (
            df["vol_trend_score"] * w_vt +
            df["beta_exp_score"]  * w_be +
            df["rel_flow_score"]  * w_rf +
            ff_filled             * w_ff
        ).round(1) / total_w
    else:
        total_w = w_vt + w_be + w_rf
        df["持仓拥挤"] = (
            df["vol_trend_score"] * w_vt +
            df["beta_exp_score"]  * w_be +
            df["rel_flow_score"]  * w_rf
        ).round(1) / total_w

    df["持仓拥挤"] = _spread_stretch(df["持仓拥挤"])
    return df


# ── 3. 估值拥挤 ───────────────────────────────────────────────────────────────

def compute_valuation(prices: pd.DataFrame, info: Dict = None) -> pd.DataFrame:
    """
    衡量市场定价是否已透支远期预期:
    52W Z-Score / 相对SPY超额 / PE/PB横截面代理（移出价格/200MA）
    """
    spy_p  = prices["SPY"].dropna() if "SPY" in prices.columns else None
    rows   = {}
    pe_vals = {}
    pb_vals = {}

    for t in SECTOR_ETFS:
        p = prices[t].dropna() if t in prices.columns else pd.Series(dtype=float)

        # 52W Z-Score → 映射到 0-100
        if len(p) >= 60:
            w   = min(252, len(p))
            mu  = p.rolling(w).mean()
            sig = p.rolling(w).std().replace(0, np.nan)
            zs  = (p - mu) / sig
            z_v = safe_float(zs.iloc[-1], 0.0)
            z_score = min(100, max(0, (z_v + 3) / 6 * 100))
        else:
            z_score, z_v = 50.0, 0.0

        # 相对 SPY 3M 超额收益
        if spy_p is not None and len(p) > 63:
            exc_s = (p.pct_change(63) - spy_p.reindex(p.index).pct_change(63)) * 100
            exc_v = safe_float(exc_s.iloc[-1])
            exc_score = hist_pct(exc_s.dropna(), exc_v)
        else:
            exc_score = 50.0

        # PE/PB 原始值（跨行业横截面排名在下方统一处理）
        if info and t in info:
            pe_raw = info[t].get("trailingPE")
            pb_raw = info[t].get("priceToBook")
            pe_vals[t] = float(pe_raw) if pe_raw and pe_raw > 0 else float("nan")
            pb_vals[t] = float(pb_raw) if pb_raw and pb_raw > 0 else float("nan")
        else:
            pe_vals[t] = float("nan")
            pb_vals[t] = float("nan")

        rows[t] = {
            "zscore_52w_raw":   round(z_v, 2),
            "zscore_52w_score": round(z_score, 1),
            "exc_score":        exc_score,
        }

    df = pd.DataFrame(rows).T
    for c in ["zscore_52w_score", "exc_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # PE/PB 横截面分位计算
    pe_s = pd.Series(pe_vals, dtype=float)
    pb_s = pd.Series(pb_vals, dtype=float)
    n_pe = int(pe_s.notna().sum())
    n_pb = int(pb_s.notna().sum())

    for t in SECTOR_ETFS:
        pe_v = pe_s.get(t, float("nan"))
        pb_v = pb_s.get(t, float("nan"))
        pe_ok = not np.isnan(pe_v) if not isinstance(pe_v, float) else not np.isnan(float(pe_v))
        pb_ok = not np.isnan(pb_v) if not isinstance(pb_v, float) else not np.isnan(float(pb_v))
        pe_rank = float((pe_s < pe_v).sum()) / max(n_pe, 1) * 100 if pe_ok else float("nan")
        pb_rank = float((pb_s < pb_v).sum()) / max(n_pb, 1) * 100 if pb_ok else float("nan")

        if not np.isnan(pe_rank) and not np.isnan(pb_rank):
            pe_proxy = (pe_rank + pb_rank) / 2
        elif not np.isnan(pe_rank):
            pe_proxy = pe_rank
        elif not np.isnan(pb_rank):
            pe_proxy = pb_rank
        else:
            pe_proxy = float("nan")

        df.loc[t, "pe_raw"]        = round(float(pe_v), 1) if pe_ok else float("nan")
        df.loc[t, "pb_raw"]        = round(float(pb_v), 2) if pb_ok else float("nan")
        df.loc[t, "pe_proxy_score"] = round(pe_proxy, 1) if not np.isnan(pe_proxy) else float("nan")

    pe_filled = pd.to_numeric(df["pe_proxy_score"], errors="coerce").fillna(50)
    df["估值拥挤"] = (
        df["zscore_52w_score"] * VALUATION_W["zscore_52w"]    +
        df["exc_score"]        * VALUATION_W["excess_vs_spy"] +
        pe_filled              * VALUATION_W["pe_proxy"]
    ).round(1)
    df["估值拥挤"] = _spread_stretch(df["估值拥挤"])
    return df


# ── 4. 叙事拥挤 ───────────────────────────────────────────────────────────────

def compute_narrative(prices: pd.DataFrame,
                      news_counts: Dict = None,
                      pcr_data: Dict = None) -> pd.DataFrame:
    """
    衡量市场叙事/预期是否已高度集中（原: 预期拥挤, 新增 PCR）:
    动量加速度 / 收益偏度 / 媒体热度代理 / PCR情绪信号
    缺失指标动态归一化权重，不使用0分填充。
    """
    if news_counts is not None:
        news_has_data = {
            t: bool((news_counts.get(t) or {}).get("has_data", False))
            for t in SECTOR_ETFS
        }
        valid_counts = {
            t: int((news_counts.get(t) or {}).get("count_7d", 0) or 0)
            for t in SECTOR_ETFS if news_has_data.get(t, False)
        }
        all_counts = list(valid_counts.values())
        n_valid    = len(all_counts)
    else:
        news_has_data = {}
        valid_counts  = {}
        all_counts    = []
        n_valid       = 0

    rows = {}
    for t in SECTOR_ETFS:
        p = prices[t].dropna() if t in prices.columns else pd.Series(dtype=float)

        # 动量加速度
        if len(p) > 63:
            r1m   = (p.iloc[-1] / p.iloc[-22] - 1) * 100
            r3m   = (p.iloc[-1] / p.iloc[-64] - 1) * 100
            accel = r1m - r3m / 3
            accel_score = min(100.0, max(0.0, (accel + 5) / 10 * 100))
        else:
            accel_score = 50.0

        # 收益偏度
        if len(p) > 30:
            skew = float(p.pct_change().dropna().iloc[-60:].skew())
            skew_score = min(100.0, max(0.0, (skew + 2) / 4 * 100))
        else:
            skew_score = 50.0

        # 媒体热度代理
        this_has_data = news_has_data.get(t, False)
        if this_has_data and n_valid >= 1:
            cnt          = valid_counts.get(t, 0)
            rank         = sum(1 for x in all_counts if x < cnt)
            news_score   = round(rank / n_valid * 100, 1)
            news_raw     = f"{cnt}条/7天"
            news_missing = False
        else:
            news_score   = float("nan")
            news_raw     = "N/A"
            news_missing = True

        # PCR 情绪（低 PCR = 过度乐观 = 叙事拥挤）
        pcr = (pcr_data or {}).get(t)
        if pcr and pcr > 0:
            pcr_score   = min(100.0, max(0.0, (2.0 - pcr) / 1.5 * 100))
            pcr_raw     = round(pcr, 3)
            pcr_missing = False
        else:
            pcr_score   = float("nan")
            pcr_raw     = None
            pcr_missing = True

        # 动态权重归一化
        w_a = NARRATIVE_W["momentum_accel"]
        w_s = NARRATIVE_W["return_skew"]
        w_n = NARRATIVE_W["news_proxy"]
        w_p = NARRATIVE_W["pcr_sentiment"]

        active = [(accel_score, w_a), (skew_score, w_s)]
        if not news_missing:
            active.append((news_score, w_n))
        if not pcr_missing:
            active.append((pcr_score, w_p))
        total_w    = sum(w for _, w in active)
        narr_score = sum(s * w for s, w in active) / (total_w + 1e-9)

        rows[t] = {
            "accel_score":  round(accel_score, 1),
            "skew_score":   round(skew_score, 1),
            "news_score":   news_score,
            "news_raw":     news_raw,
            "news_missing": news_missing,
            "pcr_score":    pcr_score,
            "pcr_raw":      pcr_raw,
            "pcr_missing":  pcr_missing,
            "accel_eff_w":  round(w_a / total_w, 4),
            "skew_eff_w":   round(w_s / total_w, 4),
            "news_eff_w":   round((w_n / total_w) if not news_missing else 0.0, 4),
            "pcr_eff_w":    round((w_p / total_w) if not pcr_missing  else 0.0, 4),
            "_narr_score":  round(narr_score, 1),
            "_renorm":      news_missing or pcr_missing,
        }

    df = pd.DataFrame(rows).T
    for c in ["accel_score", "skew_score", "accel_eff_w", "skew_eff_w",
              "news_eff_w", "pcr_eff_w", "_narr_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["叙事拥挤"] = _spread_stretch(df["_narr_score"])
    df = df.drop(columns=["_narr_score"])
    return df


# ── 5. 广度与领导权 ───────────────────────────────────────────────────────────

def compute_breadth(prices: pd.DataFrame) -> pd.DataFrame:
    """
    广度与领导权（ETF级代理，高分=广度恶化=拥挤松动信号）:
    - 距近期高点回撤（60日高点）
    - 趋势均线一致性（近20日低于50MA占比）
    - 短长期动量背离（3M超额 - 1M超额）
    """
    spy_p = prices["SPY"].dropna() if "SPY" in prices.columns else None
    rows  = {}

    for t in SECTOR_ETFS:
        p = prices[t].dropna() if t in prices.columns else pd.Series(dtype=float)

        # 距近期高点回撤（60日高点，20%回撤=满分100）
        if len(p) >= 10:
            hi60    = p.rolling(min(60, len(p))).max()
            dd      = 1.0 - p / (hi60 + 1e-9)
            dd_v    = safe_float(dd.iloc[-1], 0.0)
            dd_score = min(100.0, max(0.0, dd_v / 0.20 * 100))
        else:
            dd_score = 0.0

        # 趋势一致性（近20日低于50MA天数占比）
        if len(p) >= 50:
            ma50     = p.rolling(50).mean()
            below_50 = (p < ma50).rolling(20).mean()
            below_v  = safe_float(below_50.iloc[-1], 0.0)
            trend_score = min(100.0, max(0.0, below_v * 100))
        else:
            trend_score = 50.0

        # 动量背离（3M超额 - 1M超额 > 0 = 动量衰减）
        if spy_p is not None and len(p) > 63:
            exc_1m = safe_float(
                (p.pct_change(21) - spy_p.reindex(p.index).pct_change(21)).iloc[-1] * 100, 0.0)
            exc_3m = safe_float(
                (p.pct_change(63) - spy_p.reindex(p.index).pct_change(63)).iloc[-1] * 100, 0.0)
            div    = exc_3m - exc_1m
            div_score = min(100.0, max(0.0, (div + 5.0) / 20.0 * 100))
        else:
            div_score = 50.0

        rows[t] = {
            "dd_breadth_score": round(dd_score, 1),
            "trend_score":      round(trend_score, 1),
            "div_score":        round(div_score, 1),
        }

    df = pd.DataFrame(rows).T.astype(float)
    df["广度与领导权"] = (
        df["dd_breadth_score"] * BREADTH_W["drawdown_breadth"]    +
        df["trend_score"]      * BREADTH_W["trend_consistency"]   +
        df["div_score"]        * BREADTH_W["momentum_divergence"]
    ).round(1)
    df["广度与领导权"] = _spread_stretch(df["广度与领导权"])
    return df


# ── 6. 出清状态 ───────────────────────────────────────────────────────────────

def compute_clearance(prices: pd.DataFrame) -> pd.DataFrame:
    """
    出清状态（仅进 state machine，不影响总拥挤分，高分=出清信号强）:
    - 距252日高点回撤深度
    - 波动率突刺（10日/90日）
    - 正负收益不对称（坏消息更大=出清信号）
    """
    rows = {}
    for t in SECTOR_ETFS:
        p = prices[t].dropna() if t in prices.columns else pd.Series(dtype=float)

        # 距252日高点回撤（30%=满分100）
        if len(p) >= 10:
            hi252    = p.rolling(min(252, len(p))).max()
            dd252    = 1.0 - p / (hi252 + 1e-9)
            dd252_v  = safe_float(dd252.iloc[-1], 0.0)
            dd_score = min(100.0, max(0.0, dd252_v / 0.30 * 100))
        else:
            dd_score = 0.0

        # 波动率突刺（10日/90日）
        if len(p) > 90:
            rv10     = p.pct_change().rolling(10).std()
            rv90     = p.pct_change().rolling(90).std()
            vspike   = rv10 / (rv90 + 1e-9)
            vspike_v = safe_float(vspike.iloc[-1])
            vspike_score = hist_pct(vspike.dropna(), vspike_v)
        else:
            vspike_score = 50.0

        # 正负收益不对称（出清方向：坏消息主导→高分）
        if len(p) > 60:
            rets    = p.pct_change().dropna().iloc[-60:]
            up_rets = rets[rets > 0]
            dn_rets = rets[rets < 0]
            if len(up_rets) > 5 and len(dn_rets) > 5:
                asym = float(up_rets.mean()) / (abs(float(dn_rets.mean())) + 1e-9)
                # asym<1 = 坏消息占主导（出清）→ 高分
                asym_score = min(100.0, max(0.0, (1.5 - asym) / 1.5 * 100))
            else:
                asym_score = 50.0
        else:
            asym_score = 50.0

        rows[t] = {
            "dd_depth_score": round(dd_score, 1),
            "vspike_score":   round(vspike_score, 1),
            "rasym_score":    round(asym_score, 1),
        }

    df = pd.DataFrame(rows).T.astype(float)
    df["出清状态"] = (
        df["dd_depth_score"] * CLEARANCE_W["drawdown_depth"]    +
        df["vspike_score"]   * CLEARANCE_W["vol_spike"]         +
        df["rasym_score"]    * CLEARANCE_W["return_asymmetry"]
    ).round(1)
    df["出清状态"] = _spread_stretch(df["出清状态"])
    return df


# ── 状态机 ────────────────────────────────────────────────────────────────────

def classify_state(crowding_score: float,
                   breadth_score:  float,
                   clearance_score: float) -> dict:
    """
    7-state 行业状态机（优先级从高到低，首先匹配）:
    1. 踩踏风险区    crowding≥72 且 clearance≤35
    2. 拥挤松动中    crowding≥58 且 clearance≥60
    3. 高拥挤/赔率下降  crowding≥62
    4. 接近出清      crowding≥48 且 clearance≥65
    5. 拥挤扩张中    crowding≥48 且 breadth≥55
    6. 反向观察区    crowding<40 且 clearance≥55
    7. 低拥挤/早期升温  默认
    """
    c  = float(crowding_score)
    br = float(breadth_score)
    cl = float(clearance_score)

    if c >= 72 and cl <= 35:
        state = "踩踏风险区"
        expl  = (f"总拥挤度 {c:.0f} 分极高且出清信号极弱 {cl:.0f} 分，"
                 "持仓集中度已达危险阈值，踩踏风险最高。")
    elif c >= 58 and cl >= 60:
        state = "拥挤松动中"
        expl  = (f"总拥挤度 {c:.0f} 分偏高，但出清信号 {cl:.0f} 分已明显增强，"
                 "市场正主动或被动释放拥挤压力。")
    elif c >= 62:
        state = "高拥挤/赔率下降"
        expl  = (f"总拥挤度 {c:.0f} 分处于高位，出清信号 {cl:.0f} 分尚弱，"
                 "向上赔率已显著收窄，继续追高期望回报偏低。")
    elif c >= 48 and cl >= 65:
        state = "接近出清"
        expl  = (f"总拥挤度 {c:.0f} 分中等，出清信号 {cl:.0f} 分较强，"
                 "建议等待出清完成后再评估布局机会。")
    elif c >= 48 and br >= 55:
        state = "拥挤扩张中"
        expl  = (f"总拥挤度 {c:.0f} 分偏高且广度恶化 {br:.0f} 分，"
                 "拥挤格局仍在扩张，结构出现内部松动先兆。")
    elif c < 40 and cl >= 55:
        state = "反向观察区"
        expl  = (f"总拥挤度低至 {c:.0f} 分，出清信号 {cl:.0f} 分，"
                 "低拥挤叠加出清信号，可能处于反转积累阶段，值得逆向关注。")
    else:
        state = "低拥挤/早期升温"
        expl  = (f"总拥挤度 {c:.0f} 分偏低，各维度无极端信号，"
                 "处于低关注或早期升温阶段，配置赔率相对合理。")

    cfg = STATE_CONFIG[state]
    return {
        "state":       state,
        "action":      cfg["action"],
        "color":       cfg["color"],
        "bg":          cfg["bg"],
        "icon":        cfg["icon"],
        "explanation": expl,
    }


# ── 打分卡构建器 ──────────────────────────────────────────────────────────────

def build_scorecard(ticker: str,
                    t_df:  pd.DataFrame,
                    p_df:  pd.DataFrame,
                    v_df:  pd.DataFrame,
                    n_df:  pd.DataFrame,
                    br_df: pd.DataFrame,
                    cl_df: pd.DataFrame) -> list:
    """
    为单个 ticker 构建结构化打分卡（6维），供详情页展示。
    出清状态子指标以灰色注明不计入总分。
    """
    import math as _mth
    records = []

    # ── 交易拥挤子指标 ─────────────────────────────────────────────────────────
    if ticker in t_df.index:
        r = t_df.loc[ticker]

        # 常规指标（简单循环）
        simple_items = [
            ("RSI(14)",     f"{safe_float(r.get('rsi_raw', 50)):.1f}",
             r.get("rsi_score", 50),           TRADING_W["rsi"],
             "RSI历史分位——短期超买程度，非景气判断"),
            ("1M动量",      f"{safe_float(r.get('mom_1m_raw', 0), 0):.2f}%",
             r.get("mom_1m_score", 50),         TRADING_W["momentum_1m"],
             "1M收益率历史分位——近期涨幅强弱"),
            ("成交量Surge", "—",
             r.get("volume_surge_score", 50),   TRADING_W["volume_surge"],
             "20日均量/252日均量——成交放量程度"),
            ("波动率扩张",  "—",
             r.get("vol_exp_score", 50),        TRADING_W["vol_expansion"],
             "30日/90日波动率比——波动扩张程度"),
            ("上涨日比例",  f"{safe_float(r.get('up_day_raw', 50), 0):.0f}%" if not _mth.isnan(safe_float(r.get('up_day_raw', float('nan')), float('nan'))) else "—",
             r.get("up_day_score", 50),         TRADING_W["up_day_ratio"],
             "近30日上涨日占比（>70%=情绪偏热）"),
        ]
        for name, raw, score, w, desc in simple_items:
            sf = safe_float(score, 50)
            records.append({"维度": "交易拥挤", "子指标": name, "原始值": raw,
                             "历史分位": round(sf, 1), "维度内权重": f"{w*100:.0f}%",
                             "子项贡献": round(sf * w, 1), "说明": desc,
                             "status": "ok"})

        # 价格/50日均线
        pp50_rv  = r.get("pp50_raw", float("nan"))
        pp50_ok  = not _mth.isnan(safe_float(pp50_rv, float("nan")))
        pp50_raw = f"{safe_float(pp50_rv, 1.0):.3f}x" if pp50_ok else "—"
        pp50_sf  = safe_float(r.get("pp50_score", 50), 50)
        records.append({"维度": "交易拥挤", "子指标": "价格/50日均线",
                        "原始值": pp50_raw,
                        "历史分位": round(pp50_sf, 1),
                        "维度内权重": f"{TRADING_W['price_prox_50']*100:.0f}%",
                        "子项贡献": round(pp50_sf * TRADING_W["price_prox_50"], 1),
                        "说明": "当前价格/50日均线比值的历史分位——短期超买程度",
                        "status": "ok"})

        # 价格/200日均线（特殊处理：区分缺失/真实低分）
        pp200_rv = r.get("pp200_raw", float("nan"))
        pp200_n  = int(r.get("pp200_n", 0))
        pp200_sf = safe_float(r.get("pp200_score", 50), 50)
        pp200_w  = TRADING_W["price_prox_200"]
        pp200_ok = pp200_n > 0 and not _mth.isnan(safe_float(pp200_rv, float("nan")))

        if not pp200_ok:
            records.append({
                "维度": "交易拥挤", "子指标": "价格/200日均线",
                "原始值": "—", "历史分位": float("nan"),
                "维度内权重": "0%（缺失）", "子项贡献": 0.0,
                "说明": "数据不足200日，无法计算200日均线比值",
                "status": "missing",
            })
        else:
            pp200_rf   = safe_float(pp200_rv, 1.0)
            pp200_pct  = round(pp200_sf, 1)
            pp200_rank = round(pp200_sf / 100.0 * pp200_n)
            _low       = pp200_pct < 5.0
            pp200_desc = (
                f"当前价格为200日均线的{pp200_rf:.3f}倍，在过去{pp200_n}个有效样本中处于{pp200_pct:.0f}%分位"
                + ("，价格未偏离长期趋势上方，该指标不给交易拥挤加分。（真实低分，非数据缺失）" if _low else "")
            )
            records.append({
                "维度": "交易拥挤", "子指标": "价格/200日均线",
                "原始值": f"{pp200_rf:.3f}x",
                "历史分位": pp200_pct,
                "维度内权重": f"{pp200_w*100:.0f}%",
                "子项贡献": round(pp200_sf * pp200_w, 1),
                "说明": pp200_desc,
                "status": "ok",
                "pct_context":  f"{pp200_pct:.0f}%（{pp200_rank}/{pp200_n}）",
                "low_score_note": "真实低分" if _low else "",
            })

    # ── 持仓拥挤子指标 ─────────────────────────────────────────────────────────
    if ticker in p_df.index:
        r = p_df.loc[ticker]
        # 判断 fund_flow 是否可用
        ff_val = r.get("fund_flow_score", float("nan"))
        ff_ok  = not _mth.isnan(safe_float(ff_val, float("nan")))
        # 有效权重（动态归一化）
        _pw_active = (POSITIONING_W["volume_trend"] +
                      POSITIONING_W["beta_expansion"] +
                      POSITIONING_W["relative_flow"] +
                      (POSITIONING_W["fund_flow"] if ff_ok else 0))
        def _eff_w(key):
            return POSITIONING_W[key] / _pw_active

        items = [
            ("成交量中期趋势", "—",
             r.get("vol_trend_score", 50),   _eff_w("volume_trend"),
             "63日均量/252日均量——中期资金流入趋势"),
            ("Beta扩张",       "—",
             r.get("beta_exp_score", 50),    _eff_w("beta_expansion"),
             "近期Beta/中期Beta——追涨资金流入信号"),
            ("相对SPY资金流",  "—",
             r.get("rel_flow_score", 50),    _eff_w("relative_flow"),
             "ETF成交额/SPY成交额历史分位——资金集中度代理"),
        ]
        for name, raw, score, w, desc in items:
            sf = safe_float(score, 50)
            records.append({"维度": "持仓拥挤", "子指标": name, "原始值": raw,
                             "历史分位": round(sf, 1), "维度内权重": f"{w*100:.0f}%",
                             "子项贡献": round(sf * w, 1), "说明": desc,
                             "status": "ok"})

        # AUM 资金沉淀
        if ff_ok:
            ff_sf = safe_float(ff_val, 50)
            ff_raw_v = r.get("fund_flow_raw", float("nan"))
            ff_raw_s = f"{safe_float(ff_raw_v, 0):.0f}天" if not _mth.isnan(safe_float(ff_raw_v, float("nan"))) else "—"
            records.append({"维度": "持仓拥挤", "子指标": "AUM资金沉淀",
                             "原始值": ff_raw_s,
                             "历史分位": round(ff_sf, 1),
                             "维度内权重": f"{_eff_w('fund_flow')*100:.0f}%",
                             "子项贡献": round(ff_sf * _eff_w("fund_flow"), 1),
                             "说明": "totalAssets/日成交额——资金沉淀密度横截面排名",
                             "status": "ok"})
        else:
            records.append({"维度": "持仓拥挤", "子指标": "AUM资金沉淀",
                             "原始值": "N/A", "历史分位": float("nan"),
                             "维度内权重": "0%（缺失）", "子项贡献": 0.0,
                             "说明": "totalAssets 数据不可用，该项不参与评分",
                             "status": "missing"})

    # ── 估值拥挤子指标 ─────────────────────────────────────────────────────────
    if ticker in v_df.index:
        r = v_df.loc[ticker]

        for name, raw, score, w, desc in [
            ("52W Z-Score",   f"{safe_float(r.get('zscore_52w_raw', 0), 0):.2f}σ",
             r.get("zscore_52w_score", 50),  VALUATION_W["zscore_52w"],
             "价格偏离52周均值的标准差倍数（Z=-3→0分，Z=+3→100分）"),
            ("相对SPY超额",   "—",
             r.get("exc_score", 50),         VALUATION_W["excess_vs_spy"],
             "3M超额收益历史分位——相对估值溢价代理"),
        ]:
            sf = safe_float(score, 50)
            records.append({"维度": "估值拥挤", "子指标": name, "原始值": raw,
                             "历史分位": round(sf, 1), "维度内权重": f"{w*100:.0f}%",
                             "子项贡献": round(sf * w, 1), "说明": desc,
                             "status": "ok"})

        # PE/PB 代理（横截面分位）
        pe_v  = r.get("pe_raw",        float("nan"))
        pb_v  = r.get("pb_raw",        float("nan"))
        pe_ps = r.get("pe_proxy_score", float("nan"))
        pe_vf = safe_float(pe_v, float("nan"))
        pb_vf = safe_float(pb_v, float("nan"))
        pe_ok = not _mth.isnan(pe_vf)
        pb_ok = not _mth.isnan(pb_vf)
        ps_ok = not _mth.isnan(safe_float(pe_ps, float("nan")))

        if pe_ok or pb_ok:
            if pe_ok and pb_ok:
                pe_disp = f"PE:{pe_vf:.1f} / PB:{pb_vf:.2f}"
            elif pe_ok:
                pe_disp = f"PE:{pe_vf:.1f} / PB:N/A"
            else:
                pe_disp = f"PE:N/A / PB:{pb_vf:.2f}"
            ps_sf = safe_float(pe_ps, 50)
            records.append({"维度": "估值拥挤", "子指标": "PE/PB代理",
                             "原始值": pe_disp,
                             "历史分位": round(ps_sf, 1),
                             "维度内权重": f"{VALUATION_W['pe_proxy']*100:.0f}%",
                             "子项贡献": round(ps_sf * VALUATION_W["pe_proxy"], 1),
                             "说明": "trailingPE/priceToBook 在11行业横截面中的分位排名",
                             "status": "ok"})
        else:
            records.append({"维度": "估值拥挤", "子指标": "PE/PB代理",
                             "原始值": "N/A", "历史分位": float("nan"),
                             "维度内权重": "0%（缺失）", "子项贡献": 0.0,
                             "说明": "yfinance PE/PB 数据不可用，该项不参与评分",
                             "status": "missing"})

    # ── 叙事拥挤子指标 ─────────────────────────────────────────────────────────
    if ticker in n_df.index:
        r = n_df.loc[ticker]
        news_missing = bool(r.get("news_missing", True))
        pcr_missing  = bool(r.get("pcr_missing",  True))
        renorm_sfx   = "（已重新归一化权重）" if (news_missing or pcr_missing) else ""

        accel_eff_w = float(r.get("accel_eff_w", NARRATIVE_W["momentum_accel"]))
        skew_eff_w  = float(r.get("skew_eff_w",  NARRATIVE_W["return_skew"]))
        news_eff_w  = float(r.get("news_eff_w",  0.0))
        pcr_eff_w   = float(r.get("pcr_eff_w",   0.0))

        for name, raw, score, w, desc in [
            ("动量加速度", "—", r.get("accel_score", 50), accel_eff_w,
             "1M收益vs3M月均——叙事共振加速信号" + renorm_sfx),
            ("收益偏度",   "—", r.get("skew_score", 50),  skew_eff_w,
             "近60日收益右偏度——单边乐观程度" + renorm_sfx),
        ]:
            sf = safe_float(score, 50)
            records.append({"维度": "叙事拥挤", "子指标": name, "原始值": raw,
                             "历史分位": round(sf, 1),
                             "维度内权重": f"{w*100:.1f}%",
                             "子项贡献": round(sf * w, 1), "说明": desc,
                             "status": "ok"})

        # 媒体热度
        if news_missing:
            records.append({"维度": "叙事拥挤", "子指标": "媒体热度代理",
                             "原始值": "N/A", "历史分位": float("nan"),
                             "维度内权重": "0%（缺失）", "子项贡献": 0.0,
                             "说明": "数据获取失败，不代表媒体关注度为零 — 该项不参与本次评分",
                             "status": "missing"})
        else:
            ns = safe_float(r.get("news_score"), 50)
            records.append({"维度": "叙事拥挤", "子指标": "媒体热度代理",
                             "原始值": str(r.get("news_raw", "—")),
                             "历史分位": round(ns, 1),
                             "维度内权重": f"{news_eff_w*100:.0f}%",
                             "子项贡献": round(ns * news_eff_w, 1),
                             "说明": "yfinance新闻条目数跨行业横截面分位，衡量相对媒体关注度" + renorm_sfx,
                             "status": "ok"})

        # PCR 情绪
        if pcr_missing:
            records.append({"维度": "叙事拥挤", "子指标": "PCR情绪",
                             "原始值": "无数据", "历史分位": float("nan"),
                             "维度内权重": "0%（缺失）", "子项贡献": 0.0,
                             "说明": "该行业期权流动性不足或数据缺失，不参与评分",
                             "status": "missing"})
        else:
            ps = safe_float(r.get("pcr_score"), 50)
            pcr_disp = f"{r.get('pcr_raw', '—'):.3f}" if r.get("pcr_raw") else "—"
            records.append({"维度": "叙事拥挤", "子指标": "PCR情绪",
                             "原始值": pcr_disp,
                             "历史分位": round(ps, 1),
                             "维度内权重": f"{pcr_eff_w*100:.0f}%",
                             "子项贡献": round(ps * pcr_eff_w, 1),
                             "说明": "Put/Call Ratio，低=看涨期权需求旺盛=叙事过度乐观" + renorm_sfx,
                             "status": "ok"})

    # ── 广度与领导权子指标 ─────────────────────────────────────────────────────
    if ticker in br_df.index:
        r = br_df.loc[ticker]
        items = [
            ("广度/距高点位置", "—",
             r.get("dd_breadth_score", 50), BREADTH_W["drawdown_breadth"],
             "1−价格/60日高点（ETF级广度代理，距高点越远=广度越差）"),
            ("趋势一致性",     "—",
             r.get("trend_score", 50),      BREADTH_W["trend_consistency"],
             "近20日低于50MA天数占比——趋势断裂程度"),
            ("动量背离",       "—",
             r.get("div_score", 50),        BREADTH_W["momentum_divergence"],
             "3M超额−1M超额（3M强但1M弱=动量衰减=领导权收窄）"),
        ]
        for name, raw, score, w, desc in items:
            sf = safe_float(score, 50)
            records.append({"维度": "广度与领导权", "子指标": name, "原始值": raw,
                             "历史分位": round(sf, 1), "维度内权重": f"{w*100:.0f}%",
                             "子项贡献": round(sf * w, 1), "说明": desc,
                             "status": "ok"})

    # ── 出清状态子指标（注明不计入总分）─────────────────────────────────────────
    if ticker in cl_df.index:
        r = cl_df.loc[ticker]
        items = [
            ("回撤深度",       "—",
             r.get("dd_depth_score", 50),  CLEARANCE_W["drawdown_depth"],
             "距252日高点回撤深度（30%=满分，仅进状态机）"),
            ("波动率突刺",     "—",
             r.get("vspike_score", 50),    CLEARANCE_W["vol_spike"],
             "10日/90日波动率比历史分位（仅进状态机）"),
            ("收益不对称",     "—",
             r.get("rasym_score", 50),     CLEARANCE_W["return_asymmetry"],
             "坏消息主导程度（高分=出清中，仅进状态机）"),
        ]
        for name, raw, score, w, desc in items:
            sf = safe_float(score, 50)
            records.append({"维度": "出清状态", "子指标": name, "原始值": raw,
                             "历史分位": round(sf, 1), "维度内权重": f"{w*100:.0f}%",
                             "子项贡献": round(sf * w, 1), "说明": desc,
                             "status": "ok"})

    return records


# ── 数据完整度评估 ────────────────────────────────────────────────────────────

def compute_completeness(ticker: str,
                         t_df:  pd.DataFrame,
                         p_df:  pd.DataFrame,
                         v_df:  pd.DataFrame,
                         n_df:  pd.DataFrame,
                         br_df: pd.DataFrame,
                         cl_df: pd.DataFrame,
                         pcr_data: Dict = None) -> dict:
    """评估单个 ticker 的数据完整度与评分置信度（6维版本）"""
    dim_weights  = DIMENSION_WEIGHTS
    missing_items = []

    # PCR / 媒体热度缺失检测
    pcr_missing  = (pcr_data is None) or (pcr_data.get(ticker) is None)
    news_missing = True
    if ticker in n_df.index:
        news_missing = bool(n_df.loc[ticker].get("news_missing", True))

    # ── 各维度完整度
    dim_comp = {}

    # 交易拥挤（主要真实数据）
    dim_comp["交易拥挤"] = 95.0 if ticker in t_df.index else 50.0

    # 持仓拥挤（代理变量）
    dim_comp["持仓拥挤"] = 60.0
    missing_items.append("持仓数据（代理：成交量趋势/Beta/资金流）")

    # 估值拥挤（Z-Score/超额收益+PE/PB）
    pe_ok = False
    if ticker in v_df.index:
        pe_ok = not np.isnan(safe_float(v_df.loc[ticker].get("pe_proxy_score", float("nan")), float("nan")))
    dim_comp["估值拥挤"] = 85.0 if pe_ok else 70.0
    if not pe_ok:
        missing_items.append("PE/PB 基本面估值（yfinance 数据不可用）")

    # 叙事拥挤
    w_a = NARRATIVE_W["momentum_accel"]
    w_s = NARRATIVE_W["return_skew"]
    w_n = NARRATIVE_W["news_proxy"]
    w_p = NARRATIVE_W["pcr_sentiment"]
    narr_q = w_a * 1.0 + w_s * 0.8
    narr_q += w_n * (0.6 if not news_missing else 0.0)
    narr_q += w_p * (1.0 if not pcr_missing  else 0.0)
    dim_comp["叙事拥挤"] = round(narr_q * 100, 1)
    if news_missing:
        missing_items.append("媒体热度（新闻数据获取失败）")
    if pcr_missing:
        missing_items.append("PCR 情绪（该行业期权数据不可用）")

    # 广度与领导权（ETF代理）
    dim_comp["广度与领导权"] = 68.0 if ticker in br_df.index else 50.0
    missing_items.append("行业内个股数据（使用ETF级代理）")

    # 汇总加权完整度（出清状态不进权重）
    total_comp = sum(
        dim_comp.get(d, 50) * dim_weights.get(d, 0.2)
        for d in dim_comp
    ) / max(sum(dim_weights.values()), 1e-9)
    total_comp = round(total_comp, 1)

    confidence = "高" if total_comp >= 78 else ("中" if total_comp >= 60 else "低")

    return {
        "completeness_pct": total_comp,
        "confidence":       confidence,
        "dim_completeness": dim_comp,
        "missing_items":    missing_items,
    }
