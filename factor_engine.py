"""
五层拥挤因子计算引擎
所有子因子统一输出为 0-100 历史分位数分，独立评估，互不污染。

设计原则：
- 交易拥挤 ≠ 景气/基本面好
- RSI高 + 涨幅高 ≠ 拥挤，还需结合成交量、持仓、估值
- 每层因子尽量反映独立信息
"""

import numpy as np
import pandas as pd
from typing import Dict
from config import (
    SECTOR_ETFS,
    TRADING_W, POSITIONING_W, VALUATION_W, NARRATIVE_W, BEHAVIORAL_W,
    INDICATOR_QUALITY, DIMENSION_WEIGHTS,
)


# ── 工具 ──────────────────────────────────────────────────────────────────────

def hist_pct(series: pd.Series, value: float, window: int = 252) -> float:
    """value 在 series 最近 window 期中的历史分位数 → 0-100"""
    s = series.dropna()
    if len(s) < 10 or np.isnan(value):
        return 50.0
    hist = s.iloc[-window:] if len(s) > window else s
    return round(float((hist < value).sum() / len(hist) * 100), 1)


def safe_float(val, default=50.0) -> float:
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except Exception:
        return default


# ── 1. 交易拥挤 ───────────────────────────────────────────────────────────────

def compute_trading(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    衡量短期交易热度，不等于景气度：
    - RSI 历史分位
    - 1M 动量历史分位
    - 成交量 surge（20D / 252D 均量）
    - 波动率扩张（30D / 90D）
    - 价格在 52W 区间位置（接近高点 = 交易拥挤）
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

        # 价格 52W 位置（0=年低，100=年高）
        if len(p) >= 20:
            w = p.iloc[-252:] if len(p) >= 252 else p
            lo, hi = w.min(), w.max()
            prox = (p.iloc[-1] - lo) / (hi - lo + 1e-9) * 100
            prox_score = float(prox)
        else:
            prox_score = 50.0

        rows[t] = {
            "rsi_raw":            round(rsi_v, 1),
            "mom_1m_raw":         round(safe_float(mom1m_v, 0.0), 2),
            "rsi_score":          rsi_score,
            "mom_1m_score":       mom1m_score,
            "volume_surge_score": vol_surge_score,
            "vol_exp_score":      vol_exp_score,
            "price_prox_score":   round(prox_score, 1),
        }

    df = pd.DataFrame(rows).T
    for c in ["rsi_score","mom_1m_score","volume_surge_score","vol_exp_score","price_prox_score"]:
        df[c] = df[c].astype(float)

    df["交易拥挤"] = (
        df["rsi_score"]          * TRADING_W["rsi"]           +
        df["mom_1m_score"]       * TRADING_W["momentum_1m"]   +
        df["volume_surge_score"] * TRADING_W["volume_surge"]  +
        df["vol_exp_score"]      * TRADING_W["vol_expansion"] +
        df["price_prox_score"]   * TRADING_W["price_proximity"]
    ).round(1)
    return df


# ── 2. 持仓拥挤 ───────────────────────────────────────────────────────────────

def compute_positioning(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    衡量资金是否持续、集中流入：
    - 中期成交量趋势（63D / 252D）
    - Beta 扩张（短期 beta > 长期 beta = 资金正在追进）
    - ETF 成交额相对 SPY 的历史分位（资金集中度代理）
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
            r  = p.pct_change().dropna()
            sr = spy_p.pct_change().reindex(r.index).dropna()
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

        # 相对 SPY 资金流（成交额比值历史分位）
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
            "vol_trend_score":  vol_trend_score,
            "beta_exp_score":   round(beta_exp_score, 1),
            "rel_flow_score":   rel_flow_score,
        }

    df = pd.DataFrame(rows).T.astype(float)
    df["持仓拥挤"] = (
        df["vol_trend_score"] * POSITIONING_W["volume_trend"]   +
        df["beta_exp_score"]  * POSITIONING_W["beta_expansion"] +
        df["rel_flow_score"]  * POSITIONING_W["relative_flow"]
    ).round(1)
    return df


# ── 3. 估值拥挤 ───────────────────────────────────────────────────────────────

def compute_valuation(prices: pd.DataFrame) -> pd.DataFrame:
    """
    衡量市场定价是否已透支远期预期：
    - 52W Z-Score（价格偏离年度均值）
    - 价格/200日均线 历史分位
    - 相对 SPY 3M 超额收益历史分位
    注意：估值拥挤 ≠ 行业不好，而是预期是否已被充分定价。
    """
    spy_p = prices["SPY"].dropna() if "SPY" in prices.columns else None
    rows  = {}

    for t in SECTOR_ETFS:
        p = prices[t].dropna() if t in prices.columns else pd.Series(dtype=float)

        # 52W Z-Score → 映射到 0-100（Z=-3→0, Z=+3→100）
        if len(p) >= 60:
            w   = min(252, len(p))
            mu  = p.rolling(w).mean()
            sig = p.rolling(w).std().replace(0, np.nan)
            zs  = (p - mu) / sig
            z_v = safe_float(zs.iloc[-1], 0.0)
            z_score = min(100, max(0, (z_v + 3) / 6 * 100))
        else:
            z_score, z_v = 50.0, 0.0

        # 价格/200日均线 历史分位
        if len(p) >= 200:
            ma200    = p.rolling(200).mean()
            pta_s    = p / ma200
            pta_hist = pta_s.dropna()
            pta_v    = safe_float(pta_s.iloc[-1])
            pta_score = hist_pct(pta_hist, pta_v)
            pta_raw  = round(pta_v, 3)
            pta_n    = min(len(pta_hist), 252)
        else:
            pta_score = 50.0
            pta_raw   = float("nan")
            pta_n     = 0

        # 相对 SPY 3M 超额收益历史分位
        if spy_p is not None and len(p) > 63:
            exc_s = (p.pct_change(63) - spy_p.reindex(p.index).pct_change(63)) * 100
            exc_v = safe_float(exc_s.iloc[-1])
            exc_score = hist_pct(exc_s.dropna(), exc_v)
        else:
            exc_score = 50.0

        rows[t] = {
            "zscore_52w_raw":   round(z_v, 2),
            "zscore_52w_score": round(z_score, 1),
            "pta_raw":          pta_raw,
            "pta_n":            pta_n,
            "pta_score":        pta_score,
            "exc_score":        exc_score,
        }

    df = pd.DataFrame(rows).T
    for c in ["zscore_52w_score","pta_score","exc_score"]:
        df[c] = df[c].astype(float)

    df["估值拥挤"] = (
        df["zscore_52w_score"] * VALUATION_W["zscore_52w"]     +
        df["pta_score"]        * VALUATION_W["price_to_ma200"] +
        df["exc_score"]        * VALUATION_W["excess_vs_spy"]
    ).round(1)
    return df


# ── 4. 预期拥挤 ───────────────────────────────────────────────────────────────

def compute_narrative(prices: pd.DataFrame, news_counts: Dict = None) -> pd.DataFrame:
    """
    衡量市场叙事/预期是否已高度集中：
    - 动量加速度：1M收益 vs 3M月均收益 — 真实数据
    - 收益率正偏度：近60日收益右偏 = 市场单边乐观 — 真实数据
    - 媒体热度代理：yfinance 新闻条目数跨行业横截面分位 — 代理变量

    关键原则：
    - has_data=False（抓取失败） → news_score=NaN，不参与评分，不视为0
    - count_7d=0 且 has_data=True（真实没有新闻） → 正常评分0分
    - 媒体热度缺失时：剩余两项指标权重重新归一化，维度得分仍有意义
    """
    # ── 预先处理新闻数据（只用 has_data=True 的 ticker 做横截面排名）
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
        n_valid = len(all_counts)
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
            accel_score = min(100, max(0, (accel + 5) / 10 * 100))
        else:
            accel_score = 50.0

        # 收益偏度
        if len(p) > 30:
            skew = float(p.pct_change().dropna().iloc[-60:].skew())
            skew_score = min(100, max(0, (skew + 2) / 4 * 100))
        else:
            skew_score = 50.0

        # 媒体热度代理：区分"真实0条"和"数据缺失"
        this_has_data = news_has_data.get(t, False)
        if this_has_data and n_valid >= 1:
            cnt = valid_counts.get(t, 0)
            rank = sum(1 for x in all_counts if x < cnt)
            news_score   = round(rank / n_valid * 100, 1)
            news_raw     = f"{cnt}条/7天"
            news_missing = False
        else:
            news_score   = np.nan   # 显式NaN，不是0
            news_raw     = "N/A"
            news_missing = True

        # 预期拥挤：媒体热度缺失时重新归一化剩余指标权重
        w_a = NARRATIVE_W["momentum_accel"]
        w_s = NARRATIVE_W["return_skew"]
        w_n = NARRATIVE_W["news_proxy"]
        if news_missing:
            total_w     = w_a + w_s
            narr_score  = (accel_score * w_a + skew_score * w_s) / total_w
            accel_eff_w = round(w_a / total_w, 4)
            skew_eff_w  = round(w_s / total_w, 4)
            news_eff_w  = 0.0
        else:
            narr_score  = accel_score * w_a + skew_score * w_s + news_score * w_n
            accel_eff_w = w_a
            skew_eff_w  = w_s
            news_eff_w  = w_n

        rows[t] = {
            "accel_score":  round(accel_score, 1),
            "skew_score":   round(skew_score, 1),
            "news_score":   news_score,       # float 或 np.nan
            "news_raw":     news_raw,
            "news_missing": news_missing,
            "accel_eff_w":  accel_eff_w,
            "skew_eff_w":   skew_eff_w,
            "news_eff_w":   news_eff_w,
            "_narr_score":  round(narr_score, 1),
        }

    df = pd.DataFrame(rows).T
    for c in ["accel_score", "skew_score", "accel_eff_w", "skew_eff_w",
              "news_eff_w", "_narr_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["预期拥挤"] = df["_narr_score"].round(1)
    df = df.drop(columns=["_narr_score"])
    return df


# ── 5. 行为拥挤 ───────────────────────────────────────────────────────────────

def compute_behavioral(prices: pd.DataFrame, pcr_data: Dict) -> pd.DataFrame:
    """
    衡量市场是否正在失去风险感：
    - 近30日上涨日比例（高 = 情绪单边）
    - 正/负收益不对称性（高 = 坏消息被忽略 = 最危险信号之一）
    - P/C ratio（低 = 市场过度乐观 = 行为拥挤）
    """
    rows = {}
    for t in SECTOR_ETFS:
        p = prices[t].dropna() if t in prices.columns else pd.Series(dtype=float)

        # 上涨日比例
        if len(p) > 30:
            rets = p.pct_change().dropna().iloc[-30:]
            up_r = (rets > 0).sum() / len(rets)
            # 正常市场约 52-55%，>70% 为情绪极端
            up_score = min(100, max(0, (up_r - 0.45) / 0.30 * 100))
        else:
            up_score = 50.0

        # 正/负收益不对称（"好消息涨多、坏消息跌少"是拥挤极度危险信号）
        if len(p) > 60:
            rets = p.pct_change().dropna().iloc[-60:]
            up_rets = rets[rets > 0]
            dn_rets = rets[rets < 0]
            if len(up_rets) > 5 and len(dn_rets) > 5:
                asym = float(up_rets.mean()) / (abs(float(dn_rets.mean())) + 1e-9)
                asym_score = min(100, max(0, (asym - 0.5) / 1.5 * 100))
            else:
                asym_score = 50.0
        else:
            asym_score = 50.0

        # P/C ratio（低=乐观=行为拥挤）
        pcr = pcr_data.get(t)
        if pcr and pcr > 0:
            # PCR 约 0.5=极度乐观, 2.0=极度恐慌
            pcr_score = min(100, max(0, (2.0 - pcr) / 1.5 * 100))
        else:
            pcr_score = 50.0

        rows[t] = {
            "up_day_score":  round(up_score, 1),
            "asym_score":    round(asym_score, 1),
            "pcr_score":     round(pcr_score, 1),
            "pcr_raw":       pcr,
        }

    df = pd.DataFrame(rows).T
    for c in ["up_day_score","asym_score","pcr_score"]:
        df[c] = df[c].astype(float)

    df["行为拥挤"] = (
        df["up_day_score"] * BEHAVIORAL_W["up_day_ratio"]     +
        df["asym_score"]   * BEHAVIORAL_W["return_asymmetry"] +
        df["pcr_score"]    * BEHAVIORAL_W["pcr_proxy"]
    ).round(1)
    return df


# ── 打分卡构建器 ──────────────────────────────────────────────────────────────

def build_scorecard(ticker: str,
                    t_df: pd.DataFrame, p_df: pd.DataFrame,
                    v_df: pd.DataFrame, n_df: pd.DataFrame,
                    b_df: pd.DataFrame) -> list:
    """
    为单个 ticker 构建结构化打分卡，供详情页展示。
    返回 list[dict]，每条对应一个子指标，包含：
        维度 / 子指标 / 原始值 / 历史分位 / 维度内权重 / 子项贡献 / 说明
    """
    records = []

    # ── 交易拥挤子指标
    if ticker in t_df.index:
        r = t_df.loc[ticker]
        items = [
            ("RSI(14)",      f"{safe_float(r.get('rsi_raw', 50)):.1f}",
             r.get("rsi_score", 50),           TRADING_W["rsi"],
             "RSI历史分位——短期超买程度，非景气判断"),
            ("1M动量",       f"{safe_float(r.get('mom_1m_raw', 0), 0):.2f}%",
             r.get("mom_1m_score", 50),         TRADING_W["momentum_1m"],
             "1M收益率历史分位——近期涨幅强弱"),
            ("成交量Surge",  "—",
             r.get("volume_surge_score", 50),   TRADING_W["volume_surge"],
             "20日均量/252日均量——成交放量程度"),
            ("波动率扩张",   "—",
             r.get("vol_exp_score", 50),        TRADING_W["vol_expansion"],
             "30日/90日波动率比——波动扩张程度"),
            ("52W价格位置",  "—",
             r.get("price_prox_score", 50),     TRADING_W["price_proximity"],
             "当前价在52周区间内的位置（0=年低，100=年高）"),
        ]
        for name, raw, score, w, desc in items:
            sf = safe_float(score, 50)
            records.append({"维度": "交易拥挤", "子指标": name, "原始值": raw,
                             "历史分位": round(sf, 1), "维度内权重": f"{w*100:.0f}%",
                             "子项贡献": round(sf * w, 1), "说明": desc,
                             "status": "ok"})

    # ── 持仓拥挤子指标
    if ticker in p_df.index:
        r = p_df.loc[ticker]
        items = [
            ("成交量中期趋势", "—",
             r.get("vol_trend_score", 50),   POSITIONING_W["volume_trend"],
             "63日均量/252日均量——中期资金流入趋势"),
            ("Beta扩张",       "—",
             r.get("beta_exp_score", 50),    POSITIONING_W["beta_expansion"],
             "近期Beta/中期Beta——追涨资金流入信号"),
            ("相对SPY资金流",  "—",
             r.get("rel_flow_score", 50),    POSITIONING_W["relative_flow"],
             "ETF成交额/SPY成交额历史分位——资金集中度代理"),
        ]
        for name, raw, score, w, desc in items:
            sf = safe_float(score, 50)
            records.append({"维度": "持仓拥挤", "子指标": name, "原始值": raw,
                             "历史分位": round(sf, 1), "维度内权重": f"{w*100:.0f}%",
                             "子项贡献": round(sf * w, 1), "说明": desc,
                             "status": "ok"})

    # ── 估值拥挤子指标
    if ticker in v_df.index:
        r = v_df.loc[ticker]
        items = [
            ("52W Z-Score",    f"{safe_float(r.get('zscore_52w_raw', 0), 0):.2f}σ",
             r.get("zscore_52w_score", 50),  VALUATION_W["zscore_52w"],
             "价格偏离52周均值的标准差倍数（Z=-3→0分，Z=+3→100分）"),
            ("价格/200日均线",
             f"{safe_float(r.get('pta_raw', float('nan')), float('nan')):.3f}x" if not __import__('math').isnan(safe_float(r.get('pta_raw', float('nan')), float('nan'))) else "—",
             r.get("pta_score", 50),         VALUATION_W["price_to_ma200"],
             f"价格相对200日均线的比值\uff0c在过去{int(r.get('pta_n', 0))}日历史分布中的分位"),
            ("相对SPY超额",    "—",
             r.get("exc_score", 50),         VALUATION_W["excess_vs_spy"],
             "3M超额收益历史分位——相对估值溢价代理"),
        ]
        for name, raw, score, w, desc in items:
            sf = safe_float(score, 50)
            records.append({"维度": "估值拥挤", "子指标": name, "原始值": raw,
                             "历史分位": round(sf, 1), "维度内权重": f"{w*100:.0f}%",
                             "子项贡献": round(sf * w, 1), "说明": desc,
                             "status": "ok"})

    # ── 预期拥挤子指标
    if ticker in n_df.index:
        r = n_df.loc[ticker]
        news_missing = bool(r.get("news_missing", True))
        accel_eff_w  = float(r.get("accel_eff_w", NARRATIVE_W["momentum_accel"]))
        skew_eff_w   = float(r.get("skew_eff_w",  NARRATIVE_W["return_skew"]))
        news_eff_w   = float(r.get("news_eff_w",  NARRATIVE_W["news_proxy"]))
        renorm_sfx   = "（已重新归一化权重）" if news_missing else ""

        for name, raw, score, w, desc in [
            ("动量加速度", "—", r.get("accel_score", 50), accel_eff_w,
             "1M收益vs3M月均——叙事共振加速信号" + renorm_sfx),
            ("收益偏度",   "—", r.get("skew_score", 50),  skew_eff_w,
             "近60日收益右偏度——单边乐观程度" + renorm_sfx),
        ]:
            sf = safe_float(score, 50)
            records.append({"维度": "预期拥挤", "子指标": name, "原始值": raw,
                             "历史分位": round(sf, 1), "维度内权重": f"{w*100:.1f}%",
                             "子项贡献": round(sf * w, 1), "说明": desc,
                             "status": "ok"})

        if news_missing:
            records.append({
                "维度": "预期拥挤", "子指标": "媒体热度代理",
                "原始值": "N/A",
                "历史分位": float("nan"),
                "维度内权重": "0%（缺失）",
                "子项贡献": 0.0,
                "说明": "数据获取失败，不代表媒体关注度为零 — 该项不参与本次评分",
                "status": "missing",
            })
        else:
            ns = safe_float(r.get("news_score"), 50)
            records.append({
                "维度": "预期拥挤", "子指标": "媒体热度代理",
                "原始值": str(r.get("news_raw", "—")),
                "历史分位": round(ns, 1),
                "维度内权重": f"{news_eff_w*100:.0f}%",
                "子项贡献": round(ns * news_eff_w, 1),
                "说明": "yfinance新闻条目数跨行业横截面分位，衡量相对媒体关注度",
                "status": "ok",
            })

    # ── 行为拥挤子指标
    if ticker in b_df.index:
        r = b_df.loc[ticker]
        pcr = r.get("pcr_raw")
        pcr_disp = f"{pcr:.3f}" if pcr else "无数据"
        items = [
            ("上涨日比例",     "—",
             r.get("up_day_score", 50),   BEHAVIORAL_W["up_day_ratio"],
             "近30日上涨日占比（>70%=情绪极端）"),
            ("正负收益不对称", "—",
             r.get("asym_score", 50),     BEHAVIORAL_W["return_asymmetry"],
             "上涨日均值/跌幅绝对值——坏消息免疫程度"),
            ("P/C Ratio",      pcr_disp,
             r.get("pcr_score", 50),      BEHAVIORAL_W["pcr_proxy"],
             "看跌/看涨期权比，低=市场过度乐观"),
        ]
        for name, raw, score, w, desc in items:
            sf = safe_float(score, 50)
            records.append({"维度": "行为拥挤", "子指标": name, "原始值": raw,
                             "历史分位": round(sf, 1), "维度内权重": f"{w*100:.0f}%",
                             "子项贡献": round(sf * w, 1), "说明": desc,
                             "status": "ok"})

    return records


# ── 数据完整度评估 ────────────────────────────────────────────────────────────

def compute_completeness(ticker: str,
                         t_df: pd.DataFrame, p_df: pd.DataFrame,
                         v_df: pd.DataFrame, n_df: pd.DataFrame,
                         b_df: pd.DataFrame,
                         pcr_data: Dict = None) -> dict:
    """
    评估单个 ticker 的数据完整度与评分置信度。

    返回 dict：
        completeness_pct  : 0-100，加权完整度
        confidence        : "高" / "中" / "低"
        dim_completeness  : 各维度完整度 dict
        missing_items     : 缺失或降级的指标列表
    """
    dim_weights = DIMENSION_WEIGHTS

    # 各维度完整度：以 INDICATOR_QUALITY 质量分 × 维度内权重 加权计算
    # 行为拥挤：若 PCR 实际缺失，P/C Ratio 质量降为 0
    pcr_missing = (pcr_data is None) or (pcr_data.get(ticker) is None)
    # 预期拥挤：媒体热度数据是否成功抓取（True=失败/缺失，False=有效数据）
    news_missing = True   # 默认缺失
    if ticker in n_df.index:
        news_missing = bool(n_df.loc[ticker].get("news_missing", True))

    dim_comp = {}
    missing_items = []

    # ── 交易拥挤（全真实数据，除非价格序列过短）
    trading_q = 1.0 if (ticker in t_df.index) else 0.5
    dim_comp["交易拥挤"] = trading_q * 100

    # ── 持仓拥挤（全代理变量）
    dim_comp["持仓拥挤"] = 60.0  # 3个代理指标平均 quality=0.6
    missing_items.append("持仓数据（代理：成交量趋势/Beta/资金流）")

    # ── 估值拥挤（价格行为代理，非真正基本面估值）
    dim_comp["估值拥挤"] = 72.0  # 2×proxy(0.7) + 1×real(0.9) ÷ 3 ≈ 0.77
    missing_items.append("基本面估值（PE/PB等，用价格行为代替）")

    # ── 预期拥挤
    accel_q = 1.0; skew_q = 0.8
    news_q  = 0.0 if news_missing else 0.6  # 有代理=0.6, 无数据=0
    w_a = NARRATIVE_W["momentum_accel"]
    w_s = NARRATIVE_W["return_skew"]
    w_n = NARRATIVE_W["news_proxy"]
    narr_raw = (accel_q * w_a + skew_q * w_s + news_q * w_n) / (w_a + w_s + w_n)
    dim_comp["预期拥挤"] = round(narr_raw * 100, 1)
    if news_missing:
        missing_items.append("媒体热度（新闻数据获取失败）")
    else:
        missing_items.append("媒体热度（代理：yfinance新闻条目数）")

    # ── 行为拥挤
    up_q   = 1.0; asym_q = 0.8
    pcr_q  = 0.0 if pcr_missing else 1.0
    w_u  = BEHAVIORAL_W["up_day_ratio"]
    w_as = BEHAVIORAL_W["return_asymmetry"]
    w_p  = BEHAVIORAL_W["pcr_proxy"]
    beh_raw = (up_q * w_u + asym_q * w_as + pcr_q * w_p) / (w_u + w_as + w_p)
    dim_comp["行为拥挤"] = round(beh_raw * 100, 1)
    if pcr_missing:
        missing_items.append("P/C Ratio（该行业期权数据不可用）")

    # ── 汇总加权完整度
    total_comp = sum(dim_comp[d] * dim_weights.get(d, 0.2)
                     for d in dim_comp) / sum(dim_weights.values())
    total_comp = round(total_comp, 1)

    # ── 置信度等级
    if total_comp >= 78:
        confidence = "高"
    elif total_comp >= 60:
        confidence = "中"
    else:
        confidence = "低"

    return {
        "completeness_pct": total_comp,
        "confidence":       confidence,
        "dim_completeness": dim_comp,
        "missing_items":    missing_items,
    }
