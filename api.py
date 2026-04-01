"""
Sector Crowding API — 供外部 Agent / MCP Server 调用的结构化接口
独立于 Streamlit，纯 Python 函数，返回 dict/list。

使用方式:
    from api import get_sector_report, get_all_reports, get_signals

    # 获取单个行业报告
    report = get_sector_report("SOXX", df_scores, history_df)

    # 获取所有行业报告
    reports = get_all_reports(df_scores, history_df)

    # 获取当前触发的信号
    signals = get_signals(df_scores)
"""

import pandas as pd
from config import SECTOR_ETFS, DIMENSION_WEIGHTS
from scoring import commentary, get_level
from history import get_trend, get_acceleration

DIMS = ["叙事拥挤", "持仓拥挤", "交易拥挤", "估值拥挤", "广度与领导权"]


def get_sector_report(ticker: str,
                      df_scores: pd.DataFrame,
                      history_df: pd.DataFrame = None,
                      weights: dict = None) -> dict:
    """
    返回单个行业的完整拥挤度报告。

    Parameters:
        ticker:     ETF ticker (e.g. "SOXX")
        df_scores:  scoring.aggregate() 的输出
        history_df: history.compute_score_history() 的输出（可选）
        weights:    维度权重覆盖

    Returns:
        结构化 dict，包含分数、状态、趋势、维度明细、评论
    """
    if ticker not in df_scores.index:
        return {"error": f"Ticker {ticker} not found"}

    row = df_scores.loc[ticker]
    total = float(row.get("总拥挤度", 50))
    level, color = get_level(total)
    comm = commentary(row)

    # 维度分数
    dimensions = {}
    for d in DIMS:
        score = float(row.get(d, 50))
        trend = get_trend(history_df, ticker, d) if history_df is not None else {}
        dimensions[d] = {
            "score": round(score, 1),
            "weight": (weights or DIMENSION_WEIGHTS).get(d, 0.2),
            "contribution": round(score * (weights or DIMENSION_WEIGHTS).get(d, 0.2), 1),
            "trend_7d": trend.get("change_7d"),
            "trend_30d": trend.get("change_30d"),
        }

    # 主驱动维度
    top_dim = max(dimensions.items(), key=lambda x: x[1]["score"])

    # 总分趋势 + 加速度
    total_trend = get_trend(history_df, ticker, "总拥挤度") if history_df is not None else {}
    clearance_trend = get_trend(history_df, ticker, "出清状态") if history_df is not None else {}
    accel_data = get_acceleration(history_df, ticker) if history_df is not None else {}

    return {
        "ticker": ticker,
        "name": SECTOR_ETFS[ticker]["name"],
        "category": SECTOR_ETFS[ticker]["category"],
        "total_score": round(total, 1),
        "level": level,
        "level_color": color,
        "state": str(row.get("状态", "")),
        "action": str(row.get("操作偏向", "")),
        "clearance_score": round(float(row.get("出清状态", 50)), 1),
        "trend_7d": total_trend.get("change_7d"),
        "trend_30d": total_trend.get("change_30d"),
        "clearance_trend_7d": clearance_trend.get("change_7d"),
        "acceleration": accel_data.get("accel"),
        "accel_direction": accel_data.get("direction", "—"),
        "top_driver": top_dim[0],
        "top_driver_score": top_dim[1]["score"],
        "dimensions": dimensions,
        "commentary": comm,
    }


def get_all_reports(df_scores: pd.DataFrame,
                    history_df: pd.DataFrame = None,
                    weights: dict = None) -> list:
    """返回所有行业报告，按总拥挤度降序排列"""
    reports = []
    for ticker in df_scores.index:
        if ticker in SECTOR_ETFS:
            reports.append(get_sector_report(ticker, df_scores, history_df, weights))
    reports.sort(key=lambda r: r["total_score"], reverse=True)
    return reports


def get_market_overview(df_scores: pd.DataFrame,
                        history_df: pd.DataFrame = None) -> dict:
    """返回市场整体概览"""
    tickers = [t for t in df_scores.index if t in SECTOR_ETFS]
    scores = [float(df_scores.loc[t, "总拥挤度"]) for t in tickers]
    avg = sum(scores) / len(scores) if scores else 50

    # 分布统计
    level_counts = {"极度拥挤": 0, "高拥挤": 0, "中等拥挤": 0, "低拥挤": 0}
    for s in scores:
        lev, _ = get_level(s)
        level_counts[lev] = level_counts.get(lev, 0) + 1

    # Top 3 / Bottom 3
    sorted_rows = df_scores.loc[tickers].sort_values("总拥挤度", ascending=False)
    top3 = [
        {"ticker": t, "name": SECTOR_ETFS[t]["name"],
         "score": round(float(sorted_rows.loc[t, "总拥挤度"]), 1),
         "state": str(sorted_rows.loc[t, "状态"])}
        for t in sorted_rows.index[:3]
    ]
    bottom3 = [
        {"ticker": t, "name": SECTOR_ETFS[t]["name"],
         "score": round(float(sorted_rows.loc[t, "总拥挤度"]), 1),
         "state": str(sorted_rows.loc[t, "状态"])}
        for t in sorted_rows.index[-3:]
    ]

    # 市场平均趋势
    if history_df is not None:
        total_cols = [(t, "总拥挤度") for t in tickers
                      if (t, "总拥挤度") in history_df.columns]
        if total_cols:
            mkt_avg = history_df[total_cols].mean(axis=1).dropna()
            current = float(mkt_avg.iloc[-1]) if len(mkt_avg) > 0 else avg
            mkt_trend_7d = float(current - mkt_avg.iloc[-6]) if len(mkt_avg) >= 6 else None
            mkt_trend_30d = float(current - mkt_avg.iloc[-23]) if len(mkt_avg) >= 23 else None
        else:
            mkt_trend_7d, mkt_trend_30d = None, None
    else:
        mkt_trend_7d, mkt_trend_30d = None, None

    return {
        "market_avg_score": round(avg, 1),
        "market_trend_7d": round(mkt_trend_7d, 1) if mkt_trend_7d is not None else None,
        "market_trend_30d": round(mkt_trend_30d, 1) if mkt_trend_30d is not None else None,
        "sector_count": len(tickers),
        "level_distribution": level_counts,
        "top3_crowded": top3,
        "bottom3_crowded": bottom3,
    }


def get_signals(df_scores: pd.DataFrame,
                history_df: pd.DataFrame = None) -> list:
    """
    返回当前触发的拥挤度信号。
    四种信号类型与 app.py Tab4 一致。
    """
    signals = []

    for t in df_scores.index:
        if t not in SECTOR_ETFS:
            continue
        row = df_scores.loc[t]
        total    = float(row.get("总拥挤度", 50))
        name     = SECTOR_ETFS[t]["name"]
        dims     = {d: float(row.get(d, 50)) for d in DIMS}
        n_high   = sum(1 for v in dims.values() if v >= 60)

        trend = get_trend(history_df, t) if history_df is not None else {}

        # 信号1: 全维度共振拥挤（≥3个维度 ≥60）
        if n_high >= 3:
            high_dims = [d for d, v in dims.items() if v >= 60]
            signals.append({
                "type": "multi_dimensional",
                "type_cn": "多维共振拥挤",
                "ticker": t,
                "name": name,
                "total_score": round(total, 1),
                "detail": f"{n_high}个维度≥60: {', '.join(high_dims)}",
                "severity": "high" if n_high >= 4 else "medium",
                "trend_7d": trend.get("change_7d"),
            })

        # 信号2: 估值过热 + 动量（估值≥65 且 交易≥60）
        if dims["估值拥挤"] >= 65 and dims["交易拥挤"] >= 60:
            signals.append({
                "type": "valuation_momentum",
                "type_cn": "估值+动量过热",
                "ticker": t,
                "name": name,
                "total_score": round(total, 1),
                "detail": f"估值 {dims['估值拥挤']:.0f} + 交易 {dims['交易拥挤']:.0f}",
                "severity": "high",
                "trend_7d": trend.get("change_7d"),
            })

        # 信号3: 叙事先行（叙事≥55，交易<50，总分<62）
        if dims["叙事拥挤"] >= 55 and dims["交易拥挤"] < 50 and total < 62:
            signals.append({
                "type": "narrative_first",
                "type_cn": "叙事先行",
                "ticker": t,
                "name": name,
                "total_score": round(total, 1),
                "detail": f"叙事 {dims['叙事拥挤']:.0f}，交易仅 {dims['交易拥挤']:.0f}",
                "severity": "low",
                "trend_7d": trend.get("change_7d"),
            })

        # 信号4: 低拥挤/赔率好（总分<38，取前4）
        if total < 38:
            signals.append({
                "type": "low_crowding",
                "type_cn": "低拥挤/赔率好",
                "ticker": t,
                "name": name,
                "total_score": round(total, 1),
                "detail": f"总拥挤度 {total:.0f}，配置赔率相对合理",
                "severity": "opportunity",
                "trend_7d": trend.get("change_7d"),
            })

    # 低拥挤只保留前4
    low_signals = [s for s in signals if s["type"] == "low_crowding"]
    other_signals = [s for s in signals if s["type"] != "low_crowding"]
    low_signals.sort(key=lambda s: s["total_score"])
    signals = other_signals + low_signals[:4]

    return signals
