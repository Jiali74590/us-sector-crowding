"""
评分聚合 + 投资框架语言解读
核心原则：区分景气 vs 拥挤，区分"能涨"vs"赔率下降"
"""

import pandas as pd
import numpy as np
from config import SECTOR_ETFS, CROWDING_LEVELS, DIMENSION_WEIGHTS


def get_level(score: float):
    for lo, hi, label, color in CROWDING_LEVELS:
        if lo <= score < hi:
            return label, color
    return "低拥挤", "#1e8449"


def aggregate(trading, positioning, valuation, narrative, behavioral,
              weights: dict = None) -> pd.DataFrame:
    """合并五维评分，输出完整评分表"""
    w = weights or DIMENSION_WEIGHTS
    tickers = list(SECTOR_ETFS.keys())

    df = pd.DataFrame(index=tickers)
    df["行业"]   = [SECTOR_ETFS[t]["name"] for t in tickers]
    df["预期拥挤"] = narrative["预期拥挤"].reindex(tickers).fillna(50)
    df["持仓拥挤"] = positioning["持仓拥挤"].reindex(tickers).fillna(50)
    df["交易拥挤"] = trading["交易拥挤"].reindex(tickers).fillna(50)
    df["估值拥挤"] = valuation["估值拥挤"].reindex(tickers).fillna(50)
    df["行为拥挤"] = behavioral["行为拥挤"].reindex(tickers).fillna(50)

    df["总拥挤度"] = (
        df["预期拥挤"] * w.get("预期拥挤", 0.20) +
        df["持仓拥挤"] * w.get("持仓拥挤", 0.20) +
        df["交易拥挤"] * w.get("交易拥挤", 0.25) +
        df["估值拥挤"] * w.get("估值拥挤", 0.20) +
        df["行为拥挤"] * w.get("行为拥挤", 0.15)
    ).round(1)

    df = df.sort_values("总拥挤度", ascending=False)
    df.insert(0, "排名", range(1, len(df) + 1))
    df["拥挤等级"] = df["总拥挤度"].apply(lambda s: get_level(s)[0])
    df["等级色"]   = df["总拥挤度"].apply(lambda s: get_level(s)[1])
    return df


def _primary_driver(row: pd.Series):
    dims = ["预期拥挤", "持仓拥挤", "交易拥挤", "估值拥挤", "行为拥挤"]
    scores = {d: float(row.get(d, 50)) for d in dims}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[0][0], ranked[1][0], scores


def commentary(row: pd.Series) -> dict:
    """
    投资框架语言生成器
    区分：景气 vs 拥挤 / 基本面好 vs 交易结构脆弱 / 能涨 vs 赔率下降
    """
    total    = float(row.get("总拥挤度", 50))
    sector   = row.get("行业", "")
    p1, p2, dim_s = _primary_driver(row)

    trading     = dim_s["交易拥挤"]
    positioning = dim_s["持仓拥挤"]
    valuation   = dim_s["估值拥挤"]
    narrative   = dim_s["预期拥挤"]
    behavioral  = dim_s["行为拥挤"]

    n_high = sum(1 for v in dim_s.values() if v >= 62)
    all_crowded   = n_high >= 4
    single_driver = n_high <= 1

    # ── 综合判断 ─────────────────────────────────────────────────────────────
    if total >= 75:
        if all_crowded:
            summary = (f"{sector}处于全维度共振拥挤状态。"
                       f"预期、持仓、交易、估值、行为均已偏高，"
                       f"当前最大风险不是基本面证伪，而是预期过满后的边际降温。")
        elif p1 == "交易拥挤" and positioning < 58:
            summary = (f"{sector}高分主要由短期交易动量驱动，持仓集中度尚未极端。"
                       f"行业仍可能继续上涨，但继续上涨需要更强的新信息支持；"
                       f"若催化缺席，交易层面的过热将自然回调。")
        elif p1 == "估值拥挤":
            summary = (f"{sector}估值已进入历史高分位区间，市场对远期乐观预期定价较充分。"
                       f"当前更适合等待拥挤释放或基本面再验证，而非继续追高加仓。")
        elif p1 == "预期拥挤" and trading < 58:
            summary = (f"{sector}预期叙事已高度集中，但价格动量和持仓尚未完全跟上。"
                       f"属于'叙事先行'型拥挤，潜在风险是预期一旦落空，回调没有缓冲。")
        elif p1 == "持仓拥挤":
            summary = (f"{sector}资金持续高强度流入，持仓集中度明显上升。"
                       f"这类慢拥挤更危险——方向上没人反对，但一旦出现踩踏，没有对手盘接。")
        else:
            summary = (f"{sector}总拥挤度极高，由{p1}主导、{p2}次之。"
                       f"从赔率角度看，此处继续加仓的期望回报已显著下降。")

    elif total >= 60:
        if p1 == "交易拥挤" and valuation < 55:
            summary = (f"{sector}交易热度较高，但估值尚未极端。"
                       f"当前高分更多来自交易和动量，而非估值极端，属于高热但未完全透支。"
                       f"可以持有，但不建议此时重仓追入。")
        elif p1 == "持仓拥挤":
            summary = (f"{sector}资金流入趋势明显，持仓集中度在历史较高区间。"
                       f"景气度不等于拥挤度——即使基本面良好，资金过度集中本身是风险来源。")
        elif narrative > 65 and trading < 50:
            summary = (f"{sector}属于'预期先行、价格滞后'型潜在拥挤。"
                       f"叙事已较集中，但交易和估值尚未极端，值得跟踪是否进一步累积。")
        else:
            summary = (f"{sector}拥挤度进入中高区间，{p1}为主要来源。"
                       f"该行业仍可能继续上涨，但需要更强的新信息驱动；赔率已不占优。")

    elif total >= 40:
        if narrative > 55 and trading < 45:
            summary = (f"{sector}整体拥挤度中等，但预期层面已有升温迹象。"
                       f"价格尚未反映，属于'潜在拥挤积聚早期'，值得持续跟踪。")
        else:
            summary = (f"{sector}拥挤度处于中等区间，市场结构尚稳。"
                       f"各维度均无极端信号，当前配置赔率相对合理。")
    else:
        if narrative > 45 and valuation < 40:
            summary = (f"{sector}整体拥挤度低，各维度均处于历史低分位。"
                       f"尽管预期有轻微升温，估值仍提供一定安全边际，"
                       f"是资金和叙事尚未集中、赔率相对较好的阶段。")
        else:
            summary = (f"{sector}当前拥挤度低，做多赔率相对合理。"
                       f"主要跟踪方向是各维度是否开始同步上行。")

    # ── 结构解读 ─────────────────────────────────────────────────────────────
    if trading >= 72 and positioning < 52 and valuation < 55:
        structure = ("当前高分更多来自短期交易过热，持仓和估值均未极端。"
                     "这是'交易拥挤主导'类型——短期有回调压力，但基本面结构尚未透支。")
    elif valuation >= 68 and trading < 58:
        structure = ("价格涨幅温和，但历史估值分位已偏高，说明市场对盈利预期定价较为乐观。"
                     "基本面好 ≠ 估值不贵；若盈利不及预期，估值回归空间大。")
    elif positioning >= 68 and trading < 55:
        structure = ("资金持续进场但短期动量不强，属于'慢拥挤'：方向无争议，但一致性持仓越集中，"
                     "踩踏时反应越剧烈。这比价格动量拥挤更值得警惕。")
    elif behavioral >= 68:
        structure = ("市场对负面信息的反应明显钝化：上涨日比例偏高，负面消息后跌幅收窄。"
                     "真正危险的不是利空，而是利好也涨不动——这个阶段尚未到，但方向需关注。")
    elif narrative >= 65 and total < 55:
        structure = ("预期拥挤先于价格拥挤出现，行业涨幅一般但叙事已较集中。"
                     "这类行业属于'潜在拥挤上升期'，一旦交易和持仓跟进，总分将快速上升。")
    elif single_driver and total > 45:
        structure = (f"当前拥挤主要由{p1}单一维度驱动，其他四个维度尚未共振。"
                     f"拥挤结构尚不稳固，但{p1}的持续上升值得警惕。")
    else:
        structure = ""

    # ── 赔率判断 ─────────────────────────────────────────────────────────────
    if total >= 72:
        odds = ("从赔率角度看，当前继续做多的期望回报已显著下降："
                "向上需要不断有新的超预期信息驱动；而一旦出现哪怕轻微的不及预期，回撤可能被放大。")
    elif total >= 58:
        odds = ("赔率尚可但已不占优。该行业仍可能继续上涨，"
                "但继续加仓需要有更明确的基本面支撑或新催化，不宜仅凭趋势惯性重仓。")
    elif total >= 38:
        odds = "赔率中性，当前既无明显泡沫信号，也无明显反转触发，正常配置比例即可。"
    else:
        odds = ("低拥挤阶段，做多赔率相对占优。"
                "关注基本面催化和资金是否开始建仓，是潜在布局的相对合理时机。")

    # ── 风险提示 ─────────────────────────────────────────────────────────────
    if total >= 78:
        risk = ("当前最大风险不是基本面证伪，而是预期过满后的边际降温。"
                "建议回避追高，等待拥挤度出现明确下行信号后再重新评估。")
    elif total >= 65:
        if positioning >= 65:
            risk = ("持仓集中度上升，资金踩踏风险高于均值。"
                    "若出现负面触发，跌幅可能超出基本面应有的幅度。控制仓位并设置止损。")
        elif trading >= 70:
            risk = ("短期动量过热后存在自然降温压力，但若持仓和估值未极端，"
                    "回调更可能是机会而非风险。若未来两周无超预期催化，可能进入高位震荡。")
        else:
            risk = ("拥挤度进入高区间，建议不要仅因强势追高。"
                    "等待拥挤释放或基本面进一步验证后再决定是否加仓。")
    elif total >= 42:
        risk = ("当前无明显拥挤风险，主要需跟踪各维度是否开始加速上行。"
                "尤其关注资金流和预期层面的变化——这两个维度最早预示结构变化。")
    else:
        risk = ("低拥挤阶段的主要风险来自基本面恶化，而非拥挤结构。"
                "需关注行业基本面催化是否可持续，以及市场整体风险偏好变化。")

    # ── 操作信号 ─────────────────────────────────────────────────────────────
    if total >= 78:
        action = "⛔ 回避追高 / 考虑减仓"
    elif total >= 65:
        action = "⚠️ 控制仓位 / 等待拥挤释放"
    elif total >= 48:
        action = "👀 正常配置 / 持续跟踪"
    elif total >= 32:
        action = "✅ 可逢低布局 / 关注催化"
    else:
        action = "✅ 赔率较好 / 可积极关注"

    return {
        "summary":   summary,
        "structure": structure,
        "odds":      odds,
        "risk":      risk,
        "action":    action,
    }
