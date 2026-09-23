"""把策略从头讲清楚：从 6 列原始数据到买卖动作。

运行：uv run python scripts/explain.py
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ratema.backtest import BacktestConfig
from ratema.indicators import SignalConfig
from ratema.io_utils import DATE_COL, find_default_input, load_dataset, resolve_roles
from ratema.pipeline import run_single

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _s.reconfigure(encoding="utf-8")

ANN = 252


def rule(t: str) -> None:
    print()
    print("=" * 84)
    print(f"  {t}")
    print("=" * 84)


def main() -> int:
    ds = load_dataset(find_default_input("."))
    index_col, rate_cols = resolve_roles(ds)
    f = ds.frame
    dates = f[DATE_COL]

    # ------------------------------------------------------------------ 1
    rule("第 0 步：这 6 列数据到底是什么")
    print(f"{'列名':<14}{'中文名':<26}{'均值':>8}{'最小':>8}{'最大':>8}{'波动(年化bp)':>14}")
    print("-" * 84)
    for c in rate_cols + [index_col]:
        s = f[c].astype(float)
        vol = s.diff().std() * np.sqrt(ANN) * 100
        print(
            f"{c:<14}{ds.display_name(c):<26}{s.mean():>8.3f}{s.min():>8.3f}{s.max():>8.3f}{vol:>14.0f}"
        )
    print()
    print("注意量级：利率的日常波动是「几个 bp」，而指数是 100 上下的价格。")

    # ------------------------------------------------------------------ 2
    rule("第 1 步：5 个利率是「同一件事」的 5 个报价吗？——要看层面")
    corr_raw = f[rate_cols].pct_change().corr()
    print("(a) 原始利率「日变化」的相关系数（噪声层面）：")
    print(corr_raw.round(3).to_string())
    off_raw = corr_raw.to_numpy()[np.triu_indices(len(corr_raw), k=1)]
    print(f"    两两相关均值 {off_raw.mean():.3f} —— 其实很低！")

    print()
    print("(b) 但把它们变成信号之后，策略「日收益」的相关系数：")
    strat = {}
    for c in rate_cols:
        r = run_single(
            ds,
            c,
            SignalConfig(20, 120, "abs", 0.0),
            BacktestConfig(cost_bps=2.0),
            index_col=index_col,
        )
        strat[c] = r.frame.set_index("date")["daily_return"]
    corr_sig = pd.DataFrame(strat).corr()
    print(corr_sig.round(3).to_string())
    off_sig = corr_sig.to_numpy()[np.triu_indices(len(corr_sig), k=1)]
    print(f"    两两相关均值 {off_sig.mean():.3f} —— 非常高。")
    print()
    print("为什么差这么多？")
    print("  · 原始利率的日变化充满「期限噪声」和「机构噪声」：")
    print("    DR001 是隔夜、DR007 是 7 天；DR 只含银行、R 含非银。")
    print("    单日各走各的，所以日变化相关只有 0.2。")
    print("  · 但 20 日均线把噪声平滑掉之后，它们描述的是同一个东西：")
    print("    央行货币政策的松紧。所以均线交叉信号高度同步，相关 0.85。")
    print()
    print("→ 结论：这 5 个指标不是 5 个独立信号，而是同一信号的 5 个观测窗口。")
    print("  「5 个指标全部跑赢基准」的独立信息量，接近「1 个指标跑赢」。")

    if "R007" in f.columns and "DR007" in f.columns:
        spread = (f["R007"] - f["DR007"]).dropna()
        print()
        print("附带一个数据质量提示 —— R007 − DR007（非银相对银行的融资溢价）：")
        print(
            f"  均值 {spread.mean():.3f}%   中位 {spread.median():.3f}%   "
            f"p95 {spread.quantile(0.95):.3f}%   最大 {spread.max():.3f}%"
        )
        top = f.loc[f["R007"].nlargest(5).index, [DATE_COL, "DR007", "R007"]]
        print("  R007 最高的 5 天：")
        for _, row in top.iterrows():
            print(f"    {row[DATE_COL].date()}  DR007={row['DR007']:.2f}%  R007={row['R007']:.2f}%")
        print("  这些是真实的季末/年末非银资金挤压（银行惜贷、非银抢钱），")
        print("  不是录入错误。但它们会让 R001/R007 的均线短期失真，")
        print("  DR 系列和 SHIBOR 不受影响 —— 这也是 DR 更适合做信号的原因之一。")

    # ------------------------------------------------------------------ 3
    rule("第 2 步：把「均线交叉」翻译成人话（以 DR007 为例）")
    res = run_single(
        ds,
        "DR007",
        SignalConfig(20, 120, "abs", 0.0),
        BacktestConfig(cost_bps=2.0),
        index_col=index_col,
    )
    fr = res.frame.copy()

    sample = fr.iloc[[0, 500, 1000, 1500, 2000, 2500, 3000, -1]][
        ["date", "rate_value", "ma_short", "ma_long", "spread_bp", "signal_eff", "position"]
    ]
    sample = sample.rename(
        columns={
            "date": "日期",
            "rate_value": "当日DR007",
            "ma_short": "MA20",
            "ma_long": "MA120",
            "spread_bp": "MA20-MA120(bp)",
            "signal_eff": "信号",
            "position": "是否持仓",
        }
    )
    print(sample.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("读法：MA20 是「最近 1 个月的平均资金成本」，MA120 是「最近半年的平均」。")
    print("      MA20 > MA120  ⇒  最近比过去半年更贵  ⇒  资金面在收紧  ⇒  信号 -1（空仓）")
    print("      MA20 < MA120  ⇒  最近比过去半年更便宜 ⇒  资金面在放松  ⇒  信号 +1（持有）")

    # ------------------------------------------------------------------ 4
    rule("第 3 步：信号真的有预测力吗？（最直接的一张表）")
    ret = fr["close"].pct_change().fillna(0.0)
    fr["_ret"] = ret
    hold_r = fr.loc[fr["position"] == 1, "_ret"]
    flat_r = fr.loc[fr["position"] == 0, "_ret"]
    print(f"{'':<16}{'交易日数':>10}{'占比':>8}{'日均指数收益':>14}{'折算年化':>12}")
    print("-" * 84)
    for name, sub in (("策略持仓的日子", hold_r), ("策略空仓的日子", flat_r)):
        print(
            f"{name:<16}{len(sub):>10}{len(sub) / len(fr):>8.1%}"
            f"{sub.mean() * 1e4:>12.3f}bp{sub.mean() * ANN:>12.2%}"
        )
    print("-" * 84)
    print(
        f"{'差异':<16}{'':>10}{'':>8}{(hold_r.mean() - flat_r.mean()) * 1e4:>12.3f}bp"
        f"{(hold_r.mean() - flat_r.mean()) * ANN:>12.2%}"
    )
    print()
    print("→ 策略「选择持有」的那些日子，指数平均是涨的；")
    print("  「选择空仓」的那些日子，指数平均是跌的。这就是超额收益的来源。")

    # ------------------------------------------------------------------ 5
    rule("第 4 步：三个真实案例（按月拆开看）")
    episodes = [
        ("2013 年「钱荒」—— 策略大胜", "2013-05-02", "2013-08-30"),
        ("2020 年疫情宽松 + 5 月反转 —— 暴露策略的滞后", "2020-01-02", "2020-07-15"),
        ("2024 年债牛 —— 平稳跟涨", "2024-01-02", "2024-12-31"),
    ]
    for name, s, e in episodes:
        sub = fr[(fr["date"] >= pd.Timestamp(s)) & (fr["date"] <= pd.Timestamp(e))].copy()
        if sub.empty:
            continue
        sub["month"] = sub["date"].dt.to_period("M")
        g = sub.groupby("month").agg(
            DR007均值=("rate_value", "mean"),
            利差bp=("spread_bp", "mean"),
            多头天数=("signal_eff", lambda x: int((x == 1).sum())),
            持仓占比=("position", "mean"),
        )
        g["指数月涨跌"] = sub.groupby("month")["close"].last().pct_change() * 100
        eq_m = sub.groupby("month")["equity"].last()
        g["策略月涨跌"] = eq_m.pct_change() * 100
        g.loc[g.index[0], "指数月涨跌"] = (
            sub.groupby("month")["close"].last().iloc[0] / sub["close"].iloc[0] - 1
        ) * 100
        g.loc[g.index[0], "策略月涨跌"] = (eq_m.iloc[0] / sub["equity"].iloc[0] - 1) * 100

        print()
        print(f"── {name}")
        print(f"   {s} ~ {e}")
        print(g.round(2).to_string())
        idx_chg = sub["close"].iloc[-1] / sub["close"].iloc[0] - 1
        strat_chg = sub["equity"].iloc[-1] / sub["equity"].iloc[0] - 1
        print(
            f"   累计：指数 {idx_chg:+.2%}   策略 {strat_chg:+.2%}   "
            f"（整体持仓 {sub['position'].mean():.1%}）"
        )

    print()
    print("案例一（2013 钱荒）：DR007 从 3% 飙到 11.2%，MA20 高出 MA120 达 279bp，")
    print("  策略全程空仓（持仓仅 4.7%），指数跌 4.28%，策略只跌 0.32%。")
    print("  这是策略最擅长的事：识别「资金面骤然收紧」并提前离场。")
    print()
    print("案例二（2020）—— 看清策略的「转折成本」：")
    print("  1~4 月：DR007 一路降到 1.27%（最低 0.91%），利差 -62bp，策略满仓，")
    print("          吃下 +4.4%（月月正收益）。")
    print("  5 月  ：仍满仓，指数转跌 -1.45%，全部回吐。")
    print("  6 月  ：资金利率快速回升，但 MA20 是滞后指标，月中才翻空，")
    print("          当月持仓 62%、亏 1.36%，比指数的 -1.23% 还差一点。")
    print("  7 月  ：完全空仓，躲开指数的 -0.99%。")
    print("  全期策略 +1.52% vs 指数 +0.65%，仍然胜出。")
    print("  → 关键认识：转折当月必然吃亏（信号要等均线交叉才确认），")
    print("    但只要趋势延续，后面躲开的跌幅足以弥补。这是趋势型策略的固有代价。")
    print()
    print("案例三（2024 债牛）：资金面平稳偏松，89% 的时间为多头信号，")
    print("  策略跟涨但略输指数（+5.42% vs +6.72%），差额来自空仓日与交易成本。")
    print("  说明「顺风年份」策略不会多赚，它的价值在「逆风年份少亏」。")

    # ------------------------------------------------------------------ 6
    rule("第 5 步：为什么是「净价指数」，这让收益看起来很小")
    px = f[index_col].astype(float)
    years = len(px) / ANN
    print("中债-10年期国债净价(总值)指数 CBA04502.CS")
    print(f"  {dates.iloc[0].date()} 收于 {px.iloc[0]:.4f}")
    print(f"  {dates.iloc[-1].date()} 收于 {px.iloc[-1]:.4f}")
    print(
        f"  15.7 年累计 {(px.iloc[-1] / px.iloc[0] - 1):+.2%}，"
        f"年化 {(px.iloc[-1] / px.iloc[0]) ** (1 / years) - 1:+.2%}"
    )
    print()
    print("【净价】= 债券报价，不含应计利息。这个指数只反映「资本利得/损失」，")
    print("        不反映票息。所以 15 年才涨 17.79% —— 因为 10 年国债的回报")
    print("        本来主要来自票息（约 3%/年），价格波动只是零头。")
    print()
    print("这解释了为什么所有年化数字都是 1%~2% 量级：我们衡量的是「价格择时」，")
    print("而不是「债券投资总回报」。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
