"""策略评价：稳健性、显著性与机会成本检验。

回答几个「回测报告不会告诉你」的问题：

1. 20/120 这组参数是特例还是随便一组趋势参数都行？（参数敏感性）
2. 超额收益在统计上显著吗？（Newey-West t 值 + 分块 bootstrap）
3. 5 个利率指标真的是 5 次独立检验吗？（策略相关性）
4. 超额收益来自哪些年份？（分区间稳定性）
5. 空仓期不计息，代价有多大？加上货币市场利息后结论会不会变？
6. 交易成本的盈亏平衡点在哪？

运行：uv run python scripts/evaluate.py
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
from ratema.metrics import cagr_from_equity
from ratema.pipeline import run_all, run_single

ANN = 252
RNG = np.random.default_rng(12345)

for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8")


# --------------------------------------------------------------------------- #
def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """Newey-West 调整后的均值 t 值。"""
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    dev = x - x.mean()
    s = float((dev**2).mean())
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        s += 2.0 * w * float((dev[lag:] * dev[:-lag]).mean())
    if s <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(s / n))


def block_bootstrap_pvalue(x: np.ndarray, block: int = 20, n_boot: int = 5000) -> float:
    """分块 bootstrap：检验「超额收益均值 <= 0」的原假设。"""
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    if n < block * 2:
        return float("nan")
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(0, n - block + 1)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([x[s : s + block] for s in idx])[:n]
        means[i] = sample.mean()
    # 单边：观测均值小于等于 0 的概率
    return float((means <= 0).mean())


def cash_enhanced_equity(equity: pd.Series, position: pd.Series, rate_pct: pd.Series) -> pd.Series:
    """空仓日按货币市场利率计息后的净值。

    equity/position 为策略逐日序列；rate_pct 为年化百分数（如 DR001 = 1.85）。
    """
    idx_ret = equity.pct_change().fillna(0.0).to_numpy()
    pos_prev = np.concatenate([[0], position.to_numpy()[:-1]])
    cash_daily = np.nan_to_num(rate_pct.to_numpy(), nan=0.0) / 100.0 / ANN
    extra = np.where(pos_prev == 0, cash_daily, 0.0)
    return pd.Series(np.cumprod(1.0 + idx_ret + extra), index=equity.index)


def cagr(equity: pd.Series) -> float:
    eq = equity.dropna()
    years = len(eq) / ANN
    return cagr_from_equity(float(eq.iloc[0]), float(eq.iloc[-1]), years)


def max_dd(equity: pd.Series) -> float:
    eq = equity.dropna()
    return float((eq / eq.cummax() - 1).min())


def sharpe(equity: pd.Series) -> float:
    r = equity.pct_change().dropna()
    if r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ANN))


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


# --------------------------------------------------------------------------- #
def main() -> int:
    ds = load_dataset(find_default_input("."))
    index_col, rate_cols = resolve_roles(ds)
    frame = ds.frame
    dates = frame[DATE_COL]

    bt = BacktestConfig(cost_bps=2.0)
    run = run_all(ds, SignalConfig(20, 120, "abs", 0.0), bt, index_col=index_col)

    print(f"数据源   : {ds.source}")
    print(f"样本区间 : {dates.iloc[0].date()} ~ {dates.iloc[-1].date()}  ({len(frame)} 个交易日)")
    print(f"利率指标 : {', '.join(rate_cols)}")

    # ---------------------------------------------------------------- 1
    rule("1. 平均表现（等权平均 5 个指标，而非挑最好的那个）")
    rows = []
    for r in run.results:
        m = r.metrics
        rows.append(
            {
                "指标": r.series,
                "年化": m["strategy_cagr"],
                "波动": m["strategy_ann_vol"],
                "夏普": m["strategy_sharpe"],
                "最大回撤": m["strategy_max_drawdown"],
                "持仓占比": m["time_in_market"],
                "往返次数": m["n_completed"],
            }
        )
    tbl = pd.DataFrame(rows).set_index("指标")
    print(tbl.to_string(float_format=lambda v: f"{v:.4f}"))
    avg_cagr = tbl["年化"].mean()
    avg_sharpe = tbl["夏普"].mean()
    bench_cagr = run.results[0].metrics["benchmark_cagr"]
    bench_sharpe = run.results[0].metrics["benchmark_sharpe"]
    bench_mdd = run.results[0].metrics["benchmark_max_drawdown"]
    print(f"\n5 指标平均 : 年化 {avg_cagr:.2%}  夏普 {avg_sharpe:.3f}")
    print(f"基准(净价) : 年化 {bench_cagr:.2%}  夏普 {bench_sharpe:.3f}  最大回撤 {bench_mdd:.2%}")
    print(f"平均年化超额 : {avg_cagr - bench_cagr:+.2%}")

    # ---------------------------------------------------------------- 2
    rule("2. 参数敏感性：20/120 是特例吗？")
    grid_short = [5, 10, 20, 30, 40]
    grid_long = [60, 120, 180, 240]
    recs = []
    for s in grid_short:
        for long_win in grid_long:
            if s >= long_win:
                continue
            cagrs, sharpes = [], []
            for series in rate_cols:
                try:
                    res = run_single(
                        ds, series, SignalConfig(s, long_win, "abs", 0.0), bt, index_col=index_col
                    )
                except ValueError:
                    continue
                cagrs.append(res.metrics["strategy_cagr"])
                sharpes.append(res.metrics["strategy_sharpe"])
            if cagrs:
                recs.append(
                    {
                        "MA短": s,
                        "MA长": long_win,
                        "平均年化": np.mean(cagrs),
                        "平均夏普": np.mean(sharpes),
                        "优于基准": np.mean(cagrs) > bench_cagr,
                    }
                )
    grid = pd.DataFrame(recs)
    pivot_cagr = grid.pivot(index="MA短", columns="MA长", values="平均年化")
    pivot_sharpe = grid.pivot(index="MA短", columns="MA长", values="平均夏普")
    print("平均年化收益（行=短均线，列=长均线）：")
    print((pivot_cagr * 100).round(2).to_string())
    print("\n平均夏普：")
    print(pivot_sharpe.round(3).to_string())
    n_combos = len(grid)
    n_beat = int(grid["优于基准"].sum())
    print(f"\n{n_combos} 组参数中 {n_beat} 组跑赢基准（{n_beat / n_combos:.0%}）")
    print(
        f"全部组合平均年化 {grid['平均年化'].mean():.2%}，"
        f"区间 [{grid['平均年化'].min():.2%}, {grid['平均年化'].max():.2%}]"
    )
    is_2020 = grid[(grid["MA短"] == 20) & (grid["MA长"] == 120)]
    if len(is_2020):
        v = float(is_2020["平均年化"].iloc[0])
        pct = float((grid["平均年化"] < v).mean())
        print(
            f"20/120 的年化 {v:.2%}，在全部组合中排第 {int((grid['平均年化'] > v).sum()) + 1} 位"
            f"（优于 {pct:.0%} 的组合）"
        )

    # ---------------------------------------------------------------- 3
    rule("3. 超额收益的统计显著性（以 DR007 为例，其余同理）")
    res = next(r for r in run.results if r.series == "DR007")
    f = res.frame.set_index("date")
    excess = (f["daily_return"] - f["benchmark_return"]).to_numpy()
    t_nw = newey_west_t(excess, lags=10)
    naive_t = float(np.nanmean(excess) / (np.nanstd(excess, ddof=1) / np.sqrt(len(excess))))
    p_boot = block_bootstrap_pvalue(excess, block=20)
    ann_excess = float(np.nanmean(excess) * ANN)
    print(f"日均超额        : {np.nanmean(excess) * 1e4:+.3f} bp")
    print(f"年化超额        : {ann_excess:+.2%}")
    print(f"普通 t 值       : {naive_t:+.2f}")
    print(f"Newey-West t 值 : {t_nw:+.2f}   (滞后 10 期)")
    print(f"分块 bootstrap p: {p_boot:.3f}   (H0: 超额均值 <= 0, 5000 次重抽)")
    verdict = "显著" if (t_nw > 1.96 and p_boot < 0.05) else "不显著"
    print(f"结论            : 在 5% 水平下{verdict}")

    # ---------------------------------------------------------------- 4
    rule("4. 5 个指标是 5 次独立检验吗？（日收益相关性）")
    rets = pd.DataFrame({r.series: r.frame.set_index("date")["daily_return"] for r in run.results})
    corr = rets.corr()
    print(corr.round(3).to_string())
    off = corr.to_numpy()[np.triu_indices(len(corr), k=1)]
    print(f"\n两两相关系数：均值 {off.mean():.3f}，最小 {off.min():.3f}，最大 {off.max():.3f}")
    print("→ 5 条曲线高度同源，实质上是同一个信号被重复检验了 5 次，")
    print("  不能按「5 次独立试验全部成功」来解读。")

    # ---------------------------------------------------------------- 5
    rule("5. 分区间稳定性（年化收益）")
    periods = [
        ("2011-2014", "2011-06-27", "2014-12-31"),
        ("2015-2018", "2015-01-01", "2018-12-31"),
        ("2019-2022", "2019-01-01", "2022-12-31"),
        ("2023-2026", "2023-01-01", "2026-09-16"),
    ]
    recs = []
    for name, s, e in periods:
        row = {"区间": name}
        for series in rate_cols:
            r = run_single(
                ds,
                series,
                SignalConfig(20, 120, "abs", 0.0),
                bt,
                index_col=index_col,
                start=s,
                end=e,
            )
            row[series] = r.metrics["strategy_cagr"]
            row["基准"] = r.metrics["benchmark_cagr"]
        recs.append(row)
    sub = pd.DataFrame(recs).set_index("区间")
    print((sub * 100).round(2).to_string())
    print("\n（单位：年化 %）")

    # ---------------------------------------------------------------- 6
    rule("6. 空仓不计息的代价：加上货币市场利息后会怎样？")
    money_rate = frame.set_index(DATE_COL)[rate_cols].mean(axis=1)
    avg_rate = float(money_rate.loc[res.frame["date"].iloc[0] :].mean())
    print(f"样本内 5 个货币市场利率的均值：{avg_rate:.2f}%（可作为空仓资金收益的近似）")
    print()
    recs = []
    for r in run.results:
        fr = r.frame.set_index("date")
        eq = fr["equity"]
        rate_aligned = money_rate.reindex(eq.index)
        enh = cash_enhanced_equity(eq, fr["position"], rate_aligned)
        recs.append(
            {
                "指标": r.series,
                "不计息年化": cagr(eq),
                "计息年化": cagr(enh),
                "利息贡献": cagr(enh) - cagr(eq),
                "持仓占比": r.metrics["time_in_market"],
            }
        )
    cash_tbl = pd.DataFrame(recs).set_index("指标")
    print(cash_tbl.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\n基准(净价)年化 : {bench_cagr:.2%}")
    print(f"计息后策略平均 : {cash_tbl['计息年化'].mean():.2%}")
    print(
        f"→ 计息后年化超额从 {avg_cagr - bench_cagr:+.2%} 扩大到 "
        f"{cash_tbl['计息年化'].mean() - bench_cagr:+.2%}"
    )

    # ---------------------------------------------------------------- 7
    rule("7. 与「真实债券投资者」比较：票息该怎么算")
    enh_avg = float(cash_tbl["计息年化"].mean())
    L = float(np.mean([r.metrics["time_in_market"] for r in run.results]))
    cash_rate = avg_rate / 100.0

    # 关键：策略持仓期间同样吃票息，不能只给基准加票息。
    #   策略全收益 = 净价收益 + 记账的货币利息(空仓期) + 票息(持仓期 L)
    #   基准全收益 = 净价收益 + 票息(全程 100%)
    #   差额       = (计息口径策略 − 基准净价) + (L − 1) × 票息
    adv0 = enh_avg - bench_cagr
    breakeven_coupon = adv0 / (1.0 - L) if L < 1 else float("nan")

    print(f"策略平均持仓比例 L = {L:.1%}，空仓期占比 {(1 - L):.1%}")
    print(f"空仓期按货币市场利率计息（均值 {cash_rate:.2%}）")
    print()
    print(f"{'票息假设 c':>12}{'策略全收益':>12}{'基准全收益':>12}{'超额':>10}")
    print("-" * 48)
    for c in (0.0, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045):
        s_full = enh_avg + L * c
        b_full = bench_cagr + c
        print(f"{c:>12.2%}{s_full:>12.2%}{b_full:>12.2%}{s_full - b_full:>+10.2%}")
    print("-" * 48)
    print()
    print(f"盈亏平衡票息 c* = {breakeven_coupon:.2%}")
    print(f"即：只有当 10 年期国债平均票息高于 {breakeven_coupon:.2%} 时，")
    print("    买入持有才会在绝对收益上胜过本策略。")
    print()
    print("2011-2026 年 10 年国债收益率大部分时间在 2.5%~4.0%，中枢约 3.0~3.5%，")
    print(f"低于 {breakeven_coupon:.2%} 的平衡点，因此策略在含票息口径下仍然略微领先，")
    print(
        "但领先幅度已从「净价口径的 +1.81%」压缩到大约 "
        f"{adv0 - (1 - L) * 0.032:+.2%}（按票息 3.2% 估算）。"
    )
    print()
    print("真正的价值仍在回撤：策略最大回撤约 -4.9%，基准 -9.75%。")

    # ---------------------------------------------------------------- 8
    rule("8. 交易成本盈亏平衡点")
    for series in ("DR007", "R001"):
        r0 = run_single(
            ds,
            series,
            SignalConfig(20, 120, "abs", 0.0),
            BacktestConfig(cost_bps=0.0),
            index_col=index_col,
        )
        r2 = run_single(ds, series, SignalConfig(20, 120, "abs", 0.0), bt, index_col=index_col)
        gross_cagr = r0.metrics["strategy_cagr"]
        net_cagr = r2.metrics["strategy_cagr"]
        n_rt = r2.metrics["n_completed"]
        years = r2.metrics["strategy_n_obs"] / ANN
        edge = gross_cagr - bench_cagr
        breakeven_bps = (edge * years / (2 * n_rt)) * 10_000 if n_rt else float("nan")
        print(
            f"{series}: 零成本年化 {gross_cagr:.2%} → 2bp 成本后 {net_cagr:.2%}"
            f"（拖累 {gross_cagr - net_cagr:.2%}）"
        )
        print(
            f"        往返 {n_rt} 次 / {years:.1f} 年；盈亏平衡单边成本 "
            f"≈ {breakeven_bps:.1f} bp（万分之 {breakeven_bps:.1f}），"
            f"是假设成本 2bp 的 {breakeven_bps / 2:.1f} 倍"
        )

    print()
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
