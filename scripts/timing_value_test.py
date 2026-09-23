"""择时是否真的创造价值？—— 随机对照检验。

策略约一半时间持仓、回撤减半。问题是：这是「择时能力」还是「单纯降低仓位」？

三重对照：
  A. 恒定仓位基准（同样的平均持仓比例，但从不择时）
  B. 逐日打乱的仓位（同样的持仓天数，但时机随机）
  C. 分块打乱的仓位（同样的持仓天数，且保留持仓段长度结构）

若真实策略显著优于 B/C，说明「什么时候持有」有信息含量；
若只与 A 相当，说明收益全部来自「少拿一点仓位」。

运行：uv run python scripts/timing_value_test.py
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
from ratema.io_utils import find_default_input, load_dataset, resolve_roles
from ratema.pipeline import run_single

ANN = 252
RNG = np.random.default_rng(20240916)

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _s.reconfigure(encoding="utf-8")


def stats_from_returns(ret: np.ndarray) -> dict[str, float]:
    eq = np.cumprod(1.0 + ret)
    years = len(ret) / ANN
    cagr = float(eq[-1] ** (1 / years) - 1)
    vol = float(np.std(ret, ddof=1) * np.sqrt(ANN))
    sharpe = (
        float(np.mean(ret) / np.std(ret, ddof=1) * np.sqrt(ANN)) if np.std(ret, ddof=1) else np.nan
    )
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "mdd": mdd}


def strategy_returns(
    position: np.ndarray, index_ret: np.ndarray, cash_ret: np.ndarray, cost: float
) -> np.ndarray:
    """给定仓位序列，重算逐日收益（含换手成本）。"""
    pos_prev = np.concatenate([[0], position[:-1]])
    turn = np.abs(np.diff(np.concatenate([[0], position])))  # 1 = 建仓或平仓
    return pos_prev * index_ret + (1.0 - pos_prev) * cash_ret - turn * cost


def block_shuffle(position: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """按块重排，保留持仓段的长度结构，只打乱出现的时机。"""
    n = len(position)
    n_blocks = int(np.ceil(n / block))
    blocks = [position[i * block : (i + 1) * block] for i in range(n_blocks)]
    order = rng.permutation(n_blocks)
    return np.concatenate([blocks[i] for i in order])[:n]


def main() -> int:
    ds = load_dataset(find_default_input("."))
    index_col, rate_cols = resolve_roles(ds)

    bt = BacktestConfig(cost_bps=2.0)
    n_boot = 2000
    block = 20  # 约一个月

    print("=" * 100)
    print("  择时价值检验：真实信号 vs 恒定仓位 vs 随机时机")
    print("=" * 100)
    header = (
        f"{'指标':<10}{'平均仓位':>8}{'真实年化':>9}{'真实夏普':>9}{'真实回撤':>9}"
        f"{'恒定仓位':>9}{'随机时机':>9}{'随机p值':>8}{'真实(零成本)':>12}{'随机(零成本)':>12}{'零成本p值':>10}"
    )
    print(header)
    print("-" * 100)

    summary = []
    for series in rate_cols:
        res = run_single(ds, series, SignalConfig(20, 120, "abs", 0.0), bt, index_col=index_col)
        f = res.frame.set_index("date")
        pos = f["position"].to_numpy(dtype="float64")
        idx_ret = f["close"].pct_change().fillna(0.0).to_numpy()
        zeros = np.zeros_like(idx_ret)  # 空仓不计息，与基准口径一致
        cost = bt.cost_per_side
        w = float(pos.mean())

        real_stats = stats_from_returns(strategy_returns(pos, idx_ret, zeros, cost))
        real0_stats = stats_from_returns(strategy_returns(pos, idx_ret, zeros, 0.0))

        # A. 恒定仓位：同样的平均暴露，从不择时，建仓成本只在首日收一次
        const_ret = w * idx_ret
        const_ret[0] -= w * cost
        const_stats = stats_from_returns(const_ret)

        # B/C. 随机时机：分块打乱仓位序列（保留持仓段长度结构，只打乱时机）
        #      成本按各自实际换手计收，分别统计含成本与零成本两种口径
        boot_cagr = np.empty(n_boot)
        boot_cagr0 = np.empty(n_boot)
        for i in range(n_boot):
            shuffled = block_shuffle(pos, block, RNG)
            boot_cagr[i] = stats_from_returns(strategy_returns(shuffled, idx_ret, zeros, cost))[
                "cagr"
            ]
            boot_cagr0[i] = stats_from_returns(strategy_returns(shuffled, idx_ret, zeros, 0.0))[
                "cagr"
            ]
        p_value = float((boot_cagr >= real_stats["cagr"]).mean())
        p_value0 = float((boot_cagr0 >= real0_stats["cagr"]).mean())

        print(
            f"{series:<10}{w:>8.1%}{real_stats['cagr']:>9.2%}{real_stats['sharpe']:>9.3f}"
            f"{real_stats['mdd']:>9.2%}{const_stats['cagr']:>9.2%}"
            f"{boot_cagr.mean():>9.2%}{p_value:>8.3f}"
            f"{real0_stats['cagr']:>12.2%}{boot_cagr0.mean():>12.2%}{p_value0:>10.3f}"
        )
        summary.append(
            {
                "series": series,
                "w": w,
                "real_cagr": real_stats["cagr"],
                "real_cagr0": real0_stats["cagr"],
                "real_mdd": real_stats["mdd"],
                "real_sharpe": real_stats["sharpe"],
                "const_cagr": const_stats["cagr"],
                "const_mdd": const_stats["mdd"],
                "boot_cagr": float(boot_cagr.mean()),
                "boot_cagr0": float(boot_cagr0.mean()),
                "p_value": p_value,
                "p_value0": p_value0,
            }
        )

    df = pd.DataFrame(summary)
    print("-" * 100)
    print(
        f"{'平均':<10}{df['w'].mean():>8.1%}{df['real_cagr'].mean():>9.2%}"
        f"{df['real_sharpe'].mean():>9.3f}{df['real_mdd'].mean():>9.2%}"
        f"{df['const_cagr'].mean():>9.2%}{df['boot_cagr'].mean():>9.2%}"
        f"{df['p_value'].mean():>8.3f}{df['real_cagr0'].mean():>12.2%}"
        f"{df['boot_cagr0'].mean():>12.2%}{df['p_value0'].mean():>10.3f}"
    )
    bench = run_single(
        ds, rate_cols[0], SignalConfig(20, 120, "abs", 0.0), bt, index_col=index_col
    ).metrics["benchmark_cagr"]
    print(f"\n参考：基准（买入持有净价指数）年化 {bench:.2%}")

    print()
    print("解读：")
    print(
        f"  · 真实策略平均年化 {df['real_cagr'].mean():.2%}（含成本），"
        f"同样平均仓位 {df['w'].mean():.1%} 的恒定配置 {df['const_cagr'].mean():.2%}"
    )
    print(
        f"  · 把持仓时机分块打乱后，平均年化 {df['boot_cagr'].mean():.2%}"
        f"（零成本口径 {df['boot_cagr0'].mean():.2%}）"
    )
    print(
        f"  · 真实结果 ≥ 随机时机的概率：p 均值 {df['p_value'].mean():.3f}"
        f"（零成本口径 {df['p_value0'].mean():.3f}）"
    )
    if (df["p_value0"] < 0.05).all():
        print("  → 即使剔掉成本因素，真实时机仍显著优于随机（p<0.05），")
        print("    说明「什么时候持有」确实携带信息，不只是「少拿仓位」。")
    elif (df["p_value0"] < 0.10).all():
        print("  → 剔掉成本后边际显著（p<0.10），但未达 5% 标准。")
    else:
        print("  → 与随机时机无法区分，择时证据不足。")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
