"""独立交叉验证：用另一套（向量化）实现重算，与 ratema 的逐笔引擎对比。

这是一个「防串味」检查 —— 两套代码路径、两种写法，若结果一致，基本可以排除
引擎层面的实现错误。

运行：
    uv run python scripts/cross_check.py
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ratema.backtest import BacktestConfig, run_backtest, run_benchmark
from ratema.indicators import SignalConfig, compute_signals
from ratema.io_utils import DATE_COL, find_default_input, load_dataset, resolve_roles

for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8")


def independent_signals(values: pd.Series, short: int, long: int, tol: float) -> pd.Series:
    """独立重写：MA + 重合阈值 + 前向填充。"""
    ma_s = values.rolling(short).mean()
    ma_l = values.rolling(long).mean()
    diff = ma_s - ma_l

    raw = pd.Series(np.nan, index=values.index, dtype="float64")
    raw[diff > tol] = -1.0
    raw[diff < -tol] = 1.0
    raw[diff.abs() <= tol] = 0.0
    raw[ma_s.isna() | ma_l.isna()] = np.nan
    return raw.replace(0.0, np.nan).ffill()


def independent_equity(prices: np.ndarray, position: np.ndarray, cost: float) -> np.ndarray:
    """独立重写：逐日收益因子累乘。

    因子规则（收盘价成交）：
      持仓不变且持有      -> P_t / P_{t-1}
      空仓不变            -> 1
      当日买入（收盘）    -> 1 / (1 + cost)
      当日卖出（收盘）    -> (P_t / P_{t-1}) * (1 - cost)
    """
    n = len(prices)
    equity = np.ones(n, dtype="float64")
    factor = np.ones(n, dtype="float64")
    prev_pos = np.concatenate([[0], position[:-1]])

    for t in range(1, n):
        f = 1.0
        if prev_pos[t] == 1:
            f *= prices[t] / prices[t - 1]
        if prev_pos[t] == 0 and position[t] == 1:
            f *= 1.0 / (1.0 + cost)
        if prev_pos[t] == 1 and position[t] == 0:
            f *= 1.0 - cost
        factor[t] = f
    return equity * np.cumprod(factor)


def main() -> int:
    path = find_default_input(".")
    dataset = load_dataset(path)
    index_col, rate_cols = resolve_roles(dataset)
    frame = dataset.frame
    dates = frame[DATE_COL]
    prices_all = frame[index_col].to_numpy(dtype="float64")

    print(f"数据源      : {dataset.source}")
    print(f"标的指数    : {index_col}")
    print(f"利率指标    : {', '.join(rate_cols)}")
    print(f"样本区间    : {dates.iloc[0].date()} ~ {dates.iloc[-1].date()}  ({len(frame)} 行)")
    print()

    worst_overall = 0.0
    header = (
        f"{'指标':<12}{'tol':>8}{'交易日':>8}{'买入':>6}{'卖出':>6}"
        f"{'库净值终值':>14}{'独立重算':>14}{'最大偏差':>12}  结果"
    )
    print(header)
    print("-" * len(header))

    for tol in (0.0, 0.0005, 0.002):
        for series in rate_cols:
            values = frame[series].reset_index(drop=True)
            cfg = SignalConfig(20, 120, "abs", tol)
            sig_lib = compute_signals(dates, values, cfg, series_name=series)["signal_eff"]
            sig_ind = independent_signals(values, 20, 120, tol)

            # 信号必须逐日一致
            both = sig_lib.notna() | sig_ind.notna()
            if not np.allclose(
                sig_lib[both].fillna(-99).to_numpy(),
                sig_ind[both].fillna(-99).to_numpy(),
            ):
                print(f"信号不一致：{series} tol={tol}")
                return 1

            first = int(np.argmax(sig_lib.notna().to_numpy()))
            sl = slice(first, len(frame))
            px = prices_all[sl]
            sg = sig_lib.iloc[sl].reset_index(drop=True)

            bt_cfg = BacktestConfig(cost_bps=2.0, cost_mode="per_side")
            res = run_backtest(dates.iloc[sl].reset_index(drop=True), pd.Series(px), sg, bt_cfg)
            eq_lib = res.frame["equity"].to_numpy()
            pos = res.frame["position"].to_numpy()
            eq_ind = independent_equity(px, pos, bt_cfg.cost_per_side)

            dev = float(np.max(np.abs(eq_lib - eq_ind)))
            worst_overall = max(worst_overall, dev)
            ok = "OK" if dev < 1e-9 else "!! 不一致"

            n_buy = int((res.trades_frame["action"] == "BUY").sum())
            n_sell = int((res.trades_frame["action"] == "SELL").sum())
            print(
                f"{series:<12}{tol:>8g}{len(px):>8}{n_buy:>6}{n_sell:>6}"
                f"{eq_lib[-1]:>14.8f}{eq_ind[-1]:>14.8f}{dev:>12.2e}  {ok}"
            )
        print()

    # 基准核对
    for _series in rate_cols[:1]:
        first = 119
        px = prices_all[first:]
        bt_cfg = BacktestConfig(cost_bps=2.0)
        net, gross = run_benchmark(pd.Series(px), bt_cfg)
        expect_gross = px[-1] / px[0]
        expect_net = expect_gross / (1.0 + bt_cfg.cost_per_side)
        print(
            f"基准核对 净价指数：gross={gross.iloc[-1]:.8f} (期望 {expect_gross:.8f}), "
            f"net={net.iloc[-1]:.8f} (期望 {expect_net:.8f})"
        )

    print()
    print(f"全部对比完成，最大净值偏差 = {worst_overall:.3e}")
    if worst_overall < 1e-9:
        print("✅ 逐笔引擎与独立向量化实现完全一致")
        return 0
    print("❌ 存在偏差，请检查实现")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
