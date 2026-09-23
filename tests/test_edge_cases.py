"""边界与退化情况测试（针对代码审查中发现的问题）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ratema.backtest import BacktestConfig, run_backtest
from ratema.indicators import SignalConfig, compute_signals
from ratema.io_utils import resolve_roles
from ratema.journal import build_journal, render_journal_markdown
from ratema.metrics import annual_breakdown, cagr_from_equity, perf_stats, trade_stats
from ratema.pipeline import run_single


def _dates(n: int) -> pd.Series:
    return pd.Series(pd.bdate_range("2024-01-01", periods=n))


# --------------------------------------------------------------------------- #
# CAGR：终值 <= 0 时必须返回 NaN，不能返回复数
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("start", "end", "years"),
    [
        (1.0, -1.0, 10.0),  # 空头被击穿，净值为负
        (1.0, -0.5, 3.0),
        (1.0, 0.0, 5.0),  # 净值归零
        (0.0, 1.0, 5.0),  # 起点为 0
        (1.0, 1.2, 0.0),  # 零年数
        (1.0, 1.2, -1.0),
        (1.0, float("nan"), 5.0),
        (float("nan"), 1.0, 5.0),
    ],
)
def test_cagr_from_equity_returns_real_nan(start, end, years):
    """回归：负数开分数次方在 Python 里是**复数**，会污染全部下游指标与 JSON。

    真实触发路径：short_only / long_short 方向下标的涨幅超过 100%。
    """
    value = cagr_from_equity(start, end, years)
    assert isinstance(value, float), f"必须是 float，得到 {type(value).__name__}"
    assert np.isnan(value)


def test_cagr_from_equity_normal_cases():
    assert cagr_from_equity(1.0, 1.0, 10.0) == 0.0
    assert cagr_from_equity(1.0, 1.21, 2.0) == pytest.approx(0.10, abs=1e-12)
    assert cagr_from_equity(2.0, 1.0, 1.0) == pytest.approx(-0.5, abs=1e-12)


@pytest.mark.parametrize("n", [100, 200, 300, 1000, 3808])
def test_perf_stats_has_no_complex_values_when_equity_goes_negative(n: int):
    eq = pd.Series([1.0] * (n - 1) + [-1.0], index=pd.bdate_range("2024-01-01", periods=n))
    stats = perf_stats(eq)
    complex_keys = [k for k, v in stats.items() if isinstance(v, complex)]
    assert not complex_keys, f"出现复数指标：{complex_keys}"
    assert isinstance(stats["cagr"], float)


def test_wiped_out_short_does_not_break_metrics():
    """1 倍空头遇标的翻倍以上：净值转负，指标仍须是实数且不抛异常。"""
    res = run_backtest(
        _dates(3),
        pd.Series([100.0, 100.0, 300.0]),
        pd.Series([-1.0] * 3),
        BacktestConfig(cost_bps=0.0, direction="short_only"),
    )
    assert (res.frame["equity"] < 0).any()
    eq = pd.Series(res.frame["equity"].to_numpy(), index=pd.bdate_range("2024-01-01", periods=3))
    stats = perf_stats(eq)
    assert not any(isinstance(v, complex) for v in stats.values())


# --------------------------------------------------------------------------- #
# annual_breakdown：收益列缺失时应自动从净值推导
# --------------------------------------------------------------------------- #
def test_annual_breakdown_derives_returns_from_equity():
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=5),
            "close": [100.0] * 5,
            "position": [0] * 5,
            "equity": [1.0, 1.01, 1.02, 1.03, 1.04],
            "benchmark_equity": [1.0, 1.005, 1.01, 1.015, 1.02],
            "signal_eff": [-1.0] * 5,
        }
    )
    table = annual_breakdown(frame, None)
    assert not table.empty
    overall = table[table["year"] == "全区间"].iloc[0]
    assert overall["strategy_return"] == pytest.approx(0.04, abs=1e-12)
    assert overall["benchmark_return"] == pytest.approx(0.02, abs=1e-12)


def test_annual_breakdown_still_raises_when_nothing_to_derive_from():
    with pytest.raises(ValueError, match="可供推导"):
        annual_breakdown(pd.DataFrame({"date": [pd.Timestamp("2024-01-01")]}), None)


def test_journal_annual_is_populated_for_minimal_frame():
    """回归：build_journal 会吞掉 annual_breakdown 的 ValueError，
    导致日志静默地没有分年度段。现在应能自动推导。"""
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=5),
            "close": [100.0] * 5,
            "position": [0] * 5,
            "equity": [1.0] * 5,
            "benchmark_equity": [1.0] * 5,
            "signal_eff": [-1.0] * 5,
        }
    )
    j = build_journal(frame, series="X", index_col="Y")
    assert not j.annual.empty, "日志的分年度表不应为空"
    assert "分年度收益与胜率" in render_journal_markdown(j)


# --------------------------------------------------------------------------- #
# 退化情形
# --------------------------------------------------------------------------- #
def test_never_trading_strategy_yields_nan_not_crash():
    res = run_backtest(
        _dates(6),
        pd.Series([100.0] * 6),
        pd.Series([-1.0] * 6),
        BacktestConfig(cost_bps=2.0, direction="long_only"),
    )
    assert (res.frame["equity"] == 1.0).all()
    assert not res.trades
    stats = perf_stats(
        pd.Series(res.frame["equity"].to_numpy(), index=pd.bdate_range("2024-01-01", periods=6))
    )
    assert stats["total_return"] == 0.0
    assert np.isnan(stats["sharpe"]), "零波动时夏普应无定义"
    assert np.isnan(stats["calmar"])


def test_flat_price_charges_entry_cost_once():
    res = run_backtest(
        _dates(6), pd.Series([100.0] * 6), pd.Series([1.0] * 6), BacktestConfig(cost_bps=2.0)
    )
    assert len(res.trades) == 1
    assert res.frame["equity"].iloc[1] < 1.0
    np.testing.assert_allclose(res.frame["equity"].to_numpy()[2:], res.frame["equity"].iloc[1])


def test_all_overlap_leaves_signal_empty():
    """阈值宽到全程重合：原始信号全 0，有效信号全 NaN（没有可延续的信号）。"""
    values = pd.Series(np.linspace(1.0, 2.0, 300))
    out = compute_signals(
        pd.Series(pd.bdate_range("2024-01-01", periods=300)),
        values,
        SignalConfig(20, 120, "abs", 100.0),
    )
    valid = out["ma_long"].notna()
    assert out.loc[valid, "is_overlap"].all()
    assert out.loc[valid, "signal_raw"].eq(0).all()
    assert out["signal_eff"].isna().all()
    # 预热期不算重合（均线还没成形）
    assert not out.loc[~valid, "is_overlap"].any()


def test_single_row_and_two_row_inputs():
    one = run_backtest(_dates(1), pd.Series([100.0]), pd.Series([1.0]), BacktestConfig())
    assert np.isfinite(one.frame["equity"].iloc[-1])

    two = run_backtest(
        _dates(2), pd.Series([100.0, 101.0]), pd.Series([1.0, 1.0]), BacktestConfig()
    )
    assert np.isfinite(two.frame["equity"].iloc[-1])


def test_empty_trades_table():
    assert trade_stats(None)["n_completed"] == 0
    empty = pd.DataFrame(
        columns=["date", "action", "pnl_pct", "cost", "holding_days", "round_trip_id"]
    )
    stats = trade_stats(empty)
    assert stats["n_completed"] == 0
    assert stats["n_shorts"] == 0 and stats["n_covers"] == 0


# --------------------------------------------------------------------------- #
# 初始资金尺度不变性（回归：绝对 epsilon 会让小本金静默失效）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("capital", [1e-15, 1e-12, 1e-9, 1e-3, 1.0, 1e6, 1e12])
def test_results_are_scale_invariant_in_initial_capital(capital: float):
    """收益率必须与本金规模无关。

    回归背景：判定「是否持仓」曾用**绝对**阈值 ``units > 1e-12``。
    本金为 1e-12 时持仓份额约 1e-14，被误判为空仓，于是每个交易日都重复
    建仓（2283 笔），净值恒为初始值、年化变成 NaN，**且不报任何错**。
    现在改用精确的 0 比较。
    """
    from ratema.io_utils import load_dataset

    ds = load_dataset("data/csv/panel.csv")
    res = run_single(
        ds,
        "DR007",
        SignalConfig(20, 120, "bp", 5.0),
        BacktestConfig(cost_bps=2.0, initial_capital=capital),
        index_col="CBA04502.CS",
    )
    assert len(res.backtest.trades) == 59, f"成交笔数应恒为 59，实际 {len(res.backtest.trades)}"
    assert int((res.frame["position"] != 0).sum()) == 2283
    assert res.metrics["strategy_total_return"] == pytest.approx(0.369931, abs=1e-6)
    assert res.metrics["strategy_cagr"] == pytest.approx(0.021048, abs=1e-6)


def test_position_state_uses_exact_zero():
    from ratema.backtest import _position_state

    assert _position_state(0.0) == 0
    assert _position_state(1e-30) == 1, "极小的正持仓也必须是多头"
    assert _position_state(-1e-30) == -1
    assert _position_state(1.0) == 1
    assert _position_state(-1.0) == -1


# --------------------------------------------------------------------------- #
# 综合信号的时间过滤（回归：带 --start/--end 会直接报错）
# --------------------------------------------------------------------------- #
def _composite_inputs():
    from ratema.io_utils import load_dataset

    ds = load_dataset("data/csv/panel.csv")
    idx, rates = resolve_roles(ds)
    bt = BacktestConfig(cost_bps=2.0)
    per = [run_single(ds, s, SignalConfig(20, 120, "bp", 5.0), bt, index_col=idx) for s in rates]
    return ds, idx, bt, per


def test_composite_respects_start_and_end():
    """回归：run_composite 曾只裁剪价格、不裁剪信号，
    带 --start/--end 时必然抛「综合信号与价格序列未对齐」。"""
    from ratema.composite import run_composite

    ds, idx, bt, per = _composite_inputs()
    full = run_composite(ds, per, bt, index_col=idx, mode="score")
    part = run_composite(
        ds, per, bt, index_col=idx, mode="score", start="2020-01-01", end="2024-12-31"
    )

    assert 0 < len(part.composite_frame) < len(full.composite_frame)
    dates = pd.to_datetime(part.composite_frame["date"])
    assert dates.iloc[0] >= pd.Timestamp("2020-01-01")
    assert dates.iloc[-1] <= pd.Timestamp("2024-12-31")

    # 重叠区间上的信号必须与全窗口完全一致
    fs = full.composite_frame.set_index("date")["signal_eff"]
    ps = part.composite_frame.set_index("date")["signal_eff"]
    common = fs.index.intersection(ps.index)
    assert len(common) == len(ps)
    np.testing.assert_array_equal(fs.loc[common].to_numpy(), ps.loc[common].to_numpy())

    pdates = pd.to_datetime(part.portfolio_frame["date"])
    assert pdates.iloc[0] >= pd.Timestamp("2020-01-01")
    assert np.isfinite(part.portfolio_metrics["strategy_cagr"])


def test_composite_start_after_end_raises():
    from ratema.composite import run_composite

    ds, idx, bt, per = _composite_inputs()
    with pytest.raises(ValueError, match="窗口为空"):
        run_composite(
            ds, per, bt, index_col=idx, mode="score", start="2025-01-01", end="2020-01-01"
        )


# --------------------------------------------------------------------------- #
# 无未来函数：截断数据后重算，之前区间的结果必须不变
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("mode", "tol"), [("std", 0.5), ("q", 0.05), ("abs", 0.02)])
def test_truncating_data_does_not_change_earlier_signals(mode: str, tol: float):
    """std / q 模式的容差是滚动统计量，必须只用到 t 及以前的数据。"""
    from ratema.io_utils import Dataset, load_dataset

    ds = load_dataset("data/csv/panel.csv")
    cfg = SignalConfig(20, 120, mode, tol)
    full = run_single(ds, "DR007", cfg, BacktestConfig(), index_col="CBA04502.CS")

    truncated = Dataset(frame=ds.frame.iloc[:2500].copy(), name_map=ds.name_map, source="truncated")
    part = run_single(truncated, "DR007", cfg, BacktestConfig(), index_col="CBA04502.CS")

    n = len(part.frame)
    np.testing.assert_array_equal(
        full.frame["signal_eff"].to_numpy()[:n], part.frame["signal_eff"].to_numpy()
    )
