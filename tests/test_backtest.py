"""回测引擎测试：T+1 执行、双边成本、基准。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ratema.backtest import (
    BUY,
    SELL,
    BacktestConfig,
    run_backtest,
    run_benchmark,
)


def _dates(n: int) -> pd.Series:
    return pd.Series(pd.bdate_range("2024-01-01", periods=n))


def test_signal_is_executed_on_next_day():
    """T 日信号必须在 T+1 日成交（首日不受信号影响）。"""
    prices = pd.Series([100.0, 100.0, 100.0, 100.0])
    signals = pd.Series([1.0, 1.0, 1.0, 1.0])
    res = run_backtest(_dates(4), prices, signals, BacktestConfig(cost_bps=0.0))
    eq = res.frame["equity"].to_numpy()

    assert eq[0] == pytest.approx(1.0), "T=0 的信号只能在 T=1 成交，首日净值不变"
    assert eq[1] == pytest.approx(1.0)
    assert res.frame["position"].tolist() == [0, 1, 1, 1]
    assert res.trades[0].action == BUY
    assert res.trades[0].date == _dates(4).iloc[1]


def test_round_trip_cost_matches_formula():
    """一次买+卖后净值应等于 (1-c)/(1+c)。"""
    c = 2.0 / 10_000.0
    prices = pd.Series([100.0, 100.0, 100.0])
    signals = pd.Series([1.0, -1.0, -1.0])  # T=0 买(在 T=1)，T=1 卖(在 T=2)

    res = run_backtest(_dates(3), prices, signals, BacktestConfig(cost_bps=2.0))
    eq = res.frame["equity"].to_numpy()

    assert eq[0] == pytest.approx(1.0)
    assert eq[1] == pytest.approx(1.0 / (1.0 + c))
    assert eq[2] == pytest.approx((1.0 - c) / (1.0 + c))
    assert res.frame["position"].tolist() == [0, 1, 0]

    actions = [t.action for t in res.trades]
    assert actions == [BUY, SELL]
    assert res.n_round_trips == 1
    # 往返成本合计约等于 2c
    assert res.trades[1].pnl_pct == pytest.approx((1.0 - c) / (1.0 + c) - 1.0)


def test_zero_cost_reproduces_price_return():
    """零成本时，T 日信号在 T+1 日成交，收益从成交价起算。"""
    prices = pd.Series([100.0, 110.0, 121.0, 133.1])
    signals = pd.Series([1.0, 1.0, 1.0, 1.0])
    res = run_backtest(_dates(4), prices, signals, BacktestConfig(cost_bps=0.0))
    eq = res.frame["equity"].to_numpy()
    # 首日无持仓，净值 1.0；T=0 的信号在 T=1（价格 110）成交
    assert eq[0] == pytest.approx(1.0)
    assert eq[-1] == pytest.approx(133.1 / 110.0)


def test_cost_mode_round_trip_halves_per_side():
    cfg = BacktestConfig(cost_bps=2.0, cost_mode="round_trip")
    assert cfg.cost_per_side == pytest.approx(0.0001)
    assert cfg.round_trip_cost == pytest.approx(0.0002)

    cfg2 = BacktestConfig(cost_bps=2.0, cost_mode="per_side")
    assert cfg2.cost_per_side == pytest.approx(0.0002)
    assert cfg2.round_trip_cost == pytest.approx(0.0004)


def test_costs_reduce_equity_vs_gross():
    prices = pd.Series(np.linspace(100, 120, 60))
    signals = pd.Series(np.tile([1.0, -1.0], 30))
    res = run_backtest(_dates(60), prices, signals, BacktestConfig(cost_bps=5.0))
    net = res.frame["equity"].iloc[-1]
    gross = res.frame["equity_gross"].iloc[-1]
    assert net < gross
    assert res.total_cost > 0
    assert (res.frame["cost_paid"] > 0).sum() == len(res.trades)


def test_no_lookahead_future_prices_do_not_change_past():
    prices_a = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
    prices_b = prices_a.copy()
    prices_b.iloc[-1] = 999.0  # 只改最后一天
    signals = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])

    ra = run_backtest(_dates(5), prices_a, signals, BacktestConfig())
    rb = run_backtest(_dates(5), prices_b, signals, BacktestConfig())

    np.testing.assert_allclose(
        ra.frame["equity"].to_numpy()[:-1], rb.frame["equity"].to_numpy()[:-1]
    )
    assert ra.frame["position"].tolist()[:-1] == rb.frame["position"].tolist()[:-1]


def test_nan_signal_means_no_position():
    prices = pd.Series([100.0] * 6)
    signals = pd.Series([np.nan, np.nan, 1.0, 1.0, np.nan, -1.0])
    res = run_backtest(_dates(6), prices, signals, BacktestConfig(cost_bps=0.0))
    # T=2 的信号 -> T=3 建仓；T=4 的 NaN -> T=5 不产生动作（NaN 不延续）
    assert res.frame["position"].tolist() == [0, 0, 0, 1, 1, 1]


def test_flat_alternating_signals_produce_matching_trades():
    prices = pd.Series(np.linspace(100, 102, 20))
    signals = pd.Series(np.tile([1.0, -1.0], 10))
    res = run_backtest(_dates(20), prices, signals, BacktestConfig(cost_bps=2.0))
    assert res.frame["position"].iloc[0] == 0
    assert res.n_round_trips >= 9
    assert set(res.frame["position"].unique()) <= {0, 1}


def test_benchmark_is_buy_and_hold_with_entry_cost():
    prices = pd.Series([100.0, 100.0, 100.0, 100.0])
    cfg = BacktestConfig(cost_bps=2.0)
    net, gross = run_benchmark(prices, cfg)
    assert gross.iloc[-1] == pytest.approx(1.0)
    assert net.iloc[-1] == pytest.approx(1.0 / (1.0 + cfg.cost_per_side))
    assert net.iloc[0] == pytest.approx(gross.iloc[0] / (1.0 + cfg.cost_per_side))


def test_benchmark_zero_cost_is_exact_price_ratio():
    prices = pd.Series([100.0, 150.0])
    net, gross = run_benchmark(prices, BacktestConfig(cost_bps=0.0))
    assert gross.iloc[-1] == pytest.approx(1.5)
    assert net.iloc[-1] == pytest.approx(1.5)


def test_config_validation_and_description():
    with pytest.raises(ValueError):
        BacktestConfig(cost_mode="bogus")
    with pytest.raises(ValueError):
        BacktestConfig(cost_bps=-1)
    with pytest.raises(ValueError):
        BacktestConfig(initial_capital=0)
    assert "万分之2" in BacktestConfig(cost_bps=2.0).describe()
    assert "双边合计 4bp" in BacktestConfig(cost_bps=2.0).describe()


def test_open_position_marked_to_market_without_exit_cost():
    prices = pd.Series([100.0, 100.0, 110.0])
    signals = pd.Series([1.0, 1.0, 1.0])
    res = run_backtest(_dates(3), prices, signals, BacktestConfig(cost_bps=2.0))
    c = 0.0002
    assert res.frame["position"].iloc[-1] == 1
    assert res.frame["equity"].iloc[-1] == pytest.approx(110.0 / (100.0 * (1.0 + c)))
    assert res.n_round_trips == 1
    assert res.trades[-1].action == BUY
    assert res.frame.attrs["open_trip"] is not None


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        run_backtest(_dates(3), pd.Series([1.0, 2.0]), pd.Series([1.0, 1.0, 1.0]))


def _vectorized_equity(prices: np.ndarray, position: np.ndarray, cost: float) -> np.ndarray:
    """独立于逐笔引擎的向量化实现，用于交叉验证。"""
    n = len(prices)
    factor = np.ones(n)
    prev = np.concatenate([[0], position[:-1]])
    for t in range(1, n):
        f = 1.0
        if prev[t] == 1:
            f *= prices[t] / prices[t - 1]
        if prev[t] == 0 and position[t] == 1:
            f *= 1.0 / (1.0 + cost)
        if prev[t] == 1 and position[t] == 0:
            f *= 1.0 - cost
        factor[t] = f
    return np.cumprod(factor)


@pytest.mark.parametrize("cost_bps", [0.0, 2.0, 12.5])
def test_engine_matches_vectorized_reimplementation(cost_bps: float):
    """逐笔引擎必须与另一套向量化实现完全一致（防止方向/成本口径写错）。"""
    rng = np.random.default_rng(2024)
    n = 500
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004, n))))
    # 用与真实信号同分布的三态信号，并前向填充
    raw = pd.Series(rng.choice([-1.0, 0.0, 1.0], n, p=[0.45, 0.1, 0.45]))
    raw[:30] = np.nan
    signals = raw.mask(raw == 0.0).ffill()

    cfg = BacktestConfig(cost_bps=cost_bps)
    res = run_backtest(_dates(n), prices, signals, cfg)

    eq_lib = res.frame["equity"].to_numpy()
    eq_ind = _vectorized_equity(
        prices.to_numpy(), res.frame["position"].to_numpy(), cfg.cost_per_side
    )
    np.testing.assert_allclose(eq_lib, eq_ind, atol=1e-12)
    np.testing.assert_allclose(
        res.frame["equity_gross"].to_numpy(),
        _vectorized_equity(prices.to_numpy(), res.frame["position"].to_numpy(), 0.0),
        atol=1e-12,
    )


def test_signal_to_position_direction_is_economically_correct():
    """+1 必须对应持有多头（买入），-1 必须对应空仓（卖出）。"""
    prices = pd.Series([100.0, 100.0, 100.0, 100.0])
    res = run_backtest(
        _dates(4), prices, pd.Series([1.0, 1.0, -1.0, -1.0]), BacktestConfig(cost_bps=0.0)
    )
    assert res.frame["position"].tolist() == [0, 1, 1, 0]
    assert [t.action for t in res.trades] == [BUY, SELL]
