"""交易方向（只做多 / 只做空 / 多空双向）测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ratema.backtest import (
    BUY,
    COVER,
    DIRECTIONS,
    SELL,
    SHORT,
    BacktestConfig,
    map_target_position,
    run_backtest,
)
from ratema.indicators import SignalConfig
from ratema.io_utils import load_dataset
from ratema.journal import (
    ACTION_CLOSE,
    ACTION_CLOSE_SHORT,
    ACTION_FLAT,
    ACTION_HOLD_SHORT,
    ACTION_OPEN,
    ACTION_OPEN_SHORT,
    ACTION_TO_LONG,
    ACTION_TO_SHORT,
    build_journal,
)
from ratema.metrics import annual_breakdown
from ratema.pipeline import run_single


def _dates(n: int) -> pd.Series:
    return pd.Series(pd.bdate_range("2024-01-01", periods=n))


# --------------------------------------------------------------------------- #
# 信号 → 目标仓位 的映射
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("signal", "direction", "expected"),
    [
        (+1.0, "long_only", 1),
        (-1.0, "long_only", 0),
        (+1.0, "short_only", 0),
        (-1.0, "short_only", -1),
        (+1.0, "long_short", 1),
        (-1.0, "long_short", -1),
        (0.0, "long_only", 0),
        (0.0, "short_only", 0),
        (0.0, "long_short", 0),
        (np.nan, "long_only", 0),
        (np.nan, "long_short", 0),
        (None, "long_short", 0),
    ],
)
def test_map_target_position(signal, direction, expected):
    assert map_target_position(signal, direction) == expected


def test_map_target_position_rejects_unknown_direction():
    with pytest.raises(ValueError, match="未知 direction"):
        map_target_position(1.0, "both_ways")


def test_config_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        BacktestConfig(direction="nope")


def test_all_directions_are_advertised():
    assert set(DIRECTIONS) == {"long_only", "short_only", "long_short"}


# --------------------------------------------------------------------------- #
# 仓位取值域
# --------------------------------------------------------------------------- #
def _run(prices, signals, direction, cost_bps=0.0):
    return run_backtest(
        _dates(len(prices)),
        pd.Series(prices, dtype="float64"),
        pd.Series(signals, dtype="float64"),
        BacktestConfig(cost_bps=cost_bps, direction=direction),
    )


def test_long_only_position_domain():
    res = _run([100.0, 101.0, 99.0, 98.0], [1.0, 1.0, -1.0, -1.0], "long_only")
    assert set(res.frame["position"].unique()) <= {0, 1}
    assert {t.action for t in res.trades} == {BUY, SELL}


def test_short_only_position_domain():
    res = _run([100.0, 101.0, 99.0, 98.0], [-1.0, -1.0, 1.0, 1.0], "short_only")
    assert set(res.frame["position"].unique()) <= {0, -1}
    assert {t.action for t in res.trades} == {SHORT, COVER}


def test_long_short_position_domain_and_always_in_market():
    """多空双向：首日无信号故为空仓，之后必须始终有仓位。"""
    res = _run(
        [100.0, 101.0, 99.0, 98.0, 97.0, 96.0],
        [1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
        "long_short",
    )
    pos = res.frame["position"]
    assert pos.iloc[0] == 0, "首日没有可执行信号"
    assert set(pos.iloc[1:].unique()) <= {-1, 1}, "有信号之后不应空仓"
    assert set(pos.unique()) <= {-1, 0, 1}
    # 完整走一遍：建多 → 平多 + 建空 → 平空 + 建多
    assert {t.action for t in res.trades} == {BUY, SELL, SHORT, COVER}


# --------------------------------------------------------------------------- #
# 做空的会计口径
# --------------------------------------------------------------------------- #
def test_short_entry_leaves_equity_at_capital_over_one_plus_cost():
    """建空后的净值应与建多对称：cash/(1+c)。"""
    c = 2.0 / 10_000.0
    res = _run([100.0, 100.0], [-1.0, -1.0], "short_only", cost_bps=2.0)
    assert res.frame["equity"].iloc[0] == pytest.approx(1.0)
    assert res.frame["equity"].iloc[1] == pytest.approx(1.0 / (1.0 + c))


def test_short_gains_when_price_falls():
    """价格为 100 → 90（−10%），1 倍名义空头应赚约 10%。"""
    c = 2.0 / 10_000.0
    res = _run([100.0, 100.0, 90.0], [-1.0, -1.0, -1.0], "short_only", cost_bps=2.0)
    entry_equity = 1.0 / (1.0 + c)
    assert res.frame["equity"].iloc[2] == pytest.approx(entry_equity * 1.10)


def test_short_loses_when_price_rises():
    c = 2.0 / 10_000.0
    res = _run([100.0, 100.0, 110.0], [-1.0, -1.0, -1.0], "short_only", cost_bps=2.0)
    entry_equity = 1.0 / (1.0 + c)
    assert res.frame["equity"].iloc[2] == pytest.approx(entry_equity * 0.90)


def test_short_round_trip_cost_formula():
    """空头往返的净值应等于闭式解：C + N(1−c) − |units|·P_cover·(1+c)。"""
    c = 2.0 / 10_000.0
    capital, p_entry, p_cover = 1.0, 100.0, 90.0
    res = _run(
        [p_entry, p_entry, p_cover, p_cover],
        [-1.0, -1.0, 1.0, 1.0],
        "short_only",
        cost_bps=2.0,
    )
    notional = capital / (1.0 + c)
    units = notional / p_entry
    cash_after_short = capital + notional * (1.0 - c)
    owed = units * p_cover
    expected = cash_after_short - owed * (1.0 + c)

    assert res.frame["position"].tolist() == [0, -1, -1, 0]
    assert res.frame["equity"].iloc[3] == pytest.approx(expected)


def test_short_is_wiped_out_when_price_doubles():
    """1 倍名义空头在标的翻倍时净值归零，继续上涨才转负。"""
    flat = _run([100.0, 100.0, 200.0], [-1.0, -1.0, -1.0], "short_only", cost_bps=0.0)
    assert flat.frame["equity"].iloc[2] == pytest.approx(0.0, abs=1e-12)

    deeper = _run([100.0, 100.0, 250.0], [-1.0, -1.0, -1.0], "short_only", cost_bps=0.0)
    assert deeper.frame["equity"].iloc[2] < 0, "空头亏损无上限（真实交易中会先被强平）"


def test_open_short_round_trip_records_pnl():
    res = _run(
        [100.0, 100.0, 90.0, 90.0],
        [-1.0, -1.0, 1.0, 1.0],
        "short_only",
        cost_bps=2.0,
    )
    assert res.n_round_trips == 1
    cover = next(t for t in res.trades if t.action == COVER)
    assert cover.pnl_pct is not None and cover.pnl_pct > 0.09  # 价格跌 10%
    assert cover.holding_days == 2


def test_flip_long_to_short_charges_both_legs():
    """多翻空要拆成平多 + 建空两笔，成本相加。"""
    res = _run([100.0, 100.0, 100.0], [1.0, -1.0, -1.0], "long_short", cost_bps=2.0)
    assert [t.action for t in res.trades] == [BUY, SELL, SHORT]
    # 平多与建空发生在同一天
    assert res.trades[1].date == res.trades[2].date
    # 当天成本 = 两笔之和
    day_cost = res.frame["cost_paid"].iloc[2]
    assert day_cost == pytest.approx(res.trades[1].cost + res.trades[2].cost)


def test_gross_equity_mirror_has_no_cost():
    res = _run([100.0, 100.0, 90.0], [-1.0, -1.0, -1.0], "short_only", cost_bps=10.0)
    # 零成本镜像：建空后净值仍为 1，跌 10% 后为 1.10
    assert res.frame["equity_gross"].iloc[1] == pytest.approx(1.0)
    assert res.frame["equity_gross"].iloc[2] == pytest.approx(1.10)
    assert res.frame["equity"].iloc[2] < res.frame["equity_gross"].iloc[2]


# --------------------------------------------------------------------------- #
# 向后兼容
# --------------------------------------------------------------------------- #
def test_long_only_is_the_default():
    assert BacktestConfig().direction == "long_only"


def test_nan_signal_keeps_position_instead_of_liquidating():
    """NaN 表示「无观点」，应维持当前仓位，而不是强制平仓。"""
    res = _run([100.0] * 6, [1.0, 1.0, 1.0, np.nan, np.nan, 1.0], "long_only")
    assert res.frame["position"].tolist() == [0, 1, 1, 1, 1, 1]


def test_direction_does_not_change_benchmark():
    from ratema.backtest import run_benchmark

    prices = pd.Series([100.0, 110.0])
    for d in DIRECTIONS:
        _net, gross = run_benchmark(prices, BacktestConfig(cost_bps=2.0, direction=d))
        assert gross.iloc[-1] == pytest.approx(1.1)


# --------------------------------------------------------------------------- #
# 与真实数据 / 下游模块的集成
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_dataset():
    return load_dataset("data/csv/panel.csv")


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_real_data_runs_for_every_direction(real_dataset, direction):
    res = run_single(
        real_dataset,
        "DR007",
        SignalConfig(20, 120, "bp", 5.0),
        BacktestConfig(cost_bps=2.0, direction=direction),
    )
    m = res.metrics
    assert np.isfinite(m["strategy_cagr"])
    assert np.isfinite(m["strategy_sharpe"])
    # 持仓占比必须落在 [0,1]（曾经因取仓位均值而在做空时变成负数）
    assert 0.0 <= m["time_in_market"] <= 1.0
    pos = set(res.frame["position"].unique())
    if direction == "long_only":
        assert pos <= {0, 1}
    elif direction == "short_only":
        assert pos <= {0, -1}
    else:
        assert pos <= {-1, 0, 1}
        # 首日之外应始终有仓位
        assert set(res.frame["position"].iloc[1:].unique()) <= {-1, 1}


def test_long_only_numbers_unchanged(real_dataset):
    """回归护栏：默认方向下的结果必须与改造前一致。"""
    res = run_single(
        real_dataset, "DR007", SignalConfig(20, 120, "bp", 5.0), BacktestConfig(cost_bps=2.0)
    )
    m = res.metrics
    assert m["strategy_cagr"] == pytest.approx(0.0210, abs=5e-4)
    assert m["strategy_sharpe"] == pytest.approx(1.181, abs=5e-3)
    assert m["strategy_max_drawdown"] == pytest.approx(-0.0507, abs=5e-4)
    assert m["n_completed"] == 29


def test_short_only_counts_round_trips(real_dataset):
    """做空方向的平仓动作是 COVER，往返次数不能因为只数 SELL 而变成 0。"""
    res = run_single(
        real_dataset,
        "DR007",
        SignalConfig(20, 120, "bp", 5.0),
        BacktestConfig(cost_bps=2.0, direction="short_only"),
    )
    m = res.metrics
    assert m["n_completed"] > 0
    assert m["n_shorts"] > 0 and m["n_covers"] > 0
    assert 0.0 <= m["trade_win_rate"] <= 1.0


def test_journal_classifies_short_positions():
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=6),
            "close": [100.0] * 6,
            "position": [0, -1, -1, 0, 1, 0],
            "equity": [1.0] * 6,
            "benchmark_equity": [1.0] * 6,
            "signal_eff": [-1.0] * 6,
        }
    )
    j = build_journal(frame, series="X", index_col="Y")
    assert j.daily["action"].tolist() == [
        ACTION_FLAT,  # 首日无前值，视为空仓
        ACTION_OPEN_SHORT,
        ACTION_HOLD_SHORT,
        ACTION_CLOSE_SHORT,
        ACTION_OPEN,
        ACTION_CLOSE,
    ]
    assert j.summary["n_days_short"] == 2
    assert j.summary["n_open_short"] == 1
    assert j.summary["n_close_short"] == 1


def test_journal_classifies_direction_flips():
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=4),
            "close": [100.0] * 4,
            "position": [0, 1, -1, 1],
            "equity": [1.0] * 4,
            "benchmark_equity": [1.0] * 4,
            "signal_eff": [1.0] * 4,
        }
    )
    j = build_journal(frame, series="X", index_col="Y")
    assert j.daily["action"].tolist() == [
        ACTION_FLAT,
        ACTION_OPEN,
        ACTION_TO_SHORT,
        ACTION_TO_LONG,
    ]
    assert j.summary["n_switch"] == 2


def test_annual_breakdown_works_for_short_only(real_dataset):
    res = run_single(
        real_dataset,
        "DR007",
        SignalConfig(20, 120, "bp", 5.0),
        BacktestConfig(cost_bps=2.0, direction="short_only"),
    )
    table = annual_breakdown(res.frame, res.backtest.trades_frame)
    assert not table.empty
    overall = table[table["year"] == "全区间"].iloc[0]
    # 做空方向的持仓占比必须是正的
    assert overall["time_in_market"] > 0
    assert table["n_round_trips"].notna().all()


# --------------------------------------------------------------------------- #
# 做空方向的日志汇总（曾经显示「建仓 0、平仓 0」和「持仓 0 天」）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_journal_summary_is_self_consistent(real_dataset, direction):
    res = run_single(
        real_dataset,
        "DR007",
        SignalConfig(20, 120, "bp", 5.0),
        BacktestConfig(cost_bps=2.0, direction=direction),
    )
    j = build_journal(
        res.frame, series="DR007", index_col="CBA04502.CS", trades=res.backtest.trades_frame
    )
    s = j.summary

    # 持仓 + 空仓天数必须覆盖全部交易日
    assert s["n_days_in_market"] + s["n_days_flat"] == s["n_days"]
    assert s["n_days_long"] + s["n_days_short"] == s["n_days_in_market"]
    if direction == "short_only":
        assert s["n_days_long"] == 0
        assert s["n_days_short"] > 0

    # 持仓占比必须与天数一致（曾经出现「持仓 0 天」却写 40% 的自相矛盾）
    assert s["time_in_market"] == pytest.approx(s["n_days_in_market"] / s["n_days"], abs=1e-12)


def test_journal_action_summary_reports_short_actions(real_dataset):
    """做空方向的汇总必须报出「建空/平空」，不能显示成 0 次动作。"""
    from ratema.journal import _action_summary, render_journal_markdown

    res = run_single(
        real_dataset,
        "DR007",
        SignalConfig(20, 120, "bp", 5.0),
        BacktestConfig(cost_bps=2.0, direction="short_only"),
    )
    j = build_journal(
        res.frame, series="DR007", index_col="CBA04502.CS", trades=res.backtest.trades_frame
    )
    text = _action_summary(j.summary)
    assert "建空" in text and "平空" in text
    assert "建多 0" not in text and "平多 0" not in text
    assert j.summary["n_open_short"] > 0

    md = render_journal_markdown(j, tail=5)
    assert "建空" in md
    assert f"空头 {j.summary['n_days_short']}" in md


def test_journal_position_labels_cover_all_three_states():
    from ratema.journal import _position_label

    assert _position_label(1) == "持多"
    assert _position_label(0) == "空仓"
    assert _position_label(-1) == "持空"


# --------------------------------------------------------------------------- #
# 图表在做空方向下也必须能出图
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("direction", ["short_only", "long_short"])
def test_signal_mechanics_chart_handles_short_positions(real_dataset, tmp_path, direction):
    from ratema.charts import plot_signal_mechanics

    res = run_single(
        real_dataset,
        "DR007",
        SignalConfig(20, 120, "bp", 5.0),
        BacktestConfig(cost_bps=2.0, direction=direction),
    )
    path = plot_signal_mechanics(res, tmp_path)
    assert path.exists()
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("direction", ["short_only", "long_short"])
def test_composite_chart_handles_short_positions(real_dataset, tmp_path, direction):
    from ratema.charts import plot_composite
    from ratema.composite import run_composite

    bt = BacktestConfig(cost_bps=2.0, direction=direction)
    per = [
        run_single(real_dataset, s, SignalConfig(20, 120, "bp", 5.0), bt, index_col="CBA04502.CS")
        for s in ("DR001", "R001")
    ]
    result = run_composite(real_dataset, per, bt, mode="score")
    assert plot_composite(result, tmp_path).exists()
