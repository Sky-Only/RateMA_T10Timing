"""每日交易日志测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ratema.backtest import BacktestConfig
from ratema.composite import run_composite
from ratema.indicators import SignalConfig
from ratema.journal import (
    ACTION_CLOSE,
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_OPEN,
    DAILY_COLUMNS,
    EVENT_COLUMNS,
    append_daily_entry,
    build_journal,
    render_journal_markdown,
    write_journal,
)
from ratema.pipeline import run_single

CFG = SignalConfig(20, 120, "abs", 0.001)
BT = BacktestConfig(cost_bps=2.0)


@pytest.fixture()
def single_frame(dataset):
    return run_single(dataset, "DR001", CFG, BT, index_col="CBA04502.CS").frame


@pytest.fixture()
def composite(dataset):
    per = [run_single(dataset, s, CFG, BT, index_col="CBA04502.CS") for s in ("DR001", "R001")]
    return run_composite(dataset, per, BT, mode="score")


def test_journal_columns_and_shape(single_frame):
    j = build_journal(single_frame, series="DR001", index_col="CBA04502.CS")
    assert list(j.daily.columns) == DAILY_COLUMNS
    assert list(j.events.columns) == EVENT_COLUMNS
    assert len(j.daily) == len(single_frame)
    assert j.series == "DR001"


def test_actions_are_consistent_with_positions(single_frame):
    """动作必须是仓位变化的正确翻译。"""
    j = build_journal(single_frame, series="DR001", index_col="CBA04502.CS")
    d = j.daily
    for _, r in d.iterrows():
        prev, cur, action = r["position_prev"], r["position"], r["action"]
        if prev == cur:
            assert action == (ACTION_HOLD if cur == 1 else ACTION_FLAT)
        elif prev == 0 and cur == 1:
            assert action == ACTION_OPEN
        elif prev == 1 and cur == 0:
            assert action == ACTION_CLOSE
        else:  # pragma: no cover - 仓位只可能是 0/1
            pytest.fail(f"意外仓位变化 {prev} -> {cur}")


def test_first_day_is_never_an_open(single_frame):
    """T+1 执行：第一天不可能有仓位，因此第一条记录不可能是建仓。"""
    j = build_journal(single_frame, series="DR001", index_col="CBA04502.CS")
    assert j.daily["position"].iloc[0] == 0
    assert j.daily["action"].iloc[0] == ACTION_FLAT


def test_action_is_fully_explained_by_actionable_signal(single_frame):
    """「执行依据」列必须能唯一解释当日动作，避免日志出现自相矛盾的相邻列。

    T 日动作由 T-1 日信号决定；T 日信号要到 T+1 日才生效。
    """
    j = build_journal(single_frame, series="DR001", index_col="CBA04502.CS")
    d = j.daily
    expected = d["signal"].shift(1).fillna(0).astype(int)
    pd.testing.assert_series_equal(d["actionable_signal"], expected, check_names=False)

    for _, r in d.iterrows():
        if r["action"] == ACTION_OPEN:
            assert r["actionable_signal"] == 1, "建仓必须由执行依据 +1 触发"
        elif r["action"] == ACTION_CLOSE:
            assert r["actionable_signal"] == -1, "平仓必须由执行依据 -1 触发"
        # 当日收盘仓位必须等于执行依据的目标
        assert r["position"] == (1 if r["actionable_signal"] == 1 else 0)


def test_events_only_contain_state_changes(single_frame):
    j = build_journal(single_frame, series="DR001", index_col="CBA04502.CS")
    assert set(j.events["action"]) <= {ACTION_OPEN, ACTION_CLOSE, "换向"}
    assert len(j.events) == int((j.daily["action"].isin([ACTION_OPEN, ACTION_CLOSE, "换向"])).sum())
    # 建仓与平仓数量应基本成对
    assert j.summary["n_open"] in {j.summary["n_close"], j.summary["n_close"] + 1}


def test_summary_matches_daily(single_frame):
    j = build_journal(single_frame, series="DR001", index_col="CBA04502.CS")
    s, d = j.summary, j.daily
    assert s["n_days"] == len(d)
    # 按收盘仓位统计的天数
    assert s["n_days_in_market"] == int((d["position"] == 1).sum())
    assert s["n_days_flat"] == int((d["position"] == 0).sum())
    assert s["n_days_in_market"] + s["n_days_flat"] == s["n_days"]
    # 动作次数：持仓日 = 建仓日 + 持有日（本策略不做空，故无换向）
    assert s["n_switch"] == 0
    assert s["n_days_in_market"] == s["n_open"] + s["n_hold"]
    assert s["n_days_flat"] == s["n_close"] + s["n_flat"]
    assert s["final_equity"] == pytest.approx(float(d["equity"].iloc[-1]))
    assert s["time_in_market"] == pytest.approx(float((d["position"] == 1).mean()))


def test_composite_journal_carries_vote_columns(composite):
    """综合日志必须带上票数，否则看不出信号是怎么来的。"""
    j = build_journal(composite.composite_frame, series="COMPOSITE", index_col="CBA04502.CS")
    assert {"score", "n_long", "n_short"} <= set(j.daily.columns)
    assert j.daily["n_long"].notna().all()
    assert j.daily["score"].between(-1, 1).all()


def test_render_markdown_contains_key_sections(single_frame):
    j = build_journal(single_frame, series="DR001", index_col="CBA04502.CS")
    md = render_journal_markdown(j, tail=10)
    assert "# 每日交易日志" in md
    assert "## 概览" in md
    assert "## 动作流水（建仓 / 平仓）" in md
    assert "## 最近 10 个交易日" in md
    # 概览里应出现动作次数与持仓占比
    assert "动作次数" in md and "持仓 / 空仓天数" in md


def test_render_markdown_handles_no_events():
    """全程空仓时不应崩，且要明确写出「没有仓位变动」。"""
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
    assert j.events.empty
    assert j.summary["n_open"] == 0
    md = render_journal_markdown(j)
    assert "没有任何仓位变动" in md


def test_write_journal_artifacts(single_frame, tmp_path: Path):
    j = build_journal(single_frame, series="DR001", index_col="CBA04502.CS")
    written = write_journal(j, tmp_path, tail=15)
    assert written
    for name in ("journal.csv", "journal_events.csv", "journal.md"):
        assert (tmp_path / name).exists(), f"缺少 {name}"

    daily = pd.read_csv(tmp_path / "journal.csv", encoding="utf-8-sig")
    assert list(daily.columns) == DAILY_COLUMNS
    assert daily["date"].is_monotonic_increasing

    events = pd.read_csv(tmp_path / "journal_events.csv", encoding="utf-8-sig")
    assert list(events.columns) == EVENT_COLUMNS


def test_build_journal_rejects_missing_columns():
    bad = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "close": [1.0]})
    with pytest.raises(ValueError, match="缺少列"):
        build_journal(bad, series="X", index_col="Y")


def test_build_journal_rejects_empty():
    empty = pd.DataFrame(
        columns=["date", "close", "position", "equity", "benchmark_equity", "signal_eff"]
    )
    with pytest.raises(ValueError, match="为空"):
        build_journal(empty, series="X", index_col="Y")


def test_append_daily_entry_creates_then_dedupes(single_frame, tmp_path: Path):
    """每日追加：同日重复运行应覆盖而不是产生重复行。"""
    j = build_journal(single_frame, series="DR001", index_col="CBA04502.CS")
    path = tmp_path / "live" / "journal_tail.csv"

    append_daily_entry(j, path)
    assert path.exists()
    first = pd.read_csv(path, encoding="utf-8-sig")
    assert len(first) == 1

    # 同一天再跑一次 -> 仍然只有一行
    append_daily_entry(j, path)
    assert len(pd.read_csv(path, encoding="utf-8-sig")) == 1

    # 指定历史某天 -> 变成两行，按日期排序
    target = pd.Timestamp(j.daily["date"].iloc[-5]).strftime("%Y-%m-%d")
    append_daily_entry(j, path, date=target)
    combined = pd.read_csv(path, encoding="utf-8-sig")
    assert len(combined) == 2
    assert combined["date"].is_monotonic_increasing


def test_append_daily_entry_returns_the_row(single_frame, tmp_path: Path):
    j = build_journal(single_frame, series="DR001", index_col="CBA04502.CS")
    row = append_daily_entry(j, tmp_path / "j.csv")
    assert row["date"] == j.daily["date"].iloc[-1]
    assert row["series"] == "DR001"
