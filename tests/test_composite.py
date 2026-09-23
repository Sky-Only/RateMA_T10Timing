"""等权打分综合测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ratema.backtest import BacktestConfig
from ratema.composite import (
    COMPOSITE_MODES,
    VOTE_SOURCES,
    build_composite_signal,
    run_composite,
)
from ratema.indicators import SignalConfig
from ratema.pipeline import run_single
from ratema.writers import write_composite

CFG = SignalConfig(20, 120, "abs", 0.001)
BT = BacktestConfig(cost_bps=2.0)


@pytest.fixture()
def per_indicator(dataset):
    return [run_single(dataset, s, CFG, BT, index_col="CBA04502.CS") for s in ("DR001", "R001")]


def _signals(**kwargs) -> dict[str, pd.DataFrame]:
    """构造可控的小信号表：dates + 各指标信号。"""
    dates = pd.bdate_range("2024-01-01", periods=6)
    out = {}
    for name, eff in kwargs.items():
        out[name] = pd.DataFrame(
            {
                "date": dates,
                "signal_raw": eff,
                "signal_eff": eff,
            }
        )
    return out


def test_score_is_net_vote_share():
    sig = _signals(A=[1, 1, -1, -1, 1, 1], B=[1, -1, -1, 1, 1, -1])
    out = build_composite_signal(sig, mode="score")
    # 逐日净票数 / 2
    assert out["score"].tolist() == [1.0, 0.0, -1.0, 0.0, 1.0, 0.0]
    assert out["n_long"].tolist() == [2, 1, 0, 1, 2, 1]
    assert out["n_short"].tolist() == [0, 1, 2, 1, 0, 1]


def test_tie_falls_back_to_previous_effective_signal():
    """平票（N 为偶数）应延续上一有效信号，而不是凭空产生方向。"""
    sig = _signals(A=[1, 1, -1, -1, 1, 1], B=[1, -1, -1, 1, 1, -1])
    out = build_composite_signal(sig, mode="score")
    raw = out["signal_raw"].tolist()
    eff = out["signal_eff"].tolist()
    assert raw[1] == 0.0 and raw[3] == 0.0, "平票日的原始信号应为 0"
    assert eff[1] == eff[0] == 1.0, "平票日应延续上一有效信号"
    assert eff[3] == eff[2] == -1.0
    assert eff[5] == eff[4] == 1.0


def test_score_equals_majority_for_odd_count(per_indicator):
    """N 为奇数时，净票数 > 0 等价于简单多数。"""
    sig = {r.series: r.signals for r in per_indicator}
    a = build_composite_signal(sig, mode="score")["signal_eff"]
    b = build_composite_signal(sig, mode="majority")["signal_eff"]
    pd.testing.assert_series_equal(a, b)


def test_mode_looseness_is_monotonic(per_indicator):
    """持仓时间：unanimous ≤ score/majority ≤ any。"""
    sig = {r.series: r.signals for r in per_indicator}
    longs = {}
    for mode in COMPOSITE_MODES:
        out = build_composite_signal(sig, mode=mode)
        eff = out["signal_eff"].dropna()
        longs[mode] = float((eff == 1).mean())
    assert longs["unanimous"] <= longs["score"] + 1e-12
    assert longs["score"] <= longs["any"] + 1e-12


def test_vote_source_raw_abstains_on_overlap(dataset):
    """raw 投票：重合指标记 0 票；eff 投票：沿用上一日观点。"""
    res = [
        run_single(dataset, s, SignalConfig(20, 120, "abs", 0.01), BT, index_col="CBA04502.CS")
        for s in ("DR001", "R001")
    ]
    sig = {r.series: r.signals for r in res}
    raw_out = build_composite_signal(sig, vote_source="raw")
    eff_out = build_composite_signal(sig, vote_source="eff")
    assert raw_out.attrs["vote_source"] == "raw"
    # raw 口径下弃权票可以被统计到
    assert (raw_out["n_flat"] > 0).any()
    # eff 口径下不存在弃权（各指标已前向填充）
    assert (eff_out["n_flat"] == 0).all()
    assert set(VOTE_SOURCES) == {"eff", "raw"}


def test_invalid_options_raise(per_indicator):
    sig = {r.series: r.signals for r in per_indicator}
    with pytest.raises(ValueError, match="mode"):
        build_composite_signal(sig, mode="nope")
    with pytest.raises(ValueError, match="vote_source"):
        build_composite_signal(sig, vote_source="nope")
    with pytest.raises(ValueError, match="至少"):
        build_composite_signal({})


def test_run_composite_produces_both_flavours(per_indicator, dataset):
    result = run_composite(dataset, per_indicator, BT, mode="score")
    assert result.mode == "score"
    assert len(result.per_indicator) == 2
    # 综合信号：仓位必须是 0/1
    assert set(result.composite_frame["position"].unique()) <= {0, 1}
    # 等权组合：仓位是连续的（0, 0.5, 1）
    assert result.portfolio_frame["position"].between(0, 1).all()
    assert result.composite_metrics["strategy_n_obs"] > 0
    assert result.portfolio_metrics["strategy_n_obs"] > 0


def test_equal_weight_equity_is_exact_mean(per_indicator, dataset):
    """等权组合净值恒等于各子策略净值的算术平均（资金线性性）。"""
    result = run_composite(dataset, per_indicator, BT, mode="score")
    manual = pd.concat([r.frame.set_index("date")["equity"] for r in per_indicator], axis=1).mean(
        axis=1
    )
    got = result.portfolio_frame.set_index("date")["equity"]
    np.testing.assert_allclose(got.to_numpy(), manual.to_numpy(), atol=1e-12)


def test_equal_weight_total_return_is_mean_of_individual(per_indicator, dataset):
    """等权组合的**累计收益**恒等于各单指标累计收益的算术平均（资金线性性）。

    注意不是 CAGR：CAGR 是几何收益率，非线性，
    「均值的 CAGR」与「CAGR 的均值」相差一个二阶凸性项（此处约 1e-6）。
    """
    result = run_composite(dataset, per_indicator, BT, mode="score")
    indiv = [r.metrics["strategy_total_return"] for r in per_indicator]
    assert result.portfolio_metrics["strategy_total_return"] == pytest.approx(
        float(np.mean(indiv)), abs=1e-12
    )
    # CAGR 只应在这个数量级内接近
    cagr_gap = abs(
        result.portfolio_metrics["strategy_cagr"]
        - float(np.mean([r.metrics["strategy_cagr"] for r in per_indicator]))
    )
    assert cagr_gap < 1e-4, f"CAGR 凸性偏差应在 1e-4 以内，实际 {cagr_gap:.2e}"


def test_composite_round_trips_are_sum_of_parts(per_indicator, dataset):
    result = run_composite(dataset, per_indicator, BT, mode="score")
    total = sum(r.metrics["n_completed"] for r in per_indicator)
    assert result.portfolio_metrics["n_completed"] == total


def test_all_modes_run_end_to_end(per_indicator, dataset):
    for mode in COMPOSITE_MODES:
        result = run_composite(dataset, per_indicator, BT, mode=mode)
        m = result.composite_metrics
        assert np.isfinite(m["strategy_cagr"])
        assert np.isfinite(m["strategy_sharpe"])


def test_summary_frame_shape(per_indicator, dataset):
    result = run_composite(dataset, per_indicator, BT, mode="score")
    summary = result.summary_frame()
    # 2 个单指标 + 综合信号 + 等权组合 + 基准
    assert len(summary) == len(per_indicator) + 3
    assert set(summary["类型"]) == {"单指标", "综合信号", "等权组合", "基准"}
    assert {"年化收益", "夏普", "最大回撤", "往返次数"} <= set(summary.columns)


def test_write_composite_artifacts(per_indicator, dataset, tmp_path: Path):
    result = run_composite(dataset, per_indicator, BT, mode="score")
    written = write_composite(result, tmp_path, meta={"数据源": "synthetic"})
    assert written

    for name in (
        "composite_summary.csv",
        "composite_signals.csv",
        "composite_equity.csv",
        "composite_metrics.json",
        "composite_report.md",
    ):
        assert (tmp_path / name).exists(), f"缺少 {name}"

    eq = pd.read_csv(tmp_path / "composite_equity.csv", encoding="utf-8-sig")
    assert {
        "composite_signal_equity",
        "equal_weight_equity",
        "benchmark_equity",
        "score",
    } <= set(eq.columns)

    payload = json.loads((tmp_path / "composite_metrics.json").read_text(encoding="utf-8"))
    assert payload["config"]["mode"] == "score"
    assert "composite_signal" in payload and "equal_weight_portfolio" in payload

    report = (tmp_path / "composite_report.md").read_text(encoding="utf-8")
    assert "步骤 1：每个利率单独测试" in report
    assert "步骤 2：五利率等权打分综合" in report


def test_window_matches_individual_runs(per_indicator, dataset):
    """综合信号的评估窗口应与各单指标一致（同为 MA 预热之后）。"""
    result = run_composite(dataset, per_indicator, BT, mode="score")
    comp_dates = pd.to_datetime(result.composite_frame["date"])
    for res in per_indicator:
        assert comp_dates.iloc[0] == res.frame["date"].iloc[0]
        assert len(comp_dates) == len(res.frame)
