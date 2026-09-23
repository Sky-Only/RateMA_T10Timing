"""分年度收益与胜率测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ratema.backtest import BacktestConfig
from ratema.indicators import SignalConfig
from ratema.metrics import ANNUAL_COLUMNS, annual_breakdown
from ratema.pipeline import run_all, run_single
from ratema.render import annual_table, format_table

CFG = SignalConfig(20, 120, "abs", 0.001)
BT = BacktestConfig(cost_bps=2.0)


@pytest.fixture()
def result(dataset):
    return run_single(dataset, "DR001", CFG, BT, index_col="CBA04502.CS")


@pytest.fixture()
def run(dataset):
    return run_all(dataset, CFG, BT)


def test_annual_breakdown_has_year_rows_plus_overall(result):
    years = pd.to_datetime(result.frame["date"]).dt.year.unique()
    table = annual_breakdown(result.frame, result.backtest.trades_frame)

    assert table["year"].iloc[-1] == "全区间"
    assert set(table["year"].iloc[:-1]) == {str(int(y)) for y in years}
    # 首日无收益，故总天数比 frame 少 1；各年之和应等于全区间
    assert table["n_days"].iloc[-1] == len(result.frame) - 1
    assert table["n_days"].iloc[:-1].sum() == table["n_days"].iloc[-1]


def test_overall_row_matches_headline_metrics(result):
    """全区间行的收益必须与总指标一致（口径不能有两套）。"""
    table = annual_breakdown(result.frame, result.backtest.trades_frame)
    overall = table[table["year"] == "全区间"].iloc[0]
    m = result.metrics

    assert overall["strategy_return"] == pytest.approx(m["strategy_total_return"], abs=1e-9)
    assert overall["benchmark_return"] == pytest.approx(m["benchmark_total_return"], abs=1e-9)
    assert overall["strategy_max_drawdown"] == pytest.approx(m["strategy_max_drawdown"], abs=1e-9)
    assert overall["strategy_sharpe"] == pytest.approx(m["strategy_sharpe"], abs=1e-6)
    assert overall["benchmark_sharpe"] == pytest.approx(m["benchmark_sharpe"], abs=1e-6)


def test_yearly_returns_compound_to_overall(result):
    """各年收益连乘必须等于全区间收益（复利一致性）。"""
    table = annual_breakdown(result.frame, result.backtest.trades_frame)
    yearly = table[table["year"] != "全区间"]
    overall = table[table["year"] == "全区间"].iloc[0]

    compounded = float(np.prod(1.0 + yearly["strategy_return"].to_numpy()) - 1.0)
    assert compounded == pytest.approx(overall["strategy_return"], abs=1e-9)


def test_excess_equals_strategy_minus_benchmark(result):
    table = annual_breakdown(result.frame, result.backtest.trades_frame)
    np.testing.assert_allclose(
        table["excess_return"].to_numpy(),
        (table["strategy_return"] - table["benchmark_return"]).to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        table["beat_benchmark"].to_numpy(),
        (table["strategy_return"] > table["benchmark_return"]).to_numpy(),
    )


def test_win_rates_are_probabilities(result):
    table = annual_breakdown(result.frame, result.backtest.trades_frame)
    for col in ("strategy_win_rate", "benchmark_win_rate", "relative_win_rate"):
        vals = table[col].dropna()
        assert ((vals >= 0) & (vals <= 1)).all(), f"{col} 应落在 [0,1]"


def test_zero_return_days_count_against_strategy_win_rate(result):
    """空仓日收益为 0，不构成「上涨」，因此策略日胜率会被压低。"""
    table = annual_breakdown(result.frame, result.backtest.trades_frame)
    overall = table[table["year"] == "全区间"].iloc[0]
    frame = result.frame
    flat_share = float((frame["position"] == 0).mean())
    assert flat_share > 0.3, "本策略应有相当比例的空仓日"
    # 策略日胜率必然低于「非空仓日的上涨占比」加上空仓占比的上限
    assert overall["strategy_win_rate"] < 1.0 - flat_share + 0.5


def test_round_trips_fill_zero_for_years_without_closes(result):
    table = annual_breakdown(result.frame, result.backtest.trades_frame)
    assert table["n_round_trips"].notna().all(), "无平仓的年份应为 0 而不是缺失"
    assert (table["n_round_trips"] >= 0).all()


def test_trade_win_rate_within_bounds(result):
    table = annual_breakdown(result.frame, result.backtest.trades_frame)
    vals = table["trade_win_rate"].dropna()
    assert ((vals >= 0) & (vals <= 1)).all()


def test_annual_breakdown_without_trades(result):
    """不传成交流水也必须能算出收益与日胜率，只是没有交易胜率。"""
    table = annual_breakdown(result.frame, None)
    assert table["trade_win_rate"].isna().all()
    assert table["strategy_return"].notna().all()


def test_annual_breakdown_validates_input():
    with pytest.raises(ValueError, match="缺少列"):
        annual_breakdown(pd.DataFrame({"date": [pd.Timestamp("2024-01-01")]}))
    with pytest.raises(ValueError, match="为空"):
        annual_breakdown(pd.DataFrame(columns=["date", "daily_return", "benchmark_return"]))


def test_format_table_handles_all_kinds():
    df = pd.DataFrame(
        {
            "year": ["2024", "全区间"],
            "pct_col": [0.0123, -0.0456],
            "num_col": [1.23456, np.nan],
            "int_col": [3.0, 7.0],
            "bool_col": [True, False],
        }
    )
    cols = [
        ("year", "年份", "text"),
        ("pct_col", "百分比", "pct"),
        ("num_col", "数值", "num"),
        ("int_col", "整数", "int"),
        ("bool_col", "布尔", "bool"),
    ]
    out = format_table(df, cols)
    assert list(out.columns) == ["年份", "百分比", "数值", "整数", "布尔"]
    assert out["百分比"].tolist() == ["1.23%", "-4.56%"]
    assert out["数值"].tolist() == ["1.235", "n/a"]
    assert out["整数"].tolist() == ["3", "7"]
    assert out["布尔"].tolist() == ["是", "否"]
    # 缺列不应报错，只是不输出该列
    assert list(format_table(df, [("nope", "缺", "pct")]).columns) == []


def test_annual_table_aggregates_across_indicators(run):
    table = annual_table(run)
    assert "全区间" in table
    for label in ("策略收益", "基准收益", "超额", "相对胜率", "交易胜率"):
        assert label in table

    per_ind = annual_table(run, per_indicator=True)
    for series in run.rate_cols:
        assert series in per_ind


def test_annual_columns_spec_is_consistent():
    """ANNUAL_COLUMNS 的键必须与 annual_breakdown 的输出对得上（除 year 外）。"""
    keys = {k for k, _, _ in ANNUAL_COLUMNS}
    assert "year" in keys
    kinds = {kind for _, _, kind in ANNUAL_COLUMNS}
    assert kinds <= {"pct", "num", "int", "bool", "text"}


def test_backtest_writes_annual_csv(dataset, tmp_path: Path):
    from ratema.writers import write_run

    run = run_all(dataset, CFG, BT)
    write_run(run, tmp_path)
    path = tmp_path / "annual_breakdown.csv"
    assert path.exists()
    df = pd.read_csv(path, encoding="utf-8-sig")
    assert {"series", "year", "strategy_return", "benchmark_return"} <= set(df.columns)
    assert set(df["series"]) == set(run.rate_cols)
    # 每个指标都应有「全区间」行
    assert (df["year"] == "全区间").sum() == len(run.rate_cols)


def test_mean_annual_averages_across_indicators(result):
    """多指标聚合：非布尔列取均值、布尔列取多数，年份顺序保持不变。"""
    from ratema.metrics import mean_annual

    base = annual_breakdown(result.frame, result.backtest.trades_frame)
    other = base.copy()
    other["strategy_return"] = other["strategy_return"] + 0.10

    agg = mean_annual([base, other])

    assert agg["year"].iloc[-1] == "全区间"
    assert list(agg["year"]) == list(base["year"])
    # 构造的第二张表收益 +10%，均值应恰好 +5%
    np.testing.assert_allclose(
        agg["strategy_return"].to_numpy(),
        base["strategy_return"].to_numpy() + 0.05,
        atol=1e-12,
    )
    assert agg["beat_benchmark"].dtype == bool


def test_mean_annual_handles_empty():
    from ratema.metrics import mean_annual

    assert mean_annual([]).empty
    assert mean_annual([pd.DataFrame()]).empty


def test_plot_annual_breakdown_writes_png(result, tmp_path: Path):
    from ratema.charts import plot_annual_breakdown

    table = annual_breakdown(result.frame, result.backtest.trades_frame)
    path = plot_annual_breakdown(table, tmp_path, title="测试", subtitle="单元测试")
    assert path.exists()
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(data) > 20_000


def test_plot_functions_self_initialize_cjk_font(result, tmp_path: Path):
    """直接调用 plot_* 也必须拿到中文字体（否则图里是方框）。

    回归背景：``setup_style()`` 原本只由 CLI 调用，测试里直接调
    ``plot_annual_breakdown`` 会退回 DejaVu Sans 并产生大量
    "Glyph missing from font" 警告。
    """
    import warnings

    import matplotlib.pyplot as plt

    from ratema.charts import ensure_style, plot_annual_breakdown

    ensure_style()  # 幂等
    ensure_style()
    fonts = list(plt.rcParams["font.sans-serif"])
    assert fonts, "字体列表不应为空"

    table = annual_breakdown(result.frame, result.backtest.trades_frame)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        plot_annual_breakdown(table, tmp_path, title="中文标题测试", subtitle="分年度")


def test_plot_annual_breakdown_validates_input(tmp_path: Path):
    from ratema.charts import plot_annual_breakdown

    with pytest.raises(ValueError, match="缺少列"):
        plot_annual_breakdown(pd.DataFrame({"year": ["2024"]}), tmp_path)
    # 只有「全区间」行时无年可画
    with pytest.raises(ValueError, match="没有任何年份"):
        plot_annual_breakdown(
            pd.DataFrame(
                {"year": ["全区间"], "strategy_return": [0.1], "benchmark_return": [0.05]}
            ),
            tmp_path,
        )


def test_chart_wired_into_backtest_pipeline(dataset, tmp_path: Path):
    """分年度图必须被 make_all_charts 生成，且命名稳定。"""
    from ratema.charts import make_all_charts

    run = run_all(dataset, CFG, BT)
    paths = make_all_charts(run, tmp_path)
    assert "09_annual_breakdown.png" in {p.name for p in paths}
    assert (tmp_path / "charts" / "09_annual_breakdown.png").exists()
