"""图表输出测试。

只校验「能生成、是有效 PNG、内容合理」，不做像素级比对。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ratema.backtest import BacktestConfig
from ratema.charts import (
    annual_returns,
    make_all_charts,
    plot_drawdowns,
    plot_equity_curves,
    plot_per_series,
    plot_relative_strength,
    plot_signal_mechanics,
    plot_tol_sweep,
    setup_style,
)
from ratema.indicators import SignalConfig
from ratema.pipeline import run_all, run_single

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_png(path: Path, min_bytes: int = 8_000) -> None:
    assert path.exists(), f"图表未生成：{path}"
    data = path.read_bytes()
    assert data.startswith(PNG_MAGIC), f"{path} 不是有效 PNG"
    assert len(data) > min_bytes, f"{path} 体积过小（{len(data)} 字节），可能是空图"


@pytest.fixture()
def run(dataset):
    return run_all(dataset, SignalConfig(20, 120, "abs", 0.001), BacktestConfig())


def test_setup_style_picks_a_cjk_font():
    fonts = setup_style()
    assert fonts, "没有选中任何字体"
    # 至少不应退化成完全无法渲染中文的空配置
    import matplotlib.pyplot as plt

    assert plt.rcParams["font.sans-serif"] == fonts
    assert plt.rcParams["axes.unicode_minus"] is False


def test_make_all_charts_generates_every_figure(run, tmp_path: Path):
    paths = make_all_charts(run, tmp_path)
    names = {p.name for p in paths}

    assert "01_equity_curves.png" in names
    assert "02_per_series.png" in names
    assert "03_relative_strength.png" in names
    assert "04_drawdowns.png" in names
    assert "05_annual_returns.png" in names
    assert any(n.startswith("06_signal_mechanics_") for n in names)

    chart_dir = tmp_path / "charts"
    for path in paths:
        _assert_png(path)
    assert chart_dir.is_dir()


def test_each_plot_function_is_individually_usable(run, tmp_path: Path):
    _assert_png(plot_equity_curves(run, tmp_path))
    _assert_png(plot_per_series(run, tmp_path))
    _assert_png(plot_relative_strength(run, tmp_path))
    _assert_png(plot_drawdowns(run, tmp_path))


def test_signal_mechanics_chart_for_explicit_series_and_window(run, tmp_path: Path):
    res = next(r for r in run.results if r.series == "DR001")
    path = plot_signal_mechanics(res, tmp_path, start="2023-01-01", end="2024-12-31")
    _assert_png(path)
    assert "DR001" in path.name


def test_signal_mechanics_defaults_to_best_sharpe_series(run, tmp_path: Path):
    paths = make_all_charts(run, tmp_path)
    mech = [p for p in paths if p.name.startswith("06_signal_mechanics_")]
    assert len(mech) == 1
    best = max(
        run.results,
        key=lambda r: r.metrics.get("strategy_sharpe", float("-inf")),
    )
    assert best.series in mech[0].name


def test_signal_mechanics_rejects_empty_window(run, tmp_path: Path):
    res = run.results[0]
    with pytest.raises(ValueError, match="没有数据"):
        plot_signal_mechanics(res, tmp_path, start="2099-01-01")


def test_tol_sweep_chart(run, tmp_path: Path, dataset):
    curves = {}
    for tol in (0.0, 0.001, 0.005):
        res = run_single(dataset, "DR001", SignalConfig(20, 120, "abs", tol), BacktestConfig())
        curves[f"tol={tol:g}"] = res.frame
    _assert_png(plot_tol_sweep(curves, tmp_path, title_extra="DR001"))


def test_annual_returns_shape(run):
    table = annual_returns(run)
    assert "基准" in table.index
    assert len(table) == len(run.rate_cols) + 1
    # 每年一个数据点，且不应出现超过 100% 的异常值
    assert table.shape[1] >= 1
    assert (table.abs().max().max()) < 1.0


def test_charts_handle_zero_tolerance_gracefully(dataset, tmp_path: Path):
    """tol=0 时重合带宽度为 0，不应导致绘图报错。"""
    run0 = run_all(dataset, SignalConfig(20, 120, "abs", 0.0), BacktestConfig())
    res = run0.results[0]
    path = plot_signal_mechanics(res, tmp_path)
    _assert_png(path)
