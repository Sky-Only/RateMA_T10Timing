"""``compare_directions.py`` 方向对比模块的测试。

重点不在「跑得通」，而在四件事：
1. 三条曲线确实共用同一信号与同一评估区间（否则图上的差异归因不成立）；
2. 指标表不会因为键名取错而静默出 NaN / 张冠李戴；
3. 等权组合必须随交易方向变化（否则会画出三条一模一样的「不同方向」曲线）；
4. 年度收益的首年口径正确（从评估起点起算，而非「年内首末」）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parent.parent / "compare_directions.py"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    """按路径加载仓库根目录的 compare_directions.py（它不是包的一部分）。"""
    spec = importlib.util.spec_from_file_location("_ratema_compare_directions", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cd():
    return _load_module()


@pytest.fixture()
def cfg(cd):
    return dict(cd.CONFIG)


@pytest.fixture()
def bundle(cd, cfg, dataset):
    """在合成数据集上跑一遍三个方向（400 个交易日，2 个利率指标）。"""
    return cd.compute_all_directions(dataset, cfg)


@pytest.fixture()
def panels(cd, bundle, cfg):
    return cd.build_panels(bundle, cfg)


@pytest.fixture()
def portfolio_panels(cd, bundle, cfg):
    return cd.build_panels(bundle, cfg, "portfolio")


# --------------------------------------------------------------------------- #
# 两套曲线：综合信号 vs 等权组合
# --------------------------------------------------------------------------- #
def test_two_families_available(cd, bundle, cfg):
    for family in ("signal", "portfolio"):
        p = cd.build_panels(bundle, cfg, family)
        assert len(p) == len(cd.DIRECTION_SPECS) + 1


def test_unknown_family_rejected(cd, bundle, cfg):
    with pytest.raises(ValueError, match="family"):
        cd.build_panels(bundle, cfg, "nope")


def test_portfolio_differs_by_direction(cd, bundle):
    """回归：等权组合必须随方向变化。

    曾经的实现三个方向共用同一份 long_short 的 per_indicator，
    导致 portfolio_frame 在三个方向下完全相同 —— 图会画出三条一模一样的
    「不同方向」曲线，看起来完全正常。这条用例专门盯住它。
    """
    eqs = {
        d: r.portfolio_frame["equity"].reset_index(drop=True)
        for d, r in bundle["by_direction"].items()
    }
    for a in eqs:
        for b in eqs:
            if a < b:
                assert not eqs[a].equals(eqs[b]), f"{a} 与 {b} 的等权组合完全相同：方向没有生效"


def test_portfolio_position_signs_reflect_direction(cd, bundle):
    """等权组合的仓位区间必须体现方向 —— 与指标个数无关的严格不变量。

    刻意不用 time_in_market 来测：它是 `(平均仓位 != 0)`，而平均仓位在
    指标个数为偶数时可能恰好为 0（平票），所以那个量依赖数据集。
    这里改用符号区间：仅做多组合恒 ≥ 0，仅做空恒 ≤ 0，双向两者都有。
    """
    pos = {d: r.portfolio_frame["position"] for d, r in bundle["by_direction"].items()}
    assert (pos["long_only"] >= 0).all()
    assert (pos["short_only"] <= 0).all()
    assert (pos["long_only"] > 0).any()
    assert (pos["short_only"] < 0).any()


def test_portfolio_long_short_spans_both_sides(cd, bundle):
    """双向组合必须同时出现多头与空头暴露。"""
    p = bundle["by_direction"]["long_short"].portfolio_frame["position"]
    assert (p > 0).any(), "双向组合从未出现多头暴露"
    assert (p < 0).any(), "双向组合从未出现空头暴露"
    assert p.abs().max() <= 1.0 + 1e-12


def test_odd_indicator_count_never_ties(cd, bundle):
    """指标个数为奇数时双向组合不可能平票，持仓占比应为 1（真实数据即如此）。"""
    n = len(bundle["base"].rate_cols)
    pf = bundle["by_direction"]["long_short"].portfolio_frame
    if n % 2 == 1:
        assert (pf["position"] != 0).all()


def test_signal_eff_is_direction_free(cd, bundle):
    """底层 signal_eff 三方向必须一致，否则「差异只来自方向」的归因不成立。"""
    sigs = {
        d: r.signal_frame["signal_eff"].reset_index(drop=True)
        for d, r in bundle["by_direction"].items()
    }
    ref = sigs["long_short"]
    for direction, s in sigs.items():
        assert s.equals(ref), f"{direction} 的信号与 long_short 不同"


def test_n1_portfolio_equals_signal_curve(cd, bundle):
    """N=1 时「等权组合」就是该指标自己的净值，必须与「综合信号」逐点相同。

    这正是单利率只出一套图（不重复出 portfolio）的依据。
    """
    for item in bundle["per_indicator"]:
        for direction, res in item["bundle"]["by_direction"].items():
            sig = res.composite_frame["equity"].reset_index(drop=True)
            pf = res.portfolio_frame["equity"].reset_index(drop=True)
            assert sig.equals(pf), f"{item['series']}/{direction}: N=1 两套曲线不一致"


def test_signals_direction_free_guard(cd, bundle):
    """人为破坏信号的方向无关性，守卫必须报错。"""
    import copy

    runs = bundle["per_direction_runs"]
    broken = {d: copy.copy(b) for d, b in runs.items()}
    tampered = copy.copy(broken["short_only"])
    results = list(tampered.results)
    victim = copy.copy(results[0])
    sig = victim.signals.copy()
    sig.loc[sig.index[:5], "signal_eff"] = 999.0
    victim.signals = sig
    results[0] = victim
    tampered.results = results
    broken["short_only"] = tampered
    with pytest.raises(ValueError, match=r"与交易方向无关|信号"):
        cd._check_signals_direction_free(broken)


def test_chart_filenames_differ_between_families(cd):
    assert cd.FAMILY_FILE["signal"] != cd.FAMILY_FILE["portfolio"]


def test_portfolio_panels_use_portfolio_metrics(cd, bundle, cfg):
    """面板必须取 portfolio_metrics，而不是错拿 composite_metrics。"""
    ps = cd.build_panels(bundle, cfg, "portfolio")
    ls = next(p for p in ps if p["direction"] == "long_short")
    assert ls["metrics"] is bundle["by_direction"]["long_short"].portfolio_metrics


def test_benchmark_identical_across_families(portfolio_panels, panels):
    """基准与曲线口径无关，两套图的基准必须一致。"""
    a = next(p for p in panels if p["kind"] == "benchmark")
    b = next(p for p in portfolio_panels if p["kind"] == "benchmark")
    assert list(a["equity"]) == list(b["equity"])


@pytest.mark.parametrize("family", ["signal", "portfolio"])
def test_metrics_table_ok_for_both_families(cd, bundle, cfg, family):
    table = cd.build_metrics_table(cd.build_panels(bundle, cfg, family))
    for column in cd.REQUIRED_COLUMNS:
        assert not table[column].isna().any(), f"{family}/{column} 出现 NaN"


# --------------------------------------------------------------------------- #
# 年度收益热力图（仿 charts/05_annual_returns.png）
# --------------------------------------------------------------------------- #
def _two_year_equity():
    """2020 年末 1.10、2021 年末 1.21，起点 1.00。"""
    import numpy as np

    idx = pd.bdate_range("2020-01-02", "2021-12-31")
    mask = idx.year == 2020
    vals = np.empty(len(idx))
    vals[mask] = np.linspace(1.00, 1.10, int(mask.sum()))
    vals[~mask] = np.linspace(1.10, 1.21, int((~mask).sum()))
    return pd.Series(vals, index=idx)


def test_annual_from_equity_basic(cd):
    out = cd.annual_from_equity(_two_year_equity())
    assert list(out.index) == [2020, 2021]
    assert out[2020] == pytest.approx(0.10, rel=1e-9)
    assert out[2021] == pytest.approx(0.10, rel=1e-9)


def test_annual_from_equity_counts_first_day_move(cd):
    """回归：首年必须从「评估起点」起算，而不是「年内第一天」。

    用「年内首末净值」相除会把年初第一天的涨跌整段漏掉。
    本用例专门区分两种口径：正确 = +2%，错误 = 0%。
    """
    idx = pd.bdate_range("2020-01-02", "2020-12-31")
    eq = pd.Series(1.02, index=idx)
    eq.iloc[0] = 1.00  # 第一天涨 2%，之后一直横盘
    out = cd.annual_from_equity(eq)
    assert out[2020] == pytest.approx(0.02, rel=1e-9)


def test_annual_from_equity_partial_first_year(cd):
    """评估起点在年中时，首年收益从起点算，而不是按整年。"""
    idx = pd.bdate_range("2020-07-01", "2020-12-31")
    eq = pd.Series(1.05, index=idx)
    eq.iloc[0] = 1.00
    out = cd.annual_from_equity(eq)
    assert list(out.index) == [2020]
    assert out[2020] == pytest.approx(0.05, rel=1e-9)


def test_annual_from_equity_empty(cd):
    assert cd.annual_from_equity(pd.Series(dtype="float64")).empty


def test_annual_from_equity_counts_year_boundary_gap(cd):
    """跨年跳空必须计入**新年**的收益。

    这是区分两种口径的关键用例：
      正确（年末环比）：2021 = 1.20 / 1.10 − 1 = +9.09%
      错误（年内首末）：2021 = 1.20 / 1.20 − 1 = 0%（把跨年那一跳丢了）
    """
    import numpy as np

    idx20 = pd.bdate_range("2020-01-02", "2020-12-31")
    idx21 = pd.bdate_range("2021-01-04", "2021-12-31")
    eq = pd.concat(
        [
            pd.Series(np.linspace(1.00, 1.10, len(idx20)), index=idx20),
            pd.Series(1.20, index=idx21),  # 新年首个交易日直接跳到 1.20 后横盘
        ]
    )
    out = cd.annual_from_equity(eq)
    assert out[2020] == pytest.approx(0.10, rel=1e-9)
    assert out[2021] == pytest.approx(1.20 / 1.10 - 1.0, rel=1e-9)


def test_annual_from_equity_matches_charts_reference(cd, dataset):
    """与项目自带实现 ratema.charts.annual_returns 交叉验证（独立代码路径）。

    compare_directions 是仿 05_annual_returns.png 画的，
    两者对「基准」的逐年收益必须完全一致，否则仿出来的图与 05 口径不符。
    """
    from ratema.backtest import BacktestConfig
    from ratema.charts import annual_returns as charts_annual
    from ratema.indicators import SignalConfig
    from ratema.pipeline import run_all

    run = run_all(dataset, SignalConfig(20, 120, "abs", 0.001), BacktestConfig())
    ref = charts_annual(run)  # 行=指标+基准，列=年份
    ref_bench = ref.loc["基准"]

    frame = run.results[0].frame
    mine = cd.annual_from_equity(
        pd.Series(frame["benchmark_equity"].to_numpy(), index=pd.to_datetime(frame["date"]))
    )
    assert list(mine.index) == list(ref_bench.index)
    for year in ref_bench.index:
        assert mine[year] == pytest.approx(ref_bench[year], rel=1e-9, abs=1e-12)


def test_annual_matrix_shape(cd, panels):
    df = cd.annual_matrix(panels)
    assert len(df) == len(panels)
    assert list(df.index) == [p["label"] for p in panels]
    assert list(df.columns) == sorted(df.columns)
    assert df.notna().all().all()


def test_annual_matrix_benchmark_matches_own_curve(cd, panels):
    """基准那一行必须等于基准净值自己的逐年收益。"""
    df = cd.annual_matrix(panels)
    bench = next(p for p in panels if p["kind"] == "benchmark")
    direct = cd.annual_from_equity(
        pd.Series(bench["equity"].to_numpy(), index=pd.to_datetime(bench["date"]))
    )
    row = df.loc[bench["label"]]
    for year in direct.index:
        assert row[year] == pytest.approx(direct[year], rel=1e-12)


def test_indicator_annual_matrix_rows(cd, bundle, cfg):
    df = cd.indicator_annual_matrix(bundle, cfg, "long_short")
    labels = list(df.index)
    assert labels[0] == cd.COMPOSITE_ROW_LABEL
    assert cd.BENCH_LABEL in labels
    for s in bundle["base"].rate_cols:
        assert s in labels
    # 2 个综合口径 + 各利率 + 基准
    assert len(df) == len(bundle["base"].rate_cols) + 3


def test_indicator_matrix_composite_row_matches_direction_matrix(cd, bundle, cfg):
    """同一方向下，两张热力图里的综合行必须是同一组数字。"""
    dm = cd.annual_matrix(cd.build_panels(bundle, cfg, "signal"))
    im = cd.indicator_annual_matrix(bundle, cfg, "long_short")
    ls_label = cd.DIRECTION_SPECS[0][1]
    for year in dm.columns:
        assert im.loc[cd.COMPOSITE_ROW_LABEL, year] == pytest.approx(
            dm.loc[ls_label, year], rel=1e-12
        )


def test_indicator_matrix_varies_by_direction(cd, bundle, cfg):
    a = cd.indicator_annual_matrix(bundle, cfg, "long_only")
    b = cd.indicator_annual_matrix(bundle, cfg, "short_only")
    assert not a.loc[cd.COMPOSITE_ROW_LABEL].equals(b.loc[cd.COMPOSITE_ROW_LABEL])


# --------------------------------------------------------------------------- #
# 两个综合口径必须在场，且不能张冠李戴
# --------------------------------------------------------------------------- #
def test_two_composite_labels_are_distinct(cd):
    """两个「综合」标签必须不同 —— 曾用「等权综合」这种含糊说法导致误读。"""
    assert cd.COMPOSITE_ROW_LABEL != cd.PORTFOLIO_ROW_LABEL
    assert "等权综合" not in (cd.COMPOSITE_ROW_LABEL, cd.PORTFOLIO_ROW_LABEL)


def test_indicator_matrix_has_both_composites(cd, bundle, cfg):
    df = cd.indicator_annual_matrix(bundle, cfg, "long_short")
    labels = list(df.index)
    assert cd.COMPOSITE_ROW_LABEL in labels, "缺少打分综合"
    assert cd.PORTFOLIO_ROW_LABEL in labels, "缺少等权组合"
    # 行数 = 2 个综合 + 各利率 + 基准
    assert len(df) == len(bundle["base"].rate_cols) + 3


def test_composite_row_is_signal_not_portfolio(cd, bundle, cfg):
    """标签为「打分综合」的那一行必须真的是综合信号，不能被写成等权组合。"""
    df = cd.indicator_annual_matrix(bundle, cfg, "long_short")
    signal = cd.annual_matrix(cd.build_panels(bundle, cfg, "signal")).loc["多空双向"]
    got = df.loc[cd.COMPOSITE_ROW_LABEL]
    for year in signal.index:
        assert got[year] == pytest.approx(signal[year], rel=1e-12)


def test_portfolio_row_is_equal_weight(cd, bundle, cfg):
    """标签为「等权组合」的那一行必须真的是等权组合。"""
    df = cd.indicator_annual_matrix(bundle, cfg, "long_short")
    port = cd.annual_matrix(cd.build_panels(bundle, cfg, "portfolio")).loc["多空双向"]
    got = df.loc[cd.PORTFOLIO_ROW_LABEL]
    for year in port.index:
        assert got[year] == pytest.approx(port[year], rel=1e-12)


def test_two_composite_rows_differ(cd, bundle, cfg):
    """打分综合与等权组合是两条不同的曲线，数值不应相同。"""
    df = cd.indicator_annual_matrix(bundle, cfg, "long_short")
    assert not df.loc[cd.COMPOSITE_ROW_LABEL].equals(df.loc[cd.PORTFOLIO_ROW_LABEL])


def test_all_metrics_table_has_both_composites(cd, bundle, cfg):
    """跨口径汇总表也必须两个都在，且每个 4 条曲线。"""
    table = cd.build_all_metrics_table(bundle, cfg)
    labels = table["指标"].unique().tolist()
    assert cd.COMPOSITE_ROW_LABEL in labels
    assert cd.PORTFOLIO_ROW_LABEL in labels
    for label in (cd.COMPOSITE_ROW_LABEL, cd.PORTFOLIO_ROW_LABEL):
        sub = table[table["指标"] == label]
        assert len(sub) == len(cd.DIRECTION_SPECS) + 1


def test_rate_matrix_documents_both_concepts(cd, bundle, cfg):
    md = cd.render_rate_matrix(cd.build_all_metrics_table(bundle, cfg), cfg)
    assert cd.COMPOSITE_ROW_LABEL in md
    assert cd.PORTFOLIO_ROW_LABEL in md
    assert cd.COMPOSITE_ROW_NOTE in md
    assert cd.PORTFOLIO_ROW_NOTE in md


def test_annual_heatmap_renders(cd, panels, tmp_path):
    df = cd.annual_matrix(panels)
    path = cd.plot_annual_heatmap(df, tmp_path, title="测试年度收益", filename="heat.png")
    assert path.exists()
    assert path.read_bytes()[:8] == PNG_MAGIC
    assert path.stat().st_size > 8_000


def test_annual_heatmap_handles_all_nan(cd, tmp_path):
    """全 NaN 的矩阵不能崩（颜色范围要有兜底）。"""
    df = pd.DataFrame([[float("nan")]], index=["x"], columns=[2020])
    path = cd.plot_annual_heatmap(df, tmp_path, title="空", filename="empty.png")
    assert path.read_bytes()[:8] == PNG_MAGIC


def test_annual_heatmap_values_are_decimals(cd, panels):
    """热力图存的是小数（渲染时 ×100），与 05 的标注口径一致。"""
    data = cd.annual_matrix(panels).to_numpy(dtype="float64")
    assert data.max() < 1.0
    assert data.min() > -1.0


# --------------------------------------------------------------------------- #
# 年度收益 + 年内最大回撤 数据导出
# --------------------------------------------------------------------------- #
def test_annual_drawdown_basic(cd):
    """先涨到 1.2 再跌到 0.9：年内回撤应为 0.9/1.2 - 1 = -25%。"""
    idx = pd.bdate_range("2020-01-02", "2020-12-31")
    vals = np.linspace(1.0, 1.2, len(idx) // 2)
    vals = np.concatenate([vals, np.linspace(1.2, 0.9, len(idx) - len(vals))])
    eq = pd.Series(vals, index=idx)
    out = cd.annual_drawdown_from_equity(eq)
    assert out[2020] == pytest.approx(0.9 / 1.2 - 1.0, rel=1e-9)


def test_annual_drawdown_counts_year_start_move(cd):
    """默认把「上年末 -> 年初首个交易日」的下跌计入本年。"""
    idx20 = pd.bdate_range("2020-01-02", "2020-12-31")
    idx21 = pd.bdate_range("2021-01-04", "2021-12-31")
    eq = pd.concat(
        [
            pd.Series(1.00, index=idx20),
            pd.Series(0.90, index=idx21),  # 2021 首个交易日即跌 10%，其后横盘
        ]
    )
    default = cd.annual_drawdown_from_equity(eq)
    strict = cd.annual_drawdown_from_equity(eq, include_year_start=False)
    assert default[2021] == pytest.approx(-0.10, rel=1e-9)  # 计入
    assert strict[2021] == pytest.approx(0.0, rel=1e-9)  # 不计入


def test_annual_drawdown_strict_matches_annual_breakdown(cd, dataset):
    """严格口径必须与 metrics.annual_breakdown 的年内回撤逐格一致。"""
    from ratema.backtest import BacktestConfig
    from ratema.indicators import SignalConfig
    from ratema.metrics import annual_breakdown
    from ratema.pipeline import run_single

    cfg = SignalConfig(20, 120, "abs", 0.001)
    res = run_single(dataset, "DR001", cfg, BacktestConfig(), index_col="CBA04502.CS")
    eq = pd.Series(res.frame["equity"].to_numpy(), index=pd.to_datetime(res.frame["date"]))
    mine = cd.annual_drawdown_from_equity(eq, include_year_start=False)
    ref = annual_breakdown(res.frame, res.backtest.trades_frame)
    ref = ref[ref["year"] != "全区间"]
    for _, row in ref.iterrows():
        year = int(row["year"])
        if year in mine.index:
            assert mine[year] == pytest.approx(
                float(row["strategy_max_drawdown"]), rel=1e-12, abs=1e-12
            )


def test_annual_drawdown_never_positive(cd, bundle, cfg):
    """回撤只能是 0 或负数。"""
    for direction, *_ in cd.DIRECTION_SPECS:
        dd = cd.indicator_annual_drawdown_matrix(bundle, cfg, direction)
        assert (dd.to_numpy(dtype="float64") <= 1e-12).all()


def test_return_and_drawdown_tables_share_shape(cd, bundle, cfg):
    """两张表必须同形状同行列，否则没法逐格对照。"""
    tables = cd.annual_return_drawdown_tables(bundle, cfg)
    for direction, per in tables.items():
        ret, dd = per["年度收益"], per["年内最大回撤"]
        assert list(ret.index) == list(dd.index), f"{direction} 行不一致"
        assert list(ret.columns) == list(dd.columns), f"{direction} 列不一致"
        assert ret.notna().all().all()
        assert dd.notna().all().all()


def test_annual_return_dd_covers_three_directions(cd, bundle, cfg):
    tables = cd.annual_return_drawdown_tables(bundle, cfg)
    assert set(tables) == {d for d, *_ in cd.DIRECTION_SPECS}


def test_annual_return_dd_tidy(cd, bundle, cfg):
    tidy = cd.annual_return_drawdown_tidy(cd.annual_return_drawdown_tables(bundle, cfg))
    assert list(tidy.columns) == ["方向", "direction", "口径", "年份", "年度收益", "年内最大回撤"]
    n_years = tidy["年份"].nunique()
    n_rows = tidy["口径"].nunique()
    assert len(tidy) == 3 * n_rows * n_years
    assert tidy["年内最大回撤"].max() <= 1e-12
    assert tidy["年份"].between(2000, 2100).all()


def test_annual_return_dd_md_mentions_both(cd, bundle, cfg):
    md = cd.render_annual_return_drawdown_md(cd.annual_return_drawdown_tables(bundle, cfg), cfg)
    for direction, label, *_ in cd.DIRECTION_SPECS:
        assert label in md
        assert direction in md
    assert md.count("### 年度收益") == 3
    assert md.count("### 年内最大回撤") == 3
    # 口径差异必须写清楚，否则用户会拿它跟 annual_breakdown.csv 对不上而困惑
    assert "annual_breakdown" in md


def test_equity_map_rows_match_annual_matrix(cd, bundle, cfg):
    """indicator_equity_map 的行必须与年度表完全一致（共享底座）。"""
    eqs = cd.indicator_equity_map(bundle, cfg, "long_short")
    mat = cd.indicator_annual_matrix(bundle, cfg, "long_short")
    assert list(eqs) == list(mat.index)
    assert cd.COMPOSITE_ROW_LABEL in eqs
    assert cd.PORTFOLIO_ROW_LABEL in eqs
    assert cd.BENCH_LABEL in eqs


# --------------------------------------------------------------------------- #
# 年度最大回撤热力图（三张，排版与收益图一致）
# --------------------------------------------------------------------------- #
def test_heatmap_color_range_symmetric(cd):
    """收益图：以 0 为中心对称，绿正红负。"""
    assert cd.heatmap_color_range(0.086, symmetric=True) == (-0.086, 0.086)


def test_heatmap_color_range_drawdown(cd):
    """回撤图：[-最深, 0]，配合白→红色带即 0=白、最深=红。"""
    lo, hi = cd.heatmap_color_range(0.086, symmetric=False)
    assert lo == -0.086
    assert hi == 0.0, "回撤图的上界必须是 0（纯白端），否则最深的值染不到纯红"


def test_drawdown_cmap_is_white_to_red(cd):
    """回撤色带必须「0 = 白、最深 = 红」—— 按**实际取值→颜色**的映射来验。

    只看 cmap(0)/cmap(1) 是不够的：cmap 的 t 还要经过 vmin/vmax 归一化，
    真正的对应关系是 t = (v - vmin) / (vmax - vmin)。
    曾经把色带列表写成 ["#FFFFFF","#A50026"]，于是最深回撤(t=0)染成白、
    0 回撤(t=1)染成深红 —— 整张图反了。这条用例专门盯住这个。
    """
    limit = 0.0856
    vmin, vmax = cd.heatmap_color_range(limit, symmetric=False)

    def color(v: float):
        t = (v - vmin) / (vmax - vmin)
        return np.asarray(cd.DD_CMAP(t))[:3]

    zero = color(0.0)
    deep = color(-limit)
    mid = color(-limit / 2)

    assert zero == pytest.approx((1.0, 1.0, 1.0)), "0（无回撤）必须是纯白"
    assert deep[0] > 0.5, "最深回撤必须是红（红通道高）"
    assert deep[1] < 0.3 and deep[2] < 0.3, "最深端不能偏粉/偏白"
    # 越深越红：绿通道单调下降
    assert zero[1] > mid[1] > deep[1], "从 0 到最深必须越来越红"


def test_return_heatmap_still_uses_diverging_cmap(cd):
    """收益图仍用 RdYlGn 对称色带 —— 改配置不应波及收益图。"""
    assert cd.ANNUAL_CMAP == "RdYlGn"
    assert cd.DD_CMAP is not cd.ANNUAL_CMAP


def test_drawdown_color_limit_is_worst_across_directions(cd, bundle, cfg):
    """三张图必须共用同一个颜色下限，取三方向里的最深回撤。"""
    tables = cd.annual_return_drawdown_tables(bundle, cfg)
    limit = cd.drawdown_color_limit(tables)
    worst = min(
        float(per["年内最大回撤"].to_numpy(dtype="float64").min()) for per in tables.values()
    )
    assert limit == pytest.approx(abs(worst), rel=1e-12)
    for per in tables.values():
        local = float(per["年内最大回撤"].to_numpy(dtype="float64").min())
        assert limit >= abs(local) - 1e-12


def test_drawdown_color_limit_handles_all_nan(cd):
    empty = {"x": {"年内最大回撤": pd.DataFrame([[float("nan")]], index=["a"], columns=[2020])}}
    assert cd.drawdown_color_limit(empty) > 0


def test_plot_annual_drawdown_heatmaps_three_files(cd, bundle, cfg, tmp_path):
    tables = cd.annual_return_drawdown_tables(bundle, cfg)
    paths = cd.plot_annual_drawdown_heatmaps(tables, tmp_path)
    assert len(paths) == 3
    names = [p.name for p in paths]
    assert len(set(names)) == 3, "三张图文件名不能重复"
    for direction in ("long_short", "long_only", "short_only"):
        assert f"compare_annual_dd_{direction}.png" in names
    for p in paths:
        assert p.exists()
        assert p.read_bytes()[:8] == PNG_MAGIC
        assert p.stat().st_size > 8_000


def test_drawdown_heatmap_rows_match_return_heatmap(cd, bundle, cfg):
    """回撤图与收益图必须同形（同行同列），否则没法并排对照。"""
    for direction, *_ in cd.DIRECTION_SPECS:
        ret = cd.indicator_annual_matrix(bundle, cfg, direction)
        dd = cd.indicator_annual_drawdown_matrix(bundle, cfg, direction)
        assert list(ret.index) == list(dd.index)
        assert list(ret.columns) == list(dd.columns)


# --------------------------------------------------------------------------- #
# 单利率：N=1 的 run_composite 必须等价于「该指标自己的信号」
# --------------------------------------------------------------------------- #
def test_one_bundle_per_indicator(bundle):
    """每个利率都要有自己的一份 bundle，数量与指标数一致。"""
    items = bundle["per_indicator"]
    assert [i["series"] for i in items] == bundle["base"].rate_cols
    assert len(items) == bundle["n_indicators"] == len(bundle["base"].rate_cols)
    for item in items:
        assert item["display"]  # 中文名非空
        assert set(item["bundle"]["by_direction"]) == {
            "long_short",
            "long_only",
            "short_only",
        }
        assert item["bundle"]["is_composite"] is False


def test_composite_bundle_is_flagged(bundle):
    assert bundle["is_composite"] is True


@pytest.mark.parametrize("direction", ["long_only", "short_only", "long_short"])
def test_single_element_equals_run_single(cd, cfg, dataset, bundle, direction):
    """核心等价性：单元素综合 == 该指标单独回测（逐日区间 + 全部绩效）。

    这条是五张单利率图正确性的根基 —— 若不等，图上画的就不是该利率的信号。
    """
    from ratema.backtest import BacktestConfig
    from ratema.indicators import SignalConfig
    from ratema.pipeline import run_single

    sig_cfg = SignalConfig(
        short_window=cfg["short_window"],
        long_window=cfg["long_window"],
        tol_mode=cfg["tol_mode"],
        tol=cfg["tol"],
        std_window=cfg["std_window"],
    )
    for item in bundle["per_indicator"]:
        series = item["series"]
        bt = BacktestConfig(
            cost_bps=cfg["cost_bps"],
            cost_mode=cfg["cost_mode"],
            direction=direction,
            initial_capital=cfg["initial_capital"],
        )
        ref = run_single(dataset, series, sig_cfg, bt, index_col=bundle["base"].index_col)
        got = item["bundle"]["by_direction"][direction]

        ref_dates = ref.frame["date"].reset_index(drop=True)
        got_dates = got.composite_frame["date"].reset_index(drop=True)
        assert ref_dates.equals(got_dates), f"{series}/{direction} 评估区间不一致"

        for key in (
            "strategy_cagr",
            "strategy_sharpe",
            "strategy_max_drawdown",
            "strategy_end_value",
            "n_completed",
            "time_in_market",
        ):
            a = ref.metrics.get(key)
            b = got.composite_metrics.get(key)
            if a is None or b is None:
                continue
            assert float(a) == pytest.approx(float(b), rel=1e-12, abs=1e-12), (
                f"{series}/{direction}/{key}: run_single={a} vs 单元素={b}"
            )


def test_single_indicator_signal_is_its_own(bundle):
    """N=1 时 score 只能取 ±1，且与综合用的 signal_eff 同源。"""
    for item in bundle["per_indicator"]:
        for res in item["bundle"]["by_direction"].values():
            score = res.composite_frame["score"].dropna()
            assert set(score.unique()) <= {1.0, -1.0}


def test_each_indicator_bundle_is_internally_aligned(bundle):
    """每张单利率图内部也必须三条曲线对齐（compute_* 里已校验，这里再确认）。"""
    for item in bundle["per_indicator"]:
        frames = {d: r.composite_frame for d, r in item["bundle"]["by_direction"].items()}
        ref = frames["long_short"]["date"].reset_index(drop=True)
        for direction, f in frames.items():
            assert f["date"].reset_index(drop=True).equals(ref), (
                f"{item['series']}/{direction} 未对齐"
            )


def test_indicator_panels_and_charts(cd, cfg, bundle, tmp_path):
    """每个利率都能出图，文件名带指标代码，且互不覆盖。"""
    names = []
    for item in bundle["per_indicator"]:
        sub = cd.build_panels(item["bundle"], cfg)
        assert len(sub) == len(cd.DIRECTION_SPECS) + 1
        label = cd.indicator_chart_name(item["series"])
        assert item["series"] in label
        names.append(label)
        path = cd.plot_equity_comparison(
            sub,
            tmp_path,
            cfg,
            signal_label=item["bundle"]["signal_label"],
            title=f"{item['series']} 对比",
            filename=label,
        )
        assert path.exists() and path.read_bytes()[:8] == PNG_MAGIC
    assert len(set(names)) == len(names), "单利率图文件名重复会互相覆盖"


def test_indicator_drawdown_name_differs(cd):
    assert cd.indicator_chart_name("DR007") != cd.indicator_chart_name("DR007", drawdown=True)


def test_select_indicators_respects_scope(cd, cfg, bundle):
    assert cd.select_indicators(bundle, {**cfg, "scope": "composite"}) == []
    everything = cd.select_indicators(bundle, {**cfg, "scope": "both"})
    assert len(everything) == bundle["n_indicators"]
    # 合成数据集只有 DR001 / R001，别写死真实数据里的指标名
    first = bundle["base"].rate_cols[0]
    subset = cd.select_indicators(bundle, {**cfg, "scope": "both", "per_indicator_names": [first]})
    assert [i["series"] for i in subset] == [first]


def test_select_indicators_rejects_unknown(cd, cfg, bundle):
    with pytest.raises(SystemExit):
        cd.select_indicators(bundle, {**cfg, "scope": "both", "per_indicator_names": ["NOPE"]})


def test_all_metrics_table_covers_every_rate(cd, cfg, bundle):
    """跨口径长表要包含两个综合 + 每个利率，且每行 4 条曲线。"""
    table = cd.build_all_metrics_table(bundle, cfg)
    labels = table["指标"].unique().tolist()
    assert labels[0] == cd.COMPOSITE_ROW_LABEL
    assert set(labels) == {
        cd.COMPOSITE_ROW_LABEL,
        cd.PORTFOLIO_ROW_LABEL,
        *bundle["base"].rate_cols,
    }
    counts = table.groupby("指标").size().unique().tolist()
    assert counts == [len(cd.DIRECTION_SPECS) + 1]


def test_rate_matrix_renders(cd, cfg, bundle):
    md = cd.render_rate_matrix(cd.build_all_metrics_table(bundle, cfg), cfg)
    for series in bundle["base"].rate_cols:
        assert series in md
    assert cd.COMPOSITE_ROW_LABEL in md
    for _d, label, *_ in cd.DIRECTION_SPECS:
        assert label in md
    for note in ("年化收益率", "夏普比率", "最大回撤", "往返次数"):
        assert note in md


# --------------------------------------------------------------------------- #
# 对齐：三条曲线必须同信号同区间
# --------------------------------------------------------------------------- #
def test_three_directions_present(bundle, cd):
    keys = set(bundle["by_direction"])
    assert keys == {d for d, *_ in cd.DIRECTION_SPECS}
    assert keys == {"long_short", "long_only", "short_only"}


def test_directions_share_identical_dates(bundle):
    frames = {d: r.composite_frame for d, r in bundle["by_direction"].items()}
    dates = {d: f["date"].reset_index(drop=True) for d, f in frames.items()}
    ref = dates["long_short"]
    for direction, series in dates.items():
        assert series.equals(ref), f"{direction} 的交易日与 long_short 不一致"


def test_directions_share_identical_benchmark(bundle):
    """基准与方向无关，四条线必须站在同一个基准上。"""
    frames = {d: r.composite_frame for d, r in bundle["by_direction"].items()}
    ref = frames["long_short"]["benchmark_equity"].reset_index(drop=True)
    for direction, f in frames.items():
        got = f["benchmark_equity"].reset_index(drop=True)
        assert got.equals(ref), f"{direction} 的基准净值与 long_short 不一致"


def test_directions_share_identical_signal(bundle):
    """同一份打分信号喂给三个方向，signal_eff 必须逐日相同。"""
    sigs = {
        d: r.signal_frame["signal_eff"].reset_index(drop=True)
        for d, r in bundle["by_direction"].items()
    }
    ref = sigs["long_short"]
    for direction, series in sigs.items():
        assert series.equals(ref), f"{direction} 的信号与 long_short 不一致"


def test_alignment_guard_rejects_mismatch(cd, bundle):
    """把某个方向的区间人为改掉，对齐校验必须报错而不是照画。"""
    import copy

    broken = copy.copy(bundle["by_direction"])
    res = broken["short_only"]
    trimmed = copy.copy(res)
    trimmed.composite_frame = res.composite_frame.iloc[:-5]
    broken["short_only"] = trimmed
    with pytest.raises(ValueError, match=r"评估区间|交易日|基准"):
        cd._check_alignment(broken)


# --------------------------------------------------------------------------- #
# 面板与指标表
# --------------------------------------------------------------------------- #
def test_panels_are_three_plus_benchmark(panels, cd):
    assert len(panels) == len(cd.DIRECTION_SPECS) + 1
    assert [p["kind"] for p in panels] == ["strategy"] * 3 + ["benchmark"]


def test_panels_normalised_to_one(panels):
    """净值必须归一化到 1.0 起，否则换 initial_capital 图就没法看。"""
    for p in panels:
        assert p["equity"].iloc[0] == pytest.approx(1.0, abs=0.01)


def test_normalisation_independent_of_capital(cd, cfg, dataset):
    """改 initial_capital 不应改变图上任何一条曲线的形状。"""
    base = cd.build_panels(cd.compute_all_directions(dataset, cfg), cfg)
    scaled = cd.build_panels(
        cd.compute_all_directions(dataset, {**cfg, "initial_capital": 1_000_000.0}),
        {**cfg, "initial_capital": 1_000_000.0},
    )
    for a, b in zip(base, scaled, strict=True):
        assert a["label"] == b["label"]
        assert list(a["equity"]) == pytest.approx(list(b["equity"]), rel=1e-12)


def test_metrics_table_has_no_nan(panels, cd):
    table = cd.build_metrics_table(panels)
    for column in cd.REQUIRED_COLUMNS:
        assert column in table.columns
        assert not table[column].isna().any(), f"{column} 出现 NaN"


def test_benchmark_cost_drag_is_blank(panels, cd):
    """基准没有「成本拖累」口径，不能把策略的值显示成基准的。"""
    table = cd.build_metrics_table(panels)
    bench = table.loc[table["方案"].str.startswith("基准"), "成本拖累(年化)"]
    assert len(bench) == 1
    assert bench.isna().all()


def test_pick_does_not_leak_strategy_keys_to_benchmark(cd):
    """直接把策略侧键名喂给 benchmark 口径，必须取不到而不是取到策略值。"""
    metrics = {"cost_drag_cagr": 0.5, "strategy_cost_drag_cagr": 0.7, "n_completed": 9}
    assert cd._pick(metrics, "cost_drag_cagr", "benchmark") != 0.5
    assert cd._pick(metrics, "cost_drag_cagr", "strategy") == 0.7
    # 策略侧的不带前缀键名应能取到
    assert cd._pick(metrics, "n_completed", "strategy") == 9
    assert cd._pick(metrics, "n_completed", "benchmark") == 0  # BENCHMARK_FIXED


def test_benchmark_time_in_market_is_one(cd):
    """买入持有按定义始终满仓。"""
    assert cd._pick({}, "time_in_market", "benchmark") == 1.0


def test_round_trips_add_up(panels):
    """多空双向的往返次数应等于仅做多 + 仅做空 —— 内部一致性的强校验。"""
    by_label = {p["label"]: p["metrics"] for p in panels}
    ls = by_label["多空双向"]["n_completed"]
    lo = by_label["仅做多"]["n_completed"]
    so = by_label["仅做空"]["n_completed"]
    assert ls == lo + so


def test_long_short_time_in_market_is_full(panels):
    """多空双向始终有仓位，持仓占比应接近 1（仅首日建仓前为空）。"""
    ls = next(p for p in panels if p["direction"] == "long_short")
    assert ls["metrics"]["time_in_market"] > 0.99


def test_two_sided_beats_each_side_on_return(panels):
    """本样本内：双向年化高于任一单边（仅作一致性回归，不构成投资建议）。"""
    fin = {p["direction"]: p["cagr"] for p in panels if p["direction"]}
    assert fin["long_short"] > fin["long_only"]
    assert fin["long_short"] > fin["short_only"]


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("tol_mode", "ppt"),
        ("composite_mode", "magic"),
        ("vote_source", "nope"),
        ("cost_mode", "nope"),
        ("tol", -1.0),
        ("cost_bps", -1.0),
        ("initial_capital", 0.0),
        ("short_window", 200),  # >= long_window
    ],
)
def test_validate_rejects_bad_config(cd, cfg, key, value):
    with pytest.raises(SystemExit):
        cd.validate({**cfg, key: value})


def test_validate_accepts_defaults(cd, cfg):
    cd.validate(cfg)


def test_validate_rejects_reversed_window(cd, cfg):
    with pytest.raises(SystemExit):
        cd.validate({**cfg, "start": "2024-01-01", "end": "2020-01-01"})


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def _plot_main(cd, panels, outdir, cfg, name="compare_directions.png"):
    return cd.plot_equity_comparison(
        panels,
        outdir,
        cfg,
        signal_label="测试信号",
        title="测试标题",
        filename=name,
    )


def test_charts_and_tables_are_written(cd, panels, cfg, tmp_path):
    outdir = tmp_path / "cmp"
    main = _plot_main(cd, panels, outdir, cfg)
    dd = cd.plot_drawdown_comparison(panels, outdir, cfg)
    for path in (main, dd):
        assert path.exists()
        assert path.read_bytes()[:8] == PNG_MAGIC
        assert path.stat().st_size > 8_000

    table = cd.build_metrics_table(panels)
    csv_path = outdir / "compare_metrics.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    assert csv_path.exists()

    md = cd.render_metrics_markdown(table, cfg, panels)
    assert "方向对比" in md
    for label in ("多空双向", "仅做多", "仅做空", "基准 买入持有"):
        assert label in md
    assert "成本：成本" not in md  # 曾出现「成本：成本 …」的重复措辞


def test_log_scale_chart_renders(cd, panels, cfg, tmp_path):
    path = _plot_main(cd, panels, tmp_path, {**cfg, "log_scale": True})
    assert path.read_bytes()[:8] == PNG_MAGIC


def test_render_markdown_mentions_config(cd, panels, cfg):
    md = cd.render_metrics_markdown(cd.build_metrics_table(panels), cfg, panels)
    assert f"MA{cfg['short_window']} / MA{cfg['long_window']}" in md
    assert cfg["composite_mode"] in md


# --------------------------------------------------------------------------- #
# 逐月收益热力图（同样的 8 行，列 = 1~12 月）
# --------------------------------------------------------------------------- #
def test_monthly_from_equity_basic(cd):
    """两个月末值受控，逐月收益应精确可算。"""
    import numpy as np

    idx = pd.bdate_range("2021-01-04", "2021-02-26")
    eq = pd.Series(1.0, index=idx)
    jan = idx.month == 1
    eq[jan] = np.linspace(1.00, 1.10, int(jan.sum()))
    eq[~jan] = 1.21
    out = cd.monthly_from_equity(eq)
    assert [str(p) for p in out.index] == ["2021-01", "2021-02"]
    assert out.iloc[0] == pytest.approx(0.10, rel=1e-9)
    assert out.iloc[1] == pytest.approx(0.10, rel=1e-9)


def test_monthly_counts_cross_month_gap(cd):
    """跨月跳空必须计入**新月份**（与年度口径同源，不能用「月内首末」）。"""
    idx1 = pd.bdate_range("2021-01-04", "2021-01-29")
    idx2 = pd.bdate_range("2021-02-01", "2021-02-26")
    eq = pd.concat(
        [
            pd.Series([1.00, *([1.10] * (len(idx1) - 1))], index=idx1),
            pd.Series(1.20, index=idx2),
        ]
    )
    out = cd.monthly_from_equity(eq)
    assert out.iloc[0] == pytest.approx(0.10, rel=1e-9)
    assert out.iloc[1] == pytest.approx(1.20 / 1.10 - 1.0, rel=1e-9)


def test_monthly_from_equity_empty(cd):
    assert cd.monthly_from_equity(pd.Series(dtype="float64")).empty


def test_monthly_matrix_same_rows_as_annual(cd, bundle, cfg):
    """逐月表必须与年度表**同样的 8 行**，且顺序一致。"""
    am = cd.indicator_annual_matrix(bundle, cfg, "long_short")
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    assert list(mm.index) == list(am.index)
    assert len(mm) == len(bundle["base"].rate_cols) + 3
    assert cd.COMPOSITE_ROW_LABEL in mm.index
    assert cd.PORTFOLIO_ROW_LABEL in mm.index
    assert cd.BENCH_LABEL in mm.index


def test_monthly_compounds_to_annual(cd, bundle, cfg):
    """核心一致性：年内逐月复利必须等于该年的年度收益。

    这是逐月表正确性的最强校验 —— 两者必须同源（月末环比 + 评估起点作基期）。
    """
    am = cd.indicator_annual_matrix(bundle, cfg, "long_short")
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    import numpy as np

    worst = 0.0
    for year in am.columns:
        frame = cd.monthly_year_frame(mm, int(year))
        compounded = (1.0 + frame).prod(axis=1, skipna=True) - 1.0
        for label in am.index:
            a, c = am.loc[label, year], compounded[label]
            if np.isfinite(a) and np.isfinite(c):
                worst = max(worst, abs(float(a) - float(c)))
    assert worst < 1e-12, f"逐月复利与年度收益不一致，最大偏差 {worst:.3e}"


def test_monthly_year_frame_has_twelve_columns(cd, bundle, cfg):
    """无论该年数据是否完整，列都固定是 1~12 月。"""
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    for year in cd.monthly_years(mm, cfg):
        frame = cd.monthly_year_frame(mm, year)
        assert list(frame.columns) == cd.MONTH_LABELS
        assert frame.shape[1] == 12
        assert frame.notna().any().any(), f"{year} 一行数据都没有"


def test_monthly_year_frame_blank_months_are_nan(cd, bundle, cfg):
    """没有数据的月份必须是空，不能补 0 —— 补 0 会被读成「当月持平」。"""
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    years = cd.monthly_years(mm, cfg)
    # 合成数据的评估起点在年中（MA120 预热后才起算），首年必然有空月
    frame = cd.monthly_year_frame(mm, years[0])
    assert frame.isna().any().any(), "首年应当有空月"
    blank_cols = [c for c in frame.columns if frame[c].isna().all()]
    assert blank_cols, "首年应当有整月无数据"
    filled = frame.dropna(axis=1, how="all")
    assert (filled.fillna(0) == 0).sum().sum() < filled.notna().sum().sum()


def test_monthly_years_spec(cd, bundle, cfg):
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    everything = cd.monthly_years(mm, {**cfg, "monthly_years": None})
    assert everything == sorted(everything)
    assert len(everything) >= 2
    assert cd.monthly_years(mm, {**cfg, "monthly_years": [everything[0]]}) == [everything[0]]
    lo, hi = everything[0], everything[1]
    assert cd.monthly_years(mm, {**cfg, "monthly_years": f"{lo}-{hi}"}) == [lo, hi]


def test_monthly_years_rejects_unknown(cd, bundle, cfg):
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    with pytest.raises(SystemExit):
        cd.monthly_years(mm, {**cfg, "monthly_years": [1800]})


def test_monthly_color_limit_is_shared(cd, bundle, cfg):
    """颜色范围必须全局共用，否则各年文件的深浅不可比。"""
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    limit = cd.monthly_color_limit(mm)
    for year in cd.monthly_years(mm, cfg):
        frame = cd.monthly_year_frame(mm, year)
        local = frame.to_numpy(dtype="float64")
        assert np.nanmax(np.abs(local)) <= limit + 1e-12


def test_monthly_heatmap_renders_custom_limit(cd, bundle, cfg, tmp_path):
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    frame = cd.monthly_year_frame(mm, 2015)
    path = cd.plot_annual_heatmap(
        frame,
        tmp_path,
        title="2015 逐月",
        filename="m2015.png",
        cbar_label="月度收益",
        limit=cd.monthly_color_limit(mm),
        fontsize=7,
    )
    assert path.exists()
    assert path.read_bytes()[:8] == PNG_MAGIC


def test_monthly_various_by_direction(cd, bundle, cfg):
    a = cd.monthly_matrix(bundle, cfg, "long_only")
    b = cd.monthly_matrix(bundle, cfg, "short_only")
    assert not a.loc[cd.COMPOSITE_ROW_LABEL].equals(b.loc[cd.COMPOSITE_ROW_LABEL])


# --------------------------------------------------------------------------- #
# 单文件版：行 = 年（两种排布）
# --------------------------------------------------------------------------- #
def test_monthly_year_by_month_shape(cd, bundle, cfg):
    """行=年、列=1~12月；年数应与矩阵里的年份数一致。"""
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    df = cd.monthly_year_by_month(mm, cd.COMPOSITE_ROW_LABEL)
    years = sorted({p.year for p in mm.columns})
    assert list(df.index) == years
    assert list(df.columns) == cd.MONTH_LABELS


def test_monthly_year_by_month_matches_matrix(cd, bundle, cfg):
    """展开后的每一格都要与逐月矩阵一一对应，不能错位。"""
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    for label in mm.index:
        df = cd.monthly_year_by_month(mm, label)
        for year in df.index:
            for month in range(1, 13):
                period = pd.Period(f"{year}-{month:02d}", freq="M")
                cell = df.loc[year, f"{month}月"]
                if period in mm.columns:
                    expected = mm.loc[label, period]
                    if np.isfinite(expected):
                        assert cell == pytest.approx(float(expected), rel=1e-12), (
                            f"{label} {year}-{month} 错位"
                        )
                    else:
                        assert not np.isfinite(cell)
                else:
                    assert not np.isfinite(cell)


def test_monthly_year_by_month_blank_is_nan(cd, bundle, cfg):
    """没有数据的月份留空，不能是 0。"""
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    df = cd.monthly_year_by_month(mm, cd.COMPOSITE_ROW_LABEL)
    assert df.isna().any().any()
    assert (df.dropna(how="all") != 0).any().any()


def test_pairwise_layouts_may_differ_rows(cd, bundle, cfg):
    """纵向版：行=年（16 行）；横向版把所有口径拼在一行里。

    这里只校验两种排布共用的底座（monthly_year_by_month）返回同样的年行。
    """
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    a = cd.monthly_year_by_month(mm, cd.COMPOSITE_ROW_LABEL)
    b = cd.monthly_year_by_month(mm, cd.PORTFOLIO_ROW_LABEL)
    assert list(a.index) == list(b.index), "两种口径的年份行必须对齐"


def test_plot_monthly_strategy_strip_renders(cd, bundle, cfg, tmp_path):
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    path = cd.plot_monthly_strategy_strip(mm, tmp_path, title="测试横条", filename="s.png")
    assert path.exists()
    assert path.read_bytes()[:8] == PNG_MAGIC
    assert path.stat().st_size > 20_000


@pytest.mark.parametrize("cell", [0.10, 0.15, 0.25])
def test_strip_cells_are_square(cd, cell):
    """横条图的每一格必须是**正方形**，且边长等于请求值。

    曾经只给个大概的 figsize，aspect=1 会把坐标区压缩、格子缩到 0.10"
    （请求 0.15"），而 bbox_inches="tight" 又把痕迹裁掉，肉眼看不出。
    """
    n_rows, n_cols = 8, 184
    (fig_w, fig_h), rect, _cax = cd.strip_geometry(n_rows, n_cols, cell)
    cell_w = rect[2] * fig_w / n_cols
    cell_h = rect[3] * fig_h / n_rows
    assert cell_w == pytest.approx(cell, rel=1e-12)
    assert cell_h == pytest.approx(cell, rel=1e-12)
    assert cell_w == pytest.approx(cell_h, rel=1e-12), "格子不是正方形"


def test_strip_data_area_aspect_matches_grid(cd):
    """数据区宽高比必须等于 列数:行数，否则格子会被拉成长方形。"""
    n_rows, n_cols = 8, 184
    (fig_w, fig_h), rect, _cax = cd.strip_geometry(n_rows, n_cols, 0.15)
    data_w = rect[2] * fig_w
    data_h = rect[3] * fig_h
    assert data_w / data_h == pytest.approx(n_cols / n_rows, rel=1e-12)


def test_strip_geometry_scales_with_cell(cd):
    """格子调大，数据区按比例等比放大（整图变长是预期的）。"""
    a = cd.strip_geometry(8, 184, 0.10)
    b = cd.strip_geometry(8, 184, 0.20)
    aw, ah = a[1][2] * a[0][0], a[1][3] * a[0][1]
    bw, bh = b[1][2] * b[0][0], b[1][3] * b[0][1]
    assert bw / aw == pytest.approx(2.0, rel=1e-12)
    assert bh / ah == pytest.approx(2.0, rel=1e-12)


@pytest.mark.parametrize("horizontal", [False, True])
def test_plot_monthly_year_panels_renders(cd, bundle, cfg, tmp_path, horizontal):
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    path = cd.plot_monthly_year_panels(
        mm, tmp_path, title="测试面板", filename=f"p{horizontal}.png", horizontal=horizontal
    )
    assert path.exists()
    assert path.read_bytes()[:8] == PNG_MAGIC
    assert path.stat().st_size > 20_000


# --------------------------------------------------------------------------- #
# 坐标轴方向：横轴必须是月份、纵轴必须是策略
# （曾经把两者弄反过，这里把方向钉死）
# --------------------------------------------------------------------------- #
def test_year_panel_data_is_strategy_by_month(cd, bundle, cfg):
    """面板数据必须是 行=策略 × 列=12 个月，不能转置。

    行数由数据决定（真实数据 5 利率 → 8 行；合成数据 2 利率 → 5 行）。
    """
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    labels = list(mm.index)
    frames = [cd.monthly_year_by_month(mm, lab) for lab in labels]
    year = sorted({p.year for p in mm.columns})[0]
    # 用 np.array 而不是 column_stack：后者会得到 (12, 策略数)，正好转置
    data = np.array([f.loc[year].to_numpy(dtype="float64") for f in frames])
    assert data.shape == (len(labels), 12), "应当是 行=策略、列=月份"
    assert len(labels) == len(bundle["base"].rate_cols) + 3


def test_strip_matrix_is_strategy_by_month(cd, bundle, cfg):
    """横条图直接吃逐月矩阵：行=策略、列=月份（多月）。"""
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    assert mm.shape[0] == len(bundle["base"].rate_cols) + 3
    assert mm.shape[1] > 12, "列应当是全部月份，而不是只有 12 个月"
    assert list(mm.index)[:2] == [cd.COMPOSITE_ROW_LABEL, cd.PORTFOLIO_ROW_LABEL]
    assert all(isinstance(p, pd.Period) for p in mm.columns)


def test_year_panel_cells_match_matrix(cd, bundle, cfg):
    """面板里的每一格都要等于逐月矩阵对应月份的值（防止年/月错位）。"""
    mm = cd.monthly_matrix(bundle, cfg, "long_short")
    labels = list(mm.index)
    frames = [cd.monthly_year_by_month(mm, lab) for lab in labels]
    years = sorted({p.year for p in mm.columns})
    for li, lab in enumerate(labels):
        for year in years:
            for mi in range(12):
                period = pd.Period(f"{year}-{mi + 1:02d}", freq="M")
                got = frames[li].loc[year].to_numpy(dtype="float64")[mi]
                if period in mm.columns:
                    want = mm.loc[lab, period]
                    if np.isfinite(want):
                        assert got == pytest.approx(float(want), rel=1e-12)
                    else:
                        assert not np.isfinite(got)
                else:
                    assert not np.isfinite(got)


@pytest.mark.parametrize("layout", ["strip", "panel", "row", "all"])
def test_monthly_layout_valid_values(cd, cfg, layout):
    cd.validate({**cfg, "monthly_layout": layout})


def test_monthly_layout_rejects_unknown(cd, cfg):
    with pytest.raises(SystemExit):
        cd.validate({**cfg, "monthly_layout": "diagonal"})
