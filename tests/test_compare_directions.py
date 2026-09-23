"""``compare_directions.py`` 方向对比模块的测试。

重点不在「跑得通」，而在三件事：
1. 三条曲线确实共用同一信号与同一评估区间（否则图上的差异归因不成立）；
2. 指标表不会因为键名取错而静默出 NaN / 张冠李戴；
3. 独立入口的配置校验与参数覆盖生效。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
    """跨利率长表要包含等权综合 + 每个利率，且每行 4 条曲线。"""
    table = cd.build_all_metrics_table(bundle, cfg)
    labels = table["指标"].unique().tolist()
    assert labels[0] == cd.COMPOSITE_ROW_LABEL
    assert set(labels) == {cd.COMPOSITE_ROW_LABEL, *bundle["base"].rate_cols}
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
