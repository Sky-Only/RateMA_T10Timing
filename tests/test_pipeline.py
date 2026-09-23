"""端到端流程与产出文件测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _data import make_dataset
from ratema.backtest import BacktestConfig
from ratema.indicators import SignalConfig
from ratema.io_utils import Dataset, resolve_roles
from ratema.pipeline import run_all, run_single
from ratema.render import d, fmt
from ratema.writers import _jsonable, _sanitize, write_run, write_sweep


def test_run_all_covers_every_rate_column(dataset: Dataset):
    run = run_all(dataset, SignalConfig(20, 120), BacktestConfig(cost_bps=2.0))
    assert run.index_col == "CBA04502.CS"
    assert run.rate_cols == ["DR001", "R001"]
    assert [r.series for r in run.results] == ["DR001", "R001"]
    for res in run.results:
        assert res.warmup_days >= 119, "MA120 需要 120 个观测"
        assert len(res.frame) == len(dataset.frame) - res.warmup_days
        assert res.metrics["strategy_n_obs"] > 0


def test_warmup_is_excluded_from_evaluation(dataset: Dataset):
    res = run_single(dataset, "DR001", SignalConfig(20, 120), BacktestConfig())
    assert not res.frame["signal_eff"].isna().any()
    assert res.frame["date"].iloc[0] == dataset.frame["date"].iloc[119]


def test_evaluation_window_starts_at_first_valid_signal_day(dataset: Dataset):
    """评估起点必须恰好是 signal_eff 首次非空的交易日。"""
    res = run_single(dataset, "DR001", SignalConfig(20, 120), BacktestConfig())
    w = res.window

    first_valid = int(res.full_signals["signal_eff"].notna().to_numpy().argmax())
    assert w["eval_start"] == dataset.frame["date"].iloc[first_valid]
    assert w["eval_start"] == res.frame["date"].iloc[0]
    assert w["warmup_days"] == first_valid
    # 预热期最后一天应是评估起点的前一个交易日
    assert w["warmup_end"] == dataset.frame["date"].iloc[first_valid - 1]
    assert w["n_eval_days"] == len(dataset.frame) - first_valid
    # 基准与策略同起点、同初始资金
    assert w["benchmark_entry_date"] == w["eval_start"]
    assert w["benchmark_entry_price"] == pytest.approx(
        float(dataset.frame["CBA04502.CS"].iloc[first_valid])
    )
    # 首日只可能有信号、不可能有持仓（T+1 执行）
    assert res.frame["position"].iloc[0] == 0
    assert res.frame["equity"].iloc[0] == pytest.approx(1.0)
    assert res.window["first_position_lag_days"] >= 1


def test_evaluation_window_is_consistent_across_series(dataset: Dataset):
    run = run_all(dataset, SignalConfig(20, 120), BacktestConfig())
    starts = {r.window["eval_start"] for r in run.results}
    assert len(starts) == 1, f"各指标评估起点应一致，实际 {starts}"
    assert {r.window["n_eval_days"] for r in run.results} == {run.results[0].window["n_eval_days"]}


def test_wider_tolerance_never_reduces_overlap_days(dataset: Dataset):
    counts = []
    for tol in (0.0, 0.001, 0.005, 0.01):
        res = run_single(dataset, "DR001", SignalConfig(20, 120, "abs", tol), BacktestConfig())
        counts.append(res.overlap["n_overlap"])
    assert counts == sorted(counts)


def test_overwide_tolerance_reports_clear_error():
    """阈值宽到所有交易日都重合时，应给出可操作的错误提示。"""
    ds = make_dataset(n=300)
    with pytest.raises(ValueError, match="都落在重合区间内"):
        run_single(ds, "DR001", SignalConfig(20, 120, "abs", 99.0), BacktestConfig())


def test_strategy_and_benchmark_share_start_point(dataset: Dataset):
    res = run_single(dataset, "DR001", SignalConfig(20, 120), BacktestConfig())
    eq = res.frame["equity"]
    bench = res.frame["benchmark_equity"]
    assert eq.iloc[0] == pytest.approx(1.0)
    assert bench.iloc[0] == pytest.approx(1.0 / (1.0 + 0.0002))


def test_start_and_end_filters(dataset: Dataset):
    res = run_single(
        dataset,
        "DR001",
        SignalConfig(20, 120),
        BacktestConfig(),
        index_col="CBA04502.CS",
        start=str(dataset.frame["date"].iloc[200].date()),
        end=str(dataset.frame["date"].iloc[300].date()),
    )
    assert res.frame["date"].iloc[0] >= dataset.frame["date"].iloc[200]
    assert res.frame["date"].iloc[-1] <= dataset.frame["date"].iloc[300]
    assert len(res.frame) == 101


def test_insufficient_history_raises():
    ds = make_dataset(n=60)
    with pytest.raises(ValueError, match="不足以形成"):
        run_single(ds, "DR001", SignalConfig(20, 120), BacktestConfig())


def test_write_run_produces_all_artifacts(dataset: Dataset, tmp_path: Path):
    run = run_all(dataset, SignalConfig(20, 120, "abs", 0.01), BacktestConfig())
    written = write_run(run, tmp_path)

    for name in (
        "summary.csv",
        "metrics.json",
        "spread_calibration.csv",
        "evaluation_window.csv",
        "equity_curves.csv",
        "report.md",
        "columns.json",
    ):
        assert (tmp_path / name).exists(), f"缺少 {name}"

    for series in run.rate_cols:
        assert (tmp_path / "details" / f"{series}.csv").exists()
        assert (tmp_path / "equity" / f"{series}.csv").exists()

    summary = pd.read_csv(tmp_path / "summary.csv", encoding="utf-8-sig")
    assert len(summary) == len(run.rate_cols)
    assert {"strategy_cagr", "strategy_sharpe", "overlap_days"} <= set(summary.columns)

    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["run"]["index_col"] == "CBA04502.CS"
    assert payload["run"]["signal_config"]["tol_mode"] == "abs"
    assert payload["run"]["signal_config"]["tol"] == 0.01
    assert set(payload["strategies"]) == set(run.rate_cols)

    curves = pd.read_csv(tmp_path / "equity_curves.csv", encoding="utf-8-sig")
    assert "BENCHMARK" in curves.columns
    for series in run.rate_cols:
        assert series in curves.columns

    win = pd.read_csv(tmp_path / "evaluation_window.csv", encoding="utf-8-sig")
    assert len(win) == len(run.rate_cols)
    assert {"series", "eval_start", "warmup_days", "first_signal_value"} <= set(win.columns)

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "20/120 日均线利率择时策略回测报告" in report
    assert "评估窗口与起算口径" in report
    assert "首个有效信号日" in report
    assert "重合度阈值标定" in report
    assert written


def test_write_sweep_produces_grid_outputs(tmp_path: Path):
    rows = []
    for tol in (0.0, 0.01, 0.05):
        for series in ("DR001", "R001"):
            rows.append(
                {
                    "tol": tol,
                    "series": series,
                    "series_name": series,
                    "strategy_cagr": 0.03 + tol,
                    "strategy_sharpe": 0.5 + tol * 10,
                    "strategy_max_drawdown": -0.05,
                    "strategy_ann_vol": 0.02,
                    "strategy_calmar": 0.6,
                    "excess_cagr": 0.01,
                    "overlap_days": int(tol * 1000),
                    "overlap_share": tol,
                    "signal_flips": 10,
                    "n_completed": 5,
                    "time_in_market": 0.5,
                }
            )
    written = write_sweep(rows, tmp_path, tol_mode="abs", meta={"数据源": "synthetic"})
    assert (tmp_path / "sweep_abs.csv").exists()
    assert (tmp_path / "sweep_abs.md").exists()
    assert (tmp_path / "pivot_strategy_sharpe_abs.csv").exists()
    assert written
    pivot = pd.read_csv(tmp_path / "pivot_strategy_sharpe_abs.csv", encoding="utf-8-sig")
    assert len(pivot) == 2
    assert next(iter(pivot.columns)) == "series"


def test_resolve_roles_with_custom_index(dataset: Dataset):
    index_col, rates = resolve_roles(dataset, index_col="DR001", rate_cols=["R001"])
    assert index_col == "DR001"
    assert rates == ["R001"]


def test_json_serialization_handles_nat_and_missing_values():
    """DataFrame 往返会把缺失日期变成 pd.NaT（不是 Timestamp 实例），
    序列化器必须能处理，否则 json 会递归到 "Circular reference detected"。"""
    assert _jsonable(pd.NaT) is None
    assert _jsonable(pd.NA) is None
    assert _jsonable(None) is None
    assert _jsonable(np.float64("nan")) is None
    assert _jsonable(np.float64("inf")) is None
    assert _jsonable(pd.Timestamp("2024-01-02")) == "2024-01-02"
    assert _jsonable(np.int64(3)) == 3
    assert _jsonable(np.bool_(True)) is True
    assert _jsonable(Path("a/b")) in {"a/b", "a\\b"}
    assert _jsonable(object()) is not None  # 未知类型兜底为字符串

    # 真实回归场景：某些行没有指标值，日期列会被 pandas 转成 datetime64 + NaT
    frame = pd.DataFrame(
        {
            "tol": [0.0, 5.0],
            "error": ["boom", None],
            "strategy_start": [pd.Timestamp("2022-06-17"), None],
        }
    )
    payload = {"rows": frame.to_dict("records"), "cagr": float("nan")}
    text = json.dumps(json.loads(json.dumps(_sanitize(payload))), ensure_ascii=False)
    assert "NaN" not in text, "非法 JSON 字面量 NaN 不应出现"
    restored = json.loads(text)
    assert restored["rows"][0]["strategy_start"] == "2022-06-17"
    assert restored["rows"][1]["strategy_start"] is None
    assert restored["rows"][1]["error"] is None
    assert restored["cagr"] is None


def test_sweep_payload_survives_mixed_success_and_error_rows(tmp_path: Path):
    """成功行与失败行混在一起时，sweep 的 JSON 也必须能写出。"""
    rows = [
        {
            "tol": 0.0,
            "series": "DR001",
            "strategy_cagr": 0.02,
            "strategy_start": pd.Timestamp("2022-06-17"),
            "strategy_max_drawdown_peak": pd.Timestamp("2022-07-26"),
        },
        {"tol": 5.0, "series": "DR001", "error": "全部重合，无初始信号"},
    ]
    written = write_sweep(rows, tmp_path, tol_mode="bp", meta={"数据源": "synthetic"})
    assert written
    payload = json.loads((tmp_path / "sweep_bp.json").read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 2


def test_render_helpers_format_consistently():
    assert fmt(0.0123, "pct") == "1.23%"
    assert fmt(3.9, "int") == "3"
    assert fmt(float("nan"), "pct") == "n/a"
    assert fmt(None, "pct") == ""
    assert d(pd.Timestamp("2024-01-02")) == "2024-01-02"
    assert d(None) == "n/a"
