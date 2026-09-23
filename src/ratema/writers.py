"""产物落盘：CSV / JSON / Markdown。"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from datetime import date
from datetime import datetime as datetime_
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import COLUMNS_META_FILE
from .metrics import annual_breakdown
from .pipeline import RunResult
from .render import (
    SWEEP_COLUMNS,
    fmt,
    render_composite_markdown,
    render_markdown,
    render_sweep_markdown,
)


def _jsonable(value: Any) -> Any:
    """把 pandas / numpy 标量转成 JSON 可序列化的原生类型。

    注意 ``pd.NaT`` 并不是 ``pd.Timestamp`` 的实例，必须单独处理；
    末尾也不能直接 ``return value``，否则 json 会对同一对象反复调用 default
    并最终抛出 "Circular reference detected"。
    """
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (pd.Timedelta,)):
        return str(value)
    if isinstance(value, (datetime_, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray,)):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, (str, int, float, bool)):
        return value
    # 未知类型统一转字符串，保证序列化永不失败
    return str(value)


def _sanitize(value: Any) -> Any:
    """递归地把 payload 转成严格 JSON 类型。

    ``json.dumps`` 对普通的 ``float('nan')`` 会直接输出 ``NaN``（不是合法 JSON）
    且不会调用 ``default``，因此必须在 dumps 之前先清洗一遍。
    """
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return _jsonable(value)


def _dump_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")


def write_run(run: RunResult, outdir: str | Path) -> list[str]:
    """写出一次完整运行的全部产物，返回文件清单。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # ---- 逐指标明细 ----
    for res in run.results:
        detail = res.frame.copy()
        detail["date"] = detail["date"].dt.strftime("%Y-%m-%d")
        path = outdir / "details" / f"{res.series}.csv"
        _write_csv(detail, path)
        written.append(str(path))

        trades = res.backtest.trades_frame
        if not trades.empty:
            trades = trades.copy()
            trades["date"] = pd.to_datetime(trades["date"]).dt.strftime("%Y-%m-%d")
        tpath = outdir / "trades" / f"{res.series}.csv"
        _write_csv(trades, tpath)
        written.append(str(tpath))

        eq = res.frame[
            [
                "date",
                "equity",
                "equity_gross",
                "benchmark_equity",
                "benchmark_equity_gross",
                "daily_return",
                "benchmark_return",
                "excess_return",
                "drawdown",
                "position",
            ]
        ].copy()
        eq["date"] = eq["date"].dt.strftime("%Y-%m-%d")
        epath = outdir / "equity" / f"{res.series}.csv"
        _write_csv(eq, epath)
        written.append(str(epath))

    # ---- 净值曲线合并 ----
    curves = None
    for res in run.results:
        col = res.frame.set_index("date")["equity"].rename(res.series)
        curves = col.to_frame() if curves is None else curves.join(col, how="outer")
    if curves is not None and run.results:
        bench = run.results[0].frame.set_index("date")["benchmark_equity"].rename("BENCHMARK")
        curves = curves.join(bench, how="left").reset_index()
        curves["date"] = curves["date"].dt.strftime("%Y-%m-%d")
        cpath = outdir / "equity_curves.csv"
        _write_csv(curves, cpath)
        written.append(str(cpath))

    # ---- 汇总表 ----
    summary = run.summary_frame()
    spath = outdir / "summary.csv"
    _write_csv(summary, spath)
    written.append(str(spath))

    # ---- 阈值标定 ----
    calib = pd.DataFrame([r.calibration for r in run.results])
    calib.insert(1, "series_name", [r.display_name for r in run.results])
    cpath = outdir / "spread_calibration.csv"
    _write_csv(calib, cpath)
    written.append(str(cpath))

    # ---- 分年度收益与胜率 ----
    annual_frames = []
    for res in run.results:
        t = annual_breakdown(res.frame, res.backtest.trades_frame)
        t.insert(0, "series", res.series)
        t.insert(1, "series_name", res.display_name)
        annual_frames.append(t)
    if annual_frames:
        apath = outdir / "annual_breakdown.csv"
        _write_csv(pd.concat(annual_frames, ignore_index=True), apath)
        written.append(str(apath))

    # ---- 评估窗口与起算口径 ----
    win_rows = []
    for res in run.results:
        w = dict(res.window)
        w["series"] = res.series
        w["series_name"] = res.display_name
        win_rows.append(w)
    window = pd.DataFrame(win_rows)
    if not window.empty:
        ordered = ["series", "series_name"] + [
            c for c in window.columns if c not in {"series", "series_name"}
        ]
        window = window[ordered]
        for col in window.columns:
            if window[col].dtype.kind == "M":
                window[col] = window[col].dt.strftime("%Y-%m-%d")
        wpath = outdir / "evaluation_window.csv"
        _write_csv(window, wpath)
        written.append(str(wpath))

    # ---- 指标 JSON ----
    metrics_payload = {
        "run": {
            "input": run.dataset.source,
            "index_col": run.index_col,
            "index_name": run.dataset.display_name(run.index_col),
            "rate_cols": run.rate_cols,
            "start": run.start,
            "end": run.end,
            "eval_window_rule": "评估起点 = 首个有效信号日（signal_eff 首次非空），"
            "预热期不计入；策略与基准同起点归一",
            "signal_config": run.signal_cfg.to_dict(),
            "backtest_config": run.backtest_cfg.to_dict(),
        },
        "benchmark": run.benchmark,
        "strategies": {
            res.series: {
                "series_name": res.display_name,
                "warmup_days": res.warmup_days,
                "window": res.window,
                "overlap": res.overlap,
                "open_trip": res.open_trip,
                "metrics": res.metrics,
            }
            for res in run.results
        },
    }
    mpath = outdir / "metrics.json"
    _dump_json(metrics_payload, mpath)
    written.append(str(mpath))

    # ---- Markdown 报告 ----
    report = render_markdown(run)
    rpath = outdir / "report.md"
    rpath.write_text(report, encoding="utf-8")
    written.append(str(rpath))

    # ---- 列名映射副本，方便单独使用 CSV ----
    npath = outdir / COLUMNS_META_FILE
    _dump_json(run.dataset.name_map, npath)
    written.append(str(npath))

    return written


def write_sweep(
    rows: Iterable[dict[str, Any]],
    outdir: str | Path,
    *,
    tol_mode: str,
    meta: dict[str, Any] | None = None,
) -> list[str]:
    """写出阈值敏感性分析结果。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    df = pd.DataFrame(list(rows))
    if df.empty:
        return written

    path = outdir / f"sweep_{tol_mode}.csv"
    _write_csv(df, path)
    written.append(str(path))

    # 透视：行=指标，列=阈值
    for key, _label, kind in SWEEP_COLUMNS:
        if key not in df.columns:
            continue
        pivot = df.pivot_table(index="series", columns="tol", values=key, aggfunc="last")
        pivot = pivot.map(lambda v, k=kind: fmt(v, k))
        pivot.columns = [f"tol={c:g}" for c in pivot.columns]
        pivot = pivot.reset_index()
        ppath = outdir / f"pivot_{key}_{tol_mode}.csv"
        _write_csv(pivot, ppath)
        written.append(str(ppath))

    payload = {"tol_mode": tol_mode, "meta": meta or {}, "rows": df.to_dict("records")}
    jpath = outdir / f"sweep_{tol_mode}.json"
    _dump_json(payload, jpath)
    written.append(str(jpath))

    md = render_sweep_markdown(df, tol_mode=tol_mode, meta=meta or {})
    mpath = outdir / f"sweep_{tol_mode}.md"
    mpath.write_text(md, encoding="utf-8")
    written.append(str(mpath))
    return written


def write_composite(result, outdir: str | Path, meta: dict[str, Any] | None = None) -> list[str]:
    """写出两步走结果。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    summary = result.summary_frame()
    spath = outdir / "composite_summary.csv"
    _write_csv(summary, spath)
    written.append(str(spath))

    sig = result.signal_frame.copy()
    sig["date"] = pd.to_datetime(sig["date"]).dt.strftime("%Y-%m-%d")
    sig_path = outdir / "composite_signals.csv"
    _write_csv(sig, sig_path)
    written.append(str(sig_path))

    eq = result.composite_frame[
        ["date", "equity", "benchmark_equity", "position", "score", "n_long", "n_short"]
    ].copy()
    eq = eq.rename(
        columns={"equity": "composite_signal_equity", "position": "composite_signal_position"}
    )
    eq["equal_weight_equity"] = result.portfolio_frame["equity"].to_numpy()
    eq["equal_weight_position"] = result.portfolio_frame["position"].to_numpy()
    eq["date"] = pd.to_datetime(eq["date"]).dt.strftime("%Y-%m-%d")
    epath = outdir / "composite_equity.csv"
    _write_csv(eq, epath)
    written.append(str(epath))

    payload = {
        "config": {
            "mode": result.mode,
            "vote_source": result.vote_source,
            "index_col": result.index_col,
            "rate_cols": result.rate_cols,
            **(meta or {}),
        },
        "per_indicator": {r.series: r.metrics for r in result.per_indicator},
        "composite_signal": result.composite_metrics,
        "equal_weight_portfolio": result.portfolio_metrics,
    }
    jpath = outdir / "composite_metrics.json"
    _dump_json(payload, jpath)
    written.append(str(jpath))

    md = render_composite_markdown(result, meta=meta)
    mpath = outdir / "composite_report.md"
    mpath.write_text(md, encoding="utf-8")
    written.append(str(mpath))
    return written
