"""``ratema sweep`` —— 重合度阈值敏感性分析。

支持对**单指标**（``--target indicator``）或**等权综合信号**（``--target composite``）扫描。
"""

from __future__ import annotations

import argparse

import pandas as pd

from ..composite import COMPOSITE_MODES, run_composite
from ..indicators import TOL_MODE_HELP, SignalConfig
from ..io_utils import load_dataset, resolve_roles
from ..writers import write_sweep
from ._shared import (
    make_backtest_config,
    make_signal_config,
    output_dir,
    parse_grid,
    parse_rate_cols,
    resolve_input,
    rule,
    setup_console,
)


def _sweep_indicators(dataset, rate_cols, index_col, grid, signal_cfg, backtest_cfg, args):
    """对每个单指标逐阈值回测。"""
    from ..pipeline import run_single

    rows: list[dict] = []
    for tol in grid:
        cfg = SignalConfig(
            short_window=signal_cfg.short_window,
            long_window=signal_cfg.long_window,
            tol_mode=signal_cfg.tol_mode,
            tol=tol,
            std_window=signal_cfg.std_window,
        )
        for series in rate_cols:
            row: dict = {
                "target": "indicator",
                "tol": tol,
                "series": series,
                "series_name": dataset.display_name(series),
                "tol_description": cfg.describe_tol(),
            }
            try:
                res = run_single(
                    dataset,
                    series,
                    cfg,
                    backtest_cfg,
                    index_col=index_col,
                    start=args.start,
                    end=args.end,
                )
            except ValueError as exc:
                row["error"] = str(exc)
                rows.append(row)
                print(f"  [跳过] tol={tol:g} {series}: {exc}")
                continue
            row.update(res.metrics)
            row["overlap_days"] = res.overlap.get("n_overlap")
            row["overlap_share"] = res.overlap.get("overlap_share")
            row["signal_flips"] = res.overlap.get("signal_flips")
            row["raw_signal_flips"] = res.overlap.get("raw_signal_flips")
            rows.append(row)
    return rows


def _sweep_composite(dataset, rate_cols, index_col, grid, signal_cfg, backtest_cfg, args):
    """对等权综合信号逐阈值回测。"""
    from ..pipeline import run_single

    rows: list[dict] = []
    for tol in grid:
        cfg = SignalConfig(
            short_window=signal_cfg.short_window,
            long_window=signal_cfg.long_window,
            tol_mode=signal_cfg.tol_mode,
            tol=tol,
            std_window=signal_cfg.std_window,
        )
        row: dict = {
            "target": "composite",
            "tol": tol,
            "series": "COMPOSITE",
            "series_name": "等权综合信号",
            "tol_description": cfg.describe_tol(),
        }
        try:
            per = [
                run_single(
                    dataset,
                    s,
                    cfg,
                    backtest_cfg,
                    index_col=index_col,
                    start=args.start,
                    end=args.end,
                )
                for s in rate_cols
            ]
            result = run_composite(
                dataset,
                per,
                backtest_cfg,
                index_col=index_col,
                mode=args.mode,
                vote_source=args.vote_source,
                start=args.start,
                end=args.end,
            )
        except ValueError as exc:
            row["error"] = str(exc)
            rows.append(row)
            print(f"  [跳过] tol={tol:g} 综合: {exc}")
            continue
        m = result.composite_metrics
        row.update(m)
        row["overlap_days"] = m.get("overlap_days")
        row["overlap_share"] = m.get("overlap_share")
        row["signal_flips"] = m.get("signal_flips")
        row["raw_signal_flips"] = m.get("raw_signal_flips")
        rows.append(row)
    return rows


def cmd_sweep(args: argparse.Namespace) -> int:
    setup_console()
    path = resolve_input(args)
    dataset = load_dataset(path)
    signal_cfg = make_signal_config(args)
    backtest_cfg = make_backtest_config(args)
    grid = parse_grid(args.tol_grid or _default_grid_for(args.target, signal_cfg.tol_mode))

    if signal_cfg.tol_mode == "q":
        grid = [g for g in grid if 0.0 <= g <= 1.0]
        if not grid:
            raise SystemExit("tol-mode=q 时，--tol-grid 必须落在 [0, 1]")

    index_col, rate_cols = resolve_roles(dataset, args.index_col, parse_rate_cols(args))

    rule("重合度阈值敏感性分析")
    print(f"  扫描对象    : {args.target}")
    print(f"  模式        : {signal_cfg.tol_mode} —— {TOL_MODE_HELP[signal_cfg.tol_mode]}")
    print(f"  阈值网格    : {', '.join(f'{g:g}' for g in grid)}")
    print(f"  标的指数    : {dataset.source}")
    if args.target == "composite":
        print(f"  综合方式    : {args.mode} / 投票源 {args.vote_source}")
    rule()

    if args.target == "composite":
        rows = _sweep_composite(dataset, rate_cols, index_col, grid, signal_cfg, backtest_cfg, args)
    else:
        rows = _sweep_indicators(
            dataset, rate_cols, index_col, grid, signal_cfg, backtest_cfg, args
        )

    outdir = output_dir(args) / "sweep"
    written = write_sweep(
        rows,
        outdir,
        tol_mode=signal_cfg.tol_mode,
        meta={
            "数据源": dataset.source,
            "标的指数": index_col,
            "扫描对象": args.target,
            "均线": f"MA{signal_cfg.short_window}/MA{signal_cfg.long_window}",
            "成本": backtest_cfg.describe(),
        },
    )

    df = pd.DataFrame(rows)
    ok = df[df.get("strategy_sharpe").notna()] if "strategy_sharpe" in df.columns else df
    if not ok.empty and "strategy_sharpe" in ok.columns:
        pivot = ok.pivot_table(index="series", columns="tol", values="strategy_sharpe")
        pivot.columns = [f"tol={c:g}" for c in pivot.columns]
        print("策略夏普（行=指标，列=阈值）")
        print(pivot.round(3).to_string())
        print()
        pivot_cagr = ok.pivot_table(index="series", columns="tol", values="strategy_cagr")
        pivot_cagr.columns = [f"tol={c:g}" for c in pivot_cagr.columns]
        print("策略年化收益（行=指标，列=阈值）")
        print((pivot_cagr * 100).round(2).to_string())
        print()

    if getattr(args, "charts", True) and not ok.empty and "strategy_sharpe" in ok.columns:
        _plot_sweep(
            dataset, rate_cols, index_col, grid, signal_cfg, backtest_cfg, args, ok, outdir, written
        )

    rule("产出文件")
    for p in written:
        print(f"  - {p}")
    return 0


def _plot_sweep(
    dataset, rate_cols, index_col, grid, signal_cfg, backtest_cfg, args, ok, outdir, written
) -> None:
    """阈值对比图：挑网格内平均夏普最高的对象画净值曲线。"""
    from ..charts import plot_tol_sweep, setup_style
    from ..pipeline import run_single

    setup_style()
    scored = (
        ok.dropna(subset=["strategy_sharpe"])
        .groupby("series")["strategy_sharpe"]
        .mean()
        .sort_values(ascending=False)
    )
    chart_series = str(scored.index[0]) if len(scored) else rate_cols[0]
    curves: dict[str, pd.DataFrame] = {}
    for tol in grid:
        cfg = SignalConfig(
            short_window=signal_cfg.short_window,
            long_window=signal_cfg.long_window,
            tol_mode=signal_cfg.tol_mode,
            tol=tol,
            std_window=signal_cfg.std_window,
        )
        try:
            res = run_single(
                dataset,
                chart_series,
                cfg,
                backtest_cfg,
                index_col=index_col,
                start=args.start,
                end=args.end,
            )
        except ValueError:
            continue
        curves[f"tol={tol:g}（重合 {res.overlap['n_overlap']} 天）"] = res.frame
    if curves:
        written.append(
            str(
                plot_tol_sweep(
                    curves,
                    output_dir(args) / "charts",
                    title_extra=f"{chart_series} · {TOL_MODE_HELP[signal_cfg.tol_mode]}",
                )
            )
        )


def _default_grid_for(target: str, tol_mode: str) -> str:
    from ..parser import default_grid

    return default_grid(tol_mode)


__all__ = ["COMPOSITE_MODES", "cmd_sweep"]
