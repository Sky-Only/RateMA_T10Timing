"""``ratema composite`` —— 两步走：每利率单独测试 + 等权打分综合。"""

from __future__ import annotations

import argparse

from ..composite import MODE_HELP, VOTE_SOURCE_HELP, run_composite
from ..io_utils import load_dataset, resolve_roles
from ..render import COMPOSITE_COLUMNS
from ..writers import write_composite
from ._shared import (
    fmt_value,
    make_backtest_config,
    make_signal_config,
    output_dir,
    parse_rate_cols,
    resolve_input,
    rule,
    setup_console,
)


def cmd_composite(args: argparse.Namespace) -> int:
    setup_console()
    from ..pipeline import run_single

    path = resolve_input(args)
    dataset = load_dataset(path)
    index_col, rate_cols = resolve_roles(dataset, args.index_col, parse_rate_cols(args))
    signal_cfg = make_signal_config(args)
    backtest_cfg = make_backtest_config(args)

    rule("两步走：单指标测试 → 五利率等权综合")
    print(f"  数据源      : {dataset.source}")
    print(f"  标的指数    : {index_col} ({dataset.display_name(index_col)})")
    print(f"  利率指标    : {', '.join(rate_cols)}  （共 {len(rate_cols)} 个）")
    print(f"  均线        : MA{signal_cfg.short_window} / MA{signal_cfg.long_window}")
    print(f"  重合阈值    : {signal_cfg.describe_tol()}")
    print(f"  交易成本    : {backtest_cfg.describe()}")
    print(f"  综合方式    : {args.mode} —— {MODE_HELP[args.mode]}")
    print(f"  投票来源    : {args.vote_source} —— {VOTE_SOURCE_HELP[args.vote_source]}")
    rule()

    # ---- 步骤 1：每利率单独回测 ------------------------------------------ #
    print("【步骤 1】每个利率单独回测")
    per_indicator = []
    for series in rate_cols:
        res = run_single(
            dataset,
            series,
            signal_cfg,
            backtest_cfg,
            index_col=index_col,
            start=args.start,
            end=args.end,
        )
        per_indicator.append(res)
        print(f"    ✓ {series:<10} ({res.display_name})")

    # ---- 步骤 2：等权打分综合 -------------------------------------------- #
    print()
    print("【步骤 2】五利率等权打分综合")
    result = run_composite(
        dataset,
        per_indicator,
        backtest_cfg,
        index_col=index_col,
        mode=args.mode,
        vote_source=args.vote_source,
        start=args.start,
        end=args.end,
    )
    first = result.composite_frame["date"].iloc[0]
    last = result.composite_frame["date"].iloc[-1]
    print(f"    ✓ 综合信号 + 等权组合（评估区间 {first.date()} ~ {last.date()}）")
    rule()

    if not args.quiet:
        summary = result.summary_frame()
        show = summary.copy()
        for src, label, kind in COMPOSITE_COLUMNS:
            if src in show.columns:
                show[label] = show[src].map(lambda v, k=kind: fmt_value(v, k))
        keep = ["方案", "类型"] + [c for _, c, _ in COMPOSITE_COLUMNS if c in show.columns]
        print(show[keep].to_string(index=False))
        print()

        single = summary[summary["类型"] == "单指标"]
        cm = result.composite_metrics
        pm = result.portfolio_metrics
        if len(single):
            print(
                f"  单指标平均夏普 : {single['夏普'].mean():.3f}"
                f"   最高 {single['夏普'].max():.3f}"
                f"   最低 {single['夏普'].min():.3f}"
            )
        print(
            f"  综合信号夏普   : {cm['strategy_sharpe']:.3f}"
            f"   年化 {cm['strategy_cagr']:.2%}"
            f"   回撤 {cm['strategy_max_drawdown']:.2%}"
        )
        print(
            f"  等权组合夏普   : {pm['strategy_sharpe']:.3f}"
            f"   年化 {pm['strategy_cagr']:.2%}"
            f"   回撤 {pm['strategy_max_drawdown']:.2%}"
        )
        print()

    outdir = output_dir(args)
    written = write_composite(
        result,
        outdir,
        meta={
            "数据源": dataset.source,
            "标的指数": index_col,
            "均线": f"MA{signal_cfg.short_window}/MA{signal_cfg.long_window}",
            "重合阈值": signal_cfg.describe_tol(),
            "交易成本": backtest_cfg.describe(),
        },
    )

    if getattr(args, "charts", True):
        from ..charts import plot_composite, setup_style

        setup_style()
        written.append(str(plot_composite(result, outdir / "charts")))

    rule("产出文件")
    for p in written:
        print(f"  - {p}")
    print(f"\n报告：{(outdir / 'composite_report.md').resolve()}")
    return 0


__all__ = ["cmd_composite"]
