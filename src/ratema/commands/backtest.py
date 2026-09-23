"""``ratema backtest`` —— 跑一次完整回测。"""

from __future__ import annotations

import argparse

from ..io_utils import load_dataset
from ..render import calibration_table, summary_table
from ..writers import write_run
from ._shared import (
    make_backtest_config,
    make_signal_config,
    output_dir,
    parse_rate_cols,
    print_run_banner,
    resolve_input,
    rule,
    setup_console,
)


def cmd_backtest(args: argparse.Namespace) -> int:
    setup_console()
    path = resolve_input(args)
    dataset = load_dataset(path)

    from ..pipeline import run_all

    run = run_all(
        dataset,
        make_signal_config(args),
        make_backtest_config(args),
        index_col=args.index_col,
        rate_cols=parse_rate_cols(args),
        start=args.start,
        end=args.end,
    )
    print_run_banner(run)

    if not args.quiet:
        print(summary_table(run))
        print()
        print("重合度阈值标定（|MA快-MA慢| 分位数，单位 bp）")
        print(calibration_table(run))
        print()

    outdir = output_dir(args)
    written = write_run(run, outdir)
    rule("产出文件")
    for artifact in written:
        print(f"  - {artifact}")

    if getattr(args, "charts", True):
        from ..charts import make_all_charts
        from ..parser import parse_signal_window

        charts = make_all_charts(
            run,
            outdir,
            signal_series=getattr(args, "signal_series", None),
            signal_window=parse_signal_window(getattr(args, "signal_window", None)),
        )
        rule("图表")
        for chart in charts:
            print(f"  - {chart}")

    print(f"\n报告：{(outdir / 'report.md').resolve()}")
    return 0


__all__ = ["cmd_backtest"]
