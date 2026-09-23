"""``ratema journal`` —— 输出每日交易日志。

这是本项目的「输出终点」：把策略在每个交易日的信号、仓位、动作、净值
记成一份可读可追溯的日志。不做下单、不做对账。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..composite import run_composite
from ..io_utils import load_dataset, resolve_roles
from ..journal import append_daily_entry, build_journal, write_journal
from ._shared import (
    make_backtest_config,
    make_signal_config,
    output_dir,
    parse_rate_cols,
    resolve_input,
    rule,
    setup_console,
)

COMPOSITE_LABEL = "COMPOSITE"


def _pick_frame(args, dataset, index_col, rate_cols, signal_cfg, backtest_cfg):
    """返回 (逐日明细 DataFrame, 序列名, 补充说明)。"""
    if args.series.upper() == COMPOSITE_LABEL:
        from ..pipeline import run_single

        per = [
            run_single(
                dataset,
                s,
                signal_cfg,
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
        note = f"等权综合（{args.mode} / 投票源 {args.vote_source}，共 {len(rate_cols)} 个指标）"
        return result.composite_frame, COMPOSITE_LABEL, note

    if args.series not in rate_cols:
        raise SystemExit(
            f"未知指标 {args.series!r}；可选：{COMPOSITE_LABEL}, {', '.join(rate_cols)}"
        )
    from ..pipeline import run_single

    res = run_single(
        dataset,
        args.series,
        signal_cfg,
        backtest_cfg,
        index_col=index_col,
        start=args.start,
        end=args.end,
    )
    return res.frame, args.series, dataset.display_name(args.series)


def cmd_journal(args: argparse.Namespace) -> int:
    setup_console()
    path = resolve_input(args)
    dataset = load_dataset(path)
    index_col, rate_cols = resolve_roles(dataset, args.index_col, parse_rate_cols(args))
    signal_cfg = make_signal_config(args)
    backtest_cfg = make_backtest_config(args)

    frame, series_name, note = _pick_frame(
        args, dataset, index_col, rate_cols, signal_cfg, backtest_cfg
    )

    journal = build_journal(frame, series=series_name, index_col=index_col)

    rule("每日交易日志")
    print(f"  数据源      : {dataset.source}")
    print(f"  日志对象    : {series_name} —— {note}")
    print(f"  均线        : MA{signal_cfg.short_window} / MA{signal_cfg.long_window}")
    print(f"  重合阈值    : {signal_cfg.describe_tol()}")
    print(f"  评估区间    : {journal.summary['start'].date()} ~ {journal.summary['end'].date()}")
    print(f"  交易日数    : {journal.summary['n_days']}")
    print(f"  动作次数    : 建仓 {journal.summary['n_open']} / 平仓 {journal.summary['n_close']}")
    print(
        f"  期末净值    : {journal.summary['final_equity']:.4f}"
        f"（基准 {journal.summary['final_benchmark']:.4f}）"
    )
    rule()

    if not args.quiet:
        show = journal.events if args.events_only else journal.tail(args.tail)
        label = "全部动作日" if args.events_only else f"最近 {args.tail} 个交易日"
        print(f"【{label}】")
        if show.empty:
            print("  （无记录）")
        else:
            cols = (
                [
                    "date",
                    "action",
                    "signal",
                    "n_long",
                    "n_short",
                    "close",
                    "equity",
                    "cumulative_return",
                    "days_in_prev_state",
                ]
                if args.events_only
                else [
                    "date",
                    "signal",
                    "n_long",
                    "n_short",
                    "position",
                    "action",
                    "close",
                    "daily_return",
                    "excess_return",
                    "equity",
                    "drawdown",
                ]
            )
            print(show[cols].to_string(index=False))
        print()

    outdir = output_dir(args) / "journal"
    written = write_journal(journal, outdir, tail=args.tail)

    if args.append:
        append_path = Path(args.append)
        row = append_daily_entry(journal, append_path, date=args.append_date)
        written.append(str(append_path))
        print(f"  已追加单日记录：{pd.Timestamp(row['date']).date()} → {append_path}")
        print()

    rule("产出文件")
    for p in written:
        print(f"  - {p}")
    print(f"\n日志：{(outdir / 'journal.md').resolve()}")
    return 0


__all__ = ["COMPOSITE_LABEL", "cmd_journal"]
