"""``ratema signal`` —— 查看最新一天的交易信号。"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from ..daily import CONSENSUS_HELP, latest_snapshot
from ..io_utils import load_dataset
from ._shared import (
    make_signal_config,
    parse_rate_cols,
    resolve_input,
    rule,
    setup_console,
)


def _render(snap, source: str) -> None:
    rule("最新交易信号")
    print(f"  数据源      : {source}")
    print(f"  信号日期    : {snap.asof.date()}（数据最后一行）")
    print(f"  标的收盘    : {snap.index_col} = {snap.index_close:.4f}")
    print(
        f"  均线        : MA{snap.signal_config.short_window} / MA{snap.signal_config.long_window}"
    )
    print(f"  重合阈值    : {snap.signal_config.describe_tol()}")
    print(f"  合成方式    : {snap.mode}（{snap.votes_long}/{snap.n_indicators} 个指标当前为多头）")
    if snap.stale_warning:
        print()
        print(
            f"  ⚠️  数据已 {snap.data_stale_days} 天未更新"
            f"（最后一行 {snap.data_latest.date()}），信号可能已失效，请先更新数据。"
        )
    rule()

    table = pd.DataFrame(
        [
            {
                "指标": r["series"],
                "利率": f"{r['rate']:.3f}",
                "MA短": f"{r['ma_short']:.3f}",
                "MA长": f"{r['ma_long']:.3f}",
                "差(bp)": f"{r['spread_bp']:+.1f}",
                "容差(bp)": f"{r['tolerance_bp']:.1f}",
                "重合": "是" if r["is_overlap"] else "",
                "昨日持仓": "是" if r["hold_prev"] else "否",
                "信号": f"{int(r['signal_eff']):+d}",
                "明日目标": "持有多头" if r["hold_target"] else "空仓",
            }
            for r in snap.per_indicator.to_dict("records")
        ]
    )
    print(table.to_string(index=False))
    rule()
    print(f"  >>> 明日操作：{snap.action}")
    rule()
    print("  口径说明：")
    print("    · T 日收盘后确定信号，T+1 日以收盘价执行（与回测一致）")
    print("    · 只有信号翻转时才有动作，多数交易日无需操作")
    print("    · 空仓期间不计利息（回测口径）；如需计入货币收益请自行调整")


def cmd_signal(args: argparse.Namespace) -> int:
    setup_console()
    path = resolve_input(args)
    dataset = load_dataset(path)
    cfg = make_signal_config(args)

    try:
        snap = latest_snapshot(
            dataset,
            cfg,
            index_col=args.index_col,
            rate_cols=parse_rate_cols(args),
            asof=args.asof,
            mode=args.mode,
            max_stale_days=args.max_stale_days,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.json:
        print(json.dumps(snap.to_dict(), ensure_ascii=False, indent=2, default=str))
    else:
        _render(snap, dataset.source)

    # 退出码 2 = 数据过期，便于调度系统熔断
    return 2 if snap.stale_warning else 0


__all__ = ["CONSENSUS_HELP", "cmd_signal"]
