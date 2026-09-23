"""``ratema all`` —— 一键跑完全流程。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..io_utils import EXCEL_SUFFIXES
from .backtest import cmd_backtest
from .composite import cmd_composite
from .convert import cmd_convert
from .journal import cmd_journal
from .sweep import cmd_sweep


def cmd_all(args: argparse.Namespace) -> int:
    csv_dir = Path(args.csv_dir)

    if not args.skip_convert:
        cmd_convert(
            argparse.Namespace(
                input=args.input,
                outdir=str(csv_dir),
                header_rows=2,
                all=False,
            )
        )

    panel = csv_dir / "panel.csv"
    if args.input and Path(args.input).suffix.lower() in EXCEL_SUFFIXES:
        source = args.input
    else:
        source = str(panel) if panel.exists() else (args.input or str(panel))

    def _sub(**overrides) -> argparse.Namespace:
        ns = argparse.Namespace(**vars(args))
        ns.input = source
        for key, value in overrides.items():
            setattr(ns, key, value)
        return ns

    cmd_backtest(_sub(quiet=False))
    cmd_composite(_sub(quiet=False, mode="score", vote_source="eff"))
    cmd_journal(_sub(quiet=False))
    cmd_sweep(_sub())
    return 0


__all__ = ["cmd_all"]
