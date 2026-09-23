"""``ratema convert`` —— 把文件夹内的 Excel 转成 CSV。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..io_utils import (
    EXCEL_SUFFIXES,
    convert_excel_to_csv,
    find_excel_input,
)
from ._shared import rule, setup_console


def cmd_convert(args: argparse.Namespace) -> int:
    setup_console()
    rule("Excel -> CSV")
    if args.all:
        candidates = sorted(
            p
            for p in Path().glob("*.xls*")
            if not p.name.startswith("~$") and p.suffix.lower() in EXCEL_SUFFIXES
        )
        if not candidates:
            raise SystemExit("当前目录下没有找到 Excel 文件")
    else:
        # convert 只处理 Excel，不能把已生成的 panel.csv 当成输入
        candidates = [Path(args.input) if args.input else find_excel_input(".")]

    for xlsx in candidates:
        report = convert_excel_to_csv(xlsx, args.outdir, header_rows=args.header_rows)
        print(f"  输入      : {report.input_file}")
        print(f"  工作表    : {', '.join(report.sheets)}")
        print(f"  数据行数  : {report.rows}")
        print(f"  指标列    : {', '.join(report.columns)}")
        print(f"  日期范围  : {report.date_min} ~ {report.date_max}")
        print(f"  输出目录  : {Path(report.output_dir).resolve()}")
        for path in report.files:
            print(f"    - {path}")
        rule()

    print("转换完成。可直接运行：ratema backtest data/csv/panel.csv")
    return 0


__all__ = ["cmd_convert"]
