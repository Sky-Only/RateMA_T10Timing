"""子命令之间共享的控制台 / 配置 / 路径辅助。"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from pathlib import Path

from ..backtest import BacktestConfig
from ..indicators import SignalConfig
from ..io_utils import find_default_input
from ..pipeline import RunResult

RULE_WIDTH = 78


def setup_console() -> None:
    """把 stdout / stderr 切到 UTF-8，保证中文在管道中不乱码。"""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def rule(title: str = "") -> None:
    """打印一条分隔线（可带标题）。"""
    line = "─" * RULE_WIDTH
    print(line if not title else f"── {title} " + "─" * max(0, RULE_WIDTH - 4 - len(title)))


def fmt_value(value: object, kind: str) -> str:
    """轻量格式化，与 :func:`ratema.render.fmt` 同口径。"""
    if value is None:
        return ""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f) or math.isinf(f):
        return "n/a"
    if kind == "pct":
        return f"{f * 100:.2f}%"
    if kind == "int":
        return f"{int(f)}"
    return f"{f:.3f}"


def make_signal_config(args: argparse.Namespace) -> SignalConfig:
    return SignalConfig(
        short_window=args.short,
        long_window=args.long,
        tol_mode=args.tol_mode,
        tol=args.tol,
        std_window=args.std_window,
    )


def make_backtest_config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        cost_bps=args.cost_bps,
        cost_mode=args.cost_mode,
        direction=getattr(args, "direction", "long_only"),
        initial_capital=getattr(args, "initial_capital", 1.0),
        annualization=args.annualization,
        risk_free=args.risk_free,
    )


def resolve_input(args: argparse.Namespace) -> Path:
    if args.input:
        path = Path(args.input)
        if not path.exists():
            raise SystemExit(f"输入文件不存在：{path}")
        return path
    return find_default_input(".")


def parse_rate_cols(args: argparse.Namespace) -> list[str] | None:
    if not getattr(args, "rate_cols", None):
        return None
    return [c.strip() for c in args.rate_cols.split(",") if c.strip()]


def parse_grid(text: str) -> list[float]:
    """解析逗号分隔的阈值网格，去重并排序。"""
    values: list[float] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise SystemExit("--tol-grid 不能为空")
    return sorted(set(values))


def output_dir(args: argparse.Namespace) -> Path:
    base = Path(getattr(args, "outdir", "output"))
    tag = getattr(args, "tag", None)
    return base / tag if tag else base


def print_run_banner(run: RunResult) -> None:
    cfg = run.signal_cfg
    rule("运行配置")
    print(f"  数据源      : {run.dataset.source}")
    print(f"  标的指数    : {run.index_col} ({run.dataset.display_name(run.index_col)})")
    print(f"  利率指标    : {', '.join(run.rate_cols)}")
    print(f"  评估区间    : {run.start} ~ {run.end}")
    print(f"  均线        : MA{cfg.short_window} / MA{cfg.long_window}")
    print(f"  重合判定    : mode={cfg.tol_mode}, tol={cfg.tol:g}  ->  {cfg.describe_tol()}")
    print(f"  交易成本    : {run.backtest_cfg.describe()}")
    print("  执行方式    : T 日信号 -> T+1 日收盘价成交；空仓不计利息")
    rule()


__all__ = [
    "RULE_WIDTH",
    "fmt_value",
    "make_backtest_config",
    "make_signal_config",
    "output_dir",
    "parse_grid",
    "parse_rate_cols",
    "print_run_banner",
    "resolve_input",
    "rule",
    "setup_console",
]
