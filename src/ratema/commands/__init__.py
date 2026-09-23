"""命令行子命令实现。

每个子命令一个模块，只负责「取参数 → 调业务层 → 打印/落盘」，
业务逻辑放在 ``pipeline`` / ``composite`` / ``daily`` / ``journal`` 等层。
"""

from __future__ import annotations

from .all import cmd_all
from .backtest import cmd_backtest
from .composite import cmd_composite
from .convert import cmd_convert
from .journal import cmd_journal
from .signal import cmd_signal
from .sweep import cmd_sweep

COMMANDS = {
    "convert": cmd_convert,
    "backtest": cmd_backtest,
    "signal": cmd_signal,
    "composite": cmd_composite,
    "journal": cmd_journal,
    "sweep": cmd_sweep,
    "all": cmd_all,
}

__all__ = [
    "COMMANDS",
    "cmd_all",
    "cmd_backtest",
    "cmd_composite",
    "cmd_convert",
    "cmd_journal",
    "cmd_signal",
    "cmd_sweep",
]
