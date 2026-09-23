"""基于 20/120 日简单移动均线的利率择时策略回测工具包。

标的：中债-10年期国债净价(总值)指数（CBA04502.CS）

分层
----
L0 算法层（无内部依赖，可独立单测）：
    ``io_utils`` / ``indicators`` / ``backtest`` / ``metrics`` / ``charts``
L1 编排层：``pipeline``
L2 输出与入口：``report`` / ``cli``

注意：``charts`` 是**惰性导入**的（见下方 ``__getattr__``）。
因为 ``charts`` 在模块级调用 ``matplotlib.use("Agg")``，属于全局副作用，
不应让 ``import ratema`` 就触发。只有真正用到绘图时才加载 matplotlib。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

from .backtest import BacktestConfig, BacktestResult, run_backtest, run_benchmark
from .composite import CompositeResult, build_composite_signal, run_composite
from .daily import DailySnapshot, latest_snapshot
from .indicators import (
    SIGNAL_FLAT,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    TOL_MODES,
    SignalConfig,
    compute_signals,
    spread_calibration,
)
from .io_utils import convert_excel_to_csv, load_dataset
from .journal import Journal, build_journal, render_journal_markdown
from .metrics import perf_stats
from .pipeline import RunResult, StrategyResult, run_all, run_single

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查器
    from .charts import make_all_charts

_LAZY = {"make_all_charts": "charts"}


def __getattr__(name: str) -> Any:
    """PEP 562 惰性属性：把重量级/带全局副作用的模块推迟到真正使用时导入。"""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # 缓存，后续访问不再走 __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__))


__all__ = [
    "SIGNAL_FLAT",
    "SIGNAL_LONG",
    "SIGNAL_SHORT",
    "TOL_MODES",
    "BacktestConfig",
    "BacktestResult",
    "CompositeResult",
    "DailySnapshot",
    "Journal",
    "RunResult",
    "SignalConfig",
    "StrategyResult",
    "__version__",
    "build_composite_signal",
    "build_journal",
    "compute_signals",
    "convert_excel_to_csv",
    "latest_snapshot",
    "load_dataset",
    "make_all_charts",
    "perf_stats",
    "render_journal_markdown",
    "run_all",
    "run_backtest",
    "run_benchmark",
    "run_composite",
    "run_single",
    "spread_calibration",
]
