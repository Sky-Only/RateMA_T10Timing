"""命令行参数定义。

``--help`` 的组织方式刻意保持「一个文件能看全」，因此所有 subparser 都集中在这里；
具体执行放在 :mod:`ratema.commands`。
"""

from __future__ import annotations

import argparse

from . import __version__
from .backtest import COST_MODES
from .composite import COMPOSITE_MODES, MODE_HELP, VOTE_SOURCE_HELP, VOTE_SOURCES
from .daily import CONSENSUS_HELP, CONSENSUS_MODES
from .indicators import TOL_MODE_HELP, TOL_MODES, SignalConfig

#: 各阈值模式下的默认扫描网格
DEFAULT_TOL_GRID: dict[str, str] = {
    # abs：单位是「百分点」，1bp = 0.01 个百分点 -> 0 / 0.5 / 1 / 2 / 5 / 10 bp
    "abs": "0,0.005,0.01,0.02,0.05,0.1",
    # bp：单位是基点（推荐用这个模式做扫描，不容易搞错量级）
    "bp": "0,0.5,1,2,5,10",
    # rel：相对比例
    "rel": "0,0.001,0.002,0.005,0.01,0.02",
    # std：标准差倍数
    "std": "0,0.1,0.25,0.5,1,2",
    # q：分位数
    "q": "0,0.01,0.02,0.05,0.1,0.2",
}


def default_grid(tol_mode: str) -> str:
    """返回某个阈值模式下的默认扫描网格。"""
    return DEFAULT_TOL_GRID.get(tol_mode, DEFAULT_TOL_GRID["abs"])


# --------------------------------------------------------------------------- #
# 参数组
# --------------------------------------------------------------------------- #
def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="输入文件（.xlsx 或 data/csv/panel.csv）。默认自动查找。",
    )
    parser.add_argument("--index-col", default=None, help="标的指数列名，默认自动识别 CBA04502.CS")
    parser.add_argument(
        "--rate-cols",
        default=None,
        help="底层利率指标列名，逗号分隔。默认为除标的外的全部列。",
    )
    parser.add_argument("--start", default=None, help="回测起始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="回测结束日期 YYYY-MM-DD")


def add_signal_args(parser: argparse.ArgumentParser) -> None:
    # argparse 会对 help 文本做 % 格式化，先把裸 % 转义掉
    mode_help = "；".join(f"{k}: {v}" for k, v in TOL_MODE_HELP.items()).replace("%", "%%")
    # 默认值统一从 SignalConfig 取，避免「改了 dataclass 默认值但命令行不受影响」
    d = SignalConfig()
    parser.add_argument(
        "--short", type=int, default=d.short_window, help=f"短均线窗口，默认 {d.short_window}"
    )
    parser.add_argument(
        "--long", type=int, default=d.long_window, help=f"长均线窗口，默认 {d.long_window}"
    )
    parser.add_argument(
        "--tol-mode",
        choices=TOL_MODES,
        default=d.tol_mode,
        help=f"重合度阈值模式，默认 {d.tol_mode}。" + mode_help,
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=d.tol,
        help=(
            "重合度阈值（含义随 --tol-mode 变化）。"
            "abs: 百分点，0.01=1bp；bp: 基点，1=1bp；rel: 比例，0.01=1%%；"
            "std: 标准差倍数；q: 分位数 0~1。"
            f"默认 {d.tol:g}（{'严格相等才重合' if d.tol == 0 else d.describe_tol()}）"
        ),
    )
    parser.add_argument(
        "--std-window",
        type=int,
        default=d.std_window,
        help=f"std / q 模式下的滚动窗口，默认 {d.std_window}",
    )


def add_backtest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=2.0,
        help="交易成本（基点），默认 2bp = 万分之二",
    )
    parser.add_argument(
        "--cost-mode",
        choices=COST_MODES,
        default="per_side",
        help="per_side: 单边各收 cost-bps（默认）；round_trip: cost-bps 为往返合计",
    )
    parser.add_argument("--annualization", type=int, default=252, help="年化交易日数")
    parser.add_argument("--risk-free", type=float, default=0.0, help="年化无风险利率")


def add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--outdir", default="output", help="输出目录，默认 output")
    parser.add_argument("--tag", default=None, help="输出子目录名（可选）")
    parser.add_argument(
        "--charts",
        dest="charts",
        action="store_true",
        default=True,
        help="生成 PNG 图表（默认开启）",
    )
    parser.add_argument("--no-charts", dest="charts", action="store_false", help="不生成图表")
    parser.add_argument(
        "--signal-series",
        default=None,
        help="信号机理图使用哪个指标，默认取夏普最高者",
    )
    parser.add_argument(
        "--signal-window",
        default=None,
        metavar="START:END",
        help="信号机理图的时间窗口，如 2020-01-01:2022-12-31（默认全区间）",
    )


def parse_signal_window(text: str | None) -> tuple[str | None, str | None]:
    """解析 ``START:END`` 形式的时间窗口。"""
    if not text:
        return (None, None)
    if ":" not in text:
        raise SystemExit("--signal-window 需形如 START:END，例如 2020-01-01:2022-12-31")
    start, end = text.split(":", 1)
    return (start.strip() or None, end.strip() or None)


# --------------------------------------------------------------------------- #
# 顶层
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ratema",
        description="20/120 日均线利率择时策略回测（中债-10年期国债净价指数 CBA04502.CS）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"ratema {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert", help="把文件夹内的 Excel 转成 CSV")
    p_convert.add_argument("input", nargs="?", default=None, help="Excel 文件路径")
    p_convert.add_argument("--outdir", default="data/csv", help="CSV 输出目录")
    p_convert.add_argument("--header-rows", type=int, default=2, help="表头行数，默认 2")
    p_convert.add_argument("--all", action="store_true", help="转换文件夹内所有 Excel")

    p_bt = sub.add_parser("backtest", help="运行回测")
    add_common_data_args(p_bt)
    add_signal_args(p_bt)
    add_backtest_args(p_bt)
    add_output_args(p_bt)
    p_bt.add_argument("--quiet", action="store_true", help="不打印汇总表")

    p_signal = sub.add_parser("signal", help="查看最新一天的交易信号")
    add_common_data_args(p_signal)
    add_signal_args(p_signal)
    p_signal.add_argument(
        "--asof",
        default=None,
        metavar="YYYY-MM-DD",
        help="以该日期为「今天」计算信号，默认数据最后一行",
    )
    p_signal.add_argument(
        "--mode",
        choices=CONSENSUS_MODES,
        default="majority",
        help="多指标合成方式，默认 majority。"
        + "；".join(f"{k}: {v}" for k, v in CONSENSUS_HELP.items()),
    )
    p_signal.add_argument(
        "--max-stale-days",
        type=int,
        default=7,
        help="数据超过多少个自然日未更新则告警（退出码 2），默认 7",
    )
    p_signal.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_comp = sub.add_parser(
        "composite",
        help="两步走：每利率单独测试 + 五利率等权打分综合",
        description=(
            "步骤 1：对每个底层利率指标单独回测；\n"
            "步骤 2：把 N 个指标等权打分（净票数/N）合成一个信号，并额外给出等权组合对照。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_data_args(p_comp)
    add_signal_args(p_comp)
    add_backtest_args(p_comp)
    add_output_args(p_comp)
    p_comp.add_argument(
        "--mode",
        choices=COMPOSITE_MODES,
        default="score",
        help="综合方式，默认 score。" + "；".join(f"{k}: {v}" for k, v in MODE_HELP.items()),
    )
    p_comp.add_argument(
        "--vote-source",
        choices=VOTE_SOURCES,
        default="eff",
        help="投票来源，默认 eff。" + "；".join(f"{k}: {v}" for k, v in VOTE_SOURCE_HELP.items()),
    )
    p_comp.add_argument("--quiet", action="store_true", help="不打印汇总表")

    p_journal = sub.add_parser(
        "journal",
        help="输出每日交易日志（信号 / 仓位 / 动作 / 净值）",
        description=(
            "把策略在每个交易日的状态与动作记成日志：\n"
            "journal.csv 逐日全量、journal_events.csv 动作流水、journal.md 人读版本。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_data_args(p_journal)
    add_signal_args(p_journal)
    add_backtest_args(p_journal)
    add_output_args(p_journal)
    p_journal.add_argument(
        "--series",
        default="COMPOSITE",
        help="日志对象：COMPOSITE（等权综合，默认）或某个指标代码",
    )
    p_journal.add_argument(
        "--mode",
        choices=COMPOSITE_MODES,
        default="score",
        help="综合方式（--series COMPOSITE 时生效）",
    )
    p_journal.add_argument(
        "--vote-source",
        choices=VOTE_SOURCES,
        default="eff",
        help="投票来源（--series COMPOSITE 时生效）",
    )
    p_journal.add_argument("--tail", type=int, default=30, help="打印/记录最近 N 个交易日，默认 30")
    p_journal.add_argument("--events-only", action="store_true", help="只打印动作日")
    p_journal.add_argument("--quiet", action="store_true", help="不打印明细")
    p_journal.add_argument(
        "--append",
        default=None,
        metavar="FILE",
        help="把单日记录追加（或覆盖同日）到指定 CSV 台账",
    )
    p_journal.add_argument(
        "--append-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="配合 --append 指定要追加哪一天，默认最后一天",
    )

    p_sweep = sub.add_parser("sweep", help="重合度阈值敏感性分析")
    add_common_data_args(p_sweep)
    add_signal_args(p_sweep)
    add_backtest_args(p_sweep)
    add_output_args(p_sweep)
    p_sweep.add_argument(
        "--target",
        choices=("indicator", "composite"),
        default="indicator",
        help="扫描对象：indicator 单指标（默认）/ composite 等权综合信号",
    )
    p_sweep.add_argument(
        "--mode", choices=COMPOSITE_MODES, default="score", help="--target composite 时的综合方式"
    )
    p_sweep.add_argument(
        "--vote-source", choices=VOTE_SOURCES, default="eff", help="--target composite 时的投票来源"
    )
    p_sweep.add_argument(
        "--tol-grid",
        default=None,
        help="阈值网格，逗号分隔。默认随 --tol-mode 给出："
        + "；".join(f"{k}->{v}" for k, v in DEFAULT_TOL_GRID.items()),
    )

    p_all = sub.add_parser("all", help="convert + backtest + composite + journal + sweep")
    add_common_data_args(p_all)
    add_signal_args(p_all)
    add_backtest_args(p_all)
    add_output_args(p_all)
    p_all.add_argument("--csv-dir", default="data/csv", help="CSV 中间目录")
    p_all.add_argument("--tol-grid", default=None)
    p_all.add_argument("--skip-convert", action="store_true", help="跳过 Excel->CSV 转换")
    p_all.add_argument(
        "--target",
        choices=("indicator", "composite"),
        default="indicator",
        help="阈值扫描对象，默认 indicator",
    )
    p_all.add_argument(
        "--mode",
        choices=COMPOSITE_MODES,
        default="score",
        help="综合方式（composite / journal 生效）",
    )
    p_all.add_argument(
        "--vote-source",
        choices=VOTE_SOURCES,
        default="eff",
        help="投票来源（composite / journal 生效）",
    )
    p_all.add_argument("--series", default="COMPOSITE", help="日志对象，默认 COMPOSITE")
    p_all.add_argument("--tail", type=int, default=30, help="日志最近 N 日")
    p_all.add_argument("--events-only", action="store_true", help="日志只打印动作日")
    p_all.add_argument("--append", default=None, help="日志追加台账路径")
    p_all.add_argument("--append-date", default=None)
    return parser


__all__ = [
    "DEFAULT_TOL_GRID",
    "add_backtest_args",
    "add_common_data_args",
    "add_output_args",
    "add_signal_args",
    "build_parser",
    "default_grid",
    "parse_signal_window",
]
