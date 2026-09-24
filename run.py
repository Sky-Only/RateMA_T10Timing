"""════════════════════════════════════════════════════════════════════════
策略回测 · 统一入口

用法：
     uv run python run.py                 # 用下面的 CONFIG 跑
     uv run python run.py 我的方案A        # 临时改输出文件夹名再跑

所有可调参数都在下面的 CONFIG 里，**只改这一个文件**即可。
每个参数都注明了含义、单位和推荐值，改完直接运行。

产出会写到  <outdir>/<run_name>/  ，例如 output/我的方案A/report.md
════════════════════════════════════════════════════════════════════════"""

from __future__ import annotations

import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════
#  配置区 —— 需要改的都在这
# ══════════════════════════════════════════════════════════════════════════
CONFIG: dict = {
    # ─────────────────────────────── 输出 ───────────────────────────────
    # 输出根目录 + 本次运行的文件夹名。产出落在 <outdir>/<run_name>/
    "outdir": "output",
    "run_name": "long_short",  # ← 改这里换文件夹名，例如 "5bp_多空"
    # ───────────────────────── 要跑哪些步骤 ─────────────────────────────
    # 关掉不需要的步骤可以显著加快（尤其是 sweep）
    "steps": {
        "convert": True,  # Excel → CSV（数据已转换过可关掉）
        "backtest": True,  # 5 个利率指标各自回测 + 图表
        "composite": True,  # 五利率等权打分综合
        "journal": True,  # 每日交易日志
        "sweep": True,  # 重合阈值敏感性扫描
    },
    # ─────────────────────────── 数据来源 ───────────────────────────────
    # None = 自动查找：优先 data/csv/panel.csv，没有则找根目录的 .xlsx
    # 也可写死路径，例如 "中债-10年期国债净价指数与利率数据.xlsx"
    "input": None,
    "csv_outdir": "data/csv",  # convert 步骤的输出目录
    "header_rows": 2,  # Excel 表头行数（中文名 + 代码 = 2 行）
    "index_col": None,  # 标的指数列名，None = 自动识别 CBA04502.CS
    "rate_cols": None,  # 利率指标列，None = 除标的外的全部列
    # 只要其中几个就写列表，例如 ["DR007", "R007"]
    # ─────────────────────────── 回测区间 ───────────────────────────────
    # None = 用全部数据。格式 "YYYY-MM-DD"
    "start": None,  # 例如 "2020-01-01"
    "end": None,  # 例如 "2025-12-31"
    # ────────────────────────── 策略指标 ────────────────────────────────
    "short_window": 20,  # 短均线窗口（交易日）
    "long_window": 120,  # 长均线窗口（交易日），必须大于短均线
    # ────────────────────────── 重合阈值 ────────────────────────────────
    # 规则 3：|MA短 − MA长| ≤ 容差 即视为两线「重合」，当日标记 0，
    #         并延续上一交易日的有效信号（不改变持仓、不产生交易）。
    #
    # 容差怎么算由 tol_mode 决定，**tol 的含义随 tol_mode 变化** —— 这是本文件
    # 最容易踩坑的地方。五种模式列在下面（都折成 bp 便于横向比较）：
    #
    #   "bp"   容差 = tol                 单位「基点」    tol=5     → 5bp
    #   "abs"  容差 = tol                 单位「百分点」  tol=0.05  → 5bp
    #          └ 这两个是同一个东西，只是单位相差 100 倍：abs 的 0.01 = bp 的 1。
    #            实测 abs/0.05 与 bp/5.0 逐日结果完全一致，可互为校验。
    #   "rel"  容差 = |MA120| × tol       tol 是**比例**  tol=0.026 → 约 5bp
    #          └ 容差随利率水平自动伸缩：利率高时容差大、利率低时容差小。
    #            3% 必须写 0.03；写成 3（= 300%）会被校验直接拦下。
    #   "std"  容差 = 滚动标准差(价差, std_window) × tol
    #          └ 随价差**自身波动率**伸缩，与利率绝对水平无关。
    #   "q"    容差 = 滚动 |价差| 的 tol 分位数，tol ∈ [0, 1]
    #          └ 用前 std_window 日滚动估计，重合日占比天然 ≈ tol。
    #
    # 该选哪个？—— 本样本（2011-06-27 ~ 2026-09-16，五利率等权打分，多空双向）
    # 实测：**固定 bp 最好，且最优值在两个时代都落在 5bp**，故默认 bp / 5.0。
    #     bp / 5.0     夏普  全期 1.534 ｜ 2011-2018 1.693 ｜ 2019-2026 1.360
    #     rel / 0.026  夏普  全期 1.422 ｜ 2011-2018 1.499 ｜ 2019-2026 1.349
    #     std / 0.5    夏普  全期 0.998
    #     q / 0.10     夏普  全期 1.177
    #   自适应方案（rel/std/q）都没能超过固定 5bp；std 近期收缩过头，q 恒定
    #   占比反而更不稳。所以默认用最简单的固定 bp，不必为了「自适应」而换。
    #
    # 一个容易误判的点：利率水平从 3.14% 降到 1.92%、价差中位数从 25.5bp 降到
    # 13.3bp，所以**固定 5bp 的相对松紧确实在变松**（重合日占比 20% → 31%）。
    # 但把它收紧到 1~3bp、或改用 rel/std/q，近期表现都更差 —— 低利率时代小价差
    # 多半是噪声，少动是对的。**不要因为「5bp 相对变小了」就去收紧它。**
    #
    # 其它取值：tol=0（只有严格相等才算重合）实测重合日 = 0 天、规则 3 从不触发，
    # 即退化为原始的纯 MA 交叉策略，适合与本文口径做对照。
    "tol_mode": "rel",
    "tol": 0.026,  # rel 下是**比例**：0.026 = 利率水平的 2.6%（≈ 当前的 5bp）
    "std_window": 120,  # "std" / "q" 估计滚动统计量所用的窗口（交易日）
    # ─────────────────────────── 交易设置 ───────────────────────────────
    # 交易方向：
    #   "long_only"  信号 +1 持有多头，-1 空仓（原始策略口径）
    #   "short_only" 信号 -1 持有空头，+1 空仓
    #   "long_short" 信号 +1 持有多头，-1 持有空头（始终有仓位）
    "direction": "long_short",
    # 交易成本。per_side = 单边各收 cost_bps（默认，对应「双边万分之二」）
    #            round_trip = cost_bps 为往返合计，单边各收一半
    "cost_bps": 2.0,
    "cost_mode": "per_side",
    "initial_capital": 1.0,  # 初始资金（只影响净值绝对水平，不影响收益率）
    "annualization": 252,  # 年化交易日数
    "risk_free": 0.0,  # 年化无风险利率（只影响夏普/索提诺）
    # ────────────────────────── 多指标综合 ──────────────────────────────
    # 打分公式：score = (看多票数 − 看空票数) / N
    #   "score"     净票数 > 0 即多头（等价于简单多数）
    #   "majority"  严格过半数为多头
    #   "unanimous" 全票一致才改变观点（最保守，持仓时间最短）
    #   "any"       任一指标看多即持有（最激进，持仓时间最长）
    "composite_mode": "score",
    # 投票来源："eff" 重合的指标沿用前一日观点；"raw" 重合的指标弃权
    "vote_source": "eff",
    # ────────────────────────── 每日交易日志 ────────────────────────────
    "journal_series": "COMPOSITE",  # "COMPOSITE" 或某个指标代码，如 "DR007"
    "journal_tail": 30,  # 日志里打印/记录最近多少个交易日
    "journal_events_only": False,  # True = 日志只列动作日（建仓/平仓）
    "journal_append": None,  # 写台账路径，如 "output/ledger.csv"；同日覆盖
    # ────────────────────────── 阈值敏感性扫描 ──────────────────────────
    # "indicator" 逐指标扫描 / "composite" 扫描等权综合信号
    "sweep_target": "indicator",
    # 网格，逗号分隔的字符串。None = 按 tol_mode 取默认网格
    #   bp  默认 "0,0.5,1,2,5,10"      abs 默认 "0,0.005,0.01,0.02,0.05,0.1"
    #   rel 默认 "0,0.001,0.002,0.005,0.01,0.02"
    #   std 默认 "0,0.1,0.25,0.5,1,2"  q   默认 "0,0.01,0.02,0.05,0.1,0.2"
    "sweep_grid": None,
    # ──────────────────────────── 图表 ──────────────────────────────────
    "charts": True,  # False = 不出图（快很多）
    "signal_series": None,  # 信号机理图用哪个指标，None = 夏普最高者
    # 仅缩放「信号机理图」的横轴范围，例如 ("2020-01-01", "2022-12-31")。
    # 注意：这不是回测区间，不影响任何计算；限制回测区间请用上面的 start / end。
    # 两端都可以写 None，表示该端不设限。
    "signal_window": (None, None),
}
# ══════════════════════════════════════════════════════════════════════════
#  以下为实现，一般不需要修改
# ══════════════════════════════════════════════════════════════════════════

import argparse

from ratema.backtest import COST_MODES, DIRECTIONS
from ratema.commands import COMMANDS
from ratema.commands._shared import rule, setup_console
from ratema.composite import COMPOSITE_MODES, VOTE_SOURCES
from ratema.indicators import TOL_MODES


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def validate(cfg: dict) -> None:
    """把明显写错的参数拦在跑之前，并给出可操作的提示。"""
    problems: list[str] = []

    if cfg["tol_mode"] not in TOL_MODES:
        problems.append(f"tol_mode 必须是 {TOL_MODES} 之一，现在是 {cfg['tol_mode']!r}")
    if cfg["direction"] not in DIRECTIONS:
        problems.append(f"direction 必须是 {DIRECTIONS} 之一，现在是 {cfg['direction']!r}")
    if cfg["cost_mode"] not in COST_MODES:
        problems.append(f"cost_mode 必须是 {COST_MODES} 之一，现在是 {cfg['cost_mode']!r}")
    if cfg["composite_mode"] not in COMPOSITE_MODES:
        problems.append(f"composite_mode 必须是 {COMPOSITE_MODES} 之一")
    if cfg["vote_source"] not in VOTE_SOURCES:
        problems.append(f"vote_source 必须是 {VOTE_SOURCES} 之一")
    if cfg["sweep_target"] not in ("indicator", "composite"):
        problems.append("sweep_target 只能是 'indicator' 或 'composite'")
    if cfg["short_window"] >= cfg["long_window"]:
        problems.append(
            f"short_window({cfg['short_window']}) 必须小于 long_window({cfg['long_window']})"
        )
    if cfg["short_window"] < 1 or cfg["long_window"] < 2:
        problems.append("均线窗口必须为正整数")
    if cfg["cost_bps"] < 0:
        problems.append("cost_bps 不能为负")
    if cfg["initial_capital"] <= 0:
        problems.append("initial_capital 必须为正")
    if cfg["tol"] < 0:
        problems.append("tol 不能为负")
    if cfg["tol_mode"] == "q" and not 0.0 <= cfg["tol"] <= 1.0:
        problems.append('tol_mode="q" 时 tol 必须落在 [0, 1]')
    if cfg["tol_mode"] == "rel" and cfg["tol"] > 1.0:
        problems.append(
            f'tol_mode="rel" 时 tol 是**比例**：想要 3% 要写 0.03，不是 3（现在是 {cfg["tol"]:g}）'
        )
    if not any(cfg["steps"].values()):
        problems.append("steps 里至少要开一个步骤")
    if cfg["start"] and cfg["end"] and str(cfg["start"]) > str(cfg["end"]):
        problems.append(f"start({cfg['start']}) 晚于 end({cfg['end']})")

    if problems:
        rule("配置有误")
        for p in problems:
            print(f"  ✗ {p}")
        print("\n请修改 run.py 顶部的 CONFIG 后重试。")
        raise SystemExit(2)


# --------------------------------------------------------------------------- #
# 组装命令行参数（复用已测试的 commands，保证行为与 CLI 完全一致）
# --------------------------------------------------------------------------- #
def _rate_cols_arg(value) -> str | None:
    """配置里写列表（更好读），命令层要的是逗号分隔字符串。"""
    if not value:
        return None
    if isinstance(value, str):
        return value
    return ",".join(str(c).strip() for c in value if str(c).strip())


def _signal_window_arg(value) -> str | None:
    """配置里写 (start, end) 元组，命令层要的是 "START:END" 字符串。"""
    if not value:
        return None
    start, end = (list(value) + [None, None])[:2]
    if not start and not end:
        return None
    return f"{start or ''}:{end or ''}"


def build_args(cfg: dict) -> argparse.Namespace:
    return argparse.Namespace(
        # 数据
        input=cfg["input"],
        index_col=cfg["index_col"],
        rate_cols=_rate_cols_arg(cfg["rate_cols"]),
        start=cfg["start"],
        end=cfg["end"],
        header_rows=cfg["header_rows"],
        # 信号
        short=cfg["short_window"],
        long=cfg["long_window"],
        tol_mode=cfg["tol_mode"],
        tol=cfg["tol"],
        std_window=cfg["std_window"],
        # 回测
        cost_bps=cfg["cost_bps"],
        cost_mode=cfg["cost_mode"],
        direction=cfg["direction"],
        initial_capital=cfg["initial_capital"],
        annualization=cfg["annualization"],
        risk_free=cfg["risk_free"],
        # 输出
        outdir=cfg["outdir"],
        tag=cfg["run_name"] or None,
        charts=cfg["charts"],
        signal_series=cfg["signal_series"],
        signal_window=_signal_window_arg(cfg["signal_window"]),
        quiet=False,
        # 综合
        mode=cfg["composite_mode"],
        vote_source=cfg["vote_source"],
        # 日志
        series=cfg["journal_series"],
        tail=cfg["journal_tail"],
        events_only=cfg["journal_events_only"],
        append=cfg["journal_append"],
        append_date=None,
        # 扫描
        target=cfg["sweep_target"],
        tol_grid=cfg["sweep_grid"],
        # convert
        all=False,
    )


def resolve_source(cfg: dict, panel: Path) -> str:
    """决定后续步骤读哪个文件：显式指定的 Excel / 已转换的 panel.csv。"""
    given = cfg["input"]
    if given and Path(given).suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return given
    if panel.exists():
        return str(panel)
    return given or str(panel)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    setup_console()
    cfg = dict(CONFIG)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]:
        cfg["run_name"] = argv[0]  # 允许命令行临时改文件夹名

    validate(cfg)

    outdir = Path(cfg["outdir"]) / cfg["run_name"] if cfg["run_name"] else Path(cfg["outdir"])
    panel = Path(cfg["csv_outdir"]) / "panel.csv"

    # ---- 打印生效配置 ---------------------------------------------------- #
    rule("生效配置")
    print(f"  输出目录    : {outdir.resolve()}")
    print(f"  执行步骤    : {', '.join(k for k, v in cfg['steps'].items() if v)}")
    print(f"  数据来源    : {cfg['input'] or '自动查找'}")
    if cfg["rate_cols"]:
        print(f"  利率指标    : {', '.join(cfg['rate_cols'])}")
    print(f"  回测区间    : {cfg['start'] or '起点'} ~ {cfg['end'] or '终点'}")
    print(f"  均线        : MA{cfg['short_window']} / MA{cfg['long_window']}")
    print(f"  重合阈值    : mode={cfg['tol_mode']}, tol={cfg['tol']:g}")

    # 阈值单位提醒：abs 与 bp 相差 100 倍，是这里最容易踩的坑
    if cfg["tol_mode"] == "abs" and cfg["tol"] == 0:
        print("                └─ 严格相等才算重合（原始策略口径）；实测重合日 = 0 天，")
        print("                   规则 3 不会触发。想让它生效可改成 tol_mode='bp', tol=5.0")
    elif cfg["tol_mode"] == "abs":
        print(f"                └─ 即 {cfg['tol'] * 100:g} bp")
    elif cfg["tol_mode"] == "bp":
        print(f"                └─ 即 {cfg['tol'] / 100:g} 个百分点")
    elif cfg["tol_mode"] == "rel":
        print(f"                └─ 即利率水平的 {cfg['tol'] * 100:g}%（容差随利率高低自动伸缩）")
        print("                   注意 tol 是比例：3% 要写 0.03")
    elif cfg["tol_mode"] == "std":
        print(f"                └─ 即滚动标准差的 {cfg['tol']:g} 倍（窗口 {cfg['std_window']} 日）")
    elif cfg["tol_mode"] == "q":
        print(
            f"                └─ 即滚动 {cfg['tol'] * 100:g}% 分位（窗口 {cfg['std_window']} 日）"
        )

    print(f"  交易方向    : {cfg['direction']}")
    print(f"  交易成本    : {cfg['cost_bps']:g}bp / {cfg['cost_mode']}")
    print(
        f"  综合方式    : {cfg['composite_mode']}（投票源 {cfg['vote_source']}）"
        if cfg["steps"]["composite"]
        else "  综合方式    : （未启用）"
    )
    print(f"  图表        : {'开' if cfg['charts'] else '关'}")
    rule()

    # ---- 逐步执行 -------------------------------------------------------- #
    if cfg["steps"]["convert"]:
        # 注意：convert 的 outdir 是「CSV 输出目录」，与回测产出的 outdir 不是一回事，
        # 必须单独构造，否则 CSV 会被误写进 output/
        COMMANDS["convert"](
            argparse.Namespace(
                input=cfg["input"],
                outdir=cfg["csv_outdir"],
                header_rows=cfg["header_rows"],
                all=False,
            )
        )

    source = resolve_source(cfg, panel)
    if not Path(source).exists():
        print(f"\n找不到输入文件：{source}")
        print("请把 Excel 放到项目根目录，或在 CONFIG['input'] 里写死路径。")
        return 2

    args = build_args(cfg)
    args.input = source

    if cfg["steps"]["backtest"]:
        COMMANDS["backtest"](args)

    if cfg["steps"]["composite"]:
        COMMANDS["composite"](args)

    if cfg["steps"]["journal"]:
        COMMANDS["journal"](args)

    if cfg["steps"]["sweep"]:
        COMMANDS["sweep"](args)

    # ---- 收尾 ------------------------------------------------------------ #
    rule("完成")
    print(f"  产出目录 : {outdir.resolve()}")
    print(f"  报告     : {(outdir / 'report.md').resolve()}")
    if cfg["steps"]["composite"]:
        print(f"  综合报告 : {(outdir / 'composite_report.md').resolve()}")
    if cfg["steps"]["journal"]:
        print(f"  交易日志 : {(outdir / 'journal' / 'journal.md').resolve()}")
    if cfg["charts"]:
        print(f"  图表     : {(outdir / 'charts').resolve()}")
    print()
    print("  下一步建议先看：")
    print("    1) report.md 第 4.1 节 —— 分年度收益与四个胜率口径")
    print("    2) charts/09_annual_breakdown.png —— 分年度四联图")
    print("    3) summary.csv —— 5 个指标的绩效汇总表")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
