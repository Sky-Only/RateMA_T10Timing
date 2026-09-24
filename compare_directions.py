"""════════════════════════════════════════════════════════════════════════
方向对比图 · 独立入口

把「同一套五利率等权打分信号」在三种交易方向下的净值，和买入持有基准
画到**同一张图**上：

    1. 多空双向  long_short   信号 +1 持多头，-1 持空头（始终有仓位）
    2. 仅做多    long_only    信号 +1 持多头，-1 空仓
    3. 仅做空    short_only   信号 -1 持空头，+1 空仓
    4. 基准      买入持有      期初买入并持有到期末（含一次建仓成本）

四者共用**完全相同**的信号与评估区间，所以图上的差异只来自交易方向，
可以直接看出「做空那一段到底贡献了多少」。

用法：
    uv run python compare_directions.py                # 用下面的 CONFIG
    uv run python compare_directions.py 5bp方案         # 换输出文件夹名
    uv run python compare_directions.py --log           # 纵轴取对数
    uv run python compare_directions.py --no-drawdown   # 不出回撤附图

本文件刻意**不依赖 run.py**，可单独运行；参数在下面的 CONFIG 里改。
产出：<outdir>/<name>/compare_directions.png
      <outdir>/<name>/compare_drawdowns.png
      <outdir>/<name>/compare_metrics.csv / .md
═══════════════════════════════════════════════════════════════════════"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════
#  配置区 —— 需要改的都在这
# ══════════════════════════════════════════════════════════════════════════
CONFIG: dict = {
    # ─────────────────────────────── 输出 ───────────────────────────────
    "outdir": "output",
    "name": "direction_compare",  # 产出写到 <outdir>/<name>/
    # ─────────────────────────── 数据来源 ───────────────────────────────
    # None = 自动查找：优先 data/csv/panel.csv，没有则找根目录的 .xlsx
    "input": None,
    "csv_outdir": "data/csv",
    "header_rows": 2,
    "index_col": None,  # None = 自动识别 CBA04502.CS
    "rate_cols": None,  # None = 除标的外的全部列（本项目为 5 个利率）
    # ─────────────────────────── 回测区间 ───────────────────────────────
    "start": None,  # 例如 "2020-01-01"；None = 全部数据
    "end": None,
    # ────────────────────────── 策略指标 ────────────────────────────────
    "short_window": 20,
    "long_window": 120,
    # 重合阈值：|MA短 − MA长| ≤ 容差 即视为重合，标记 0 并延续前一日有效信号。
    # 容差的算法由 tol_mode 决定，**tol 的含义随之变化**（五种模式）：
    #   "bp"   容差 = tol                单位基点    tol=5     → 5bp  ← 默认
    #   "abs"  容差 = tol                单位百分点  tol=0.05  → 5bp（与 bp 差 100 倍）
    #   "rel"  容差 = |MA120| × tol      tol 是比例  0.026 → 约 5bp（随利率水平伸缩）
    #   "std"  容差 = 滚动标准差(价差) × tol
    #   "q"    容差 = 滚动 |价差| 的 tol 分位数，tol ∈ [0, 1]
    # 实测（五利率等权打分，多空双向）：固定 bp/5.0 最好（夏普 1.534），
    # 自适应方案 rel 0.026 为 1.422、q 0.10 为 1.177、std 0.5 为 0.998，均未超过。
    # 完整对比与「为什么不该因为利率下行使去收紧阈值」见 run.py 顶部同一段注释。
    "tol_mode": "bp",
    "tol": 5.0,  # 5bp 以内视为两线重合
    "std_window": 120,  # "std" / "q" 估计滚动统计量所用的窗口（交易日）
    # ────────────────────────── 打分方式 ────────────────────────────────
    # 打分公式：score = (看多票数 − 看空票数) / N
    #   "score"     净票数 > 0 即多头（默认，即通常说的「打分策略」）
    #   "majority"  严格过半数为多头
    #   "unanimous" 全票一致才改变观点
    #   "any"       任一指标看多即持有
    "composite_mode": "score",
    "vote_source": "eff",  # eff = 重合的指标沿用前一日观点；raw = 弃权
    # ─────────────────────────── 交易设置 ───────────────────────────────
    # 成本按「单边各收 cost_bps」计，2.0 对应题目口径的「双边万分之二」
    "cost_bps": 2.0,
    "cost_mode": "per_side",
    "initial_capital": 1.0,  # 图上会归一化到 1.0，改这个不影响图形状
    "annualization": 252,
    "risk_free": 0.0,
    # ──────────────────────────── 绘图 ──────────────────────────────────
    # 出图范围：
    #   "both"         综合 + 每个利率各一张（默认，共 1+N 张）
    #   "composite"    只出综合那一张
    #   "per_indicator" 只出每个利率各一张
    "scope": "both",
    # 画哪一套曲线：
    #   "both"      打分综合（五票合成一个仓位）+ 等权组合（五份资金各跟一个信号）
    #   "signal"    只画打分综合
    #   "portfolio" 只画等权组合
    "curve": "both",
    # 只对其中几个利率出图，例如 ["DR007", "R007"]；None = 全部
    "per_indicator_names": None,
    "log_scale": False,  # True = 纵轴取对数，早期差异看得更清
    "drawdown_chart": True,  # True = 额外出一张综合的回撤对比附图
    "per_indicator_drawdown": False,  # True = 每个利率也各出一张回撤附图
    # 年度收益热力图（仿 charts/05_annual_returns.png 的样式：绿=赚 红=亏）
    "annual_heatmap": True,
    #   "direction"  行 = 三个方向 + 基准（每套曲线一张）
    #   "indicator"  行 = 打分综合 + 等权组合 + 各利率 + 基准（每个方向一张）
    #   "both"       两者都出
    "annual_heatmap_rows": "both",
    # 年度收益 + 年内最大回撤 的数据导出（三方向 × 各口径）
    # 热力图里只有年度收益，回撤不在图上，所以单独出数据文件
    "annual_return_dd_data": True,
    # 三张「年度最大回撤」热力图（三方向各一张，排版与年度收益图一致）
    # 色带：**白 → 红**，0 = 白、最深回撤 = 红
    "annual_dd_heatmap": True,
    # 逐月收益热力图（同样的 8 行 = 打分综合/等权组合/各利率/基准，列 = 1~12 月）
    # 15 年 × 12 月 ≈ 184 列塞不进一张可读的图，因此**按年拆成多个文件**，
    # 每个文件 8 行 × 12 月，并共用同一个颜色范围以保证各年文件可比。
    "monthly_heatmap": True,
    # 逐月表看哪个方向（三个方向 × 每年一张会产出太多文件，故只取一个）
    "monthly_direction": "long_short",
    # 只出其中几年：None = 全部；也可写 [2024, 2025] 或 "2020-2023"
    "monthly_years": None,
    # 单文件版（**横轴 = 月份，纵轴 = 策略种类**）：
    #   "strip"  一张图：8 行（策略）× 全部 184 个月，横向拉通
    #   "panel"  一年一个面板（每个面板 8 策略 × 12 月），16 个面板竖排
    #   "row"    同上，但 16 个面板横排（所有年份在同一行）
    #   "all"    三种都出
    "monthly_layout": "all",
    # 一张图那版的**单格边长**（英寸）。格子是正方形，所以整图 ≈ 23:1 的长条；
    # 调大即整体等比放大、格里数字更清楚（0.15 时整图约 31×2.8 英寸）。
    "monthly_cell_inch": 0.15,
    "figsize": (13.5, 7.2),
}
# ══════════════════════════════════════════════════════════════════════════
#  以下为实现，一般不需要修改
# ══════════════════════════════════════════════════════════════════════════

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from ratema.backtest import COST_MODES, DIRECTIONS, BacktestConfig
from ratema.charts import ensure_style
from ratema.commands._shared import rule, setup_console
from ratema.composite import COMPOSITE_MODES, VOTE_SOURCES, run_composite
from ratema.indicators import TOL_MODES, SignalConfig
from ratema.io_utils import find_default_input, load_dataset
from ratema.pipeline import run_all

#: 三条曲线 = 三个交易方向。顺序即图例顺序：主线在最前。
#: color / width / style 都写死在这里，保证每张图配色一致、可跨图对比。
#: 每项为 (方向键, 图例名, 颜色, 线型, 线宽, 说明)。
DIRECTION_SPECS = (
    # key           图例名        颜色       线型    线宽   说明
    ("long_short", "多空双向", "#1F4E79", "-", 2.4, "信号 +1 持多头 / −1 持空头"),
    ("long_only", "仅做多", "#C0392B", "-", 1.8, "信号 +1 持多头 / −1 空仓"),
    ("short_only", "仅做空", "#1E8449", "-", 1.8, "信号 −1 持空头 / +1 空仓"),
)
BENCH_LABEL = "基准 买入持有"
BENCH_COLOR = "#7F8C8D"
BENCH_STYLE = "--"
BENCH_WIDTH = 2.0
GRID = {"alpha": 0.25, "linestyle": "-", "linewidth": 0.6}


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def validate(cfg: dict) -> None:
    """把明显写错的参数拦在跑之前。"""
    problems: list[str] = []
    if cfg["tol_mode"] not in TOL_MODES:
        problems.append(f"tol_mode 必须是 {TOL_MODES} 之一，现在是 {cfg['tol_mode']!r}")
    if cfg["composite_mode"] not in COMPOSITE_MODES:
        problems.append(f"composite_mode 必须是 {COMPOSITE_MODES} 之一")
    if cfg["curve"] not in ("both", "signal", "portfolio"):
        problems.append('curve 必须是 "both" / "signal" / "portfolio" 之一')
    if cfg["annual_heatmap_rows"] not in ("direction", "indicator", "both"):
        problems.append('annual_heatmap_rows 必须是 "direction" / "indicator" / "both" 之一')
    if cfg["monthly_direction"] not in [d for d, *_ in DIRECTION_SPECS]:
        problems.append(f"monthly_direction 必须是 {[d for d, *_ in DIRECTION_SPECS]} 之一")
    if cfg["monthly_layout"] not in ("strip", "panel", "row", "all"):
        problems.append('monthly_layout 必须是 "strip" / "panel" / "row" / "all" 之一')
    if not 0.05 <= cfg["monthly_cell_inch"] <= 0.6:
        problems.append("monthly_cell_inch 应在 0.05 ~ 0.6 英寸之间")
    if cfg["scope"] not in ("both", "composite", "per_indicator"):
        problems.append('scope 必须是 "both" / "composite" / "per_indicator" 之一')
    if cfg["vote_source"] not in VOTE_SOURCES:
        problems.append(f"vote_source 必须是 {VOTE_SOURCES} 之一")
    if cfg["cost_mode"] not in COST_MODES:
        problems.append(f"cost_mode 必须是 {COST_MODES} 之一")
    if cfg["short_window"] >= cfg["long_window"]:
        problems.append(
            f"short_window({cfg['short_window']}) 必须小于 long_window({cfg['long_window']})"
        )
    if cfg["tol"] < 0:
        problems.append("tol 不能为负")
    if cfg["tol_mode"] == "q" and not 0.0 <= cfg["tol"] <= 1.0:
        problems.append('tol_mode="q" 时 tol 必须落在 [0, 1]')
    if cfg["tol_mode"] == "rel" and cfg["tol"] > 1.0:
        problems.append(
            f'tol_mode="rel" 时 tol 是**比例**：想要 3% 要写 0.03，不是 3（现在是 {cfg["tol"]:g}）'
        )
    if cfg["cost_bps"] < 0:
        problems.append("cost_bps 不能为负")
    if cfg["initial_capital"] <= 0:
        problems.append("initial_capital 必须为正")
    if cfg["start"] and cfg["end"] and str(cfg["start"]) > str(cfg["end"]):
        problems.append(f"start({cfg['start']}) 晚于 end({cfg['end']})")
    for direction, *_ in DIRECTION_SPECS:
        if direction not in DIRECTIONS:
            problems.append(f"内部错误：未知方向 {direction!r}")

    if problems:
        rule("配置有误")
        for p in problems:
            print(f"  x {p}")
        print("\n请修改 compare_directions.py 顶部的 CONFIG 后重试。")
        raise SystemExit(2)


# --------------------------------------------------------------------------- #
# 数据与回测
# --------------------------------------------------------------------------- #
def resolve_source(cfg: dict) -> str:
    """决定读哪个文件：显式指定的 Excel / 已转换的 panel.csv。"""
    given = cfg["input"]
    panel = Path(cfg["csv_outdir"]) / "panel.csv"
    if given and Path(given).suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return given
    if panel.exists():
        return str(panel)
    if given:
        return given
    try:
        return str(find_default_input("."))
    except SystemExit:
        return str(panel)


def make_backtest_config(cfg: dict, direction: str) -> BacktestConfig:
    return BacktestConfig(
        cost_bps=cfg["cost_bps"],
        cost_mode=cfg["cost_mode"],
        direction=direction,
        initial_capital=cfg["initial_capital"],
        annualization=cfg["annualization"],
        risk_free=cfg["risk_free"],
    )


def _num(metrics: dict, key: str) -> float:
    v = metrics.get(key)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if f == f else float("nan")  # NaN 归一


#: run_composite 同时产出两套曲线，两套的「方向」语义不同，必须分别取用：
#:   signal    —— 五票合成一个仓位（composite_frame）
#:   portfolio —— 五份资金各跟一个信号后等权平均（portfolio_frame）
FAMILY_FRAME = {"signal": "composite_frame", "portfolio": "portfolio_frame"}
FAMILY_METRICS = {"signal": "composite_metrics", "portfolio": "portfolio_metrics"}
FAMILY_LABEL = {
    "signal": "综合信号（五票合成一个仓位）",
    "portfolio": "等权组合（五份资金各跟一个信号）",
}
FAMILY_TITLE = {
    "signal": "五利率等权打分策略 · 三种交易方向 vs 买入持有",
    "portfolio": "五利率等权组合（各 1/5 资金）· 三种交易方向 vs 买入持有",
}
FAMILY_FILE = {"signal": "compare_directions", "portfolio": "compare_portfolio"}


def family_frame(res, family: str):
    return getattr(res, FAMILY_FRAME[family])


def family_metrics(res, family: str) -> dict:
    return getattr(res, FAMILY_METRICS[family])


def compute_direction_bundle(
    dataset,
    per_indicator_by_direction,
    cfg: dict,
    *,
    index_col: str | None = None,
    signal_label: str = "",
) -> dict:
    """给定「每个方向各自的一组指标回测结果」，跑出三个方向的净值。

    综合图与单利率图走的是**同一条代码路径**：单利率只需传一个只含该指标的
    列表。N=1 时 ``score = (看多票数 − 看空票数) / 1 ∈ {−1, +1}``，
    与「该指标自己的 signal_eff」逐日等价，因此单利率图不需要另写一套逻辑，
    也就自动继承了同一套对齐校验。

    参数必须是 **按方向分开** 的 ``{方向: [StrategyResult, ...]}``，不能三个方向
    共用同一份。原因：``portfolio_frame``（等权组合）是把各指标的净值曲线做
    等权平均，而每条净值曲线都是由该指标在**该方向**下的回测得到的。
    若三个方向共用同一份（例如都用 long_short 的结果），等权组合会变成
    「三个方向完全相同的一条曲线」，图上看起来正常、其实是错的。
    """
    expected = [d for d, *_ in DIRECTION_SPECS]
    missing = [d for d in expected if d not in per_indicator_by_direction]
    if missing:
        raise ValueError(f"per_indicator_by_direction 缺少方向：{missing}")

    results: dict = {}
    for direction in expected:
        per_indicator = list(per_indicator_by_direction[direction])
        if not per_indicator:
            raise ValueError(f"{direction} 的 per_indicator 不能为空")
        results[direction] = run_composite(
            dataset,
            per_indicator,
            make_backtest_config(cfg, direction),
            index_col=index_col,
            mode=cfg["composite_mode"],
            vote_source=cfg["vote_source"],
            start=cfg["start"],
            end=cfg["end"],
        )

    for family in ("signal", "portfolio"):
        _check_alignment(results, family)
    series = [r.series for r in per_indicator_by_direction[expected[0]]]
    return {
        "by_direction": results,
        "series": series,
        "signal_label": signal_label,
        "is_composite": len(series) > 1,
    }


def compute_all_directions(dataset, cfg: dict) -> dict:
    """同一个打分信号跑三个方向，返回 {方向: CompositeResult} 与基准。

    关键点：三个方向必须共用同一份信号和同一个评估区间，否则图上的
    差异就说不清是「方向不同」还是「窗口不同」。这里显式校验对齐。

    返回的 bundle 里，**综合信号**与**等权组合**两套曲线都是方向正确的：
    每个方向都单独跑了一遍 ``run_all``，因为等权组合是对各指标的净值曲线
    求平均，而净值曲线本身依赖交易方向。信号只取决于均线、与方向无关，
    因此三个方向拿到的 ``signal_eff`` 仍然逐日相同（下面会断言这一点）。
    """
    signal_cfg = SignalConfig(
        short_window=cfg["short_window"],
        long_window=cfg["long_window"],
        tol_mode=cfg["tol_mode"],
        tol=cfg["tol"],
        std_window=cfg["std_window"],
    )
    # 每个方向各跑一次：等权组合需要「该方向下」各指标的净值曲线
    per_direction: dict = {}
    for direction, *_ in DIRECTION_SPECS:
        per_direction[direction] = run_all(
            dataset,
            signal_cfg,
            make_backtest_config(cfg, direction),
            index_col=cfg["index_col"],
            rate_cols=cfg["rate_cols"],
            start=cfg["start"],
            end=cfg["end"],
        )

    base = per_direction[DIRECTION_SPECS[0][0]]
    n = len(base.rate_cols)
    _check_signals_direction_free(per_direction)

    bundle = compute_direction_bundle(
        dataset,
        {d: b.results for d, b in per_direction.items()},
        cfg,
        index_col=base.index_col,
        signal_label=f"五利率等权打分（{cfg['composite_mode']} / 投票源 {cfg['vote_source']}）",
    )
    bundle["base"] = base
    bundle["per_direction_runs"] = per_direction
    # 每个利率各自一张图：只把该指标一个元素喂进去，N=1 即它自己的信号。
    # 注意 N=1 时「等权组合」与「综合信号」是同一条曲线（组合=单条净值），
    # 所以单利率只出 signal 一套，不重复出 portfolio。
    bundle["per_indicator"] = [
        {
            "series": series,
            "display": dataset.display_name(series),
            "bundle": compute_direction_bundle(
                dataset,
                {d: [b.results[i]] for d, b in per_direction.items()},
                cfg,
                index_col=base.index_col,
                signal_label=(f"单利率信号 · {series}（{dataset.display_name(series)}）"),
            ),
        }
        for i, series in enumerate(base.rate_cols)
    ]
    bundle["n_indicators"] = n
    return bundle


def _check_signals_direction_free(per_direction: dict) -> None:
    """信号必须与交易方向无关 —— 这是「三方向共用同一信号」的前提。

    每个方向都单独跑了一次 run_all，所以这里显式验证三次得到的
    ``signal_eff`` 逐日完全相同。若哪天信号开始依赖方向（例如加入
    做空专属过滤），这张对比图的归因就不再成立，必须立刻报错。
    """
    ref_dir = None
    ref = None
    for direction, base in per_direction.items():
        sig = [r.signals["signal_eff"].reset_index(drop=True) for r in base.results]
        if ref is None:
            ref_dir, ref = direction, sig
            continue
        for i, s in enumerate(sig):
            if not s.equals(ref[i]):
                raise ValueError(
                    f"{direction} 第 {i + 1} 个指标的信号与 {ref_dir} 不同："
                    "信号不再与交易方向无关，三方向共用一个信号的前提被破坏"
                )


def _check_alignment(results: dict, family: str = "signal") -> None:
    """三个方向必须逐日对齐，且基准曲线完全相同。"""
    attr = FAMILY_FRAME[family]
    reference = None
    for direction, res in results.items():
        f = getattr(res, attr)
        key = (
            len(f),
            str(f["date"].iloc[0].date()),
            str(f["date"].iloc[-1].date()),
        )
        if reference is None:
            reference = (direction, key, f)
            continue
        ref_dir, ref_key, ref_f = reference
        if key != ref_key:
            raise ValueError(
                f"[{family}] {direction} 与 {ref_dir} 的评估区间不一致：{key} vs {ref_key}；"
                "对比图必须建立在同一区间上，请检查 start/end 与数据完整性"
            )
        if not f["date"].reset_index(drop=True).equals(ref_f["date"].reset_index(drop=True)):
            raise ValueError(f"{direction} 与 {ref_dir} 的交易日不完全一致，无法同图对比")
        if (
            not f["benchmark_equity"]
            .reset_index(drop=True)
            .equals(ref_f["benchmark_equity"].reset_index(drop=True))
        ):
            raise ValueError(f"{direction} 与 {ref_dir} 的基准净值不一致")


def build_panels(bundle: dict, cfg: dict, family: str = "signal") -> list[dict]:
    """把三条策略 + 基准整理成绘图用的行。

    ``family`` 决定画哪一套曲线：``signal`` = 综合信号，``portfolio`` = 等权组合。
    """
    if family not in FAMILY_FRAME:
        raise ValueError(f"未知 family: {family!r}，应为 {tuple(FAMILY_FRAME)}")
    scale = float(cfg["initial_capital"])
    by_direction = bundle["by_direction"]
    any_res = next(iter(by_direction.values()))
    n_days = len(family_frame(any_res, family))

    panels: list[dict] = []
    for direction, label, color, style, width, _desc in DIRECTION_SPECS:
        res = by_direction[direction]
        m = family_metrics(res, family)
        frame = family_frame(res, family)
        panels.append(
            {
                "kind": "strategy",
                "label": label,
                "direction": direction,
                "color": color,
                "style": style,
                "width": width,
                "date": frame["date"],
                "equity": frame["equity"] / scale,
                "metrics": m,
                "cagr": _num(m, "strategy_cagr"),
                "sharpe": _num(m, "strategy_sharpe"),
                "mdd": _num(m, "strategy_max_drawdown"),
                "final": _num(m, "strategy_end_value") / scale,
            }
        )

    bm = family_metrics(any_res, family)
    bench = family_frame(any_res, family)
    panels.append(
        {
            "kind": "benchmark",
            "label": BENCH_LABEL,
            "direction": None,
            "color": BENCH_COLOR,
            "style": BENCH_STYLE,
            "width": BENCH_WIDTH,
            "date": bench["date"],
            "equity": bench["benchmark_equity"] / scale,
            "metrics": bm,
            "cagr": _num(bm, "benchmark_cagr"),
            "sharpe": _num(bm, "benchmark_sharpe"),
            "mdd": _num(bm, "benchmark_max_drawdown"),
            "final": _num(bm, "benchmark_end_value") / scale,
        }
    )
    for p in panels:
        p["n_days"] = n_days
    return panels


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def _cost_text(cfg: dict) -> str:
    """把成本说成人话：per_side 就是「单边各收」。"""
    side = "单边各收" if cfg["cost_mode"] == "per_side" else "往返合计"
    return f"成本 {cfg['cost_bps']:g}bp（{side}）"


def _subtitle(cfg: dict, n_days: int, start, end, signal_label: str) -> str:
    return (
        f"信号：{signal_label}"
        f"    MA{cfg['short_window']}/{cfg['long_window']}"
        f"    重合阈值 {cfg['tol']:g} {cfg['tol_mode']}"
        f"    {_cost_text(cfg)}\n"
        f"{start} ~ {end}（{n_days} 个交易日）"
        f"    四条曲线共用同一信号与同一区间，差异只来自交易方向"
    )


def _legend_text(p: dict) -> str:
    """图例自带关键数字，图单独拿出去也能看懂。"""
    return (
        f"{p['label']}   净值 {p['final']:.4f}"
        f"   年化 {p['cagr'] * 100:+.2f}%"
        f"   夏普 {p['sharpe']:.3f}"
        f"   最大回撤 {p['mdd'] * 100:.2f}%"
    )


def _finalize_axes(ax, cfg: dict) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, axis="x", **GRID)
    if cfg["log_scale"]:
        ax.set_yscale("log")


def plot_equity_comparison(
    panels: list[dict],
    outdir: Path,
    cfg: dict,
    *,
    signal_label: str,
    title: str,
    filename: str,
) -> Path:
    """主图：四条净值曲线画在同一张图上（综合与单利率共用本函数）。"""
    ensure_style()
    fig, ax = plt.subplots(figsize=tuple(cfg["figsize"]))

    first = panels[0]
    for p in panels:
        ax.plot(
            p["date"],
            p["equity"],
            color=p["color"],
            linestyle=p["style"],
            linewidth=p["width"],
            label=_legend_text(p),
            zorder=3 if p["kind"] == "benchmark" else 4,
        )

    ax.axhline(1.0, color="#333333", linewidth=0.9, alpha=0.6, zorder=1)
    ax.set_ylabel("净值（起点 = 1.0，含双边成本）")
    ax.set_title(
        f"{title}\n"
        + _subtitle(
            cfg,
            first["n_days"],
            first["date"].iloc[0].date(),
            first["date"].iloc[-1].date(),
            signal_label,
        ),
        loc="left",
    )
    _finalize_axes(ax, cfg)
    ax.legend(loc="upper left", ncols=1, fontsize=9.5)
    fig.autofmt_xdate()
    return _save(fig, outdir / filename)


def plot_drawdown_comparison(
    panels: list[dict],
    outdir: Path,
    cfg: dict,
    *,
    title: str = "回撤对比 · 三种交易方向 vs 买入持有",
    filename: str = "compare_drawdowns.png",
) -> Path:
    """附图：回撤对比 —— 看收益之外，代价是什么。"""
    ensure_style()
    width, height = cfg["figsize"]
    fig, ax = plt.subplots(figsize=(width, max(4.6, height * 0.64)))

    for p in panels:
        dd = p["equity"] / p["equity"].cummax() - 1.0
        ax.plot(
            p["date"],
            dd * 100.0,
            color=p["color"],
            linestyle=p["style"],
            linewidth=p["width"] * 0.85,
            label=f"{p['label']}   最大回撤 {p['mdd'] * 100:.2f}%",
            zorder=3 if p["kind"] == "benchmark" else 4,
        )
    ax.axhline(0.0, color="#333333", linewidth=0.9, alpha=0.6)
    ax.set_ylabel("回撤（%）")
    ax.set_title(title, loc="left")
    _finalize_axes(ax, cfg)
    ax.legend(loc="lower left", ncols=2, fontsize=9.5)
    fig.autofmt_xdate()
    return _save(fig, outdir / filename)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

from matplotlib.colors import ListedColormap
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    return ListedColormap(cmap(np.linspace(minval, maxval, n)))
cmap_full = plt.get_cmap("RdYlGn")

# --------------------------------------------------------------------------- #
# 年度收益热力图（仿 charts/05_annual_returns.png）
# --------------------------------------------------------------------------- #
ANNUAL_CMAP = "RdYlGn"  # 与 05_annual_returns.png 同一色带：绿=正 红=负

#: 回撤专用色带。列表顺序 = **归一化参数 t 的顺序**，不是「数值从大到小」：
#:   t=0（对应 vmin = 最深回撤）→ 纯红 #A50026
#:   t=1（对应 vmax = 0）       → 纯白 #FFFFFF
#: 视觉上就是「白 → 红」：**0 是白、越深越红**。
#: 深端取 #A50026，与 RdYlGn 的红端一致 —— 回撤图最深的红与收益图最差的红
#: 是同一种红，两张图并排看不会串味。
#:
#: 注意别把列表写成 ["#FFFFFF", "#A50026"]：那样 t=0 变白，而 t=0 恰恰是
#: 最深回撤，整张图会反过来 —— 0 回撤染成深红、最深的回撤染成白。
DD_CMAP = truncate_colormap(cmap_full, minval=0, maxval=0.5)

def annual_from_equity(equity: pd.Series) -> pd.Series:
    """把一条净值曲线转成逐年收益（年末环比）。

    首年用「评估起点净值」作基期，否则第一年只有半年数据却按整年显示。
    注意不能用「年内首末」，那会漏掉年初第一天的涨跌。
    """
    eq = pd.Series(equity).astype("float64")
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index().dropna()
    if eq.empty:
        return pd.Series(dtype="float64")
    year_end = eq.resample("YE").last()
    base_index = eq.index[0] - pd.Timedelta(days=1)
    chained = pd.concat([pd.Series([eq.iloc[0]], index=[base_index]), year_end])
    out = chained.pct_change().dropna()
    out.index = out.index.year
    return out


def annual_matrix(panels: list[dict]) -> pd.DataFrame:
    """行 = 各条曲线，列 = 年份。直接吃 build_panels 的输出。"""
    rows: dict[str, pd.Series] = {}
    for p in panels:
        s = pd.Series(p["equity"].to_numpy(), index=pd.to_datetime(p["date"]))
        rows[p["label"]] = annual_from_equity(s)
    df = pd.DataFrame(rows).T
    return df.reindex(sorted(df.columns), axis=1)


def indicator_equity_map(bundle: dict, cfg: dict, direction: str) -> dict[str, pd.Series]:
    """行 = 打分综合 + 等权组合 + 各利率 + 基准 → 各自的净值曲线。

    年度收益表与年度回撤表共用这一份，保证两张表的行完全对齐、不会各算各的。
    各利率那一行只列一次：N=1 时两种综合口径是同一条曲线。
    """

    def _curve(frame) -> pd.Series:
        return pd.Series(frame["equity"].to_numpy(), index=pd.to_datetime(frame["date"]))

    res = bundle["by_direction"][direction]
    rows: dict[str, pd.Series] = {
        COMPOSITE_ROW_LABEL: _curve(res.composite_frame),
        PORTFOLIO_ROW_LABEL: _curve(res.portfolio_frame),
    }
    for item in bundle["per_indicator"]:
        sub = item["bundle"]["by_direction"][direction].composite_frame
        rows[item["series"]] = _curve(sub)
    comp = res.composite_frame
    rows[BENCH_LABEL] = pd.Series(
        comp["benchmark_equity"].to_numpy(), index=pd.to_datetime(comp["date"])
    )
    return rows


def annual_drawdown_from_equity(equity: pd.Series, *, include_year_start: bool = True) -> pd.Series:
    """**年内**最大回撤：只看该年自身的峰谷，不跨年累计。

    ``include_year_start=True``（默认）把**上一年最后一个净值**当作该年路径的
    起点，于是「年初第一个交易日的涨跌」也算进这一年的回撤 —— 这样回撤与年度
    收益覆盖同一段时间（都是上年末 → 本年末），两者可以逐格对照。

    ``include_year_start=False`` 从该年**第一个交易日收盘**起算，与
    ``metrics.annual_breakdown`` 的 ``strategy_max_drawdown`` 完全一致
    （它窗口内净值是 ``(1+ret).cumprod()``，首个收益被当成基期）。
    两种口径只在「年初第一天是负收益」时有差，实测最大 0.44 个百分点。
    """
    eq = pd.Series(equity).astype("float64")
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index().dropna()
    out: dict[int, float] = {}
    for year, grp in eq.groupby(eq.index.year):
        if include_year_start:
            prior = eq[eq.index < grp.index[0]]
            seg = pd.concat([prior.tail(1), grp]) if len(prior) else grp
        else:
            seg = grp
        out[int(year)] = float((seg / seg.cummax() - 1.0).min())
    return pd.Series(out)


def indicator_annual_matrix(bundle: dict, cfg: dict, direction: str) -> pd.DataFrame:
    """行 = 打分综合 + 等权组合 + 各利率 + 基准，列 = 年份（指定方向下）。

    两个综合口径都要列，且标签必须写清楚 —— 它们不是一回事：
    打分综合是五票合成一个仓位，等权组合是五份资金各跟一个信号。
    """
    eqs = indicator_equity_map(bundle, cfg, direction)
    df = pd.DataFrame({k: annual_from_equity(v) for k, v in eqs.items()}).T
    return df.reindex(sorted(df.columns), axis=1)


def indicator_annual_drawdown_matrix(bundle: dict, cfg: dict, direction: str) -> pd.DataFrame:
    """行/列与 :func:`indicator_annual_matrix` 完全一致，值为**年内最大回撤**。"""
    eqs = indicator_equity_map(bundle, cfg, direction)
    df = pd.DataFrame({k: annual_drawdown_from_equity(v) for k, v in eqs.items()}).T
    return df.reindex(sorted(df.columns), axis=1)


def heatmap_color_range(limit: float, *, symmetric: bool = True) -> tuple[float, float]:
    """热力图的颜色范围（配合 ``cmap`` 决定「哪个值是什么颜色」）。

    ``symmetric=True``（收益 + ``RdYlGn``）：``[-limit, +limit]``，
    0 居中，绿正红负。

    ``symmetric=False``（回撤 + ``DD_CMAP``）：``[-limit, 0]``。
    配合白→红色带即：**0 = 纯白（没有回撤）、-limit = 纯红（最深）**。
    回撤全是非正数，若还用对称色带，一半的色域永远用不上。
    """
    limit = max(float(limit), 1e-6)
    return (-limit, limit) if symmetric else (-limit, 0.0)


def plot_annual_heatmap(
    df: pd.DataFrame,
    outdir: Path,
    *,
    title: str,
    filename: str,
    cbar_label: str = "年度收益",
    limit: float | None = None,
    fontsize: float = 8,
    figsize: tuple[float, float] | None = None,
    symmetric: bool = True,
    cmap=None,
) -> Path:
    """年度热力图 —— 收益图样式对齐 charts/05_annual_returns.png。

    两套用法：

    - **收益**（默认）：``cmap=RdYlGn`` + ``symmetric=True``，
      颜色以 0 为中心对称，绿正红负。
    - **回撤**：``cmap=DD_CMAP`` + ``symmetric=False``，
      颜色范围 ``[-limit, 0]``，配白→红色带即
      **0 = 纯白（没有回撤）、越深越红**。

    格子内标数值（%，不带百分号，与 05 一致）。

    ``limit`` 可显式指定颜色范围；拆成多个文件的那几套图必须共用同一个
    ``limit``，否则各文件之间颜色深浅不可比。
    """
    ensure_style()
    if cmap is None:
        cmap = ANNUAL_CMAP
    if figsize is None:
        figsize = (13.5, 1.0 + 0.62 * len(df))
    fig, ax = plt.subplots(figsize=figsize)
    data = df.to_numpy(dtype="float64")
    finite = np.isfinite(data)
    if limit is None:
        limit = float(np.nanmax(np.abs(data))) if finite.any() else 1.0
    limit = max(float(limit), 1e-6)

    vmin, vmax = heatmap_color_range(limit, symmetric=symmetric)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(df.columns)), [str(c) for c in df.columns], rotation=45)
    ax.set_yticks(range(len(df.index)), list(df.index))
    ax.grid(False)  # 全局样式开了网格，热力图必须关掉
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v * 100:.1f}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="#111111",
                )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7, label=cbar_label)
    fig.tight_layout()
    return _save(fig, outdir / filename)


# --------------------------------------------------------------------------- #
# 逐月收益热力图（同样的 8 行，列=1~12 月）
# --------------------------------------------------------------------------- #
MONTH_LABELS = [f"{m}月" for m in range(1, 13)]


def monthly_from_equity(equity: pd.Series) -> pd.Series:
    """把一条净值曲线转成逐月收益（月末环比），index 为 PeriodIndex('M')。

    与年度口径同源：首月用「评估起点净值」作基期，因此不会漏掉月初的涨跌。
    """
    eq = pd.Series(equity).astype("float64")
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index().dropna()
    if eq.empty:
        return pd.Series(dtype="float64")
    month_end = eq.resample("ME").last()
    base_index = eq.index[0] - pd.Timedelta(days=1)
    chained = pd.concat([pd.Series([eq.iloc[0]], index=[base_index]), month_end])
    out = chained.pct_change().dropna()
    out.index = out.index.to_period("M")
    return out


def monthly_matrix(bundle: dict, cfg: dict, direction: str) -> pd.DataFrame:
    """行 = 打分综合 + 等权组合 + 各利率 + 基准，列 = 全部月份（与年度表同样的 8 行）。"""
    eqs = indicator_equity_map(bundle, cfg, direction)
    df = pd.DataFrame({k: monthly_from_equity(v) for k, v in eqs.items()}).T
    return df.reindex(sorted(df.columns), axis=1)


def monthly_year_frame(matrix: pd.DataFrame, year: int) -> pd.DataFrame:
    """从逐月矩阵里取出某一年，列补齐为 1~12 月（缺月留空）。"""
    cols = [p for p in matrix.columns if p.year == year]
    sub = matrix.loc[:, cols]
    sub.columns = [f"{p.month}月" for p in cols]
    return sub.reindex(columns=MONTH_LABELS)


def monthly_color_limit(matrix: pd.DataFrame) -> float:
    """所有年份共用的颜色范围 —— 否则各年文件的深浅不可比。"""
    data = matrix.to_numpy(dtype="float64")
    if not np.isfinite(data).any():
        return 1e-6
    return max(float(np.nanmax(np.abs(data))), 1e-6)


def monthly_years(matrix: pd.DataFrame, cfg: dict) -> list[int]:
    """要出图的年份；``monthly_years`` 可写 [2024, 2025] 或 "2020-2023"。"""
    available = sorted({p.year for p in matrix.columns})
    spec = cfg.get("monthly_years")
    if not spec:
        return available
    if isinstance(spec, str):
        if "-" in spec:
            lo, hi = (int(x) for x in spec.split("-", 1))
            wanted = [y for y in available if lo <= y <= hi]
        else:
            wanted = [int(spec)]
    else:
        wanted = [int(y) for y in spec]
    unknown = [y for y in wanted if y not in available]
    if unknown:
        raise SystemExit(f"monthly_years 里有数据里没有的年份：{unknown}；可用 {available}")
    return wanted


# --------------------------------------------------------------------------- #
# 单文件版：行 = 年，列 = 12 个月
# --------------------------------------------------------------------------- #
def monthly_year_by_month(matrix: pd.DataFrame, label: str) -> pd.DataFrame:
    """把某个口径的逐月序列转成 行=年、列=1~12月 的表。"""
    series = matrix.loc[label]
    lookup = {p: float(v) for p, v in series.items()}
    years = sorted({p.year for p in matrix.columns})
    data = np.full((len(years), 12), np.nan)
    for i, year in enumerate(years):
        for month in range(1, 13):
            data[i, month - 1] = lookup.get(pd.Period(f"{year}-{month:02d}", freq="M"), np.nan)
    return pd.DataFrame(data, index=years, columns=MONTH_LABELS)


def _annotate_cells(ax, data: np.ndarray, *, fontsize: float) -> None:
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v * 100:.1f}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="#111111",
                )


#: 横条图四周留给标题 / 轴标签 / 色条的边距（英寸）
STRIP_MARGINS = {"left": 1.25, "right": 1.35, "top": 0.80, "bottom": 0.95}


def strip_geometry(n_rows: int, n_cols: int, cell_inch: float):
    """算出「每格严格为 cell_inch 正方形」所需的画布尺寸与坐标区位置。

    返回 ``((fig_w, fig_h), axes_rect, cax_rect)``，三者都是英寸/比例，
    不依赖 matplotlib，便于直接断言格子确实是正方形。
    """
    m = STRIP_MARGINS
    data_w = cell_inch * n_cols
    data_h = cell_inch * n_rows
    fig_w = m["left"] + data_w + m["right"]
    fig_h = m["top"] + data_h + m["bottom"]
    axes_rect = [m["left"] / fig_w, m["bottom"] / fig_h, data_w / fig_w, data_h / fig_h]
    cax_rect = [
        (m["left"] + data_w + 0.35) / fig_w,
        m["bottom"] / fig_h,
        0.18 / fig_w,
        data_h / fig_h,
    ]
    return (fig_w, fig_h), axes_rect, cax_rect


def plot_monthly_strategy_strip(
    matrix: pd.DataFrame,
    outdir: Path,
    *,
    title: str,
    filename: str,
    cell_inch: float = 0.15,
) -> Path:
    """单文件 · 一张图：**纵轴 = 策略种类、横轴 = 全部月份**，每格为**正方形**。

    格子是严格的正方形，因此图会很长很扁（184 列 : 8 行 ≈ 23:1）——
    这是方格子的必然结果，不是排版问题。
    ``cell_inch`` 是单格边长（英寸），调大则整体等比放大、格子里的数字更清楚。
    """
    ensure_style()
    labels = list(matrix.index)
    periods = list(matrix.columns)
    limit = monthly_color_limit(matrix)
    data = matrix.to_numpy(dtype="float64")

    n_rows, n_cols = data.shape
    jan_positions = [i for i, p in enumerate(periods) if p.month == 1]

    # 坐标区**严格**按「格子边长 × 行列数」摆放，格子因此必然是正方形。
    # 不能只给个大概的 figsize：那样 aspect=1 会把坐标区压缩、格子跟着变小
    # （实测 0.15" 被压到 0.10"），而 bbox_inches="tight" 又会把痕迹裁掉。
    (fig_w, fig_h), axes_rect, cax_rect = strip_geometry(n_rows, n_cols, cell_inch)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes(axes_rect)

    im = ax.imshow(data, cmap=ANNUAL_CMAP, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(n_rows), labels, fontsize=10)
    ax.set_xticks(jan_positions, [str(periods[i].year) for i in jan_positions], fontsize=9)
    ax.grid(False)
    # 年份之间画细线，方便横向定位
    for i in jan_positions[1:]:
        ax.axvline(i - 0.5, color="white", linewidth=1.2)
    # 字号跟着格子走，格子越大字越大
    _annotate_cells(ax, data, fontsize=max(3.5, cell_inch * 38))
    ax.set_title(title, fontsize=13, pad=12, loc="left")

    fig.colorbar(im, cax=fig.add_axes(cax_rect), label="月度收益")
    return _save(fig, outdir / filename)


def plot_monthly_year_panels(
    matrix: pd.DataFrame,
    outdir: Path,
    *,
    title: str,
    filename: str,
    horizontal: bool = False,
) -> Path:
    """单文件 · 一年一个面板：每个面板**纵轴 = 策略（8 行）、横轴 = 12 个月**。

    ``horizontal=False`` 时 16 个年面板**竖排**（每年占一条）；
    ``horizontal=True`` 时 16 个年面板**横排**（所有年份在同一行）。
    """
    ensure_style()
    labels = list(matrix.index)
    years = sorted({p.year for p in matrix.columns})
    limit = monthly_color_limit(matrix)
    n_years = len(years)
    frames = [monthly_year_by_month(matrix, label) for label in labels]

    if horizontal:
        fig, axes = plt.subplots(
            1,
            n_years,
            figsize=(0.118 * 12 * n_years + 2.6, 0.34 * len(labels) + 3.0),
            squeeze=False,
            layout="constrained",
        )
    else:
        fig, axes = plt.subplots(
            n_years,
            1,
            figsize=(13.5, 0.6 + 1.62 * n_years),
            squeeze=False,
            layout="constrained",
        )
    flat = [ax for row in axes for ax in row]
    im = None
    for ax, year in zip(flat, years, strict=True):
        # 必须是「行=策略、列=月份」：np.array 堆叠得到 (策略数, 12)。
        # 不能用 np.column_stack —— 那会得到 (12, 策略数)，正好转置。
        data = np.array([f.loc[year].to_numpy(dtype="float64") for f in frames])
        assert data.shape == (len(labels), 12), f"面板形状应为 (策略, 12)，实际 {data.shape}"
        im = ax.imshow(data, cmap=ANNUAL_CMAP, vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(12), MONTH_LABELS, fontsize=7 if horizontal else 7.5)
        if horizontal:
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
        # 纵轴只最左边那个面板标策略名，其余留白，避免 16 遍重复
        if ax is flat[0] or not horizontal:
            ax.set_yticks(range(len(labels)), labels, fontsize=8)
        else:
            ax.set_yticks(range(len(labels)), [""] * len(labels))
        ax.grid(False)
        _annotate_cells(ax, data, fontsize=6.4 if horizontal else 6.8)
        ax.set_title(str(year), fontsize=10.5, pad=3)
    fig.suptitle(title, fontsize=13.5)
    fig.colorbar(im, ax=flat, shrink=0.35, label="月度收益")
    return _save(fig, outdir / filename)


# --------------------------------------------------------------------------- #
# 指标表
# --------------------------------------------------------------------------- #
METRIC_ROWS: tuple[tuple[str, str, str], ...] = (
    ("年化收益", "cagr", "pct"),
    ("年化波动", "ann_vol", "pct"),
    ("夏普", "sharpe", "num"),
    ("索提诺", "sortino", "num"),
    ("最大回撤", "max_drawdown", "pct"),
    ("卡玛", "calmar", "num"),
    ("日胜率", "daily_win_rate", "pct"),
    ("持仓占比", "time_in_market", "pct"),
    ("往返次数", "n_completed", "int"),
    ("成本拖累(年化)", "cost_drag_cagr", "pct"),
)

#: 基准没有「自身相对自己」的口径，这两项按定义直接给定，比填 NaN 更有信息量。
#: 买入持有 = 始终满仓；期初那一笔从未平仓，故已完成的往返为 0。
BENCHMARK_FIXED: dict[str, float] = {"time_in_market": 1.0, "n_completed": 0.0}


def _pick(metrics: dict, key: str, kind: str) -> float:
    """按 metrics 里的真实命名取数。

    ``summarize`` 的键名并不统一：策略统计是 ``strategy_*``、基准是
    ``benchmark_*``，但 ``time_in_market`` / ``n_completed`` / ``cost_drag_cagr``
    这几个是**不带前缀**的，而且都是策略侧口径。

    因此：策略先试 ``strategy_*`` 再退回原名；基准只认 ``benchmark_*``。
    基准**不能**退回原名 —— 否则会把策略的成本拖累当成基准的显示出来，
    看上去合理、其实是错的。
    """
    if kind == "benchmark":
        if key in BENCHMARK_FIXED:
            return BENCHMARK_FIXED[key]
        if f"benchmark_{key}" in metrics:
            return _num(metrics, f"benchmark_{key}")
        return float("nan")  # 基准没有这个口径，留空比填错值好

    for candidate in (f"strategy_{key}", key):
        if candidate in metrics:
            return _num(metrics, candidate)
    return float("nan")


def build_metrics_table(panels: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for p in panels:
        kind = "benchmark" if p["kind"] == "benchmark" else "strategy"
        row: dict = {"方案": p["label"]}
        if p["direction"]:
            row["direction"] = p["direction"]
        row["期末净值"] = p["final"]
        row["总收益"] = p["final"] - 1.0
        for label, key, _fmt in METRIC_ROWS:
            row[label] = _pick(p["metrics"], key, kind)
        rows.append(row)
    table = pd.DataFrame(rows)
    _check_metrics_complete(table)
    return table


#: 这些列一旦是 NaN，说明 metrics 的键名变了，必须报错而不是默默出一张空表
REQUIRED_COLUMNS = ("年化收益", "年化波动", "夏普", "最大回撤", "卡玛", "持仓占比")


def _check_metrics_complete(table: pd.DataFrame) -> None:
    for column in REQUIRED_COLUMNS:
        if column not in table.columns:
            raise ValueError(f"指标表缺少列：{column}")
        missing = table.loc[table[column].isna(), "方案"].tolist()
        if missing:
            raise ValueError(
                f"「{column}」在 {missing} 上取不到值 —— "
                "多半是 ratema.metrics 的键名改了，请更新 METRIC_ROWS / _pick"
            )
    # 基准没有「成本拖累」这个口径，留空才对。若这里有值，说明取数又退回了
    # 不带前缀的 cost_drag_cagr，等于把策略的成本拖累显示成基准的。
    bench = table.loc[table["方案"].str.startswith("基准"), "成本拖累(年化)"]
    if len(bench) and not bench.isna().all():
        raise ValueError("基准行的「成本拖累」不应有值：那是策略侧口径，不能套到基准上")


def _fmt_cell(value, fmt: str) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "-"
    if f != f:
        return "-"
    if fmt == "pct":
        return f"{f * 100:.2f}%"
    if fmt == "int":
        return f"{round(f)}"
    return f"{f:.3f}"


def render_metrics_markdown(
    table: pd.DataFrame, cfg: dict, panels: list[dict], *, family: str = "signal"
) -> str:
    first = panels[0]
    lines = [
        f"# 方向对比 · 五利率等权打分策略（{FAMILY_LABEL[family]}）",
        "",
        f"- 信号：五利率等权打分（`{cfg['composite_mode']}`，投票源 `{cfg['vote_source']}`）",
        f"- 均线：MA{cfg['short_window']} / MA{cfg['long_window']}",
        f"- 重合阈值：{cfg['tol']:g} {cfg['tol_mode']}",
        f"- {_cost_text(cfg)}",
        f"- 区间：{first['date'].iloc[0].date()} ~ {first['date'].iloc[-1].date()}"
        f"（{first['n_days']} 个交易日）",
        f"- 曲线口径：{FAMILY_LABEL[family]}",
        "- 四条曲线共用同一信号与同一评估区间，差异只来自交易方向",
        "",
        "| 方案 | " + " | ".join(["期末净值", "总收益"] + [r[0] for r in METRIC_ROWS]) + " |",
        "| --- | " + " | ".join(["---"] * (2 + len(METRIC_ROWS))) + " |",
    ]
    for _, row in table.iterrows():
        cells = [str(row["方案"]), f"{row['期末净值']:.4f}", f"{row['总收益'] * 100:+.2f}%"]
        cells += [_fmt_cell(row[label], fmt) for label, _k, fmt in METRIC_ROWS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 跨利率汇总
# --------------------------------------------------------------------------- #
#: 两个「综合」的含义完全不同，标签必须能区分，不能用「等权综合」这种含糊说法
#: （它既能读成「等权组合」也能读成「等权打分」，曾经因此让人以为打分综合没画）。
#:   打分综合 —— 五票合成一个仓位（composite_frame）
#:   等权组合 —— 五份资金各跟一个信号后等权平均（portfolio_frame）
COMPOSITE_ROW_LABEL = "打分综合"
PORTFOLIO_ROW_LABEL = "等权组合"

#: 热力图/汇总表里综合口径那一行的完整说明
COMPOSITE_ROW_NOTE = "五票合成一个仓位，每天要么满仓要么空仓"
PORTFOLIO_ROW_NOTE = "五份资金各跟一个信号后等权平均，仓位比例连续"


# --------------------------------------------------------------------------- #
# 年度收益 + 年内最大回撤 数据导出
# --------------------------------------------------------------------------- #
def annual_return_drawdown_tables(bundle: dict, cfg: dict) -> dict[str, dict[str, pd.DataFrame]]:
    """{方向: {"年度收益": 表, "年内最大回撤": 表}}，行=口径、列=年份。

    两张表用同一份净值、同一套行列，所以可以逐格对照：
    「这一年赚了多少、途中最多回撤了多少」。
    """
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for direction, *_ in DIRECTION_SPECS:
        out[direction] = {
            "年度收益": indicator_annual_matrix(bundle, cfg, direction),
            "年内最大回撤": indicator_annual_drawdown_matrix(bundle, cfg, direction),
        }
    return out


def render_annual_return_drawdown_md(tables: dict[str, dict[str, pd.DataFrame]], cfg: dict) -> str:
    labels = {d: lab for d, lab, *_ in DIRECTION_SPECS}
    lines = [
        "# 年度收益 与 年内最大回撤（三方向 × 各口径）",
        "",
        f"- 均线：MA{cfg['short_window']} / MA{cfg['long_window']}"
        f"    重合阈值：{cfg['tol']:g} {cfg['tol_mode']}    {_cost_text(cfg)}",
        f"- 打分方式：`{cfg['composite_mode']}`（投票源 `{cfg['vote_source']}`）",
        f"- 「{COMPOSITE_ROW_LABEL}」= {COMPOSITE_ROW_NOTE}",
        f"- 「{PORTFOLIO_ROW_LABEL}」= {PORTFOLIO_ROW_NOTE}",
        "- 各利率那一行是**该利率单独**的策略（N=1 时两种综合口径等价，故只列一次）",
        "",
        "口径说明：",
        "",
        "- **年度收益**：年末净值环比；首年从评估起点起算，"
        "因此跨年跳空计入新一年、首年不是整年也不失真",
        "- **年内最大回撤**：只看该年自身的峰谷，不跨年累计；"
        "该年路径从**上年末净值**接起（首年从评估起点起），"
        "因此年初第一个交易日的涨跌计入本年 —— 与年度收益覆盖同一段时间",
        "- 两者是两条独立的信息：一年可以正收益但途中深跌",
        "",
        "> 与 `annual_breakdown.csv` 的 `strategy_max_drawdown` 可能有极小差异"
        "（实测最大 0.44 个百分点）：那一列把年初首个收益当作基期、不计入回撤，"
        "本表则计入。两种口径都只是约定，本表取「与收益率同起止」的那种。"
        "用 `include_year_start=False` 可复现前者（已验证逐格一致）。",
        "",
    ]

    def _table(df: pd.DataFrame, pct: bool) -> list[str]:
        years = [str(c) for c in df.columns]
        out = [
            "| 口径 | " + " | ".join(years) + " |",
            "| --- | " + " | ".join(["---"] * len(years)) + " |",
        ]
        for label in df.index:
            cells = [
                f"{df.loc[label, c] * 100:+.2f}%" if pct else f"{df.loc[label, c] * 100:.2f}%"
                for c in df.columns
            ]
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        return out

    for direction, lab in labels.items():
        lines.append(f"## {lab}（{direction}）")
        lines.append("")
        lines.append("### 年度收益")
        lines.append("")
        lines += _table(tables[direction]["年度收益"], pct=True)
        lines.append("")
        lines.append("### 年内最大回撤")
        lines.append("")
        lines += _table(tables[direction]["年内最大回撤"], pct=False)
        lines.append("")
    return "\n".join(lines)


def annual_return_drawdown_tidy(
    tables: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """长表：方向 / 口径 / 年份 / 年度收益 / 年内最大回撤（便于在 Excel 里透视）。"""
    labels = {d: lab for d, lab, *_ in DIRECTION_SPECS}
    rows: list[dict] = []
    for direction, per in tables.items():
        ret, dd = per["年度收益"], per["年内最大回撤"]
        for label in ret.index:
            for year in ret.columns:
                rows.append(
                    {
                        "方向": labels[direction],
                        "direction": direction,
                        "口径": label,
                        "年份": int(year),
                        "年度收益": float(ret.loc[label, year]),
                        "年内最大回撤": float(dd.loc[label, year]),
                    }
                )
    return pd.DataFrame(rows)


def drawdown_color_limit(tables: dict[str, dict[str, pd.DataFrame]]) -> float:
    """三张回撤图共用的颜色下限 —— 各图深浅必须可比。"""
    worst = 0.0
    for per in tables.values():
        data = per["年内最大回撤"].to_numpy(dtype="float64")
        if np.isfinite(data).any():
            worst = min(worst, float(np.nanmin(data)))
    return max(abs(worst), 1e-6)


def plot_annual_drawdown_heatmaps(
    tables: dict[str, dict[str, pd.DataFrame]], outdir: Path
) -> list[Path]:
    """三张年度最大回撤热力图，排版与收益图一致。

    与收益图的差别只有配色：**白 → 红**，``vmax=0`` 使 0 落在纯白端、
    最深回撤落在纯红端。越红 = 回撤越深。
    """
    limit = drawdown_color_limit(tables)
    labels = {d: lab for d, lab, *_ in DIRECTION_SPECS}
    out: list[Path] = []
    for direction, per in tables.items():
        out.append(
            plot_annual_heatmap(
                per["年内最大回撤"],
                outdir,
                title=f"年度最大回撤（%，行=口径，列=年份）· {labels[direction]}",
                filename=f"compare_annual_dd_{direction}.png",
                cbar_label="年内最大回撤",
                limit=limit,
                symmetric=False,
                cmap=DD_CMAP,
            )
        )
    return out


def build_all_metrics_table(bundle: dict, cfg: dict) -> pd.DataFrame:
    """把两个综合口径与每个利率各自的三方向绩效拼成一张长表。"""
    frames: list[pd.DataFrame] = []

    composite = build_metrics_table(build_panels(bundle, cfg, "signal"))
    composite.insert(0, "指标", COMPOSITE_ROW_LABEL)
    frames.append(composite)

    portfolio = build_metrics_table(build_panels(bundle, cfg, "portfolio"))
    portfolio.insert(0, "指标", PORTFOLIO_ROW_LABEL)
    frames.append(portfolio)

    for item in bundle["per_indicator"]:
        table = build_metrics_table(build_panels(item["bundle"], cfg, "signal"))
        table.insert(0, "指标", item["series"])
        table.insert(1, "指标名称", item["display"])
        frames.append(table)

    return pd.concat(frames, ignore_index=True)


def render_rate_matrix(all_table: pd.DataFrame, cfg: dict) -> str:
    """按「口径 × 方向」排列的对照表 —— 一眼看出哪个利率适合哪种方向。"""
    seen: list[str] = []
    for series in [COMPOSITE_ROW_LABEL, PORTFOLIO_ROW_LABEL, *all_table["指标"].tolist()]:
        if series not in seen:
            seen.append(series)
    directions = [label for _d, label, *_ in DIRECTION_SPECS] + [BENCH_LABEL]

    lines = [
        "# 各口径 · 三方向对比（同一打分口径）",
        "",
        f"- 均线：MA{cfg['short_window']} / MA{cfg['long_window']}"
        f"    重合阈值：{cfg['tol']:g} {cfg['tol_mode']}    {_cost_text(cfg)}",
        f"- 打分方式：`{cfg['composite_mode']}`（投票源 `{cfg['vote_source']}`）",
        f"- 「{COMPOSITE_ROW_LABEL}」= {COMPOSITE_ROW_NOTE}",
        f"- 「{PORTFOLIO_ROW_LABEL}」= {PORTFOLIO_ROW_NOTE}",
        "- 其余每行是该利率**单独**的策略（N=1 时两种综合口径等价，故只列一次）",
        "- 每行的四条曲线共用同一信号与同一评估区间，差异只来自交易方向",
        "",
    ]

    for metric, fmt, note in (
        ("年化收益", "pct", "年化收益率"),
        ("夏普", "num", "夏普比率"),
        ("最大回撤", "pct", "最大回撤（越接近 0 越好）"),
        ("往返次数", "int", "已完成往返次数"),
    ):
        lines.append(f"## {note}")
        lines.append("")
        lines.append("| 指标 | " + " | ".join(directions) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(directions)) + " |")
        for series in seen:
            sub = all_table[all_table["指标"] == series]
            cells: list[str] = []
            for label in directions:
                row = sub[sub["方案"] == label]
                cells.append(_fmt_cell(row[metric].iloc[0], fmt) if len(row) else "-")
            lines.append(f"| {series} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def select_indicators(bundle: dict, cfg: dict) -> list[dict]:
    """按 scope / per_indicator_names 决定要给哪些利率出图。"""
    if cfg["scope"] == "composite":
        return []
    items = bundle["per_indicator"]
    wanted = cfg["per_indicator_names"]
    if not wanted:
        return items
    available = {i["series"]: i for i in items}
    unknown = [w for w in wanted if w not in available]
    if unknown:
        raise SystemExit(
            f"per_indicator_names 里有未知指标：{unknown}；可用指标为 {list(available)}"
        )
    return [available[w] for w in wanted]


def indicator_chart_name(series: str, *, drawdown: bool = False) -> str:
    suffix = "_drawdowns" if drawdown else ""
    return f"compare_directions_{series}{suffix}.png"


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(
        prog="compare_directions.py",
        description="同一打分信号下 long_short / long / short / 基准 的同图对比",
    )
    parser.add_argument("name", nargs="?", default=None, help="输出文件夹名，覆盖 CONFIG['name']")
    parser.add_argument("--log", action="store_true", help="纵轴取对数")
    parser.add_argument("--no-drawdown", action="store_true", help="不出回撤附图")
    parser.add_argument("--only-composite", action="store_true", help="只出综合那一张图")
    parser.add_argument("--only-indicators", action="store_true", help="只出每个利率各一张图")
    parser.add_argument(
        "--rates", default=None, help="只对这几个利率出图，逗号分隔，例如 DR007,R007"
    )
    parser.add_argument(
        "--indicator-drawdown", action="store_true", help="每个利率也各出一张回撤附图"
    )
    parser.add_argument(
        "--curve",
        choices=("both", "signal", "portfolio"),
        default=None,
        help="画哪一套：signal=综合信号（五票合成一个仓位）/ portfolio=等权组合（各 1/5 资金）",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    cfg = dict(CONFIG)
    if args.name:
        cfg["name"] = args.name
    if args.log:
        cfg["log_scale"] = True
    if args.no_drawdown:
        cfg["drawdown_chart"] = False
    if args.indicator_drawdown:
        cfg["per_indicator_drawdown"] = True
    if args.curve:
        cfg["curve"] = args.curve
    if args.only_composite and args.only_indicators:
        print("--only-composite 与 --only-indicators 不能同时使用")
        return 2
    if args.only_composite:
        cfg["scope"] = "composite"
    elif args.only_indicators:
        cfg["scope"] = "per_indicator"
    if args.rates:
        cfg["per_indicator_names"] = [c.strip() for c in args.rates.split(",") if c.strip()]

    validate(cfg)

    outdir = Path(cfg["outdir"]) / cfg["name"]
    source = resolve_source(cfg)
    if not Path(source).exists():
        print(f"\n找不到输入文件：{source}")
        print("请把 Excel 放到项目根目录，或在 CONFIG['input'] 里写死路径。")
        return 2

    rule("方向对比")
    print(f"  数据来源    : {source}")
    print(f"  输出目录    : {outdir.resolve()}")
    print(f"  均线        : MA{cfg['short_window']} / MA{cfg['long_window']}")
    print(f"  重合阈值    : {cfg['tol']:g} {cfg['tol_mode']}")
    print(f"  打分方式    : {cfg['composite_mode']}（投票源 {cfg['vote_source']}）")
    print(f"  交易成本    : {cfg['cost_bps']:g}bp / {cfg['cost_mode']}")
    print("  对比方向    : " + "、".join(label for _d, label, *_ in DIRECTION_SPECS) + "、买入持有")
    print(f"  出图范围    : {cfg['scope']}")

    dataset = load_dataset(source)
    bundle = compute_all_directions(dataset, cfg)

    families = ["signal", "portfolio"] if cfg["curve"] == "both" else [cfg["curve"]]
    panels_by_family = {f: build_panels(bundle, cfg, f) for f in families}

    first = panels_by_family[families[0]][0]
    print(
        f"\n  评估区间    : {first['date'].iloc[0].date()} ~ {first['date'].iloc[-1].date()}"
        f"（{first['n_days']} 个交易日）"
    )
    rate_cols = bundle["base"].rate_cols
    print(f"  参与打分的指标: {', '.join(rate_cols)}（共 {len(rate_cols)} 个）")
    print(f"  曲线类型    : {cfg['curve']}")

    selected = select_indicators(bundle, cfg)
    if selected:
        print(f"  单利率出图  : {', '.join(i['series'] for i in selected)}")

    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    show = ["方案", "期末净值", "总收益", "年化收益", "夏普", "最大回撤", "持仓占比", "往返次数"]

    # ---- 综合曲线（综合信号 / 等权组合） --------------------------------- #
    tables: dict[str, pd.DataFrame] = {}
    for family in families:
        rule(FAMILY_LABEL[family])
        panels = panels_by_family[family]
        table = build_metrics_table(panels)
        tables[family] = table
        print(table[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        if cfg["scope"] in ("composite", "both"):
            written.append(
                plot_equity_comparison(
                    panels,
                    outdir,
                    cfg,
                    signal_label=bundle["signal_label"],
                    title=FAMILY_TITLE[family],
                    filename=f"{FAMILY_FILE[family]}.png",
                )
            )
            if cfg["drawdown_chart"]:
                written.append(
                    plot_drawdown_comparison(
                        panels,
                        outdir,
                        cfg,
                        title=f"回撤对比 · {FAMILY_LABEL[family]}",
                        filename=f"{FAMILY_FILE[family]}_drawdowns.png",
                    )
                )

    # ---- 每个利率单独一张（只出综合信号一套） ---------------------------- #
    # N=1 时「等权组合」就是该指标自己的那条净值，与综合信号曲线完全重合，
    # 所以单利率不需要再出一套 portfolio，避免出两张一模一样的图。
    if selected:
        rule(f"每个利率单独一张（共 {len(selected)} 张）")
        for item in selected:
            sub_panels = build_panels(item["bundle"], cfg, "signal")
            sub_table = build_metrics_table(sub_panels)
            print(f"\n  【{item['series']}】{item['display']}")
            cols = ["方案", "期末净值", "年化收益", "夏普", "最大回撤", "往返次数"]
            print(sub_table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            written.append(
                plot_equity_comparison(
                    sub_panels,
                    outdir,
                    cfg,
                    signal_label=item["bundle"]["signal_label"],
                    title=(
                        f"{item['series']}（{item['display']}）单利率信号"
                        " · 三种交易方向 vs 买入持有"
                    ),
                    filename=indicator_chart_name(item["series"]),
                )
            )
            if cfg["per_indicator_drawdown"]:
                written.append(
                    plot_drawdown_comparison(
                        sub_panels,
                        outdir,
                        cfg,
                        title=f"回撤对比 · {item['series']}（{item['display']}）单利率信号",
                        filename=indicator_chart_name(item["series"], drawdown=True),
                    )
                )

    # ---- 年度收益热力图（仿 05_annual_returns.png） ---------------------- #
    if cfg["annual_heatmap"]:
        mode = cfg["annual_heatmap_rows"]
        if mode in ("direction", "both"):
            rule("年度收益热力图（行=方向）")
            for family in families:
                df = annual_matrix(panels_by_family[family])
                tag = "" if family == "signal" else "_portfolio"
                path = plot_annual_heatmap(
                    df,
                    outdir,
                    title=f"年度收益（%，行=方向，列=年份）· {FAMILY_LABEL[family]}",
                    filename=f"compare_annual_returns{tag}.png",
                )
                written.append(path)
                print(f"  - {path}")
                print(df.to_string(float_format=lambda v: f"{v * 100:+.2f}%"))
        if mode in ("indicator", "both"):
            rule("年度收益热力图（打分综合 + 等权组合 + 各利率 + 基准）")
            for direction, label, *_ in DIRECTION_SPECS:
                df = indicator_annual_matrix(bundle, cfg, direction)
                path = plot_annual_heatmap(
                    df,
                    outdir,
                    title=f"年度收益（%，行=综合/各利率/基准，列=年份）· {label}",
                    filename=f"compare_annual_indicator_{direction}.png",
                )
                written.append(path)
                print(f"  - {path}")

    # ---- 逐月收益热力图（同样的 8 行，按年拆文件） ------------------------ #
    if cfg["monthly_heatmap"]:
        direction = cfg["monthly_direction"]
        labels = {d: lab for d, lab, *_ in DIRECTION_SPECS}
        if direction not in labels:
            raise SystemExit(f"monthly_direction 必须是 {list(labels)} 之一，现在是 {direction!r}")
        rule(f"逐月收益热力图（行=综合/各利率/基准，列=1~12 月）· {labels[direction]}")
        matrix = monthly_matrix(bundle, cfg, direction)
        limit = monthly_color_limit(matrix)
        years = monthly_years(matrix, cfg)
        print(f"  共用颜色范围 ±{limit * 100:.2f}%（各年文件深浅可比）")
        for year in years:
            frame = monthly_year_frame(matrix, year)
            path = plot_annual_heatmap(
                frame,
                outdir,
                title=(
                    f"{year} 年逐月收益（%，行=综合/各利率/基准，列=月份）· {labels[direction]}"
                ),
                filename=f"compare_monthly_{year}.png",
                cbar_label="月度收益",
                limit=limit,
                fontsize=7,
            )
            written.append(path)
            print(f"  - {path}")

        monthly_csv = outdir / "compare_monthly.csv"
        out_matrix = matrix.copy()
        out_matrix.columns = [str(p) for p in out_matrix.columns]
        out_matrix.to_csv(monthly_csv, encoding="utf-8-sig")
        print(f"  - {monthly_csv}")

        # 单文件版：横轴 = 月份，纵轴 = 策略种类
        layout = cfg["monthly_layout"]
        span = f"{years[0]}–{years[-1]}"
        base = f"逐月收益（%，横轴=月份，纵轴=策略）· {labels[direction]}"
        if layout in ("strip", "all"):
            path = plot_monthly_strategy_strip(
                matrix,
                outdir,
                title=f"{span} {base} · 一张图（正方形格子）",
                filename="compare_monthly_by_strategy.png",
                cell_inch=cfg["monthly_cell_inch"],
            )
            written.append(path)
            print(f"  - {path}")
        if layout in ("panel", "all"):
            path = plot_monthly_year_panels(
                matrix,
                outdir,
                title=f"{span} {base} · 一年一个面板（竖排）",
                filename="compare_monthly_years_panel.png",
            )
            written.append(path)
            print(f"  - {path}")
        if layout in ("row", "all"):
            path = plot_monthly_year_panels(
                matrix,
                outdir,
                title=f"{span} {base} · 一年一个面板（横排，同一行）",
                filename="compare_monthly_years_row.png",
                horizontal=True,
            )
            written.append(path)
            print(f"  - {path}")

    # ---- 年度收益 + 年内最大回撤 ------------------------------------------ #
    if cfg["annual_return_dd_data"] or cfg["annual_dd_heatmap"]:
        rule("年度收益 与 年内最大回撤")
        # 变量别叫 tables —— 上面存绩效表用的就是这个名字，会把它覆盖掉
        annual_tables = annual_return_drawdown_tables(bundle, cfg)

        if cfg["annual_dd_heatmap"]:
            for path in plot_annual_drawdown_heatmaps(annual_tables, outdir):
                written.append(path)
                print(f"  - {path}")
            print(
                f"  三张图共用颜色下限 -{drawdown_color_limit(annual_tables) * 100:.2f}%"
                "（各图深浅可比）"
            )

        if cfg["annual_return_dd_data"]:
            tidy = annual_return_drawdown_tidy(annual_tables)
            csv_path = outdir / "compare_annual_return_dd.csv"
            md_path = outdir / "compare_annual_return_dd.md"
            tidy.to_csv(csv_path, index=False, encoding="utf-8-sig")
            md_path.write_text(
                render_annual_return_drawdown_md(annual_tables, cfg), encoding="utf-8"
            )
            print(f"  - {csv_path}")
            print(f"  - {md_path}")
            for direction, lab in {d: lab for d, lab, *_ in DIRECTION_SPECS}.items():
                ret = annual_tables[direction]["年度收益"]
                dd = annual_tables[direction]["年内最大回撤"]
                print(f"\n  【{lab}】年度收益")
                print(ret.to_string(float_format=lambda v: f"{v * 100:+.2f}%"))
                print(f"  【{lab}】年内最大回撤")
                print(dd.to_string(float_format=lambda v: f"{v * 100:.2f}%"))

    # ---- 表格 ------------------------------------------------------------ #
    rule("产出文件")
    for path in written:
        print(f"  - {path}")

    for family in families:
        tag = "" if family == "signal" else "_portfolio"
        csv_path = outdir / f"compare_metrics{tag}.csv"
        md_path = outdir / f"compare_metrics{tag}.md"
        tables[family].to_csv(csv_path, index=False, encoding="utf-8-sig")
        md_path.write_text(
            render_metrics_markdown(tables[family], cfg, panels_by_family[family], family=family),
            encoding="utf-8",
        )
        print(f"  - {csv_path}")
        print(f"  - {md_path}")

    if cfg["scope"] != "composite":
        all_table = build_all_metrics_table(bundle, cfg)
        by_rate_csv = outdir / "compare_by_rate.csv"
        by_rate_md = outdir / "compare_by_rate.md"
        all_table.to_csv(by_rate_csv, index=False, encoding="utf-8-sig")
        by_rate_md.write_text(render_rate_matrix(all_table, cfg), encoding="utf-8")
        print(f"  - {by_rate_csv}")
        print(f"  - {by_rate_md}")

    rule("完成")
    print(f"  共 {len(written)} 张图，目录：{outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
