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
    # 重合阈值：|MA短 − MA长| ≤ 阈值 即视为重合，标记 0 并延续前一日信号
    #   "bp"  单位基点，1 = 1bp        ← 推荐
    #   "abs" 单位百分点，0.01 = 1bp   ← 与 bp 相差 100 倍
    #   "rel" 相对比例；"std" 标准差倍数；"q" 分位数(0~1)
    "tol_mode": "bp",
    "tol": 5.0,  # 5bp 以内视为两线重合
    "std_window": 120,
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
    #   "both"         等权综合 + 每个利率各一张（默认，共 1+N 张）
    #   "composite"    只出等权综合那一张
    #   "per_indicator" 只出每个利率各一张
    "scope": "both",
    # 只对其中几个利率出图，例如 ["DR007", "R007"]；None = 全部
    "per_indicator_names": None,
    "log_scale": False,  # True = 纵轴取对数，早期差异看得更清
    "drawdown_chart": True,  # True = 额外出一张综合的回撤对比附图
    "per_indicator_drawdown": False,  # True = 每个利率也各出一张回撤附图
    "figsize": (13.5, 7.2),
}
# ══════════════════════════════════════════════════════════════════════════
#  以下为实现，一般不需要修改
# ══════════════════════════════════════════════════════════════════════════

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

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


def compute_direction_bundle(
    dataset, per_indicator, cfg: dict, *, index_col: str | None = None, signal_label: str = ""
) -> dict:
    """给定「一组指标的信号」，跑出三个方向的净值。

    综合图与单利率图走的是**同一条代码路径**：单利率只需传一个只含该指标的
    列表。N=1 时 ``score = (看多票数 − 看空票数) / 1 ∈ {−1, +1}``，
    与「该指标自己的 signal_eff」逐日等价，因此单利率图不需要另写一套逻辑，
    也就自动继承了同一套对齐校验。
    """
    per_indicator = list(per_indicator)
    if not per_indicator:
        raise ValueError("per_indicator 不能为空")

    results: dict = {}
    for direction, *_ in DIRECTION_SPECS:
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

    _check_alignment(results)
    return {
        "by_direction": results,
        "series": [r.series for r in per_indicator],
        "signal_label": signal_label,
        "is_composite": len(per_indicator) > 1,
    }


def compute_all_directions(dataset, cfg: dict) -> dict:
    """同一个打分信号跑三个方向，返回 {方向: CompositeResult} 与基准。

    关键点：三个方向必须共用同一份信号和同一个评估区间，否则图上的
    差异就说不清是「方向不同」还是「窗口不同」。这里显式校验对齐。

    额外返回 ``per_indicator_bundles``：每个利率各自一张图用的 bundle。
    """
    signal_cfg = SignalConfig(
        short_window=cfg["short_window"],
        long_window=cfg["long_window"],
        tol_mode=cfg["tol_mode"],
        tol=cfg["tol"],
        std_window=cfg["std_window"],
    )
    # 单指标回测只为拿到各指标的信号；信号只取决于均线，与交易方向无关，
    # 因此这里跑一次就够三个方向复用。
    base = run_all(
        dataset,
        signal_cfg,
        make_backtest_config(cfg, "long_short"),
        index_col=cfg["index_col"],
        rate_cols=cfg["rate_cols"],
        start=cfg["start"],
        end=cfg["end"],
    )

    n = len(base.rate_cols)
    bundle = compute_direction_bundle(
        dataset,
        base.results,
        cfg,
        index_col=base.index_col,
        signal_label=f"五利率等权打分（{cfg['composite_mode']} / 投票源 {cfg['vote_source']}）",
    )
    bundle["base"] = base
    # 每个利率各自一张图：只把该指标一个元素喂进去，N=1 即它自己的信号
    bundle["per_indicator"] = [
        {
            "series": series,
            "display": dataset.display_name(series),
            "bundle": compute_direction_bundle(
                dataset,
                [res],
                cfg,
                index_col=base.index_col,
                signal_label=(f"单利率信号 · {series}（{dataset.display_name(series)}）"),
            ),
        }
        for series, res in zip(base.rate_cols, base.results, strict=True)
    ]
    bundle["n_indicators"] = n
    return bundle


def _check_alignment(results: dict) -> None:
    """三个方向必须逐日对齐，且基准曲线完全相同。"""
    reference = None
    for direction, res in results.items():
        f = res.composite_frame
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
                f"{direction} 与 {ref_dir} 的评估区间不一致：{key} vs {ref_key}；"
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


def build_panels(bundle: dict, cfg: dict) -> list[dict]:
    """把三条策略 + 基准整理成绘图用的行。"""
    scale = float(cfg["initial_capital"])
    by_direction = bundle["by_direction"]
    any_res = next(iter(by_direction.values()))
    n_days = len(any_res.composite_frame)

    panels: list[dict] = []
    for direction, label, color, style, width, _desc in DIRECTION_SPECS:
        res = by_direction[direction]
        m = res.composite_metrics
        frame = res.composite_frame
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

    bm = any_res.composite_metrics
    bench = any_res.composite_frame
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


def render_metrics_markdown(table: pd.DataFrame, cfg: dict, panels: list[dict]) -> str:
    first = panels[0]
    lines = [
        "# 方向对比 · 五利率等权打分策略",
        "",
        f"- 信号：五利率等权打分（`{cfg['composite_mode']}`，投票源 `{cfg['vote_source']}`）",
        f"- 均线：MA{cfg['short_window']} / MA{cfg['long_window']}",
        f"- 重合阈值：{cfg['tol']:g} {cfg['tol_mode']}",
        f"- {_cost_text(cfg)}",
        f"- 区间：{first['date'].iloc[0].date()} ~ {first['date'].iloc[-1].date()}"
        f"（{first['n_days']} 个交易日）",
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
COMPOSITE_ROW_LABEL = "等权综合"


def build_all_metrics_table(bundle: dict, cfg: dict) -> pd.DataFrame:
    """把等权综合与每个利率各自的三方向绩效拼成一张长表。"""
    frames: list[pd.DataFrame] = []

    composite = build_metrics_table(build_panels(bundle, cfg))
    composite.insert(0, "指标", COMPOSITE_ROW_LABEL)
    frames.append(composite)

    for item in bundle["per_indicator"]:
        table = build_metrics_table(build_panels(item["bundle"], cfg))
        table.insert(0, "指标", item["series"])
        table.insert(1, "指标名称", item["display"])
        frames.append(table)

    return pd.concat(frames, ignore_index=True)


def render_rate_matrix(all_table: pd.DataFrame, cfg: dict) -> str:
    """按「指标 × 方向」排列的对照表 —— 一眼看出哪个利率适合哪种方向。"""
    seen: list[str] = []
    for series in [COMPOSITE_ROW_LABEL, *all_table["指标"].tolist()]:
        if series not in seen:
            seen.append(series)
    directions = [label for _d, label, *_ in DIRECTION_SPECS] + [BENCH_LABEL]

    lines = [
        "# 各利率 · 三方向对比（同一打分口径）",
        "",
        f"- 均线：MA{cfg['short_window']} / MA{cfg['long_window']}"
        f"    重合阈值：{cfg['tol']:g} {cfg['tol_mode']}    {_cost_text(cfg)}",
        f"- 打分方式：`{cfg['composite_mode']}`（投票源 `{cfg['vote_source']}`）",
        "- 「等权综合」= 五利率等权打分；其余每行是该利率**单独**的信号",
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
    parser.add_argument("--only-composite", action="store_true", help="只出等权综合那一张图")
    parser.add_argument("--only-indicators", action="store_true", help="只出每个利率各一张图")
    parser.add_argument(
        "--rates", default=None, help="只对这几个利率出图，逗号分隔，例如 DR007,R007"
    )
    parser.add_argument(
        "--indicator-drawdown", action="store_true", help="每个利率也各出一张回撤附图"
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
    panels = build_panels(bundle, cfg)

    first = panels[0]
    print(
        f"\n  评估区间    : {first['date'].iloc[0].date()} ~ {first['date'].iloc[-1].date()}"
        f"（{first['n_days']} 个交易日）"
    )
    rate_cols = bundle["base"].rate_cols
    print(f"  参与打分的指标: {', '.join(rate_cols)}（共 {len(rate_cols)} 个）")

    selected = select_indicators(bundle, cfg)
    if selected:
        print(f"  单利率出图  : {', '.join(i['series'] for i in selected)}")

    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # ---- 等权综合 -------------------------------------------------------- #
    rule("等权综合")
    table = build_metrics_table(panels)
    show = ["方案", "期末净值", "总收益", "年化收益", "夏普", "最大回撤", "持仓占比", "往返次数"]
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if cfg["scope"] in ("composite", "both"):
        written.append(
            plot_equity_comparison(
                panels,
                outdir,
                cfg,
                signal_label=bundle["signal_label"],
                title="五利率等权打分策略 · 三种交易方向 vs 买入持有",
                filename="compare_directions.png",
            )
        )
        if cfg["drawdown_chart"]:
            written.append(plot_drawdown_comparison(panels, outdir, cfg))

    # ---- 每个利率单独一张 ------------------------------------------------ #
    if selected:
        rule(f"每个利率单独一张（共 {len(selected)} 张）")
        for item in selected:
            sub_panels = build_panels(item["bundle"], cfg)
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

    # ---- 表格 ------------------------------------------------------------ #
    rule("产出文件")
    for path in written:
        print(f"  - {path}")

    csv_path = outdir / "compare_metrics.csv"
    md_path = outdir / "compare_metrics.md"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(render_metrics_markdown(table, cfg, panels), encoding="utf-8")
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
