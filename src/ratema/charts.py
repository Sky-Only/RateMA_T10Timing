"""图表输出：把净值变化画出来。

产出（默认写到 ``<outdir>/charts/``）：

* ``01_equity_curves.png``    全部策略 + 基准的净值曲线
* ``02_per_series.png``       每个底层指标一张：策略 vs 基准 + 超额填充
* ``03_relative_strength.png``策略/基准 相对强弱
* ``04_drawdowns.png``        回撤对比
* ``05_annual_returns.png``   年度收益热力图（含基准）
* ``06_signal_mechanics.png`` 信号机理：利率双均线 / spread 与重合带 / 实际仓位
* ``07_tol_sweep.png``        不同重合度阈值下的净值（可选）
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.figure import Figure

CJK_FONTS = [
    "Microsoft YaHei",
    "SimHei",
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
    "DengXian",
    "SimSun",
    "DejaVu Sans",
]

PALETTE = ["#2E5C8A", "#C0392B", "#1E8449", "#7D3C98", "#B9770E", "#117A8B"]
BENCH_COLOR = "#7F8C8D"
GRID = {"alpha": 0.25, "linestyle": "-", "linewidth": 0.6}


# --------------------------------------------------------------------------- #
# 样式
# --------------------------------------------------------------------------- #
def setup_style() -> list[str]:
    """挑选可用的中文字体并设置统一风格，返回实际使用的字体列表。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = [f for f in CJK_FONTS if f in available]
    plt.rcParams["font.sans-serif"] = picked or ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False  # 负号显示
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#B0B0B0",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return picked


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _date_axis(ax, *, years: int = 2) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(years))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, axis="x", **GRID)


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:+.{digits}f}%"


def _bench_frame(run) -> pd.DataFrame:
    """取基准列（各指标的基准相同）。"""
    return run.results[0].frame[["date", "benchmark_equity", "benchmark_equity_gross"]].copy()


# --------------------------------------------------------------------------- #
# 01 净值曲线总览
# --------------------------------------------------------------------------- #
def plot_equity_curves(run, outdir: Path, *, log_scale: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 7))

    bench = _bench_frame(run)
    for i, res in enumerate(run.results):
        f = res.frame
        ax.plot(
            f["date"],
            f["equity"],
            color=PALETTE[i % len(PALETTE)],
            linewidth=1.7,
            label=f"{res.series}  {f['equity'].iloc[-1]:.4f}",
        )
    ax.plot(
        bench["date"],
        bench["benchmark_equity"],
        color=BENCH_COLOR,
        linewidth=2.2,
        linestyle="-",
        label=f"基准 买入持有  {bench['benchmark_equity'].iloc[-1]:.4f}",
    )
    ax.axhline(1.0, color="#333333", linewidth=0.9, alpha=0.6)
    ax.set_ylabel("净值（起点 = 1.0，含双边成本）")
    ax.set_title(
        f"20/120 日均线利率择时策略 · 净值曲线\n"
        f"{run.start} ~ {run.end}（{len(run.results[0].frame)} 个交易日）"
        f"   重合阈值：{run.signal_cfg.describe_tol()}"
    )
    _date_axis(ax)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(loc="upper left", ncols=2)
    fig.autofmt_xdate()
    return _save(fig, Path(outdir) / "01_equity_curves.png")


# --------------------------------------------------------------------------- #
# 02 每个指标一张：策略 vs 基准 + 超额
# --------------------------------------------------------------------------- #
def plot_per_series(run, outdir: Path) -> Path:
    results = run.results
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(13.5, 3.0 * n), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, res in zip(axes, results):
        f = res.frame
        color = PALETTE[results.index(res) % len(PALETTE)]
        ax.plot(
            f["date"],
            f["benchmark_equity"],
            color=BENCH_COLOR,
            linewidth=1.6,
            linestyle="-",
            label="基准（买入持有）",
        )
        ax.plot(f["date"], f["equity"], color=color, linewidth=1.8, label="策略")
        ax.fill_between(
            f["date"],
            f["equity"],
            f["benchmark_equity"],
            where=f["equity"] >= f["benchmark_equity"],
            color=color,
            alpha=0.16,
            interpolate=True,
        )
        ax.fill_between(
            f["date"],
            f["equity"],
            f["benchmark_equity"],
            where=f["equity"] < f["benchmark_equity"],
            color="#C0392B",
            alpha=0.14,
            interpolate=True,
        )
        m = res.metrics
        ax.set_title(
            f"{res.series}（{res.display_name}）   "
            f"策略年化 {_pct(m.get('strategy_cagr', float('nan')), 2)}  "
            f"夏普 {m.get('strategy_sharpe', float('nan')):.2f}  "
            f"最大回撤 {_pct(m.get('strategy_max_drawdown', float('nan')), 2)}   |   "
            f"基准年化 {_pct(m.get('benchmark_cagr', float('nan')), 2)}  "
            f"最大回撤 {_pct(m.get('benchmark_max_drawdown', float('nan')), 2)}",
            loc="left",
        )
        ax.set_ylabel("净值")
        ax.axhline(1.0, color="#333333", linewidth=0.8, alpha=0.5)
        ax.legend(loc="upper left")

    _date_axis(axes[-1])
    fig.suptitle("各底层利率指标：策略 vs 基准", y=1.001, fontsize=14)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, Path(outdir) / "02_per_series.png")


# --------------------------------------------------------------------------- #
# 03 相对强弱
# --------------------------------------------------------------------------- #
def plot_relative_strength(run, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 6))
    for i, res in enumerate(run.results):
        f = res.frame
        rel = f["equity"] / f["benchmark_equity"]
        ax.plot(
            f["date"],
            rel,
            color=PALETTE[i % len(PALETTE)],
            linewidth=1.6,
            label=f"{res.series}  {rel.iloc[-1]:.4f}",
        )
    ax.axhline(1.0, color="#333333", linewidth=1.2)
    ax.set_ylabel("策略净值 / 基准净值")
    ax.set_title("相对强弱（>1 表示跑赢买入持有基准）")
    _date_axis(ax)
    ax.legend(loc="upper left", ncols=2)
    fig.autofmt_xdate()
    return _save(fig, Path(outdir) / "03_relative_strength.png")


# --------------------------------------------------------------------------- #
# 04 回撤
# --------------------------------------------------------------------------- #
def plot_drawdowns(run, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 6))
    bench = _bench_frame(run)
    bench_dd = bench["benchmark_equity"] / bench["benchmark_equity"].cummax() - 1.0
    ax.plot(
        bench["date"],
        bench_dd,
        color=BENCH_COLOR,
        linewidth=2.0,
        linestyle="-",
        label=f"基准  {bench_dd.min() * 100:.2f}%",
    )
    ax.fill_between(bench["date"], bench_dd, 0, color=BENCH_COLOR, alpha=0.12)

    for i, res in enumerate(run.results):
        dd = res.frame["drawdown"]
        ax.plot(
            res.frame["date"],
            dd,
            color=PALETTE[i % len(PALETTE)],
            linewidth=1.5,
            label=f"{res.series}  {dd.min() * 100:.2f}%",
        )
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_ylabel("回撤")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.set_title("回撤对比（越浅越好）")
    _date_axis(ax)
    ax.legend(loc="lower left", ncols=3)
    fig.autofmt_xdate()
    return _save(fig, Path(outdir) / "04_drawdowns.png")


# --------------------------------------------------------------------------- #
# 05 年度收益热力图
# --------------------------------------------------------------------------- #
def annual_returns(run) -> pd.DataFrame:
    """行=指标（含基准），列=年份，值=年度收益。"""
    rows: dict[str, pd.Series] = {}
    for res in run.results:
        f = res.frame.set_index("date")["equity"]
        rows[res.series] = (
            f.resample("YE")
            .last()
            .pct_change()
            .fillna(f.resample("YE").last().iloc[0] / f.iloc[0] - 1.0)
        )
    f = run.results[0].frame.set_index("date")["benchmark_equity"]
    rows["基准"] = (
        f.resample("YE")
        .last()
        .pct_change()
        .fillna(f.resample("YE").last().iloc[0] / f.iloc[0] - 1.0)
    )
    df = pd.DataFrame(rows).T
    df.columns = [c.year for c in df.columns]
    return df


def plot_annual_returns(run, outdir: Path) -> Path:
    df = annual_returns(run)
    fig, ax = plt.subplots(figsize=(13.5, 1.0 + 0.62 * len(df)))
    data = df.to_numpy(dtype="float64")
    limit = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    limit = max(limit, 1e-6)

    im = ax.imshow(data, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(df.columns)), [str(c) for c in df.columns], rotation=45)
    ax.set_yticks(range(len(df.index)), list(df.index))
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(
                    j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=8, color="#111111"
                )
    ax.set_title("年度收益（%，行=指标，列=年份）")
    fig.colorbar(im, ax=ax, shrink=0.7, label="年度收益")
    fig.tight_layout()
    return _save(fig, Path(outdir) / "05_annual_returns.png")


# --------------------------------------------------------------------------- #
# 06 信号机理
# --------------------------------------------------------------------------- #
def plot_signal_mechanics(
    res,
    outdir: Path,
    *,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    """利率双均线 / spread 与重合带 / 实际仓位 —— 三行共享时间轴。"""
    f = res.frame.copy()
    if start:
        f = f[f["date"] >= pd.Timestamp(start)]
    if end:
        f = f[f["date"] <= pd.Timestamp(end)]
    if f.empty:
        raise ValueError("信号机理图：指定的窗口内没有数据")

    fig, axes = plt.subplots(
        3, 1, figsize=(13.5, 9.5), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1]}
    )

    # --- 利率与双均线 ---
    ax = axes[0]
    ax.plot(f["date"], f["rate_value"], color="#95A5A6", linewidth=1.0, label="利率（原始值）")
    ax.plot(f["date"], f["ma_short"], color="#C0392B", linewidth=1.8, label="MA20（短）")
    ax.plot(f["date"], f["ma_long"], color="#2E5C8A", linewidth=1.8, label="MA120（长）")
    overlap = f[f["is_overlap"].fillna(False)]
    if len(overlap):
        ax.scatter(
            overlap["date"],
            overlap["ma_short"],
            s=14,
            color="#F1C40F",
            zorder=5,
            label=f"重合日（{len(overlap)} 天）",
        )
    ax.set_ylabel("利率 (%)")
    band = ""
    if "overlap_band" in f.columns and len(f):
        band = str(f["overlap_band"].iloc[0]).lstrip("±")
    note = ""
    max_tol_bp = float(f["tolerance_bp"].abs().max()) if len(f) else 0.0
    if not np.isfinite(max_tol_bp) or max_tol_bp <= 0:
        note = "\n（当前阈值为 0，即仅严格相等才算重合，图中重合带宽度为 0）"
    ax.set_title(
        f"{res.series}（{res.display_name}） 信号机理：MA20 vs MA120 与重合判定\n"
        f"重合阈值：{band}" + note
    )
    ax.legend(loc="upper right", ncols=2)

    # --- spread 与重合带 ---
    ax = axes[1]
    spread_bp = f["spread_bp"]
    tol_bp = f["tolerance_bp"]
    ax.plot(f["date"], spread_bp, color="#2C3E50", linewidth=1.3, label="MA20 − MA120")
    ax.fill_between(f["date"], -tol_bp, tol_bp, color="#F1C40F", alpha=0.35, label="重合区间 ±阈值")
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_ylabel("spread (bp)")
    ax.legend(loc="upper right", ncols=2)

    # --- 仓位 ---
    ax = axes[2]
    ax.fill_between(f["date"], f["position"], step="post", color="#1E8449", alpha=0.55)
    ax.set_ylabel("仓位")
    ax.set_yticks([0, 1], ["空仓", "持有多头"])
    ax.set_ylim(-0.05, 1.25)
    ax.grid(True, axis="x", **GRID)

    _date_axis(axes[-1], years=1)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, Path(outdir) / f"06_signal_mechanics_{res.series}.png")


# --------------------------------------------------------------------------- #
# 07 阈值对比
# --------------------------------------------------------------------------- #
def plot_tol_sweep(
    curves: Mapping[str, pd.DataFrame],
    outdir: Path,
    *,
    title_extra: str = "",
) -> Path:
    """``curves``: {标签: 含 date/equity 的 DataFrame}。"""
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(13.5, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    first = True
    for i, (label, df) in enumerate(curves.items()):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(df["date"], df["equity"], color=color, linewidth=1.6, label=label)
        ax2.plot(
            df["date"],
            df["equity"] / df["benchmark_equity"],
            color=color,
            linewidth=1.4,
            label=label,
        )
        if first:
            ax.plot(
                df["date"],
                df["benchmark_equity"],
                color=BENCH_COLOR,
                linewidth=2.2,
                linestyle="-",
                label="基准（买入持有）",
            )
            first = False

    ax.axhline(1.0, color="#333333", linewidth=0.9, alpha=0.6)
    ax.set_ylabel("净值")
    ax.set_title(f"不同重合度阈值下的净值对比{('　' + title_extra) if title_extra else ''}")
    ax.legend(loc="upper left", ncols=3)

    ax2.axhline(1.0, color="#333333", linewidth=1.0)
    ax2.set_ylabel("策略 / 基准")
    ax2.legend(loc="upper left", ncols=3)
    _date_axis(ax2)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, Path(outdir) / "07_tol_sweep.png")


# --------------------------------------------------------------------------- #
# 08 综合信号 / 等权组合
# --------------------------------------------------------------------------- #
def plot_composite(result, outdir: Path) -> Path:
    """五指标等权综合：净值对比 + 票数/仓位结构。"""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.5, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4, 1.2]},
    )

    # --- 净值对比 ---
    ax = axes[0]
    for i, res in enumerate(result.per_indicator):
        ax.plot(
            res.frame["date"],
            res.frame["equity"],
            color=PALETTE[i % len(PALETTE)],
            linewidth=1.0,
            alpha=0.55,
            label=f"{res.series}（单指标）",
        )
    cf = result.composite_frame
    pf = result.portfolio_frame
    ax.plot(
        cf["date"],
        cf["benchmark_equity"],
        color=BENCH_COLOR,
        linewidth=2.2,
        linestyle="--",
        label=f"基准 买入持有  {cf['benchmark_equity'].iloc[-1]:.4f}",
    )
    ax.plot(
        pf["date"],
        pf["equity"],
        color="#117A8B",
        linewidth=2.0,
        label=f"等权组合 1/N  {pf['equity'].iloc[-1]:.4f}",
    )
    ax.plot(
        cf["date"],
        cf["equity"],
        color="#C0392B",
        linewidth=2.6,
        label=f"综合信号 {result.mode}  {cf['equity'].iloc[-1]:.4f}",
    )
    ax.axhline(1.0, color="#333333", linewidth=0.9, alpha=0.6)
    ax.set_ylabel("净值")
    cm = result.composite_metrics
    pm = result.portfolio_metrics
    ax.set_title(
        f"五利率等权综合（{result.mode} / 投票源={result.vote_source}）\n"
        f"综合信号：年化 {cm.get('strategy_cagr', float('nan')):.2%}  "
        f"夏普 {cm.get('strategy_sharpe', float('nan')):.2f}  "
        f"回撤 {cm.get('strategy_max_drawdown', float('nan')):.2%}   |   "
        f"等权组合：年化 {pm.get('strategy_cagr', float('nan')):.2%}  "
        f"夏普 {pm.get('strategy_sharpe', float('nan')):.2f}  "
        f"回撤 {pm.get('strategy_max_drawdown', float('nan')):.2%}"
    )
    ax.legend(loc="upper left", ncols=3, fontsize=8)

    # --- 票数结构 ---
    ax = axes[1]
    sf = result.signal_frame.dropna(subset=["score"])
    ax.fill_between(
        sf["date"], 0, sf["n_long"], step="mid", color="#1E8449", alpha=0.65, label="看多票数"
    )
    ax.fill_between(
        sf["date"], 0, -sf["n_short"], step="mid", color="#C0392B", alpha=0.65, label="看空票数"
    )
    ax.axhline(0, color="#333333", linewidth=1.0)
    n = len(result.rate_cols)
    ax.axhline(n / 2, color="#7F8C8D", linewidth=1.0, linestyle=":")
    ax.set_ylim(-n - 0.5, n + 0.5)
    ax.set_ylabel("票数")
    ax.set_title(f"逐日投票结构（共 {n} 个指标，虚线为过半门槛）", fontsize=11)
    ax.legend(loc="upper left", ncols=2)

    # --- 仓位对比 ---
    ax = axes[2]
    ax.fill_between(
        cf["date"],
        cf["position"],
        step="post",
        color="#C0392B",
        alpha=0.45,
        label="综合信号仓位（0/1）",
    )
    ax.plot(
        pf["date"], pf["position"], color="#117A8B", linewidth=1.2, label="等权组合仓位（连续 0~1）"
    )
    ax.set_ylabel("仓位")
    ax.set_ylim(-0.05, 1.25)
    ax.legend(loc="upper left", ncols=2)
    ax.grid(True, axis="x", **GRID)

    _date_axis(axes[-1])
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, Path(outdir) / "08_composite.png")


# --------------------------------------------------------------------------- #
# 汇总入口
# --------------------------------------------------------------------------- #
def make_all_charts(
    run,
    outdir: str | Path,
    *,
    signal_series: str | None = None,
    signal_window: tuple[str | None, str | None] | None = None,
    sweep_curves: Mapping[str, pd.DataFrame] | None = None,
    sweep_title: str = "",
) -> list[Path]:
    """生成全套图表，返回文件路径列表。"""
    setup_style()
    outdir = Path(outdir) / "charts"
    written: list[Path] = []

    written.append(plot_equity_curves(run, outdir))
    written.append(plot_per_series(run, outdir))
    written.append(plot_relative_strength(run, outdir))
    written.append(plot_drawdowns(run, outdir))
    written.append(plot_annual_returns(run, outdir))

    if run.results:
        target = None
        if signal_series:
            target = next((r for r in run.results if r.series == signal_series), None)
        if target is None:
            # 默认选夏普最高的指标来展示信号机理
            target = max(
                run.results,
                key=lambda r: (
                    r.metrics.get("strategy_sharpe")
                    if isinstance(r.metrics.get("strategy_sharpe"), float)
                    and np.isfinite(r.metrics["strategy_sharpe"])
                    else -np.inf
                ),
            )
        start, end = signal_window or (None, None)
        written.append(plot_signal_mechanics(target, outdir, start=start, end=end))

    if sweep_curves:
        written.append(plot_tol_sweep(sweep_curves, outdir, title_extra=sweep_title))

    return written


__all__ = [
    "annual_returns",
    "make_all_charts",
    "plot_annual_returns",
    "plot_composite",
    "plot_drawdowns",
    "plot_equity_curves",
    "plot_per_series",
    "plot_relative_strength",
    "plot_signal_mechanics",
    "plot_tol_sweep",
    "setup_style",
]
