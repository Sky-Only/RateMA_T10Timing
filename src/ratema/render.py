"""表格与 Markdown 渲染（纯函数，不落盘）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .indicators import TOL_MODE_HELP
from .pipeline import RunResult

#: 汇总表的列定义：(metrics 键, 中文标签, 格式)
SUMMARY_METRICS: list[tuple[str, str, str]] = [
    ("strategy_total_return", "策略累计收益", "pct"),
    ("strategy_cagr", "策略年化收益", "pct"),
    ("strategy_ann_vol", "策略年化波动", "pct"),
    ("strategy_sharpe", "策略夏普", "num"),
    ("strategy_sortino", "策略索提诺", "num"),
    ("strategy_max_drawdown", "策略最大回撤", "pct"),
    ("strategy_calmar", "策略卡玛", "num"),
    ("benchmark_total_return", "基准累计收益", "pct"),
    ("benchmark_cagr", "基准年化收益", "pct"),
    ("benchmark_max_drawdown", "基准最大回撤", "pct"),
    ("excess_cagr", "年化超额", "pct"),
    ("cost_drag_cagr", "成本拖累(年化)", "pct"),
    ("time_in_market", "持仓时间占比", "pct"),
    ("n_completed", "完成往返次数", "int"),
    ("n_buys", "买入次数", "int"),
    ("n_sells", "卖出次数", "int"),
    ("trade_win_rate", "交易胜率", "pct"),
    ("avg_holding_days", "平均持有天数", "num"),
    ("overlap_days", "重合日数", "int"),
    ("overlap_share", "重合日占比", "pct"),
    ("raw_signal_flips", "原始信号切换", "int"),
    ("signal_flips", "交易信号切换", "int"),
]

#: 阈值敏感性分析的透视列
SWEEP_COLUMNS: list[tuple[str, str, str]] = [
    ("strategy_cagr", "年化收益", "pct"),
    ("strategy_ann_vol", "年化波动", "pct"),
    ("strategy_sharpe", "夏普", "num"),
    ("strategy_max_drawdown", "最大回撤", "pct"),
    ("strategy_calmar", "卡玛", "num"),
    ("excess_cagr", "年化超额", "pct"),
    ("overlap_days", "重合日数", "int"),
    ("overlap_share", "重合日占比", "pct"),
    ("signal_flips", "交易信号切换", "int"),
    ("n_completed", "往返次数", "int"),
    ("time_in_market", "持仓占比", "pct"),
]

#: 综合信号对照表的列（键名与 summary_frame 的中文列名一致）
COMPOSITE_COLUMNS: list[tuple[str, str, str]] = [
    ("年化收益", "年化收益", "pct"),
    ("年化波动", "年化波动", "pct"),
    ("夏普", "夏普", "num"),
    ("最大回撤", "最大回撤", "pct"),
    ("卡玛", "卡玛", "num"),
    ("持仓占比", "持仓占比", "pct"),
    ("往返次数", "往返次数", "int"),
    ("成本拖累", "成本拖累", "pct"),
]


def d(value: Any) -> str:
    """日期格式化，缺失值返回 n/a。"""
    if value is None or value is pd.NaT:
        return "n/a"
    ts = pd.to_datetime(value, errors="coerce")
    return "n/a" if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def fmt(value: Any, kind: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "n/a"
    if kind == "pct":
        try:
            return f"{float(value) * 100:.2f}%"
        except (TypeError, ValueError):
            return str(value)
    if kind == "int":
        try:
            return f"{int(value)}"
        except (TypeError, ValueError):
            return str(value)
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def summary_table(run: RunResult) -> str:
    """生成核心指标 Markdown 表。"""
    rows = []
    for res in run.results:
        row = {"指标": res.label}
        for key, label, kind in SUMMARY_METRICS:
            row[label] = fmt(res.metrics.get(key), kind)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("指标")
    return df.to_markdown()


def calibration_table(run: RunResult) -> str:
    """生成「重合度阈值」标定表。"""
    rows = []
    for res in run.results:
        cal = res.calibration
        row = {
            "指标": res.label,
            "|spread| 均值(bp)": f"{cal.get('mean_abs_bp', float('nan')):.2f}",
            "|spread| 中位(bp)": f"{cal.get('p50_abs_bp', float('nan')):.2f}",
            "p10(bp)": f"{cal.get('p10_abs_bp', float('nan')):.2f}",
            "p05(bp)": f"{cal.get('p05_abs_bp', float('nan')):.2f}",
            "p01(bp)": f"{cal.get('p01_abs_bp', float('nan')):.2f}",
            "最大(bp)": f"{cal.get('max_abs_bp', float('nan')):.2f}",
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("指标").to_markdown()


def render_markdown(run: RunResult) -> str:
    cfg = run.signal_cfg
    bt = run.backtest_cfg
    lines: list[str] = []
    lines.append("# 20/120 日均线利率择时策略回测报告")
    lines.append("")
    lines.append("## 1. 运行配置")
    lines.append("")
    lines.append(f"- 数据源：`{run.dataset.source}`")
    lines.append(f"- 标的指数：`{run.index_col}`（{run.dataset.display_name(run.index_col)}）")
    lines.append(f"- 底层利率指标：{', '.join(f'`{c}`' for c in run.rate_cols)}")
    lines.append(f"- 回测区间：{run.start} ~ {run.end}")
    lines.append(f"- 均线：{cfg.short_window} 日 / {cfg.long_window} 日简单移动平均")
    lines.append(f"- 重合判定模式：`{cfg.tol_mode}`，阈值 {cfg.tol:g}")
    lines.append(f"- 重合区间定义：{cfg.describe_tol()}")
    lines.append(f"- 交易成本：{bt.describe()}（价格口径：收盘价）")
    lines.append("- 执行时滞：T 日信号，T+1 日收盘执行；空仓不计利息")
    lines.append(f"- 年化系数：{bt.annualization} 个交易日")
    lines.append("")
    lines.append("## 2. 策略指标定义")
    lines.append("")
    lines.append("| 条件 | 标记 | 含义 |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| MA{cfg.short_window} > MA{cfg.long_window} | -1 | 短期资金成本高于中长期水平，流动性边际趋紧 → 债券空头 |"
    )
    lines.append(
        f"| MA{cfg.short_window} < MA{cfg.long_window} | +1 | 短期资金成本低于中长期水平，流动性相对宽松 → 债券多头 |"
    )
    lines.append(
        f"| \\|MA{cfg.short_window} - MA{cfg.long_window}\\| ≤ 重合阈值 | 0 | 两线重合，延续上一交易日的有效信号 |"
    )
    lines.append("")
    lines.append("## 3. 评估窗口与起算口径")
    lines.append("")
    lines.append(
        "**评估起点定义为「首个有效信号日」**，即 `signal_eff` 首次非空的交易日"
        "（跳过 MA 未就绪的预热期）。策略与基准在同一日期、同一初始资金上归一，"
        "预热期不计入任何绩效指标。"
    )
    lines.append("")
    if run.results:
        first = run.results[0]
        w = first.window
        lines.append(f"以 `{first.series}` 为例：")
        lines.append("")
        lines.append("| 项目 | 值 |")
        lines.append("| --- | --- |")
        lines.append(f"| 数据区间 | {d(w.get('data_start'))} ~ {d(w.get('data_end'))} |")
        lines.append(
            f"| MA{cfg.long_window} 预热期 | {d(w.get('warmup_start'))} ~ {d(w.get('warmup_end'))}"
            f"（{w.get('warmup_days')} 个交易日，信号为空，不计入） |"
        )
        lines.append(f"| **评估起点（首个有效信号日）** | **{d(w.get('eval_start'))}** |")
        lines.append(f"| 评估终点 | {d(w.get('eval_end'))} |")
        lines.append(f"| 评估交易日数 | {w.get('n_eval_days')} |")
        sig = w.get("first_signal_value")
        lines.append(
            f"| 首个有效信号 | {sig:+.0f}（{'多头' if sig == 1 else '空头/空仓'}） |"
            if sig is not None
            else "| 首个有效信号 | n/a |"
        )
        lag = w.get("first_position_lag_days")
        lines.append(
            f"| 首个建仓日 | {d(w.get('first_position_date'))}"
            f"（信号日后第 {lag} 个交易日；T+1 执行，且期间无 +1 信号） |"
        )
        lines.append(
            f"| 基准建仓 | {d(w.get('benchmark_entry_date'))} @ {w.get('benchmark_entry_price'):.4f}"
            f"（含 {bt.cost_per_side * 100:.4f}% 建仓成本） |"
        )
        lines.append("")
        if lag is not None and lag > 1:
            lines.append(
                f"> 注意：首个有效信号为 {sig:+.0f}，因此评估起点当天及之后的一段时间"
                f"策略保持空仓，直到 {d(w.get('first_position_date'))} 才首次建仓。"
                f"该空仓期正是信号本身的结论（空头区间不参与），基准同期满仓。"
            )
            lines.append("")
        lines.append(
            "各指标的评估起点一致（均为 MA 预热结束后的首个有效信号日）；"
            "逐指标明细见 `evaluation_window.csv`。"
        )
        lines.append("")
    lines.append("## 4. 策略绩效 vs 基准")
    lines.append("")
    lines.append(summary_table(run))
    lines.append("")
    lines.append(f"> 基准为买入并持有 `{run.index_col}`（首日按含成本价建仓）。")
    lines.append("")
    lines.append("## 5. 重合度阈值标定")
    lines.append("")
    lines.append("下表统计各指标的 `|MA快 - MA慢|` 分布（单位 bp，1bp = 0.01 个百分点），")
    lines.append("可用于选择「重合」阈值：例如取 p05，则历史上约 5% 的交易日会被判定为重合。")
    lines.append("")
    lines.append(calibration_table(run))
    lines.append("")
    if run.results:
        first = run.results[0]
        lines.append(f"### 5.1 当前阈值下的重合情况（以 `{first.series}` 为例）")
        lines.append("")
        lines.append(f"- 评估区间交易日数：{first.overlap.get('n_valid')}")
        lines.append(f"- 判定为重合的交易日：{first.overlap.get('n_overlap')}")
        share = first.overlap.get("overlap_share")
        lines.append(f"- 重合日占比：{'n/a' if share is None else f'{share:.2%}'}")
        lines.append(f"- 原始信号切换次数：{first.overlap.get('raw_signal_flips')}")
        lines.append(f"- 实际交易信号切换次数：{first.overlap.get('signal_flips')}")
        lines.append(f"- 被重合区间吸收的噪声切换：{first.overlap.get('absorbed_flips')}")
        lines.append("")
        if not first.overlap.get("n_overlap"):
            lines.append(
                "> ⚠️ 当前阈值下没有任何交易日被判为重合，规则 3（延续上一交易日"
                "有效信号）实际未被触发。若希望该规则生效，请放宽阈值，例如："
            )
            lines.append(">")
            lines.append("> ```bash")
            lines.append(
                "> uv run ratema sweep --tol-mode bp --tol-grid 0,1,2,5,10   # 先看阈值敏感性"
            )
            lines.append(
                "> uv run ratema backtest --tol-mode bp --tol 2             # 再选定阈值回测"
            )
            lines.append("> ```")
            lines.append("")
    lines.append("## 6. 产出文件")
    lines.append("")
    lines.append("| 路径 | 说明 |")
    lines.append("| --- | --- |")
    lines.append("| `summary.csv` | 各指标核心绩效汇总 |")
    lines.append("| `metrics.json` | 完整绩效指标与运行参数 |")
    lines.append("| `evaluation_window.csv` | 各指标评估窗口与起算口径 |")
    lines.append("| `spread_calibration.csv` | 阈值标定（spread 分位数） |")
    lines.append("| `equity_curves.csv` | 所有策略与基准的净值曲线 |")
    lines.append("| `equity/<指标>.csv` | 单指标逐日净值 / 回撤 |")
    lines.append("| `details/<指标>.csv` | 单指标逐日信号与持仓明细 |")
    lines.append("| `trades/<指标>.csv` | 单指标成交流水 |")
    lines.append("")
    return "\n".join(lines)


def render_composite_markdown(result, meta: dict[str, Any] | None = None) -> str:
    from .composite import MODE_HELP, VOTE_SOURCE_HELP

    summary = result.summary_frame()
    lines = ["# 两步走：单指标测试 → 五利率等权综合", ""]
    lines.append("## 运行配置")
    lines.append("")
    if meta:
        for key, value in meta.items():
            lines.append(f"- {key}：{value}")
    lines.append(f"- 综合方式：`{result.mode}` —— {MODE_HELP.get(result.mode, '')}")
    lines.append(
        f"- 投票来源：`{result.vote_source}` —— {VOTE_SOURCE_HELP.get(result.vote_source, '')}"
    )
    lines.append(f"- 参与综合的指标：{', '.join(f'`{c}`' for c in result.rate_cols)}")
    lines.append("")

    lines.append("## 步骤 1：每个利率单独测试")
    lines.append("")
    lines.append(composite_table(summary[summary["类型"] == "单指标"].reset_index(drop=True)))
    lines.append("")
    single = summary[summary["类型"] == "单指标"]
    if len(single):
        best = single.loc[single["夏普"].idxmax()]
        worst = single.loc[single["夏普"].idxmin()]
        lines.append(
            f"单指标中夏普最高：{best['方案']}（{fmt(best['夏普'], 'num')}），"
            f"最低：{worst['方案']}（{fmt(worst['夏普'], 'num')}）；"
            f"平均年化 {fmt(single['年化收益'].mean(), 'pct')}，"
            f"平均夏普 {fmt(single['夏普'].mean(), 'num')}。"
        )
        lines.append("")

    lines.append("## 步骤 2：五利率等权打分综合")
    lines.append("")
    lines.append(
        "打分公式：`score = (看多票数 − 看空票数) / N`，再按上面的「综合方式」映射为单一信号。"
    )
    lines.append("")
    lines.append(composite_table(summary[summary["类型"] != "单指标"].reset_index(drop=True)))
    lines.append("")

    cm, pm = result.composite_metrics, result.portfolio_metrics
    lines.append("### 两种「综合」的区别")
    lines.append("")
    lines.append("| | 综合信号 | 等权组合 |")
    lines.append("| --- | --- | --- |")
    lines.append("| 含义 | 五票合成**一个**决策，仓位 0/1 | 五份资金各跟一个信号，仓位连续 |")
    lines.append(
        f"| 年化 | {fmt(cm.get('strategy_cagr'), 'pct')} | {fmt(pm.get('strategy_cagr'), 'pct')} |"
    )
    lines.append(
        f"| 夏普 | {fmt(cm.get('strategy_sharpe'), 'num')} | {fmt(pm.get('strategy_sharpe'), 'num')} |"
    )
    lines.append(
        f"| 最大回撤 | {fmt(cm.get('strategy_max_drawdown'), 'pct')} | "
        f"{fmt(pm.get('strategy_max_drawdown'), 'pct')} |"
    )
    lines.append(
        f"| 往返次数 | {fmt(cm.get('n_completed'), 'int')} | {fmt(pm.get('n_completed'), 'int')} |"
    )
    lines.append("")
    if len(single):
        avg_sharpe = float(single["夏普"].mean())
        best_sharpe = float(single["夏普"].max())
        comp_sharpe = cm.get("strategy_sharpe")
        if isinstance(comp_sharpe, float):
            lines.append(
                f"**结论**：综合信号夏普 {comp_sharpe:.3f}，"
                f"高于单指标平均 {avg_sharpe:.3f}"
                + (
                    f"，也高于最好的单指标 {best_sharpe:.3f}。"
                    if comp_sharpe > best_sharpe
                    else f"，但低于最好的单指标 {best_sharpe:.3f}。"
                )
            )
            lines.append("")
    lines.append("> 等权组合的年化收益在数学上恒等于各单指标年化收益的算术平均")
    lines.append("> （因为每个子策略的净值对初始资金线性）。它的价值在于**降低波动**，")
    lines.append("> 而不是提高收益；综合信号则通过「只在共识足够强时持有」来改善择时。")
    lines.append("")
    return "\n".join(lines)


def composite_table(summary: pd.DataFrame) -> str:
    out = summary.copy()
    for src, label, kind in COMPOSITE_COLUMNS:
        if src in out.columns:
            out[label] = out[src].map(lambda v, k=kind: fmt(v, k))
    keep = ["方案", "类型"] + [c for _, c, _ in COMPOSITE_COLUMNS if c in out.columns]
    return out[keep].to_markdown(index=False)


def render_sweep_markdown(df: pd.DataFrame, *, tol_mode: str, meta: dict[str, Any]) -> str:
    lines = ["# 重合度阈值敏感性分析", ""]
    lines.append(f"- 阈值模式：`{tol_mode}` —— {TOL_MODE_HELP.get(tol_mode, '')}")
    for key, value in meta.items():
        lines.append(f"- {key}：{value}")
    lines.append("")

    for key, label, kind in SWEEP_COLUMNS:
        if key not in df.columns:
            continue
        lines.append(f"## {label}（行=指标，列=阈值）")
        lines.append("")
        pivot = df.pivot_table(index="series", columns="tol", values=key, aggfunc="last")
        pivot = pivot.map(lambda v, k=kind: fmt(v, k))
        pivot.columns = [f"tol={c:g}" for c in pivot.columns]
        lines.append(pivot.reset_index().to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)
