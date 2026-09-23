"""等权打分综合：把 5 个利率指标合成一个信号 / 一个组合。

两种「综合」的含义不同，本模块都实现并对比
------------------------------------------------
1. **综合信号（composite signal）**
   5 个指标各自给出 -1/+1 的投票，等权打分为净票数占比：

       score_t = (票数_多头 - 票数_空头) / N        ∈ [-1, +1]

   再按 ``mode`` 把 score 映射成单一信号，驱动**一个**仓位。
   这是「五票合成一个决策」。

2. **等权组合（equal-weight portfolio）**
   5 个子策略各分 1/N 资金独立交易，组合净值 = 各子策略净值曲线的算术平均。
   这是「五份资金各跟一个信号」，天然分散化，但换手与成本也是 5 份之和。

两者长期收益接近，但风险特征不同：综合信号只有一个仓位（每天要么全持有要么全空仓），
等权组合的仓位比例是连续的（持有票数/N），回撤更平滑。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .backtest import (
    BacktestConfig,
    BacktestResult,
    build_strategy_frame,
    run_backtest,
    run_benchmark,
)
from .indicators import overlap_stats
from .io_utils import DATE_COL, Dataset, resolve_roles
from .metrics import summarize

#: 综合信号的合成方式
COMPOSITE_MODES: tuple[str, ...] = ("score", "majority", "unanimous", "any")

MODE_HELP: dict[str, str] = {
    "score": "净票数 > 0 即为多头（等价于简单多数，默认）",
    "majority": "严格过半数为多头才算多头",
    "unanimous": "全部指标一致看多才持有（最保守，持仓时间最短）",
    "any": "只要有一个指标看多就持有（最激进，持仓时间最长）",
}

VOTE_SOURCES: tuple[str, ...] = ("eff", "raw")
VOTE_SOURCE_HELP: dict[str, str] = {
    "eff": "用有效信号投票：发生重合的指标沿用上一日观点（默认）",
    "raw": "用原始三态信号投票：发生重合的指标弃权（记 0 票）",
}


def _agg(n_long: np.ndarray, n_short: np.ndarray, n: int, mode: str) -> np.ndarray:
    """把票数按 mode 映射为 -1/+1/0（0 表示无法判定，需前向填充）。"""
    if mode == "score":
        net = (n_long - n_short) / n
        out = np.sign(net)
        return np.where(net == 0, 0.0, out)
    if mode == "majority":
        return np.where(n_long * 2 > n, 1.0, np.where(n_short * 2 > n, -1.0, 0.0))
    if mode == "unanimous":
        return np.where(n_long == n, 1.0, np.where(n_short == n, -1.0, 0.0))
    if mode == "any":
        return np.where(n_long > 0, 1.0, np.where(n_short > 0, -1.0, 0.0))
    raise ValueError(f"未知 mode: {mode!r}，应为 {COMPOSITE_MODES} 之一")


def build_composite_signal(
    signals_by_series: Mapping[str, pd.DataFrame],
    *,
    vote_source: str = "eff",
    mode: str = "score",
) -> pd.DataFrame:
    """把多个指标的信号等权打分为一个综合信号。

    Parameters
    ----------
    signals_by_series:
        {指标名: 含 ``date`` / ``signal_raw`` / ``signal_eff`` 的 DataFrame}。
        各表按 ``date`` 做外连接对齐。
    vote_source:
        ``eff`` 用有效信号投票（重合指标沿用前一日观点）；
        ``raw`` 用原始三态信号投票（重合指标弃权）。
    mode:
        见 :data:`COMPOSITE_MODES`。

    Returns
    -------
    DataFrame，列：date, n_long, n_short, n_flat, score, signal_raw, signal_eff
    """
    if not signals_by_series:
        raise ValueError("至少需要一个指标")
    if vote_source not in VOTE_SOURCES:
        raise ValueError(f"vote_source 应为 {VOTE_SOURCES} 之一")
    if mode not in COMPOSITE_MODES:
        raise ValueError(f"mode 应为 {COMPOSITE_MODES} 之一")

    names = list(signals_by_series)
    col = "signal_eff" if vote_source == "eff" else "signal_raw"

    parts = []
    for name in names:
        df = signals_by_series[name]
        missing = {"date", col} - set(df.columns)
        if missing:
            raise ValueError(f"{name} 的信号表缺少列：{sorted(missing)}")
        parts.append(df.set_index("date")[[col]].rename(columns={col: name}))

    merged = pd.concat(parts, axis=1).sort_index()
    votes = merged.to_numpy(dtype="float64")

    n = len(names)
    n_long = np.nansum(votes == 1.0, axis=1).astype("float64")
    n_short = np.nansum(votes == -1.0, axis=1).astype("float64")
    n_flat = np.nansum(votes == 0.0, axis=1).astype("float64")
    # 任一指标尚未就绪时不计票，该日整体不可判定
    ready = np.isfinite(votes).all(axis=1)

    signal_raw = _agg(n_long, n_short, n, mode)
    signal_raw = np.where(ready, signal_raw, np.nan)

    raw_series = pd.Series(signal_raw, index=merged.index, dtype="float64")
    # 规则 3 的推广：无法判定（平票 / 未就绪 / 全部弃权）时延续上一有效信号
    signal_eff = raw_series.mask(raw_series == 0.0).ffill()

    score = np.where(ready, (n_long - n_short) / n, np.nan)

    out = pd.DataFrame(
        {
            "date": merged.index,
            "n_long": n_long,
            "n_short": n_short,
            "n_flat": n_flat,
            "score": score,
            "signal_raw": raw_series.to_numpy(),
            "signal_eff": signal_eff.to_numpy(),
        }
    )
    out.attrs["mode"] = mode
    out.attrs["vote_source"] = vote_source
    out.attrs["series"] = names
    return out.reset_index(drop=True)


@dataclass
class CompositeResult:
    """两步走的完整结果。"""

    per_indicator: list[Any] = field(default_factory=list)
    """步骤 1：每个利率单独回测的 StrategyResult 列表。"""

    signal_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    """综合信号逐日明细（score / 票数 / signal_raw / signal_eff）。"""

    composite_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    """步骤 2a：用综合信号驱动单一仓位的逐日明细。"""

    composite_metrics: dict[str, Any] = field(default_factory=dict)

    composite_backtest: BacktestResult | None = None
    """综合信号的逐笔回测结果（含成交流水，供交易日志使用）。"""

    portfolio_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    """步骤 2b：等权组合（5 份资金各跟一个信号）的逐日明细。"""

    portfolio_metrics: dict[str, Any] = field(default_factory=dict)

    mode: str = "score"
    vote_source: str = "eff"
    index_col: str = ""
    rate_cols: list[str] = field(default_factory=list)

    def summary_frame(self) -> pd.DataFrame:
        """步骤 1 + 步骤 2 的一览表。"""
        rows: list[dict[str, Any]] = []
        for res in self.per_indicator:
            m = res.metrics
            rows.append(
                {
                    "方案": f"[单] {res.series}",
                    "类型": "单指标",
                    "年化收益": m.get("strategy_cagr"),
                    "年化波动": m.get("strategy_ann_vol"),
                    "夏普": m.get("strategy_sharpe"),
                    "最大回撤": m.get("strategy_max_drawdown"),
                    "卡玛": m.get("strategy_calmar"),
                    "持仓占比": m.get("time_in_market"),
                    "往返次数": m.get("n_completed"),
                    "成本拖累": m.get("cost_drag_cagr"),
                }
            )
        cm = self.composite_metrics
        rows.append(
            {
                "方案": f"[综合] 等权打分({self.mode})",
                "类型": "综合信号",
                "年化收益": cm.get("strategy_cagr"),
                "年化波动": cm.get("strategy_ann_vol"),
                "夏普": cm.get("strategy_sharpe"),
                "最大回撤": cm.get("strategy_max_drawdown"),
                "卡玛": cm.get("strategy_calmar"),
                "持仓占比": cm.get("time_in_market"),
                "往返次数": cm.get("n_completed"),
                "成本拖累": cm.get("cost_drag_cagr"),
            }
        )
        pm = self.portfolio_metrics
        rows.append(
            {
                "方案": "[综合] 等权组合(1/N)",
                "类型": "等权组合",
                "年化收益": pm.get("strategy_cagr"),
                "年化波动": pm.get("strategy_ann_vol"),
                "夏普": pm.get("strategy_sharpe"),
                "最大回撤": pm.get("strategy_max_drawdown"),
                "卡玛": pm.get("strategy_calmar"),
                "持仓占比": pm.get("time_in_market"),
                "往返次数": pm.get("n_completed"),
                "成本拖累": pm.get("cost_drag_cagr"),
            }
        )
        if self.per_indicator:
            bm = self.per_indicator[0].metrics
            rows.append(
                {
                    "方案": "[基准] 买入持有",
                    "类型": "基准",
                    "年化收益": None,
                    "年化波动": None,
                    "夏普": None,
                    "最大回撤": bm.get("benchmark_max_drawdown"),
                    "卡玛": None,
                    "持仓占比": 1.0,
                    "往返次数": 0,
                    "成本拖累": None,
                }
            )
            rows[-1]["年化收益"] = bm.get("benchmark_cagr")
            rows[-1]["年化波动"] = bm.get("benchmark_ann_vol")
            rows[-1]["夏普"] = bm.get("benchmark_sharpe")
        return pd.DataFrame(rows)


def run_composite(
    dataset: Dataset,
    per_indicator: Sequence[Any],
    backtest_cfg: BacktestConfig,
    *,
    index_col: str | None = None,
    mode: str = "score",
    vote_source: str = "eff",
    start: str | None = None,
    end: str | None = None,
) -> CompositeResult:
    """在步骤 1 的结果之上构建步骤 2 的综合信号与等权组合。

    ``per_indicator`` 为 :func:`ratema.pipeline.run_single` 的返回值列表。
    """
    if not per_indicator:
        raise ValueError("per_indicator 不能为空")

    if index_col is None:
        index_col, _ = resolve_roles(dataset, None, None)

    frame_all = dataset.frame
    dates_all = frame_all[DATE_COL]
    prices_all = frame_all[index_col]

    signals_by_series = {r.series: r.signals for r in per_indicator}
    signal_frame = build_composite_signal(signals_by_series, vote_source=vote_source, mode=mode)

    # 评估窗口：所有指标都就绪之后（各自 warm-up 的最大值）
    sig_eff = signal_frame["signal_eff"]
    valid_dates = signal_frame.loc[sig_eff.notna(), "date"]
    if valid_dates.empty:
        raise ValueError("综合信号全部为空，无法回测")
    first_date, last_date = valid_dates.iloc[0], valid_dates.iloc[-1]

    if start:
        first_date = max(first_date, pd.Timestamp(start))
    if end:
        last_date = min(last_date, pd.Timestamp(end))
    if last_date <= first_date:
        raise ValueError("综合信号的评估窗口为空")

    mask = (dates_all >= first_date) & (dates_all <= last_date)
    sub_dates = dates_all.loc[mask].reset_index(drop=True)
    sub_prices = prices_all.loc[mask].reset_index(drop=True)

    # 信号帧也必须按同一区间裁剪（此前只裁了价格，
    # 导致带 --start/--end 时长度不等、直接报错）
    ready_mask = sig_eff.notna().to_numpy()
    ready_frame = signal_frame.loc[ready_mask].reset_index(drop=True)
    ready_dates = pd.to_datetime(ready_frame["date"]).reset_index(drop=True)
    in_window = (ready_dates >= first_date) & (ready_dates <= last_date)
    ready_frame = ready_frame.loc[in_window].reset_index(drop=True)
    ready_dates = ready_dates.loc[in_window].reset_index(drop=True)

    # 显式校验对齐，避免长度不符时静默截断造成信号与日期错位
    if len(ready_dates) != len(sub_dates) or not np.array_equal(
        ready_dates.to_numpy(), sub_dates.to_numpy()
    ):
        raise ValueError(
            f"综合信号与价格序列未对齐：信号 {len(ready_dates)} 行 / 价格 {len(sub_dates)} 行"
        )
    sub_signal = ready_frame["signal_eff"].reset_index(drop=True)
    sub_sig_frame = ready_frame

    # ---- 2a 综合信号驱动单一仓位 ---------------------------------------- #
    bt = run_backtest(sub_dates, sub_prices, sub_signal, backtest_cfg)
    bench_net, bench_gross = run_benchmark(sub_prices, backtest_cfg, charge_cost=True)

    composite_signals = pd.DataFrame(
        {
            "date": sub_dates,
            "series": f"COMPOSITE({mode})",
            "value": sub_sig_frame["score"],
            "ma_short": np.nan,
            "ma_long": np.nan,
            "spread": np.nan,
            "spread_bp": np.nan,
            "tolerance": np.nan,
            "tolerance_bp": np.nan,
            "is_overlap": sub_sig_frame["signal_raw"].eq(0.0),
            "signal_raw": sub_sig_frame["signal_raw"],
            "signal_eff": sub_sig_frame["signal_eff"],
        }
    )
    composite_signals["overlap_band"] = ""
    composite_signals["signal_changed"] = composite_signals["signal_eff"].ne(
        composite_signals["signal_eff"].shift(1)
    )
    composite_signals = composite_signals[
        [
            "date",
            "series",
            "value",
            "ma_short",
            "ma_long",
            "spread",
            "spread_bp",
            "tolerance",
            "tolerance_bp",
            "overlap_band",
            "is_overlap",
            "signal_raw",
            "signal_eff",
            "signal_changed",
        ]
    ]

    composite_frame = build_strategy_frame(composite_signals, bt, bench_net, bench_gross)
    composite_frame["n_long"] = sub_sig_frame["n_long"].to_numpy()
    composite_frame["n_short"] = sub_sig_frame["n_short"].to_numpy()
    composite_frame["score"] = sub_sig_frame["score"].to_numpy()

    trades = bt.trades_frame
    composite_metrics = summarize(
        composite_frame.set_index("date")["equity"],
        composite_frame.set_index("date")["benchmark_equity"],
        gross_equity=composite_frame.set_index("date")["equity_gross"],
        trades=trades,
        position=composite_frame["position"],
        annualization=backtest_cfg.annualization,
        risk_free=backtest_cfg.risk_free,
    )
    ov = overlap_stats(composite_signals)
    composite_metrics.update(
        {
            "overlap_days": ov["n_overlap"],
            "overlap_share": ov["overlap_share"],
            "raw_signal_flips": ov["raw_signal_flips"],
            "signal_flips": ov["signal_flips"],
        }
    )

    # ---- 2b 等权组合（1/N 资金各跟一个信号） ----------------------------- #
    # 每个子策略的净值对初始资金线性，因此等权组合净值 = 各子策略净值曲线的算术平均
    eq_cols = []
    pos_cols = []
    gross_cols = []
    for res in per_indicator:
        s = res.frame.set_index("date")
        eq_cols.append(s["equity"].reindex(sub_dates))
        pos_cols.append(s["position"].reindex(sub_dates))
        gross_cols.append(s["equity_gross"].reindex(sub_dates))
    eq_mat = pd.concat(eq_cols, axis=1)
    pos_mat = pd.concat(pos_cols, axis=1)
    gross_mat = pd.concat(gross_cols, axis=1)
    for mat in (eq_mat, pos_mat, gross_mat):
        mat.columns = [r.series for r in per_indicator]

    portfolio_equity = eq_mat.mean(axis=1)
    portfolio_equity_gross = gross_mat.mean(axis=1)  # 零成本镜像的等权平均，精确
    portfolio_position = pos_mat.mean(axis=1)  # 连续仓位：持有票数 / N

    # 组合层面的成本：各子策略成本之和（已含在各自净值里）
    cost_mat = pd.concat(
        [r.frame.set_index("date")["cost_paid"].reindex(sub_dates) for r in per_indicator],
        axis=1,
    )
    portfolio_frame = pd.DataFrame(
        {
            "date": sub_dates,
            "equity": portfolio_equity.to_numpy(),
            "equity_gross": portfolio_equity_gross.to_numpy(),
            "position": portfolio_position.to_numpy(),
            "benchmark_equity": bench_net.to_numpy(),
            "benchmark_equity_gross": bench_gross.to_numpy(),
            "cost_paid": cost_mat.sum(axis=1).to_numpy() / len(per_indicator),
        }
    )
    portfolio_frame["daily_return"] = portfolio_frame["equity"].pct_change().fillna(0.0)
    portfolio_frame["benchmark_return"] = (
        portfolio_frame["benchmark_equity"].pct_change().fillna(0.0)
    )
    portfolio_frame["drawdown"] = (
        portfolio_frame["equity"] / portfolio_frame["equity"].cummax() - 1.0
    )

    portfolio_metrics = summarize(
        portfolio_frame.set_index("date")["equity"],
        portfolio_frame.set_index("date")["benchmark_equity"],
        gross_equity=portfolio_frame.set_index("date")["equity_gross"],
        trades=None,
        position=portfolio_frame["position"],
        annualization=backtest_cfg.annualization,
        risk_free=backtest_cfg.risk_free,
    )
    portfolio_metrics["n_completed"] = int(
        sum(r.metrics.get("n_completed", 0) for r in per_indicator)
    )
    portfolio_metrics["n_buys"] = int(sum(r.metrics.get("n_buys", 0) for r in per_indicator))
    portfolio_metrics["n_sells"] = int(sum(r.metrics.get("n_sells", 0) for r in per_indicator))

    return CompositeResult(
        per_indicator=list(per_indicator),
        signal_frame=signal_frame,
        composite_frame=composite_frame,
        composite_metrics=composite_metrics,
        composite_backtest=bt,
        portfolio_frame=portfolio_frame,
        portfolio_metrics=portfolio_metrics,
        mode=mode,
        vote_source=vote_source,
        index_col=index_col,
        rate_cols=[r.series for r in per_indicator],
    )


__all__ = [
    "COMPOSITE_MODES",
    "MODE_HELP",
    "VOTE_SOURCES",
    "VOTE_SOURCE_HELP",
    "CompositeResult",
    "build_composite_signal",
    "run_composite",
]
