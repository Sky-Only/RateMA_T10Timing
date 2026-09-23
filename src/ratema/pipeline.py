"""端到端流程：单个指标 -> 信号 -> 回测 -> 指标汇总。"""

from __future__ import annotations

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
from .indicators import (
    SignalConfig,
    compute_signals,
    overlap_stats,
    spread_calibration,
)
from .io_utils import DATE_COL, Dataset, resolve_roles
from .metrics import summarize


@dataclass
class StrategyResult:
    """单个底层利率指标的策略回测结果。"""

    series: str
    display_name: str
    signals: pd.DataFrame
    full_signals: pd.DataFrame
    backtest: BacktestResult
    frame: pd.DataFrame
    metrics: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    overlap: dict[str, Any] = field(default_factory=dict)
    benchmark_metrics: dict[str, Any] = field(default_factory=dict)
    warmup_days: int = 0
    open_trip: dict[str, Any] | None = None
    window: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.display_name and self.display_name != self.series:
            return f"{self.series} ({self.display_name})"
        return self.series


@dataclass
class RunResult:
    """一次完整运行（多指标）的结果集合。"""

    dataset: Dataset
    index_col: str
    rate_cols: list[str]
    signal_cfg: SignalConfig
    backtest_cfg: BacktestConfig
    results: list[StrategyResult] = field(default_factory=list)
    benchmark: dict[str, Any] = field(default_factory=dict)
    start: str = ""
    end: str = ""

    def summary_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for res in self.results:
            row: dict[str, Any] = {
                "series": res.series,
                "series_name": res.display_name,
            }
            row.update(res.metrics)
            row["warmup_days"] = res.warmup_days
            row["tol_description"] = self.signal_cfg.describe_tol()
            rows.append(row)
        return pd.DataFrame(rows)


def run_single(
    dataset: Dataset,
    series: str,
    signal_cfg: SignalConfig,
    backtest_cfg: BacktestConfig,
    *,
    index_col: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> StrategyResult:
    """对单个底层利率指标执行完整流程。

    ``index_col`` 为 None 时自动识别标的指数列（优先 CBA04502.CS）。
    """
    if index_col is None:
        index_col, _ = resolve_roles(dataset, None, None)

    frame = dataset.frame
    dates = frame[DATE_COL]
    prices_all = frame[index_col]
    values_all = frame[series]

    # 保留完整信号（含均线未就绪期），用于详细输出
    full_signals = compute_signals(dates, values_all, signal_cfg, series_name=series)

    # 评估窗口：从首个有效信号日开始（跳过 MA 未就绪的预热期）
    valid = full_signals["signal_eff"].notna() & prices_all.notna()
    if not valid.any():
        if full_signals["signal_raw"].notna().any():
            raise ValueError(
                f"{series}: 在阈值「{signal_cfg.describe_tol()}」下，全部交易日的 "
                f"|MA{signal_cfg.short_window}-MA{signal_cfg.long_window}| 都落在重合区间内，"
                f"始终无法产生第一个有效信号，因此无仓可建。"
                f"请减小 --tol，或缩短均线窗口。"
            )
        raise ValueError(
            f"{series}: 数据长度不足以形成 {signal_cfg.long_window} 日均线，无法产生信号"
        )
    first_idx = int(np.argmax(valid.to_numpy()))
    last_idx = len(frame) - 1

    if start:
        start_ts = pd.Timestamp(start)
        later = np.flatnonzero((dates >= start_ts).to_numpy())
        if later.size:
            first_idx = max(first_idx, int(later[0]))
    if end:
        end_ts = pd.Timestamp(end)
        earlier = np.flatnonzero((dates <= end_ts).to_numpy())
        if earlier.size:
            last_idx = min(last_idx, int(earlier[-1]))
    if last_idx <= first_idx:
        raise ValueError(f"{series}: 指定的回测窗口为空")

    idx = slice(first_idx, last_idx + 1)
    sub_dates = dates.iloc[idx].reset_index(drop=True)
    sub_prices = prices_all.iloc[idx].reset_index(drop=True)
    sub_signal = full_signals["signal_eff"].iloc[idx].reset_index(drop=True)

    backtest = run_backtest(sub_dates, sub_prices, sub_signal, backtest_cfg)

    bench_net, bench_gross = run_benchmark(sub_prices, backtest_cfg, charge_cost=True)
    strategy_frame = build_strategy_frame(
        full_signals.iloc[idx].reset_index(drop=True),
        backtest,
        bench_net,
        bench_gross,
    )
    strategy_frame = strategy_frame.merge(
        frame[[DATE_COL, series]].rename(columns={series: "rate_value"}),
        on=DATE_COL,
        how="left",
    )

    trades = backtest.trades_frame
    metrics = summarize(
        strategy_frame.set_index("date")["equity"],
        strategy_frame.set_index("date")["benchmark_equity"],
        gross_equity=strategy_frame.set_index("date")["equity_gross"],
        trades=trades,
        position=strategy_frame["position"],
        annualization=backtest_cfg.annualization,
        risk_free=backtest_cfg.risk_free,
    )
    metrics["strategy_start"] = strategy_frame[DATE_COL].iloc[0]
    metrics["strategy_end"] = strategy_frame[DATE_COL].iloc[-1]

    bench_metrics = summarize(
        strategy_frame.set_index("date")["benchmark_equity"],
        strategy_frame.set_index("date")["benchmark_equity_gross"],
        annualization=backtest_cfg.annualization,
        risk_free=backtest_cfg.risk_free,
    )

    overlap = overlap_stats(full_signals.iloc[idx].reset_index(drop=True))
    metrics["overlap_days"] = overlap["n_overlap"]
    metrics["overlap_share"] = overlap["overlap_share"]
    metrics["raw_signal_flips"] = overlap["raw_signal_flips"]
    metrics["signal_flips"] = overlap["signal_flips"]

    # ---- 评估窗口口径（可审计）------------------------------------------ #
    position = strategy_frame["position"].to_numpy()
    first_pos = int(np.argmax(position)) if position.any() else None
    window = {
        "data_start": dates.iloc[0],
        "data_end": dates.iloc[-1],
        "warmup_days": first_idx,
        "warmup_start": dates.iloc[0],
        "warmup_end": dates.iloc[first_idx - 1] if first_idx > 0 else None,
        # 评估起点 = signal_eff 首次非空的交易日（首个有效信号日）
        "eval_start": strategy_frame[DATE_COL].iloc[0],
        "eval_end": strategy_frame[DATE_COL].iloc[-1],
        "n_eval_days": len(strategy_frame),
        "first_signal_value": float(sub_signal.iloc[0]),
        "first_signal_date": strategy_frame[DATE_COL].iloc[0],
        # T+1 执行：信号日当天不持仓，首笔成交发生在之后的第一个可执行日
        "first_position_date": (
            strategy_frame[DATE_COL].iloc[first_pos] if first_pos is not None else None
        ),
        "first_position_lag_days": first_pos,
        "benchmark_entry_date": strategy_frame[DATE_COL].iloc[0],
        "benchmark_entry_price": float(sub_prices.iloc[0]),
        "benchmark_charge_cost": True,
    }

    return StrategyResult(
        series=series,
        display_name=dataset.display_name(series),
        signals=full_signals.iloc[idx].reset_index(drop=True),
        full_signals=full_signals,
        backtest=backtest,
        frame=strategy_frame,
        metrics=metrics,
        calibration=spread_calibration(full_signals, series_name=series),
        overlap=overlap,
        benchmark_metrics=bench_metrics,
        warmup_days=first_idx,
        open_trip=backtest.frame.attrs.get("open_trip"),
        window=window,
    )


def run_all(
    dataset: Dataset,
    signal_cfg: SignalConfig,
    backtest_cfg: BacktestConfig,
    *,
    index_col: str | None = None,
    rate_cols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> RunResult:
    """对全部底层利率指标执行完整流程。"""
    index_col, resolved_rates = resolve_roles(dataset, index_col, rate_cols)
    results = [
        run_single(
            dataset,
            series,
            signal_cfg,
            backtest_cfg,
            index_col=index_col,
            start=start,
            end=end,
        )
        for series in resolved_rates
    ]

    dates = dataset.frame[DATE_COL]
    bench_full = dataset.frame[index_col]
    b_first = dates.iloc[results[0].warmup_days] if results[0].warmup_days < len(dates) else None
    benchmark = {
        "index_col": index_col,
        "index_name": dataset.display_name(index_col),
        "start": str(b_first.date()) if b_first is not None else "",
        "end": str(dates.iloc[-1].date()),
        "n_obs": len(bench_full),
    }

    return RunResult(
        dataset=dataset,
        index_col=index_col,
        rate_cols=resolved_rates,
        signal_cfg=signal_cfg,
        backtest_cfg=backtest_cfg,
        results=results,
        benchmark=benchmark,
        start=str(results[0].frame[DATE_COL].iloc[0].date()) if results else "",
        end=str(results[0].frame[DATE_COL].iloc[-1].date()) if results else "",
    )


__all__ = ["RunResult", "StrategyResult", "run_all", "run_single"]
