"""回测引擎。

交易规则
--------
* T 日收盘后得到标记（``signal_eff``）；**T+1 日以收盘价**执行。
* T 日标记 +1：T+1 日买入（T 日无持仓）或继续持有（T 日有持仓）。
* T 日标记 -1：T+1 日卖出（T 日有持仓）或继续空仓（T 日无持仓）。
* 空仓时段不计利息（闲置资金收益为 0）。
* 交易成本默认 **万分之二、双边收取**（买入 2bp + 卖出 2bp）。

成本口径
--------
``cost_mode="per_side"``：``cost_bps`` 为**单边**成本，双边合计 2 × cost_bps（默认）。
``cost_mode="round_trip"``：``cost_bps`` 为**往返合计**成本，单边各收一半。

价格口径：收盘价。买入按 ``P*(1+c)`` 计算份额，卖出按 ``P*(1-c)`` 计算得款，
等价于对成交金额按比例计费，且成本在成交当日立即反映到净值上。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

CostMode = Literal["per_side", "round_trip"]
COST_MODES: tuple[str, ...] = ("per_side", "round_trip")

BUY = "BUY"
SELL = "SELL"


@dataclass(frozen=True)
class BacktestConfig:
    """回测参数。"""

    cost_bps: float = 2.0
    """成本，单位基点（1bp = 0.01%）。默认 2bp = 万分之二。"""

    cost_mode: str = "per_side"
    """``per_side``：cost_bps 为单边成本；``round_trip``：cost_bps 为往返合计。"""

    initial_capital: float = 1.0
    annualization: int = 252
    risk_free: float = 0.0

    def __post_init__(self) -> None:
        if self.cost_mode not in COST_MODES:
            raise ValueError(f"cost_mode 必须是 {COST_MODES} 之一")
        if self.cost_bps < 0:
            raise ValueError("cost_bps 必须非负")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital 必须为正")

    @property
    def cost_per_side(self) -> float:
        """单边成本（小数）。"""
        base = self.cost_bps / 10_000.0
        return base if self.cost_mode == "per_side" else base / 2.0

    @property
    def round_trip_cost(self) -> float:
        return 2.0 * self.cost_per_side

    def describe(self) -> str:
        if self.cost_mode == "per_side":
            return (
                f"单边 {self.cost_bps:g}bp（万分之{self.cost_bps:g}），"
                f"双边合计 {2 * self.cost_bps:g}bp"
            )
        return (
            f"往返合计 {self.cost_bps:g}bp（万分之{self.cost_bps:g}），单边 {self.cost_bps / 2:g}bp"
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "cost_bps": self.cost_bps,
            "cost_mode": self.cost_mode,
            "initial_capital": self.initial_capital,
            "annualization": self.annualization,
            "risk_free": self.risk_free,
        }
        data["cost_per_side_pct"] = self.cost_per_side * 100
        data["cost_description"] = self.describe()
        return data


@dataclass
class Trade:
    """一笔成交记录。"""

    date: pd.Timestamp
    action: str
    price: float
    units: float
    notional: float
    cost: float
    equity_after: float
    round_trip_id: int | None = None
    pnl_pct: float | None = None
    holding_days: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "action": self.action,
            "price": self.price,
            "units": self.units,
            "notional": self.notional,
            "cost": self.cost,
            "equity_after": self.equity_after,
            "round_trip_id": self.round_trip_id,
            "pnl_pct": self.pnl_pct,
            "holding_days": self.holding_days,
        }


@dataclass
class BacktestResult:
    """回测输出。"""

    frame: pd.DataFrame
    trades: list[Trade] = field(default_factory=list)
    config: BacktestConfig = field(default_factory=BacktestConfig)

    @property
    def equity(self) -> pd.Series:
        return self.frame.set_index("date")["equity"]

    @property
    def equity_gross(self) -> pd.Series:
        return self.frame.set_index("date")["equity_gross"]

    @property
    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "date",
                    "action",
                    "price",
                    "units",
                    "notional",
                    "cost",
                    "equity_after",
                    "round_trip_id",
                    "pnl_pct",
                    "holding_days",
                ]
            )
        return pd.DataFrame([t.to_dict() for t in self.trades])

    @property
    def n_round_trips(self) -> int:
        if not self.trades:
            return 0
        return len({t.round_trip_id for t in self.trades if t.round_trip_id is not None})

    @property
    def total_cost(self) -> float:
        return float(sum(t.cost for t in self.trades))


def run_backtest(
    dates: pd.Series | pd.Index,
    prices: pd.Series,
    signal_eff: pd.Series,
    cfg: BacktestConfig | None = None,
) -> BacktestResult:
    """执行回测。

    Parameters
    ----------
    dates, prices:
        交易日与标的收盘价（升序，等长）。
    signal_eff:
        T 日标记，取值 -1 / +1（重合日已在 :mod:`ratema.indicators` 中前向填充）。
        前导 NaN 表示均线未就绪，此期间不交易。
    cfg:
        见 :class:`BacktestConfig`。

    Returns
    -------
    :class:`BacktestResult`
    """
    cfg = cfg or BacktestConfig()

    dates = pd.Series(pd.to_datetime(pd.Series(dates).to_numpy())).reset_index(drop=True)
    prices = pd.Series(prices).astype("float64").reset_index(drop=True)
    signal_eff = pd.Series(signal_eff).astype("float64").reset_index(drop=True)

    if not (len(dates) == len(prices) == len(signal_eff)):
        raise ValueError("dates / prices / signal_eff 长度不一致")
    if len(dates) == 0:
        raise ValueError("输入为空")

    prices = prices.ffill()
    if prices.isna().any():
        raise ValueError("价格序列存在前导缺失值，无法回测")

    n = len(dates)
    # T 日标记 -> T+1 日执行
    actionable = signal_eff.shift(1)

    c = cfg.cost_per_side
    cash = float(cfg.initial_capital)
    units = 0.0
    units_gross = 0.0
    cash_gross = float(cfg.initial_capital)

    equity = np.empty(n, dtype="float64")
    equity_gross = np.empty(n, dtype="float64")
    cost_paid = np.zeros(n, dtype="float64")
    position = np.zeros(n, dtype="int8")

    trades: list[Trade] = []
    round_trip_id = 0
    entry_cash: float | None = None
    entry_index: int | None = None

    for t in range(n):
        price = float(prices.iloc[t])
        sig = actionable.iloc[t]

        if sig == 1.0 and units == 0.0:
            # ---- 买入 ----
            cash_before = cash
            units = cash / (price * (1.0 + c))
            traded_units = units
            notional = units * price
            cost = cash_before - notional
            cash = 0.0
            cost_paid[t] = cost
            round_trip_id += 1
            entry_cash = cash_before
            entry_index = t
            trades.append(
                Trade(
                    date=dates.iloc[t],
                    action=BUY,
                    price=price,
                    units=traded_units,
                    notional=notional,
                    cost=cost,
                    equity_after=notional,
                    round_trip_id=round_trip_id,
                )
            )
            # 无成本镜像
            units_gross = cash_gross / price
            cash_gross = 0.0
        elif sig == -1.0 and units > 0.0:
            # ---- 卖出 ----
            gross = units * price
            cost = gross * c
            proceeds = gross - cost
            cash = proceeds
            cost_paid[t] = cost
            holding_days = (t - entry_index) if entry_index is not None else None
            pnl = (proceeds / entry_cash - 1.0) if entry_cash else None
            trades.append(
                Trade(
                    date=dates.iloc[t],
                    action=SELL,
                    price=price,
                    units=units,
                    notional=gross,
                    cost=cost,
                    equity_after=proceeds,
                    round_trip_id=round_trip_id,
                    pnl_pct=pnl,
                    holding_days=holding_days,
                )
            )
            units = 0.0
            entry_cash = None
            entry_index = None
            # 无成本镜像
            cash_gross = units_gross * price
            units_gross = 0.0

        equity[t] = cash + units * price
        equity_gross[t] = cash_gross + units_gross * price
        position[t] = 1 if units > 0.0 else 0

    frame = pd.DataFrame(
        {
            "date": dates,
            "close": prices,
            "signal_eff": signal_eff,
            "actionable_signal": actionable,
            "position": position,
            "equity": equity,
            "equity_gross": equity_gross,
            "cost_paid": cost_paid,
        }
    )

    # 未平仓交易收尾：按最后一日收盘价盯市，不扣卖出成本
    open_trip = None
    if units > 0.0 and entry_cash is not None:
        open_trip = {
            "round_trip_id": round_trip_id,
            "entry_index": entry_index,
            "entry_cash": entry_cash,
            "mark_value": units * float(prices.iloc[-1]),
        }

    result = BacktestResult(frame=frame, trades=trades, config=cfg)
    result.frame.attrs["open_trip"] = open_trip  # type: ignore[attr-defined]
    return result


def build_strategy_frame(
    signals: pd.DataFrame,
    backtest: BacktestResult,
    benchmark_equity: pd.Series,
    benchmark_equity_gross: pd.Series,
) -> pd.DataFrame:
    """合并信号、持仓与净值，输出逐日明细表。

    ``signals`` 已经带有 ``signal_eff``，因此丢弃回测帧里的同名列，避免合并后
    出现 ``signal_eff_x`` / ``signal_eff_y``。
    """
    bt = backtest.frame.drop(columns=["signal_eff"], errors="ignore")
    frame = signals.merge(bt, on="date", how="inner")
    frame["benchmark_equity"] = benchmark_equity.to_numpy()
    frame["benchmark_equity_gross"] = benchmark_equity_gross.to_numpy()
    frame["daily_return"] = frame["equity"].pct_change().fillna(0.0)
    frame["benchmark_return"] = frame["benchmark_equity"].pct_change().fillna(0.0)
    frame["excess_return"] = frame["daily_return"] - frame["benchmark_return"]
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
    return frame


def run_benchmark(
    prices: pd.Series,
    cfg: BacktestConfig,
    *,
    charge_cost: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """基准：买入并持有中债-10年期国债净价(总值)指数。

    Returns
    -------
    (net_equity, gross_equity)
        ``net_equity`` 在首日按 ``P*(1+c)`` 建仓，期末仍持有（不扣卖出成本）。
    """
    prices = pd.Series(prices).astype("float64").reset_index(drop=True)
    entry = float(prices.iloc[0])
    gross = prices / entry * cfg.initial_capital
    net = gross / (1.0 + cfg.cost_per_side) if charge_cost else gross.copy()
    net.index = prices.index
    gross.index = prices.index
    return net, gross


__all__ = [
    "BUY",
    "COST_MODES",
    "SELL",
    "BacktestConfig",
    "BacktestResult",
    "Trade",
    "build_strategy_frame",
    "run_backtest",
    "run_benchmark",
]
