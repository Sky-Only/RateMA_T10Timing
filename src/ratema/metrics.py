"""绩效指标计算。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """最大回撤、峰值日、谷底日。"""
    equity = equity.dropna()
    if equity.empty:
        return float("nan"), None, None
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough = dd.idxmin()
    if pd.isna(trough):
        return float("nan"), None, None
    peak_date = equity.loc[:trough].idxmax()
    return float(dd.loc[trough]), peak_date, trough


def max_drawdown_duration(equity: pd.Series) -> int:
    """最长回撤持续天数（净值处于前高之下）。"""
    equity = equity.dropna()
    if equity.empty:
        return 0
    under = (equity < equity.cummax()).to_numpy()
    longest = current = 0
    for flag in under:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def perf_stats(
    equity: pd.Series,
    *,
    annualization: int = TRADING_DAYS,
    risk_free: float = 0.0,
) -> dict[str, Any]:
    """由净值序列计算常用绩效指标。"""
    eq = pd.Series(equity).astype("float64").dropna()
    if len(eq) < 2:
        return {"n_obs": len(eq)}

    n_obs = len(eq)
    years = n_obs / annualization
    start_value = float(eq.iloc[0])
    end_value = float(eq.iloc[-1])
    if start_value <= 0:
        return {"n_obs": n_obs}

    total_return = end_value / start_value - 1.0
    cagr = (end_value / start_value) ** (1.0 / years) - 1.0 if years > 0 else float("nan")

    returns = eq.pct_change().dropna()
    if returns.empty or returns.std(ddof=1) == 0 or not np.isfinite(returns.std(ddof=1)):
        ann_vol = 0.0
        sharpe = float("nan")
        sortino = float("nan")
    else:
        std = float(returns.std(ddof=1))
        ann_vol = std * np.sqrt(annualization)
        excess = float(returns.mean()) - risk_free / annualization
        sharpe = excess / std * np.sqrt(annualization)
        downside = returns[returns < 0]
        down_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        sortino = (
            excess * annualization / (down_std * np.sqrt(annualization))
            if down_std > 0
            else float("nan")
        )

    mdd, peak_date, trough_date = max_drawdown(eq)
    calmar = cagr / abs(mdd) if mdd and np.isfinite(mdd) and mdd < 0 else float("nan")

    var95 = float(returns.quantile(0.05)) if not returns.empty else float("nan")
    cvar95 = (
        float(returns[returns <= var95].mean())
        if not returns.empty and returns.le(var95).any()
        else float("nan")
    )

    return {
        "n_obs": n_obs,
        "start": eq.index[0],
        "end": eq.index[-1],
        "years": years,
        "start_value": start_value,
        "end_value": end_value,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "max_drawdown_peak": peak_date,
        "max_drawdown_trough": trough_date,
        "max_drawdown_days": max_drawdown_duration(eq),
        "calmar": calmar,
        "daily_win_rate": float((returns > 0).mean()) if not returns.empty else float("nan"),
        "best_day": float(returns.max()) if not returns.empty else float("nan"),
        "worst_day": float(returns.min()) if not returns.empty else float("nan"),
        "var95_daily": var95,
        "cvar95_daily": cvar95,
    }


def trade_stats(trades: pd.DataFrame) -> dict[str, Any]:
    """交易层面统计（以已平仓的往返交易为单位）。"""
    if trades is None or trades.empty:
        return {
            "n_trades": 0,
            "n_buys": 0,
            "n_sells": 0,
            "n_round_trips": 0,
            "n_completed": 0,
            "trade_win_rate": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "profit_factor": float("nan"),
            "avg_holding_days": float("nan"),
            "total_cost": 0.0,
        }

    buys = trades[trades["action"] == "BUY"]
    sells = trades[trades["action"] == "SELL"]
    closed = sells["pnl_pct"].dropna()
    wins = closed[closed > 0]
    losses = closed[closed <= 0]
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))

    return {
        "n_trades": len(trades),
        "n_buys": len(buys),
        "n_sells": len(sells),
        "n_round_trips": int(trades["round_trip_id"].nunique()),
        "n_completed": len(closed),
        "trade_win_rate": float((closed > 0).mean()) if len(closed) else float("nan"),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "best_trade": float(closed.max()) if len(closed) else float("nan"),
        "worst_trade": float(closed.min()) if len(closed) else float("nan"),
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("nan"),
        "avg_holding_days": (
            float(sells["holding_days"].dropna().mean()) if len(sells) else float("nan")
        ),
        "total_cost": float(trades["cost"].sum()),
    }


def summarize(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    *,
    gross_equity: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
    position: pd.Series | None = None,
    annualization: int = TRADING_DAYS,
    risk_free: float = 0.0,
) -> dict[str, Any]:
    """合并策略 / 基准 / 交易层面指标。"""
    stats = perf_stats(strategy_equity, annualization=annualization, risk_free=risk_free)
    stats = {f"strategy_{k}": v for k, v in stats.items()}

    bench = perf_stats(benchmark_equity, annualization=annualization, risk_free=risk_free)
    stats.update({f"benchmark_{k}": v for k, v in bench.items()})

    if gross_equity is not None and len(gross_equity) > 1:
        gross = perf_stats(gross_equity, annualization=annualization, risk_free=risk_free)
        stats["gross_cagr"] = gross.get("cagr")
        stats["gross_total_return"] = gross.get("total_return")
        stats["gross_max_drawdown"] = gross.get("max_drawdown")
        stats["gross_sharpe"] = gross.get("sharpe")
        s_cagr = stats.get("strategy_cagr")
        g_cagr = gross.get("cagr")
        # 正值 = 成本带来的年化拖累（毛收益 - 净收益）
        stats["cost_drag_cagr"] = (
            (g_cagr - s_cagr)
            if isinstance(s_cagr, float) and isinstance(g_cagr, float)
            else float("nan")
        )

    stats.update(trade_stats(trades) if trades is not None else trade_stats(None))

    if position is not None and len(position):
        stats["time_in_market"] = float(pd.Series(position).astype(float).mean())

    s_cagr = stats.get("strategy_cagr")
    b_cagr = stats.get("benchmark_cagr")
    stats["excess_cagr"] = (
        (s_cagr - b_cagr)
        if isinstance(s_cagr, float) and isinstance(b_cagr, float)
        else float("nan")
    )
    s_mdd = stats.get("strategy_max_drawdown")
    b_mdd = stats.get("benchmark_max_drawdown")
    stats["drawdown_reduction"] = (
        (s_mdd - b_mdd) if isinstance(s_mdd, float) and isinstance(b_mdd, float) else float("nan")
    )
    return stats


__all__ = [
    "TRADING_DAYS",
    "max_drawdown",
    "max_drawdown_duration",
    "perf_stats",
    "summarize",
    "trade_stats",
]
