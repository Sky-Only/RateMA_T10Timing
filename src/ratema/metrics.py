"""绩效指标计算。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .backtest import BUY, CLOSE_ACTIONS, COVER, OPEN_ACTIONS, SELL, SHORT

TRADING_DAYS = 252


def cagr_from_equity(start_value: float, end_value: float, years: float) -> float:
    """由起止净值计算年化收益（CAGR）。

    终值 <= 0 时年化收益**没有定义**（空头被击穿 / 爆仓），必须返回 NaN。
    否则「负数开分数次方」在 Python 里会得到**复数**，污染后续所有
    指标计算、Markdown 表格与 JSON 序列化。

    （真实触发路径：short_only / long_short 方向下，标的涨幅超过 100%。）
    """
    if not np.isfinite(years) or years <= 0:
        return float("nan")
    if not np.isfinite(start_value) or start_value <= 0:
        return float("nan")
    if not np.isfinite(end_value) or end_value <= 0:
        return float("nan")
    return float((end_value / start_value) ** (1.0 / years) - 1.0)


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
    cagr = cagr_from_equity(start_value, end_value, years)

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
    """交易层面统计（以已平仓的往返交易为单位）。

    开平仓动作名复用 :mod:`ratema.backtest` 的常量，
    这样 ``long_short`` 方向的 ``SHORT`` / ``COVER`` 也能被正确统计。
    """
    if trades is None or trades.empty:
        return {
            "n_trades": 0,
            "n_buys": 0,
            "n_sells": 0,
            "n_shorts": 0,
            "n_covers": 0,
            "n_opens": 0,
            "n_round_trips": 0,
            "n_completed": 0,
            "trade_win_rate": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "profit_factor": float("nan"),
            "avg_holding_days": float("nan"),
            "total_cost": 0.0,
        }

    opens = trades[trades["action"].isin(OPEN_ACTIONS)]
    closes = trades[trades["action"].isin(CLOSE_ACTIONS)]
    closed = closes["pnl_pct"].dropna()
    wins = closed[closed > 0]
    losses = closed[closed <= 0]
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))

    return {
        "n_trades": len(trades),
        "n_buys": int((trades["action"] == BUY).sum()),
        "n_sells": int((trades["action"] == SELL).sum()),
        "n_shorts": int((trades["action"] == SHORT).sum()),
        "n_covers": int((trades["action"] == COVER).sum()),
        "n_opens": len(opens),
        "n_round_trips": int(trades["round_trip_id"].nunique()),
        "n_completed": len(closed),
        "trade_win_rate": float((closed > 0).mean()) if len(closed) else float("nan"),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "best_trade": float(closed.max()) if len(closed) else float("nan"),
        "worst_trade": float(closed.min()) if len(closed) else float("nan"),
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("nan"),
        "avg_holding_days": (
            float(closes["holding_days"].dropna().mean()) if len(closes) else float("nan")
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
        # 非空仓天数占比。注意不能取仓位均值——空头是 -1，会把占比算成负数
        stats["time_in_market"] = float((pd.Series(position).astype(float) != 0).mean())

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


# --------------------------------------------------------------------------- #
# 分年度拆解
# --------------------------------------------------------------------------- #
#: 分年度表的列：(键, 中文标签, 格式)
ANNUAL_COLUMNS: list[tuple[str, str, str]] = [
    ("year", "年份", "text"),
    ("n_days", "交易日", "int"),
    ("strategy_return", "策略收益", "pct"),
    ("benchmark_return", "基准收益", "pct"),
    ("excess_return", "超额", "pct"),
    ("beat_benchmark", "跑赢", "bool"),
    ("strategy_sharpe", "策略夏普", "num"),
    ("benchmark_sharpe", "基准夏普", "num"),
    ("strategy_max_drawdown", "策略回撤", "pct"),
    ("benchmark_max_drawdown", "基准回撤", "pct"),
    ("strategy_win_rate", "策略日胜率", "pct"),
    ("benchmark_win_rate", "基准日胜率", "pct"),
    ("relative_win_rate", "相对胜率", "pct"),
    ("trade_win_rate", "交易胜率", "pct"),
    ("n_round_trips", "往返次数", "int"),
    ("time_in_market", "持仓占比", "pct"),
]


def _window_stats(
    ret: pd.Series,
    bench_ret: pd.Series,
    position: pd.Series | None,
    annualization: int,
) -> dict[str, Any]:
    """单个时间窗口内的收益 / 风险 / 胜率。"""
    ret = pd.Series(ret).astype("float64").dropna()
    bench_ret = pd.Series(bench_ret).astype("float64").reindex(ret.index)
    if ret.empty:
        return {}

    # 用日收益累乘，避免依赖窗口外的前收盘价
    strat_equity = (1.0 + ret).cumprod()
    bench_equity = (1.0 + bench_ret.fillna(0.0)).cumprod()

    def _sharpe(x: pd.Series) -> float:
        x = x.dropna()
        sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        return float(x.mean() / sd * np.sqrt(annualization)) if sd > 0 else float("nan")

    def _mdd(eq: pd.Series) -> float:
        return float((eq / eq.cummax() - 1.0).min()) if len(eq) else float("nan")

    strat_ret_total = float(strat_equity.iloc[-1] - 1.0)
    bench_ret_total = float(bench_equity.iloc[-1] - 1.0)

    out: dict[str, Any] = {
        "n_days": len(ret),
        "strategy_return": strat_ret_total,
        "benchmark_return": bench_ret_total,
        "excess_return": strat_ret_total - bench_ret_total,
        "beat_benchmark": strat_ret_total > bench_ret_total,
        "strategy_sharpe": _sharpe(ret),
        "benchmark_sharpe": _sharpe(bench_ret),
        "strategy_max_drawdown": _mdd(strat_equity),
        "benchmark_max_drawdown": _mdd(bench_equity),
        "strategy_win_rate": float((ret > 0).mean()),
        "benchmark_win_rate": float((bench_ret > 0).mean()),
        # 相对胜率：策略当日收益跑赢基准的天数占比
        "relative_win_rate": float((ret > bench_ret).mean()),
    }
    if position is not None:
        pos = pd.Series(position).reindex(ret.index)
        out["time_in_market"] = float((pos != 0).mean()) if len(pos) else float("nan")
    return out


def _trade_stats_by_year(trades: pd.DataFrame | None) -> dict[int, dict[str, Any]]:
    """按年份统计往返交易：次数与胜率。"""
    if trades is None or trades.empty:
        return {}
    df = trades.copy()
    closed = df[df["action"].isin(CLOSE_ACTIONS) & df["pnl_pct"].notna()]
    if closed.empty:
        return {}
    years = pd.to_datetime(closed["date"]).dt.year
    out: dict[int, dict[str, Any]] = {}
    for year, grp in closed.groupby(years):
        pnl = grp["pnl_pct"].astype("float64")
        out[int(year)] = {
            "n_round_trips": len(pnl),
            "trade_win_rate": float((pnl > 0).mean()),
        }
    # 往返次数也统计未平仓的建仓（按建仓年份）
    opened = df[df["action"].isin(OPEN_ACTIONS)]
    if not opened.empty:
        open_years = pd.to_datetime(opened["date"]).dt.year
        for year, grp in opened.groupby(open_years):
            out.setdefault(int(year), {})["n_trades_opened"] = len(grp)
    return out


def annual_breakdown(
    frame: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    *,
    annualization: int = TRADING_DAYS,
) -> pd.DataFrame:
    """按自然年拆解收益与胜率，并给出与基准的逐年对比。

    Parameters
    ----------
    frame:
        逐日明细，需含 ``date``；``daily_return`` / ``benchmark_return``
        缺失时会从 ``equity`` / ``benchmark_equity`` 自动推导。
        ``position`` 可选（用于持仓占比）。
    trades:
        成交流水，需含 ``date`` / ``action`` / ``pnl_pct``；可选。

    Returns
    -------
    DataFrame，索引为年份字符串，最后一行是 ``全区间``。
    列见 :data:`ANNUAL_COLUMNS`。
    """
    if "date" not in frame.columns:
        raise ValueError("分年度拆解缺少列：['date']")
    if frame.empty:
        raise ValueError("分年度拆解的输入为空")

    df = frame.copy()
    # 收益列缺失时从净值推导，避免调用方必须手工补两列
    for ret_col, eq_col in (
        ("daily_return", "equity"),
        ("benchmark_return", "benchmark_equity"),
    ):
        if ret_col not in df.columns:
            if eq_col not in df.columns:
                raise ValueError(f"分年度拆解缺少列：['{ret_col}']，且没有 '{eq_col}' 可供推导")
            df[ret_col] = pd.to_numeric(df[eq_col], errors="coerce").pct_change().fillna(0.0)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")

    # 评估首日没有「前收盘」，其 daily_return 是被 fillna(0) 补出来的假 0。
    # 把它算进胜率/夏普会同时低估均值与波动，因此与 perf_stats 保持一致：剔除。
    # 累计收益不受影响（该日收益本就是 0）。
    if len(df) > 1:
        df = df.iloc[1:]

    position = df["position"] if "position" in df.columns else None
    trade_by_year = _trade_stats_by_year(trades)

    rows: list[dict[str, Any]] = []
    for year, grp in df.groupby(df.index.year):
        stats = _window_stats(
            grp["daily_return"],
            grp["benchmark_return"],
            None if position is None else position.loc[grp.index],
            annualization,
        )
        stats["year"] = str(int(year))
        stats.update(trade_by_year.get(int(year), {}))
        rows.append(stats)

    overall = _window_stats(df["daily_return"], df["benchmark_return"], position, annualization)
    overall["year"] = "全区间"
    if trade_by_year:
        overall["n_round_trips"] = int(
            sum(v.get("n_round_trips", 0) for v in trade_by_year.values())
        )
        if trades is not None and not trades.empty:
            closed = trades[trades["action"].isin(CLOSE_ACTIONS) & trades["pnl_pct"].notna()]
            if not closed.empty:
                overall["trade_win_rate"] = float((closed["pnl_pct"].astype("float64") > 0).mean())
    rows.append(overall)

    out = pd.DataFrame(rows)
    # 没有平仓的年份，往返次数是 0 而不是缺失；交易胜率则确实无定义
    if "n_round_trips" in out.columns:
        out["n_round_trips"] = out["n_round_trips"].fillna(0)
    for col in ("trade_win_rate", "time_in_market"):
        if col not in out.columns:
            out[col] = np.nan
    ordered = [k for k, _, _ in ANNUAL_COLUMNS if k in out.columns]
    return out[ordered].reset_index(drop=True)


def mean_annual(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """把多个指标的分年度表按年聚合（等权平均）。

    布尔列取多数，其余取算术平均；年份顺序保持为「按年递增，全区间置末」。
    """
    frames = [t for t in tables if t is not None and not t.empty]
    if not frames:
        return pd.DataFrame()

    by_year: dict[str, list[pd.Series]] = {}
    for t in frames:
        for _, row in t.iterrows():
            by_year.setdefault(str(row["year"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for year in sorted(by_year, key=lambda y: (y == "全区间", y)):
        stacked = pd.DataFrame(by_year[year])
        agg: dict[str, Any] = {"year": year}
        for key, _label, kind in ANNUAL_COLUMNS:
            if key == "year" or key not in stacked.columns:
                continue
            series = pd.to_numeric(stacked[key], errors="coerce")
            if kind == "bool":
                agg[key] = bool(series.fillna(0).mean() >= 0.5)
            else:
                agg[key] = float(series.mean())
        rows.append(agg)

    out = pd.DataFrame(rows)
    if "n_round_trips" in out.columns:
        out["n_round_trips"] = out["n_round_trips"].round().astype("Int64")
    ordered = [k for k, _, _ in ANNUAL_COLUMNS if k in out.columns]
    return out[ordered].reset_index(drop=True)


__all__ = [
    "ANNUAL_COLUMNS",
    "TRADING_DAYS",
    "annual_breakdown",
    "max_drawdown",
    "max_drawdown_duration",
    "mean_annual",
    "perf_stats",
    "summarize",
    "trade_stats",
]
