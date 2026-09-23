"""每日交易日志。

本模块只做一件事：把策略在**每个交易日**的状态与动作，记成一份可读、可追溯的日志。
它不接券商、不下单、不对账——那属于交易系统，不属于研究工具。

日志由三张表组成
----------------
``daily``   逐日全量记录：信号、票数、目标仓位、当日行情、净值、累计收益
``events``  仅「动作日」：建仓 / 平仓 / 换向，附动作前后上下文
``summary`` 阶段性汇总：区间、动作次数、持仓天数占比、期末净值

产出
----
``journal.csv``         逐日全量（机器可读）
``journal_events.csv``  动作流水（机器可读）
``journal.md``          人读日志：概览 + 最近 N 日 + 全部动作日
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .metrics import ANNUAL_COLUMNS, TRADING_DAYS, annual_breakdown, cagr_from_equity

Action = Literal["建仓", "平仓", "持有", "空仓", "建空", "平空", "持空", "多翻空", "空翻多"]

ACTION_OPEN = "建仓"
ACTION_CLOSE = "平仓"
ACTION_HOLD = "持有"
ACTION_FLAT = "空仓"
#: 做空相关动作（direction != long_only 时才会出现）
ACTION_OPEN_SHORT = "建空"
ACTION_CLOSE_SHORT = "平空"
ACTION_HOLD_SHORT = "持空"
ACTION_TO_SHORT = "多翻空"
ACTION_TO_LONG = "空翻多"

#: 所有「改变了仓位」的动作（即动作流水要记录的行）
CHANGE_ACTIONS: tuple[str, ...] = (
    ACTION_OPEN,
    ACTION_CLOSE,
    ACTION_OPEN_SHORT,
    ACTION_CLOSE_SHORT,
    ACTION_TO_SHORT,
    ACTION_TO_LONG,
)

DAILY_COLUMNS = [
    "date",
    "series",
    "signal",
    "actionable_signal",
    "score",
    "n_long",
    "n_short",
    "position_prev",
    "position",
    "action",
    "close",
    "daily_return",
    "benchmark_return",
    "excess_return",
    "equity",
    "benchmark_equity",
    "drawdown",
    "cost_paid",
]

EVENT_COLUMNS = [
    "date",
    "series",
    "action",
    "signal",
    "actionable_signal",
    "score",
    "n_long",
    "n_short",
    "close",
    "equity",
    "days_in_prev_state",
    "cumulative_return",
]


# --------------------------------------------------------------------------- #
# 构建
# --------------------------------------------------------------------------- #
@dataclass
class Journal:
    """一份交易日志。"""

    daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict[str, Any] = field(default_factory=dict)
    annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    series: str = ""
    index_col: str = ""

    def tail(self, n: int) -> pd.DataFrame:
        return self.daily.tail(n)

    def since(self, start: str) -> pd.DataFrame:
        ts = pd.Timestamp(start)
        return self.daily[pd.to_datetime(self.daily["date"]) >= ts]


def _classify(prev: float, cur: float) -> str:
    """把仓位变化翻译成动作（支持 多头 / 空仓 / 空头 三态）。"""
    if prev == cur:
        if cur > 0:
            return ACTION_HOLD
        if cur < 0:
            return ACTION_HOLD_SHORT
        return ACTION_FLAT
    if prev == 0 and cur > 0:
        return ACTION_OPEN
    if prev > 0 and cur == 0:
        return ACTION_CLOSE
    if prev == 0 and cur < 0:
        return ACTION_OPEN_SHORT
    if prev < 0 and cur == 0:
        return ACTION_CLOSE_SHORT
    if prev > 0 and cur < 0:
        return ACTION_TO_SHORT
    return ACTION_TO_LONG


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """取数值列；列不存在时返回同长度的 NaN 序列（而不是标量 None）。"""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")


def build_journal(
    frame: pd.DataFrame,
    *,
    series: str,
    index_col: str,
    signal_col: str = "signal_eff",
    extra_cols: tuple[str, ...] = ("score", "n_long", "n_short"),
    trades: pd.DataFrame | None = None,
) -> Journal:
    """由一条策略的逐日明细构造日志。

    Parameters
    ----------
    frame:
        至少含 ``date`` / ``close`` / ``position`` / ``equity`` / ``benchmark_equity``，
        以及 ``signal_col`` 指定的信号列。其余列（当日收益、回撤、成本、票数等）
        缺失时按 NaN 处理。
    """
    required = {"date", "close", "position", "equity", "benchmark_equity", signal_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"构造日志缺少列：{sorted(missing)}")
    if frame.empty:
        raise ValueError("构造日志的输入为空")

    df = frame.copy().reset_index(drop=True)
    signal = _num(df, signal_col).fillna(0.0)

    position = _num(df, "position").fillna(0.0)
    # 建仓发生在 T+1，因此「前一日仓位」就是 shift(1)
    position_prev = position.shift(1).fillna(0.0)

    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "series": series,
            "signal": signal.astype(int),
            "position_prev": position_prev.astype(int),
            "position": position.astype(int),
            "close": _num(df, "close"),
            "daily_return": _num(df, "daily_return"),
            "benchmark_return": _num(df, "benchmark_return"),
            "excess_return": _num(df, "excess_return"),
            "equity": _num(df, "equity"),
            "benchmark_equity": _num(df, "benchmark_equity"),
            "drawdown": _num(df, "drawdown"),
            "cost_paid": _num(df, "cost_paid").fillna(0.0),
        }
    )
    for col in extra_cols:
        daily[col] = _num(df, col)

    daily["action"] = [
        _classify(float(p), float(c)) for p, c in zip(daily["position_prev"], daily["position"])
    ]
    # T 日的动作由 T-1 日的信号决定（T+1 执行）。把「执行依据」单独列出，
    # 否则日志里会出现「信号 +1 却平仓」这种看似矛盾的相邻列。
    daily["actionable_signal"] = signal.shift(1).fillna(0.0).astype(int)

    # ---- 动作事件 -------------------------------------------------------- #
    ev = daily[daily["action"].isin(CHANGE_ACTIONS)].copy()
    if len(ev):
        # 记录「上一状态持续了多少个交易日」
        idx = list(ev.index)
        spans = []
        prev_stop = -1
        for i in idx:
            spans.append(i - prev_stop)
            prev_stop = i
        ev["days_in_prev_state"] = spans
    else:
        ev["days_in_prev_state"] = pd.Series(dtype="int64")
    ev["cumulative_return"] = ev["equity"] - 1.0
    events = ev.reindex(columns=EVENT_COLUMNS).reset_index(drop=True)

    daily = daily.reindex(columns=DAILY_COLUMNS)

    n = len(daily)
    years = n / TRADING_DAYS
    total_return = float(daily["equity"].iloc[-1]) - 1.0
    # 非空仓天数占比（空头也算「有仓位」，不能只数 == 1）
    in_market = float((daily["position"] != 0).mean())
    summary = {
        "series": series,
        "index_col": index_col,
        "start": daily["date"].iloc[0],
        "end": daily["date"].iloc[-1],
        "n_days": n,
        "years": years,
        "n_open": int((daily["action"] == ACTION_OPEN).sum()),
        "n_close": int((daily["action"] == ACTION_CLOSE).sum()),
        "n_switch": int(daily["action"].isin([ACTION_TO_SHORT, ACTION_TO_LONG]).sum()),
        "n_open_short": int((daily["action"] == ACTION_OPEN_SHORT).sum()),
        "n_close_short": int((daily["action"] == ACTION_CLOSE_SHORT).sum()),
        "n_days_short": int((daily["position"] == -1).sum()),
        "n_hold": int((daily["action"] == ACTION_HOLD).sum()),
        "n_hold_short": int((daily["action"] == ACTION_HOLD_SHORT).sum()),
        "n_flat": int((daily["action"] == ACTION_FLAT).sum()),
        # 按「当日收盘仓位」统计，与动作次数区分开：
        # 建仓日收盘已有仓位，平仓日收盘已空仓。
        # n_days_in_market 用 != 0 统计，这样做空方向也有值
        # （曾经用 == 1，导致 short_only 显示「持仓 0 天」却又有 40% 持仓占比）
        "n_days_in_market": int((daily["position"] != 0).sum()),
        "n_days_flat": int((daily["position"] == 0).sum()),
        "n_days_long": int((daily["position"] == 1).sum()),
        "time_in_market": in_market,
        "cagr": cagr_from_equity(
            float(daily["equity"].iloc[0]), float(daily["equity"].iloc[-1]), years
        ),
        "total_return": total_return,
        "final_equity": float(daily["equity"].iloc[-1]),
        "final_benchmark": float(daily["benchmark_equity"].iloc[-1]),
        "max_drawdown": float(daily["drawdown"].min())
        if daily["drawdown"].notna().any()
        else float("nan"),
        "total_cost": float(daily["cost_paid"].sum()),
        "avg_holding_days": (float(ev["days_in_prev_state"].mean()) if len(ev) else float("nan")),
    }
    trades = trades if trades is not None else pd.DataFrame()
    try:
        annual = annual_breakdown(frame, trades if len(trades) else None)
    except ValueError:
        annual = pd.DataFrame()

    return Journal(
        daily=daily,
        events=events,
        summary=summary,
        annual=annual,
        series=series,
        index_col=index_col,
    )


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #
def _action_summary(s: dict[str, Any]) -> str:
    """按方向自适应地汇总动作次数（做多/做空分开列）。"""
    parts: list[str] = []
    if s.get("n_open") or s.get("n_close"):
        parts.append(f"建多 {s['n_open']}、平多 {s['n_close']}")
    if s.get("n_open_short") or s.get("n_close_short"):
        parts.append(f"建空 {s['n_open_short']}、平空 {s['n_close_short']}")
    if s.get("n_switch"):
        parts.append(f"换向 {s['n_switch']}")
    total = (
        s.get("n_open", 0)
        + s.get("n_close", 0)
        + s.get("n_open_short", 0)
        + s.get("n_close_short", 0)
        + s.get("n_switch", 0)
    )
    if not parts:
        parts.append("无")
    return "、".join(parts) + f"（合计 {total} 次）"


def _position_label(pos: int) -> str:
    if pos > 0:
        return "持多"
    if pos < 0:
        return "持空"
    return "空仓"


def render_journal_markdown(
    journal: Journal,
    *,
    tail: int = 30,
    title: str = "每日交易日志",
) -> str:
    s = journal.summary
    lines: list[str] = [f"# {title}", ""]

    lines.append("## 概览")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 策略 | `{s['series']}` |")
    lines.append(f"| 标的 | `{s['index_col']}` |")
    lines.append(
        f"| 区间 | {_d(s['start'])} ~ {_d(s['end'])}（{s['n_days']} 个交易日 / {s['years']:.1f} 年） |"
    )
    lines.append(
        f"| 期末净值 | 策略 **{s['final_equity']:.4f}** vs 基准 {s['final_benchmark']:.4f} |"
    )
    lines.append(f"| 累计 / 年化 | {s['total_return']:+.2%} / {s['cagr']:+.2%} |")
    lines.append(f"| 最大回撤 | {s['max_drawdown']:.2%} |")
    lines.append(f"| 动作次数 | {_action_summary(s)} |")
    holding = f"{s['n_days_in_market']} / {s['n_days_flat']}"
    if s.get("n_days_short"):
        holding += f"（其中多头 {s['n_days_long']}、空头 {s['n_days_short']}）"
    lines.append(f"| 持仓 / 空仓天数 | {holding}（持仓占比 {s['time_in_market']:.1%}） |")
    lines.append(f"| 平均持有天数 | {s['avg_holding_days']:.1f} |")
    lines.append(f"| 累计交易成本 | {s['total_cost']:.4f}（占初始资金） |")
    lines.append("")

    if journal.annual is not None and not journal.annual.empty:
        lines.append("## 分年度收益与胜率（对比基准）")
        lines.append("")
        from .render import format_table

        lines.append(format_table(journal.annual, ANNUAL_COLUMNS).to_markdown(index=False))
        lines.append("")
        lines.append("> 「策略日胜率」低于「基准日胜率」是正常的：空仓日收益为 0，不计入胜率分子。")
        lines.append("> 本类策略靠「低胜率 + 高盈亏比」取胜，应看相对胜率与超额收益。")
        lines.append("")

    lines.append("## 动作流水（建仓 / 平仓）")
    lines.append("")
    if journal.events.empty:
        lines.append("_区间内没有任何仓位变动。_")
    else:
        lines.append(_events_table(journal.events))
    lines.append("")

    lines.append(f"## 最近 {tail} 个交易日")
    lines.append("")
    lines.append(_daily_table(journal.daily.tail(tail)))
    lines.append("")
    return "\n".join(lines)


def _d(value: Any) -> str:
    if value is None or value is pd.NaT:
        return "n/a"
    ts = pd.to_datetime(value, errors="coerce")
    return "n/a" if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def _events_table(events: pd.DataFrame) -> str:
    rows = []
    for _, r in events.iterrows():
        rows.append(
            {
                "日期": _d(r["date"]),
                "动作": r["action"],
                "执行依据": f"{int(r['actionable_signal']):+d}",
                "明日信号": f"{int(r['signal']):+d}",
                "票数(多/空)": _votes(r),
                "收盘": f"{r['close']:.4f}",
                "净值": f"{r['equity']:.4f}",
                "累计收益": f"{r['cumulative_return']:+.2%}",
                "上状态持续": f"{int(r['days_in_prev_state'])} 天",
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def _daily_table(daily: pd.DataFrame) -> str:
    rows = []
    for _, r in daily.iterrows():
        rows.append(
            {
                "日期": _d(r["date"]),
                "当日信号": f"{int(r['signal']):+d}",
                "执行依据": f"{int(r['actionable_signal']):+d}",
                "票数(多/空)": _votes(r),
                "仓位": _position_label(int(r["position"])),
                "动作": r["action"],
                "收盘": f"{r['close']:.4f}",
                "当日": _pct(r["daily_return"]),
                "超额": _pct(r["excess_return"]),
                "净值": f"{r['equity']:.4f}",
                "回撤": _pct(r["drawdown"]),
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def _votes(row: pd.Series) -> str:
    nl, ns = row.get("n_long"), row.get("n_short")
    if nl is None or ns is None or pd.isna(nl) or pd.isna(ns):
        return "-"
    return f"{int(nl)}/{int(ns)}"


def _pct(v: Any) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    return f"{float(v):+.2%}"


# --------------------------------------------------------------------------- #
# 写出
# --------------------------------------------------------------------------- #
def write_journal(
    journal: Journal,
    outdir: str | Path,
    *,
    tail: int = 30,
    title: str = "每日交易日志",
) -> list[str]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    daily = journal.daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    p = outdir / "journal.csv"
    daily.to_csv(p, index=False, encoding="utf-8-sig")
    written.append(str(p))

    events = journal.events.copy()
    if not events.empty:
        events["date"] = pd.to_datetime(events["date"]).dt.strftime("%Y-%m-%d")
    p = outdir / "journal_events.csv"
    events.to_csv(p, index=False, encoding="utf-8-sig")
    written.append(str(p))

    if journal.annual is not None and not journal.annual.empty:
        p = outdir / "journal_annual.csv"
        journal.annual.to_csv(p, index=False, encoding="utf-8-sig")
        written.append(str(p))

    md = render_journal_markdown(journal, tail=tail, title=title)
    p = outdir / "journal.md"
    p.write_text(md, encoding="utf-8")
    written.append(str(p))
    return written


def append_daily_entry(
    journal: Journal,
    path: str | Path,
    *,
    date: str | None = None,
) -> pd.Series:
    """把某一天（默认最后一天）的记录追加到 CSV 台账，返回该行。

    用于「每个交易日跑一次、只追加当天」的用法。若该日期已存在则覆盖该行，
    保证重复运行不会产生重复记录。
    """
    path = Path(path)
    row = (
        journal.daily[pd.to_datetime(journal.daily["date"]) == pd.Timestamp(date)].iloc[-1]
        if date
        else journal.daily.iloc[-1]
    )
    record = row.to_frame().T
    record["date"] = pd.to_datetime(record["date"]).dt.strftime("%Y-%m-%d")

    if path.exists():
        existing = pd.read_csv(path, encoding="utf-8-sig")
        existing = existing[existing["date"] != record["date"].iloc[0]]
        combined = pd.concat([existing, record], ignore_index=True)
        combined = combined.sort_values("date").reset_index(drop=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        combined = record
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return row


__all__ = [
    "ACTION_CLOSE",
    "ACTION_FLAT",
    "ACTION_HOLD",
    "ACTION_OPEN",
    "DAILY_COLUMNS",
    "EVENT_COLUMNS",
    "Journal",
    "append_daily_entry",
    "build_journal",
    "render_journal_markdown",
    "write_journal",
]
