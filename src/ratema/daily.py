"""每日快照：给定数据与参数，算出「最新一天各指标的状态与结论」。

把这段逻辑从 CLI 里抽出来，是因为它有两个消费者：

* ``ratema signal``  —— 打印当日信号
* ``ratema journal`` —— 写入每日交易日志

两者必须用同一份计算，否则日志和信号会对不上。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from .indicators import SignalConfig, compute_signals
from .io_utils import DATE_COL, Dataset, resolve_roles

ConsensusMode = Literal["majority", "any", "all"]
CONSENSUS_MODES: tuple[str, ...] = ("majority", "any", "all")

CONSENSUS_HELP: dict[str, str] = {
    "majority": "过半数为多头即判定为多头（默认）",
    "any": "任一指标为多头即判定为多头（最激进）",
    "all": "全部指标为多头才判定为多头（最保守）",
}

ACTION_OPEN = "买入 / 建立多头"
ACTION_HOLD = "继续持有"
ACTION_CLOSE = "卖出 / 平仓"
ACTION_FLAT = "继续空仓"

SNAPSHOT_COLUMNS = [
    "series",
    "name",
    "date",
    "rate",
    "ma_short",
    "ma_long",
    "spread_bp",
    "tolerance_bp",
    "is_overlap",
    "signal_raw",
    "signal_eff",
    "hold_target",
    "hold_prev",
]


@dataclass
class DailySnapshot:
    """某一天的信号快照。"""

    asof: pd.Timestamp
    prev_date: pd.Timestamp | None
    index_col: str
    index_close: float
    per_indicator: pd.DataFrame = field(default_factory=pd.DataFrame)
    mode: str = "majority"
    signal_config: SignalConfig = field(default_factory=SignalConfig)
    data_latest: pd.Timestamp | None = None
    data_stale_days: int = 0
    stale_warning: bool = False

    @property
    def votes_long(self) -> int:
        return int(self.per_indicator["hold_target"].sum())

    @property
    def n_indicators(self) -> int:
        return len(self.per_indicator)

    @property
    def consensus_long(self) -> bool:
        n, votes = self.n_indicators, self.votes_long
        if self.mode == "majority":
            return votes * 2 > n
        if self.mode == "any":
            return votes > 0
        return votes == n

    @property
    def consensus(self) -> str:
        return "LONG" if self.consensus_long else "FLAT"

    @property
    def holding_now(self) -> bool:
        return bool(self.per_indicator["hold_prev"].any())

    @property
    def action(self) -> str:
        if self.consensus_long and not self.holding_now:
            return ACTION_OPEN
        if self.consensus_long and self.holding_now:
            return ACTION_HOLD
        if not self.consensus_long and self.holding_now:
            return ACTION_CLOSE
        return ACTION_FLAT

    def to_dict(self) -> dict[str, Any]:
        return {
            "asof": self.asof.strftime("%Y-%m-%d"),
            "prev_date": self.prev_date.strftime("%Y-%m-%d")
            if self.prev_date is not None
            else None,
            "index_col": self.index_col,
            "index_close": self.index_close,
            "signal_config": self.signal_config.to_dict(),
            "mode": self.mode,
            "per_indicator": [
                {
                    **{
                        k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v)
                        for k, v in row.items()
                    },
                    "signal": int(row["signal_eff"]),
                }
                for row in self.per_indicator.to_dict("records")
            ],
            "votes_long": self.votes_long,
            "n_indicators": self.n_indicators,
            "consensus": self.consensus,
            "action": self.action,
            "data_latest": (
                self.data_latest.strftime("%Y-%m-%d") if self.data_latest is not None else None
            ),
            "data_stale_days": self.data_stale_days,
            "stale_warning": self.stale_warning,
        }


def latest_snapshot(
    dataset: Dataset,
    signal_cfg: SignalConfig,
    *,
    index_col: str | None = None,
    rate_cols: list[str] | None = None,
    asof: str | pd.Timestamp | None = None,
    mode: str = "majority",
    max_stale_days: int = 7,
    today: pd.Timestamp | None = None,
) -> DailySnapshot:
    """计算最新一天的信号快照。

    ``asof`` 用于历史回溯（只看该日及之前的数据）；数据新鲜度始终以数据集
    真实最后一行衡量，与 ``asof`` 无关。
    """
    if mode not in CONSENSUS_MODES:
        raise ValueError(f"mode 应为 {CONSENSUS_MODES} 之一，收到 {mode!r}")

    index_col, resolved_rates = resolve_roles(dataset, index_col, rate_cols)

    frame = dataset.frame
    dates = pd.to_datetime(frame[DATE_COL])

    if asof is not None:
        cutoff = pd.Timestamp(asof)
        mask = dates <= cutoff
        if not mask.any():
            raise ValueError(f"asof {cutoff.date()} 早于数据起点 {dates.iloc[0].date()}")
        frame = frame.loc[mask].reset_index(drop=True)
        dates = pd.to_datetime(frame[DATE_COL])

    if frame.empty:
        raise ValueError("数据为空，无法计算信号")

    latest = pd.Timestamp(dates.iloc[-1])
    prev = pd.Timestamp(dates.iloc[-2]) if len(dates) > 1 else None
    last_price = float(frame[index_col].iloc[-1])

    rows: list[dict[str, Any]] = []
    for series in resolved_rates:
        sig = compute_signals(dates, frame[series], signal_cfg, series_name=series)
        valid = sig["signal_eff"].notna()
        if not valid.any():
            raise ValueError(
                f"{series}: 数据不足以形成 MA{signal_cfg.long_window}，"
                f"或阈值「{signal_cfg.describe_tol()}」过宽导致全程重合"
            )
        last = sig.loc[valid].iloc[-1]
        prev_sig = sig.loc[valid].iloc[-2] if int(valid.sum()) > 1 else None
        rows.append(
            {
                "series": series,
                "name": dataset.display_name(series),
                "date": pd.Timestamp(last["date"]),
                "rate": float(last["value"]),
                "ma_short": float(last["ma_short"]),
                "ma_long": float(last["ma_long"]),
                "spread_bp": float(last["spread_bp"]),
                "tolerance_bp": float(last["tolerance_bp"]),
                "is_overlap": bool(last["is_overlap"]),
                "signal_raw": float(last["signal_raw"]),
                "signal_eff": float(last["signal_eff"]),
                "hold_target": float(last["signal_eff"]) > 0,
                "hold_prev": bool(prev_sig is not None and float(prev_sig["signal_eff"]) > 0),
            }
        )

    per = pd.DataFrame(rows).reindex(columns=SNAPSHOT_COLUMNS)

    data_last = pd.Timestamp(dataset.frame[DATE_COL].iloc[-1])
    ref = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    stale_days = int((ref.normalize() - data_last.normalize()).days)

    return DailySnapshot(
        asof=latest,
        prev_date=prev,
        index_col=index_col,
        index_close=last_price,
        per_indicator=per,
        mode=mode,
        signal_config=signal_cfg,
        data_latest=data_last,
        data_stale_days=stale_days,
        stale_warning=stale_days > max_stale_days,
    )


__all__ = [
    "ACTION_CLOSE",
    "ACTION_FLAT",
    "ACTION_HOLD",
    "ACTION_OPEN",
    "CONSENSUS_HELP",
    "CONSENSUS_MODES",
    "SNAPSHOT_COLUMNS",
    "DailySnapshot",
    "latest_snapshot",
]
