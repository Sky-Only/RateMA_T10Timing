"""策略指标：20/120 日简单移动均线 + 可配置「重合度」阈值。

原始规则
--------
1. MA_short > MA_long  -> 债券空头信号，标记 -1
2. MA_short < MA_long  -> 债券多头信号，标记 +1
3. 两线「重合」        -> 标记 0，并延续上一交易日的有效信号

本模块的扩展
------------
第 3 条中「重合」不再要求严格相等，而是由 ``SignalConfig.tol_mode`` /
``SignalConfig.tol`` 决定的一个可配置区间::

    |MA_short - MA_long| <= tolerance  ->  视为重合

tolerance 的计算方式（tol_mode）：

* ``abs``  绝对阈值，单位与利率一致（百分点）。``--tol 0.01`` 即 1bp。
* ``bp``   以基点为单位。``--tol 1`` 即 1bp = 0.01 个百分点。
* ``rel``  相对阈值，``|spread| / |MA_long| <= tol``。``--tol 0.01`` 即 1%。
* ``std``  波动自适应，``tol * rolling_std(spread)``，随利差自身波动缩放。
* ``q``    分位数自适应，``tol`` 取 0~1，表示 ``|spread|`` 在滚动窗口内的分位数。

``tol=0``（默认）严格复现原始规则——只有两条均线数值完全相等才算重合。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

TOL_MODES: tuple[str, ...] = ("abs", "bp", "rel", "std", "q")

TOL_MODE_HELP: dict[str, str] = {
    "abs": "绝对阈值：|MA快-MA慢| <= tol，单位=利率单位(百分点)。tol=0.01 即 1bp",
    "bp": "基点阈值：|MA快-MA慢|*100 <= tol，单位=bp。tol=1 即 1bp",
    "rel": "相对阈值：|MA快-MA慢|/|MA慢| <= tol。tol=0.01 即百分之一",
    "std": "波动自适应：|MA快-MA慢| <= tol * 滚动标准差(spread, std_window)。tol=0.5 即半个标准差",
    "q": "分位数自适应：|MA快-MA慢| <= |spread| 在 std_window 内的 tol 分位数。tol 取 0~1",
}

#: 信号取值
SIGNAL_SHORT = -1.0
SIGNAL_FLAT = 0.0
SIGNAL_LONG = 1.0


#: 默认参数。改动这些会改变 README / 报告中的**全部既有结论**，
#: 因此由 tests/test_cli.py::test_documented_default_tolerance_is_zero 锁定。
#: ``DEFAULT_TOL = 0`` 表示「只有两条均线数值严格相等才算重合」——
#: 即原始策略的字面口径。
DEFAULT_SHORT_WINDOW = 20
DEFAULT_LONG_WINDOW = 120
DEFAULT_TOL_MODE = "abs"
DEFAULT_TOL = 0.0
DEFAULT_STD_WINDOW = 120


@dataclass(frozen=True)
class SignalConfig:
    """均线与重合判定参数。"""

    short_window: int = DEFAULT_SHORT_WINDOW
    long_window: int = DEFAULT_LONG_WINDOW
    tol_mode: str = DEFAULT_TOL_MODE
    tol: float = DEFAULT_TOL
    std_window: int = DEFAULT_STD_WINDOW
    min_periods_short: int | None = None
    min_periods_long: int | None = None

    def __post_init__(self) -> None:
        if self.short_window < 1:
            raise ValueError("short_window 必须 >= 1")
        if self.long_window <= self.short_window:
            raise ValueError(
                f"long_window({self.long_window}) 必须大于 short_window({self.short_window})"
            )
        if self.tol_mode not in TOL_MODES:
            raise ValueError(f"tol_mode 必须是 {TOL_MODES} 之一，收到 {self.tol_mode!r}")
        if not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError(f"tol 必须是非负有限数，收到 {self.tol!r}")
        if self.tol_mode == "q" and not 0.0 <= self.tol <= 1.0:
            raise ValueError(f"tol_mode='q' 时 tol 必须落在 [0, 1]，收到 {self.tol!r}")
        if self.tol_mode == "rel" and self.tol > 1.0:
            # rel 的 tol 是**比例**：想要 3% 要写 0.03，不是 3。
            # 写成 3 意味着容差 = 利率水平的 300%，会把几乎每一天都判成重合，
            # 信号实际上被冻住 —— 这种量级错误必须当场拦下，不能静默跑完。
            raise ValueError(
                f"tol_mode='rel' 时 tol 是比例（3% 要写 0.03），不应超过 1.0，收到 {self.tol!r}"
            )
        if self.std_window < 2:
            raise ValueError("std_window 必须 >= 2")
        for field_name in ("min_periods_short", "min_periods_long"):
            value = getattr(self, field_name)
            if value is not None and value < 1:
                raise ValueError(f"{field_name} 必须 >= 1 或为 None")

    # -- 便于打印 ---------------------------------------------------------- #
    @property
    def mp_short(self) -> int:
        return self.min_periods_short or self.short_window

    @property
    def mp_long(self) -> int:
        return self.min_periods_long or self.long_window

    def describe_tol(self) -> str:
        """人类可读的阈值描述。"""
        if self.tol_mode == "abs":
            return f"绝对 {self.tol:g} 个百分点（{self.tol * 100:g} bp）"
        if self.tol_mode == "bp":
            return f"绝对 {self.tol:g} bp（{self.tol / 100:g} 个百分点）"
        if self.tol_mode == "rel":
            return f"相对 {self.tol:.4%}（|MA快-MA慢| / |MA慢|）"
        if self.tol_mode == "std":
            return f"自适应 {self.tol:g} × 滚动{self.std_window}日标准差"
        return f"自适应 {self.tol:.2%} 分位（滚动{self.std_window}日 |spread| 分位数）"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tol_description"] = self.describe_tol()
        return data


# --------------------------------------------------------------------------- #
# 阈值计算
# --------------------------------------------------------------------------- #
def compute_tolerance(
    spread: pd.Series,
    ma_long: pd.Series,
    cfg: SignalConfig,
) -> pd.Series:
    """按 ``cfg.tol_mode`` 计算逐日容差（与 spread 同量纲）。"""
    if cfg.tol_mode == "abs":
        return pd.Series(float(cfg.tol), index=spread.index, dtype="float64")
    if cfg.tol_mode == "bp":
        return pd.Series(float(cfg.tol) / 100.0, index=spread.index, dtype="float64")
    if cfg.tol_mode == "rel":
        return (ma_long.abs() * float(cfg.tol)).astype("float64")
    if cfg.tol_mode == "std":
        std = spread.rolling(cfg.std_window, min_periods=max(2, cfg.std_window // 4)).std()
        return (std * float(cfg.tol)).astype("float64")
    if cfg.tol_mode == "q":
        quantile = (
            spread.abs()
            .rolling(cfg.std_window, min_periods=max(2, cfg.std_window // 4))
            .quantile(float(cfg.tol))
        )
        return quantile.astype("float64")
    raise ValueError(f"未知 tol_mode: {cfg.tol_mode!r}")


# --------------------------------------------------------------------------- #
# 信号生成
# --------------------------------------------------------------------------- #
SIGNAL_COLUMNS = [
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


def compute_signals(
    dates: pd.Series | pd.Index,
    values: pd.Series,
    cfg: SignalConfig,
    series_name: str = "",
) -> pd.DataFrame:
    """计算均线、重合标记与有效信号。

    Parameters
    ----------
    dates:
        交易日序列（与 values 等长，升序）。
    values:
        底层利率指标数值。
    cfg:
        见 :class:`SignalConfig`。
    series_name:
        写入 ``series`` 列的名称。

    Returns
    -------
    DataFrame，列见 :data:`SIGNAL_COLUMNS`。

    Notes
    -----
    * ``signal_raw`` ∈ {-1, 0, +1}，0 表示当日两条均线落入重合区间。
    * ``signal_eff`` ∈ {-1, +1}，把 0 用「上一交易日的有效信号」前向填充，
      即规则第 3 条。
    """
    values = pd.Series(values).astype("float64").reset_index(drop=True)
    dates = pd.Series(pd.to_datetime(pd.Series(dates).to_numpy())).reset_index(drop=True)
    if len(values) != len(dates):
        raise ValueError("dates 与 values 长度不一致")

    ma_short = values.rolling(cfg.short_window, min_periods=cfg.mp_short).mean()
    ma_long = values.rolling(cfg.long_window, min_periods=cfg.mp_long).mean()
    spread = ma_short - ma_long

    tolerance = compute_tolerance(spread, ma_long, cfg)

    valid = ma_short.notna() & ma_long.notna()
    with np.errstate(invalid="ignore"):
        is_overlap = valid & (spread.abs() <= tolerance)
    is_overlap = is_overlap.fillna(False).astype(bool)

    signal_raw = pd.Series(np.nan, index=values.index, dtype="float64")
    # 规则 1/2：MA快 > MA慢 -> 空头 -1（资金成本抬升，流动性趋紧）
    #            MA快 < MA慢 -> 多头 +1（短端成本低于中长期，流动性宽松）
    signal_raw[valid] = -np.sign(spread[valid])
    signal_raw[is_overlap] = SIGNAL_FLAT

    # 规则 3：重合时延续上一交易日的有效信号
    signal_eff = signal_raw.mask(signal_raw == SIGNAL_FLAT).ffill()

    out = pd.DataFrame(
        {
            "date": dates,
            "series": series_name,
            "value": values,
            "ma_short": ma_short,
            "ma_long": ma_long,
            "spread": spread,
            "spread_bp": spread * 100.0,
            "tolerance": tolerance,
            "tolerance_bp": tolerance * 100.0,
            "is_overlap": is_overlap,
            "signal_raw": signal_raw,
            "signal_eff": signal_eff,
        }
    )
    out["overlap_band"] = f"±{cfg.describe_tol()}"
    out["signal_changed"] = out["signal_eff"].ne(out["signal_eff"].shift(1))
    return out[SIGNAL_COLUMNS]


# --------------------------------------------------------------------------- #
# 阈值标定辅助
# --------------------------------------------------------------------------- #
SPREAD_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)


def spread_calibration(signals: pd.DataFrame, series_name: str = "") -> dict[str, Any]:
    """给出 |MA快-MA慢| 的分布，用于选择「重合度」阈值。

    这是给使用者的标定参考：例如 p10 表示历史上 10% 的交易日里，
    两线距离小于该值（单位 bp）。
    """
    if not len(signals):
        return {"series": series_name}
    spread = signals["spread"].dropna()
    abs_bp = spread.abs() * 100.0
    row: dict[str, Any] = {
        "series": series_name,
        "n_obs": len(spread),
        "mean_abs_bp": float(abs_bp.mean()),
        "max_abs_bp": float(abs_bp.max()),
    }
    if len(abs_bp):
        for q in SPREAD_QUANTILES:
            row[f"p{int(q * 100):02d}_abs_bp"] = float(abs_bp.quantile(q))
    return row


def overlap_stats(signals: pd.DataFrame) -> dict[str, Any]:
    """重合日与信号切换统计。

    ``raw_signal_flips``  原始三态信号（-1/0/+1）的切换次数；
    ``signal_flips``      实际交易信号（-1/+1，重合已前向填充）的切换次数。
    两者之差即重合区间「吸收」掉的噪声切换。
    """
    n = len(signals)
    raw = signals["signal_raw"]
    eff = signals["signal_eff"]
    valid = raw.notna()
    n_valid = int(valid.sum())
    n_overlap = int(signals["is_overlap"].sum())

    def _flips(series: pd.Series) -> int:
        prev = series.shift(1)
        changed = series.ne(prev) & series.notna() & prev.notna()
        return int(changed.sum())

    raw_flips = _flips(raw)
    eff_flips = _flips(eff)
    return {
        "n_obs": n,
        "n_valid": n_valid,
        "n_overlap": n_overlap,
        "overlap_share": (n_overlap / n_valid) if n_valid else float("nan"),
        "raw_signal_flips": raw_flips,
        "signal_flips": eff_flips,
        "absorbed_flips": raw_flips - eff_flips,
    }


__all__ = [
    "SIGNAL_COLUMNS",
    "SIGNAL_FLAT",
    "SIGNAL_LONG",
    "SIGNAL_SHORT",
    "SPREAD_QUANTILES",
    "TOL_MODES",
    "TOL_MODE_HELP",
    "SignalConfig",
    "compute_signals",
    "compute_tolerance",
    "overlap_stats",
    "spread_calibration",
]
