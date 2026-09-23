"""信号与重合阈值测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ratema.indicators import (
    SIGNAL_FLAT,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SignalConfig,
    compute_signals,
    compute_tolerance,
    overlap_stats,
    spread_calibration,
)


def _dates(n: int) -> pd.Series:
    return pd.Series(pd.bdate_range("2024-01-01", periods=n))


def test_basic_signals_and_carry_forward():
    """MA快>MA慢 -> -1（空头）；MA快<MA慢 -> +1（多头）；重合 -> 0 且延续前一日有效信号。"""
    values = pd.Series([1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 1.0])
    cfg = SignalConfig(short_window=1, long_window=3, tol_mode="abs", tol=0.0)
    out = compute_signals(_dates(len(values)), values, cfg, series_name="T")

    raw = out["signal_raw"].tolist()
    eff = out["signal_eff"].tolist()

    #           idx:  0    1     2     3     4     5     6
    #  MA1 =          1    2     3     3     3     2     1
    #  MA3 =          -    -     2   2.667   3   2.667   2
    #  spread =       -    -    +1  +0.333   0  -0.667  -1
    assert np.isnan(raw[0]) and np.isnan(raw[1])
    assert raw[2] == SIGNAL_SHORT, "MA快 > MA慢 应为空头 -1"
    assert raw[3] == SIGNAL_SHORT  # MA1=3 > MA3=2.667
    assert raw[4] == SIGNAL_FLAT, "两线相等应判为重合 0"
    assert raw[5] == SIGNAL_LONG, "MA快 < MA慢 应为多头 +1"
    assert raw[6] == SIGNAL_LONG

    assert np.isnan(eff[0]) and np.isnan(eff[1])
    assert eff[4] == SIGNAL_SHORT, "重合日必须延续上一交易日的有效信号"
    assert eff[5] == SIGNAL_LONG
    assert out["is_overlap"].tolist() == [False, False, False, False, True, False, False]
    assert out["spread"].iloc[4] == pytest.approx(0.0)


def test_long_signal_when_short_ma_below_long_ma():
    """单调下行 -> MA快 < MA慢 -> 全程 +1。"""
    values = pd.Series(np.linspace(3.0, 1.0, 150))
    out = compute_signals(_dates(150), values, SignalConfig(20, 120, "abs", 0.0))
    valid = out["signal_eff"].dropna()
    assert (valid == SIGNAL_LONG).all()

    up = pd.Series(np.linspace(1.0, 3.0, 150))
    out_up = compute_signals(_dates(150), up, SignalConfig(20, 120, "abs", 0.0))
    valid_up = out_up["signal_eff"].dropna()
    assert (valid_up == SIGNAL_SHORT).all()


def test_constant_series_is_always_overlap_with_zero_tolerance():
    """数值完全相等 -> 任意时刻都是重合，tol=0 也能识别。"""
    values = pd.Series([2.5] * 30)
    cfg = SignalConfig(short_window=5, long_window=20, tol_mode="abs", tol=0.0)
    out = compute_signals(_dates(30), values, cfg)

    valid = out["signal_raw"].notna()
    assert valid.sum() == 30 - 20 + 1
    assert out.loc[valid, "signal_raw"].eq(0.0).all()
    assert out.loc[valid, "is_overlap"].all()
    # 前面没有有效信号可以延续 -> 仍然为空
    assert out["signal_eff"].isna().all()


def test_zero_tolerance_misses_float_noise():
    """tol=0 时，极微小的浮点差异就不算重合；给个小阈值就能抓住。"""
    # 每期递增 1e-10：MA2 与 MA4 之差恒为 1e-10，但永不严格相等
    values = pd.Series(1.0 + np.arange(20) * 1e-10)
    dates = _dates(20)

    out = compute_signals(dates, values, SignalConfig(2, 4, "abs", 0.0))
    spread = out["spread"].dropna()
    assert (spread.abs() > 0).all(), "构造的数据不应出现严格相等"
    assert spread.abs().max() < 1e-6
    assert out["is_overlap"].sum() == 0, "tol=0 时不应判定为重合"

    out2 = compute_signals(dates, values, SignalConfig(2, 4, "abs", 1e-6))
    assert out2["is_overlap"].sum() > 0, "放宽到 1e-6 后应判定为重合"


def test_tolerance_absolute_and_bp_are_equivalent():
    values = pd.Series(np.linspace(1.0, 3.0, 200))
    dates = _dates(200)
    a = compute_signals(dates, values, SignalConfig(20, 120, "abs", 0.01))
    b = compute_signals(dates, values, SignalConfig(20, 120, "bp", 1.0))
    pd.testing.assert_series_equal(a["is_overlap"], b["is_overlap"])
    np.testing.assert_allclose(a["tolerance"].dropna(), 0.01)


def test_relative_tolerance_scales_with_ma_long():
    values = pd.Series(np.linspace(1.0, 2.0, 150))
    out = compute_signals(_dates(150), values, SignalConfig(20, 120, "rel", 0.01))
    expected = out["ma_long"].abs() * 0.01
    np.testing.assert_allclose(out["tolerance"].to_numpy(), expected.to_numpy(), equal_nan=True)


def test_std_mode_tolerance_is_positive_and_scaled():
    rng = np.random.default_rng(0)
    values = pd.Series(2.0 + rng.normal(0, 0.05, 400).cumsum() * 0.01)
    out = compute_signals(_dates(400), values, SignalConfig(20, 120, "std", 0.5, std_window=120))
    tol = out["tolerance"].dropna()
    assert (tol >= 0).all()
    spread = out["spread"].dropna()
    std = spread.rolling(120, min_periods=30).std() * 0.5
    np.testing.assert_allclose(tol.to_numpy(), std.dropna().to_numpy(), rtol=1e-9)


def test_quantile_mode_matches_rolling_quantile():
    rng = np.random.default_rng(1)
    values = pd.Series(2.0 + rng.normal(0, 0.1, 300))
    cfg = SignalConfig(20, 120, "q", 0.1, std_window=120)
    out = compute_signals(_dates(300), values, cfg)
    direct = out["spread"].abs().rolling(120, min_periods=30).quantile(0.1)
    np.testing.assert_allclose(out["tolerance"].to_numpy(), direct.to_numpy(), equal_nan=True)


def test_larger_tolerance_marks_more_overlap_days():
    rng = np.random.default_rng(7)
    values = pd.Series(2.0 + rng.normal(0, 0.08, 500).cumsum() * 0.02)
    dates = _dates(500)
    counts = []
    for tol in (0.0, 0.005, 0.02, 0.05):
        out = compute_signals(dates, values, SignalConfig(20, 120, "abs", tol))
        counts.append(int(out["is_overlap"].sum()))
    assert counts == sorted(counts), f"重合日数应随阈值单调不减，实际 {counts}"


def test_overlap_never_changes_signal_direction_only_fills_gap():
    """重合只会在原方向之间「抹平」，不会凭空反转方向。"""
    rng = np.random.default_rng(3)
    values = pd.Series(2.0 + rng.normal(0, 0.05, 600).cumsum() * 0.03)
    dates = _dates(600)

    base = compute_signals(dates, values, SignalConfig(20, 120, "abs", 0.0))
    wide = compute_signals(dates, values, SignalConfig(20, 120, "abs", 0.02))

    both = base["signal_eff"].notna() & wide["signal_eff"].notna()
    assert set(wide.loc[both, "signal_eff"].unique()) <= {-1.0, 1.0}
    # 阈值为 0 时不可能出现重合（除非浮点严格相等）
    assert (base["is_overlap"] & base["spread"].ne(0)).sum() == 0
    # 放宽阈值后，重合日一定不少于严格相等
    assert wide["is_overlap"].sum() >= base["is_overlap"].sum()


def test_signal_eff_equals_forward_filled_non_zero_raw():
    rng = np.random.default_rng(11)
    values = pd.Series(2.0 + rng.normal(0, 0.05, 400).cumsum() * 0.02)
    out = compute_signals(_dates(400), values, SignalConfig(20, 120, "abs", 0.01))
    expected = out["signal_raw"].mask(out["signal_raw"] == 0.0).ffill()
    pd.testing.assert_series_equal(out["signal_eff"], expected, check_names=False)
    # 非重合日的有效信号 == 原始信号
    non_overlap = ~out["is_overlap"] & out["signal_raw"].notna()
    np.testing.assert_allclose(
        out.loc[non_overlap, "signal_eff"].to_numpy(),
        out.loc[non_overlap, "signal_raw"].to_numpy(),
    )


def test_config_validation():
    with pytest.raises(ValueError):
        SignalConfig(short_window=120, long_window=20)
    with pytest.raises(ValueError):
        SignalConfig(tol=-1.0)
    with pytest.raises(ValueError):
        SignalConfig(tol_mode="nope")
    with pytest.raises(ValueError):
        SignalConfig(tol_mode="q", tol=1.5)
    with pytest.raises(ValueError):
        SignalConfig(std_window=1)
    SignalConfig(tol_mode="q", tol=0.9)  # 合法


def test_describe_tol_is_human_readable():
    assert "1 bp" in SignalConfig(tol_mode="abs", tol=0.01).describe_tol()
    assert "1 bp" in SignalConfig(tol_mode="bp", tol=1).describe_tol()
    assert "1.0000%" in SignalConfig(tol_mode="rel", tol=0.01).describe_tol()
    assert "0.5" in SignalConfig(tol_mode="std", tol=0.5).describe_tol()
    assert "120" in SignalConfig(tol_mode="std", tol=0.5, std_window=120).describe_tol()


def test_spread_calibration_and_overlap_stats():
    rng = np.random.default_rng(5)
    values = pd.Series(2.0 + rng.normal(0, 0.05, 300).cumsum() * 0.02)
    out = compute_signals(_dates(300), values, SignalConfig(20, 120, "abs", 0.01))
    cal = spread_calibration(out, "X")
    assert cal["series"] == "X"
    assert cal["p50_abs_bp"] <= cal["p90_abs_bp"] <= cal["max_abs_bp"]
    stats = overlap_stats(out)
    assert stats["n_obs"] == 300
    assert stats["n_overlap"] == int(out["is_overlap"].sum())
    assert 0.0 <= stats["overlap_share"] <= 1.0


def test_compute_tolerance_rejects_unknown_mode():
    spread = pd.Series([0.1, 0.2])
    ma_long = pd.Series([1.0, 1.0])
    # 伪造一个非法 mode（绕过 __post_init__ 的校验）
    cfg = object.__new__(SignalConfig)
    object.__setattr__(cfg, "tol_mode", "bogus")
    object.__setattr__(cfg, "tol", 0.0)
    object.__setattr__(cfg, "std_window", 120)
    with pytest.raises(ValueError):
        compute_tolerance(spread, ma_long, cfg)
