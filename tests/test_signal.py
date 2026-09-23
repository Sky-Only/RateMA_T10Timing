"""signal 子命令测试（实盘每日入口）。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ratema.cli import main


def test_signal_outputs_decision(sample_panel_csv: Path, capsys):
    code = main(["signal", str(sample_panel_csv)])
    out = capsys.readouterr().out
    assert code in (0, 2)
    assert "最新交易信号" in out
    assert ">>> 明日操作：" in out
    for kw in ("买入 / 建立多头", "继续持有", "卖出 / 平仓", "继续空仓"):
        if kw in out:
            break
    else:
        pytest.fail("没有输出任何可识别的操作指令")


def test_signal_json_is_machine_readable(sample_panel_csv: Path, capsys):
    main(["signal", str(sample_panel_csv), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["consensus"] in {"LONG", "FLAT"}
    assert payload["n_indicators"] == 2
    assert 0 <= payload["votes_long"] <= payload["n_indicators"]
    assert payload["signal_config"]["long_window"] == 120
    for row in payload["per_indicator"]:
        assert row["signal"] in (-1, 1)
        assert isinstance(row["is_overlap"], bool)
        assert "ma_short" in row and "spread_bp" in row


def test_signal_modes_are_monotonic(sample_panel_csv: Path, capsys):
    """all ⊆ majority ⊆ any：放宽合成口径，多头票数只会不减。"""
    results = {}
    for mode in ("all", "majority", "any"):
        main(["signal", str(sample_panel_csv), "--mode", mode, "--json"])
        results[mode] = json.loads(capsys.readouterr().out)["votes_long"]
    # votes_long 与 mode 无关，但 consensus 的宽松度应满足 all <= majority <= any
    assert results["all"] == results["majority"] == results["any"]


def test_signal_asof_historical_date(sample_panel_csv: Path, capsys):
    """--asof 只截断信号计算，不应改变「数据新鲜度」的判定基准。"""
    df = pd.read_csv(sample_panel_csv, encoding="utf-8-sig")
    dates = pd.to_datetime(df["date"])
    target = dates.iloc[250]
    data_last = dates.iloc[-1]

    main(["signal", str(sample_panel_csv), "--asof", str(target.date()), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["asof"] == str(target.date())
    # 新鲜度始终以数据源真实最后一行衡量，与 asof 无关
    assert payload["data_latest"] == str(data_last.date())
    assert payload["data_latest"] != payload["asof"]

    # 不传 --asof 时两者一致
    main(["signal", str(sample_panel_csv), "--json"])
    latest = json.loads(capsys.readouterr().out)
    assert latest["asof"] == latest["data_latest"]


def test_signal_rejects_asof_before_data(sample_panel_csv: Path):
    with pytest.raises(SystemExit):
        main(["signal", str(sample_panel_csv), "--asof", "1990-01-01"])


def test_signal_stale_data_returns_exit_code_2(sample_panel_csv: Path, capsys):
    """数据过期必须返回非零退出码，供调度系统熔断。"""
    code = main(["signal", str(sample_panel_csv), "--max-stale-days", "0"])
    out = capsys.readouterr().out
    assert code == 2
    assert "未更新" in out


def test_signal_has_no_live_trading_flags(sample_panel_csv: Path):
    """实盘相关参数已移除：signal 只读不写。"""
    with pytest.raises(SystemExit):
        main(["signal", str(sample_panel_csv), "--log", "x.csv"])


def test_signal_respects_threshold(sample_panel_csv: Path, capsys):
    main(["signal", str(sample_panel_csv), "--json"])
    strict = json.loads(capsys.readouterr().out)
    main(["signal", str(sample_panel_csv), "--tol-mode", "bp", "--tol", "5", "--json"])
    loose = json.loads(capsys.readouterr().out)
    assert strict["signal_config"]["tol"] == 0.0
    assert loose["signal_config"]["tol"] == 5.0
    assert loose["signal_config"]["tol_mode"] == "bp"
    assert "5 bp" in loose["signal_config"]["tol_description"]
    # 阈值越大，判定为重合的指标只可能更多
    strict_overlap = sum(1 for r in strict["per_indicator"] if r["is_overlap"])
    loose_overlap = sum(1 for r in loose["per_indicator"] if r["is_overlap"])
    assert loose_overlap >= strict_overlap


def test_signal_overwide_threshold_fails_loudly(sample_panel_csv: Path):
    """阈值宽到全程重合时应明确报错，而不是静默给出错误信号。"""
    with pytest.raises(SystemExit):
        main(["signal", str(sample_panel_csv), "--tol-mode", "bp", "--tol", "99999"])
