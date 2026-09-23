"""命令行接口测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ratema.cli import main
from ratema.commands._shared import parse_grid
from ratema.indicators import SignalConfig
from ratema.parser import build_parser, default_grid

SUBCOMMANDS = ["convert", "backtest", "signal", "composite", "journal", "sweep", "all"]


@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_help_renders_for_every_subcommand(command: str, capsys):
    """--help 必须能正常渲染（argparse 的 % 格式化很容易踩坑）。"""
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--tol-mode" in out or command == "convert"


def test_top_level_help_and_version(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "ratema" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_every_tol_mode_is_offered():
    parser = build_parser()
    action = next(
        a
        for a in parser._subparsers._group_actions[0].choices["backtest"]._actions
        if a.dest == "tol_mode"
    )
    assert set(action.choices) == {"abs", "bp", "rel", "std", "q"}


def test_default_grid_is_mode_aware():
    for mode in ("abs", "bp", "rel", "std", "q"):
        values = parse_grid(default_grid(mode))
        assert values[0] == 0.0, f"{mode} 的默认网格应从 0 开始"
        assert values == sorted(values)


def test_default_grids_agree_across_units():
    """abs 与 bp 的默认网格必须描述同一组物理阈值。

    这是防止「100 倍单位错误」的守卫：abs 的单位是百分点，bp 的单位是基点，
    tol_abs * 100 == tol_bp。
    """
    abs_bp = [round(v * 100, 9) for v in parse_grid(default_grid("abs"))]
    bp_values = parse_grid(default_grid("bp"))
    assert abs_bp == bp_values, f"abs({abs_bp}bp) 与 bp({bp_values}bp) 默认网格不一致"
    # 量级合理性：利率利差的扫描上限应是几个 bp 到几十 bp，而不是 1000bp
    assert 1.0 <= max(bp_values) <= 50.0


def test_abs_and_bp_modes_describe_the_same_physical_threshold():
    """abs 0.01 与 bp 1 必须指向同一个物理阈值，两种描述都应含两个单位的数值。"""
    a = SignalConfig(20, 120, "abs", 0.01)
    b = SignalConfig(20, 120, "bp", 1.0)
    for cfg in (a, b):
        text = cfg.describe_tol()
        assert "1 bp" in text, text
        assert "0.01 个百分点" in text, text


def test_cli_signal_defaults_come_from_signal_config():
    """命令行默认值必须与 SignalConfig 的默认值一致（单一来源）。

    回归背景：``--tol`` / ``--tol-mode`` 曾各自硬编码默认值，
    导致「改了 indicators.py 的默认值但命令行行为不变」，很容易踩。
    """
    d = SignalConfig()
    parser = build_parser()
    backtest = parser._subparsers._group_actions[0].choices["backtest"]
    defaults = {
        a.dest: a.default
        for a in backtest._actions
        if a.dest in {"short", "long", "tol", "tol_mode", "std_window"}
    }
    assert defaults == {
        "short": d.short_window,
        "long": d.long_window,
        "tol": d.tol,
        "tol_mode": d.tol_mode,
        "std_window": d.std_window,
    }


def test_documented_default_tolerance_is_zero():
    """当前默认是「严格相等才重合」。改动这个默认值会改变全部既有结论，
    所以用测试显式锁住，避免无意间改动。"""
    d = SignalConfig()
    assert d.tol_mode == "abs"
    assert d.tol == 0.0
    assert "0 bp" in d.describe_tol()


def test_parse_grid_dedupes_and_sorts():
    assert parse_grid(" 5, 1 ,0,1 ") == [0.0, 1.0, 5.0]
    with pytest.raises(SystemExit):
        parse_grid(" , ")


def test_backtest_command_writes_outputs(sample_panel_csv: Path, tmp_path: Path):
    outdir = tmp_path / "out"
    code = main(["backtest", str(sample_panel_csv), "--outdir", str(outdir), "--quiet"])
    assert code == 0
    assert (outdir / "report.md").exists()
    assert (outdir / "summary.csv").exists()
    assert (outdir / "metrics.json").exists()


def test_backtest_with_bp_threshold(sample_panel_csv: Path, tmp_path: Path):
    outdir = tmp_path / "out_bp"
    code = main(
        [
            "backtest",
            str(sample_panel_csv),
            "--tol-mode",
            "bp",
            "--tol",
            "2",
            "--outdir",
            str(outdir),
            "--quiet",
        ]
    )
    assert code == 0
    summary = pd.read_csv(outdir / "summary.csv", encoding="utf-8-sig")
    assert (summary["overlap_days"] >= 0).all()
    report = (outdir / "report.md").read_text(encoding="utf-8")
    assert "2 bp" in report


def test_sweep_command_writes_grid(sample_panel_csv: Path, tmp_path: Path):
    outdir = tmp_path / "sweep"
    code = main(
        [
            "sweep",
            str(sample_panel_csv),
            "--tol-mode",
            "bp",
            "--tol-grid",
            "0,1,5",
            "--outdir",
            str(outdir),
        ]
    )
    assert code == 0
    assert (outdir / "sweep" / "sweep_bp.csv").exists()
    assert (outdir / "sweep" / "sweep_bp.md").exists()


def test_convert_command_on_workbook(sample_workbook: Path, tmp_path: Path):
    outdir = tmp_path / "csv"
    code = main(["convert", str(sample_workbook), "--outdir", str(outdir)])
    assert code == 0
    assert (outdir / "panel.csv").exists()
    assert (outdir / "long.csv").exists()


def test_missing_input_reports_cleanly(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["backtest", str(tmp_path / "nope.csv")])


def test_all_pipeline_runs_end_to_end(sample_panel_csv: Path, tmp_path: Path, capsys):
    """`all` 会把多个子命令串起来，必须保证它提供的参数足够每个子命令使用。

    回归背景：新增 journal 子命令后，`all` 的 Namespace 缺少 mode / vote_source，
    整条流水线在最后一步才崩掉。这个测试覆盖完整的串联。
    """
    outdir = tmp_path / "out"
    code = main(
        [
            "all",
            str(sample_panel_csv),
            "--skip-convert",
            "--tol-grid",
            "0,1",
            "--outdir",
            str(outdir),
            "--no-charts",
        ]
    )
    capsys.readouterr()
    assert code == 0

    for name in (
        "report.md",
        "summary.csv",
        "composite_report.md",
        "composite_summary.csv",
        "journal/journal.csv",
        "journal/journal_events.csv",
        "journal/journal.md",
        "sweep/sweep_abs.csv",
    ):
        assert (outdir / name).exists(), f"all 未产出 {name}"


def test_journal_command_writes_daily_log(sample_panel_csv: Path, tmp_path: Path, capsys):
    outdir = tmp_path / "out"
    code = main(["journal", str(sample_panel_csv), "--outdir", str(outdir), "--tail", "5"])
    capsys.readouterr()
    assert code == 0

    daily = pd.read_csv(outdir / "journal" / "journal.csv", encoding="utf-8-sig")
    assert daily["action"].notna().all()
    report = (outdir / "journal" / "journal.md").read_text(encoding="utf-8")
    assert "每日交易日志" in report


def test_journal_single_indicator_and_append(sample_panel_csv: Path, tmp_path: Path, capsys):
    outdir = tmp_path / "out"
    ledger = tmp_path / "ledger.csv"
    code = main(
        [
            "journal",
            str(sample_panel_csv),
            "--series",
            "DR001",
            "--outdir",
            str(outdir),
            "--append",
            str(ledger),
            "--quiet",
        ]
    )
    capsys.readouterr()
    assert code == 0
    assert ledger.exists()
    assert len(pd.read_csv(ledger, encoding="utf-8-sig")) == 1


def test_journal_rejects_unknown_series(sample_panel_csv: Path, tmp_path: Path):
    with pytest.raises(SystemExit, match="未知指标"):
        main(
            [
                "journal",
                str(sample_panel_csv),
                "--series",
                "NOT_A_SERIES",
                "--outdir",
                str(tmp_path),
            ]
        )


def test_sweep_composite_target(sample_panel_csv: Path, tmp_path: Path, capsys):
    """sweep 现在也支持对等权综合信号扫描。"""
    outdir = tmp_path / "sweep"
    code = main(
        [
            "sweep",
            str(sample_panel_csv),
            "--target",
            "composite",
            "--tol-mode",
            "bp",
            "--tol-grid",
            "0,2",
            "--outdir",
            str(outdir),
            "--no-charts",
        ]
    )
    capsys.readouterr()
    assert code == 0
    df = pd.read_csv(outdir / "sweep" / "sweep_bp.csv", encoding="utf-8-sig")
    assert set(df["target"]) == {"composite"}
    assert set(df["series"]) == {"COMPOSITE"}


@pytest.mark.parametrize("direction", ["long_only", "short_only", "long_short"])
def test_backtest_accepts_every_direction(
    sample_panel_csv: Path, tmp_path: Path, capsys, direction: str
):
    outdir = tmp_path / direction
    code = main(
        [
            "backtest",
            str(sample_panel_csv),
            "--direction",
            direction,
            "--outdir",
            str(outdir),
            "--no-charts",
            "--quiet",
        ]
    )
    capsys.readouterr()
    assert code == 0

    payload = json.loads((outdir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["run"]["backtest_config"]["direction"] == direction

    detail = pd.read_csv(outdir / "details" / "DR001.csv", encoding="utf-8-sig")
    positions = set(detail["position"].unique())
    if direction == "long_only":
        assert positions <= {0, 1}
    elif direction == "short_only":
        assert positions <= {0, -1}
    else:
        assert positions <= {-1, 0, 1}


def test_backtest_rejects_unknown_direction(sample_panel_csv: Path):
    with pytest.raises(SystemExit):
        main(["backtest", str(sample_panel_csv), "--direction", "sideways"])


def test_journal_runs_for_short_direction(sample_panel_csv: Path, tmp_path: Path, capsys):
    outdir = tmp_path / "j"
    code = main(
        [
            "journal",
            str(sample_panel_csv),
            "--direction",
            "short_only",
            "--series",
            "DR001",
            "--outdir",
            str(outdir),
            "--quiet",
        ]
    )
    capsys.readouterr()
    assert code == 0
    daily = pd.read_csv(outdir / "journal" / "journal.csv", encoding="utf-8-sig")
    assert set(daily["position"].unique()) <= {0, -1}
    report = (outdir / "journal" / "journal.md").read_text(encoding="utf-8")
    assert "每日交易日志" in report
