"""``run.py`` 全局配置入口的回归测试。

这里的用例专门盯住「CONFIG 的值 → 命令层 argparse.Namespace」这段桥接。
默认配置里 ``rate_cols=None``、``signal_window=(None, None)``，
两条转换分支都不会被走到，因此类型不匹配的缺陷在默认参数下完全不可见。
下面每个用例都用**非默认**取值把对应分支逼出来。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from ratema.backtest import DIRECTIONS, BacktestConfig
from ratema.commands._shared import make_backtest_config, parse_rate_cols
from ratema.parser import parse_signal_window

RUN_PY = Path(__file__).resolve().parent.parent / "run.py"


def _load_run_module():
    """按路径加载仓库根目录的 run.py（它不是包的一部分）。"""
    spec = importlib.util.spec_from_file_location("_ratema_run_entry", RUN_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_mod():
    return _load_run_module()


@pytest.fixture()
def cfg(run_mod):
    """CONFIG 的浅拷贝，避免用例之间互相污染。"""
    return dict(run_mod.CONFIG)


# --------------------------------------------------------------------------- #
# rate_cols：配置里写列表，命令层要的是逗号分隔字符串
# --------------------------------------------------------------------------- #
def test_none_rate_cols_stays_none(run_mod, cfg):
    """rate_cols=None（自动识别全部列）不应产生空字符串参数。

    注意这里显式传 None，而不是断言 CONFIG 的当前取值 —— CONFIG 是使用者
    按需修改的地方，测试不该因为改了配置就变红。
    """
    assert run_mod.build_args({**cfg, "rate_cols": None}).rate_cols is None


def test_rate_cols_list_is_joined(run_mod, cfg):
    rates = ["DR001", "DR007", "R001", "R007", "M0017139"]
    ns = run_mod.build_args({**cfg, "rate_cols": rates})
    assert ns.rate_cols == "DR001,DR007,R001,R007,M0017139"
    assert parse_rate_cols(ns) == rates


def test_rate_cols_list_strips_blanks(run_mod):
    assert run_mod._rate_cols_arg([" A ", "", "B", "  "]) == "A,B"
    assert run_mod._rate_cols_arg([]) is None


def test_rate_cols_string_passthrough(run_mod, cfg):
    assert parse_rate_cols(run_mod.build_args({**cfg, "rate_cols": "A,B"})) == ["A", "B"]


def test_single_rate_col_list(run_mod, cfg):
    """单元素列表不能退化成逐字符拆分。"""
    ns = run_mod.build_args({**cfg, "rate_cols": [" DR001 "]})
    assert parse_rate_cols(ns) == ["DR001"]


# --------------------------------------------------------------------------- #
# signal_window：配置里写二元组，命令层要的是 "START:END"
# --------------------------------------------------------------------------- #
def test_none_signal_window_stays_none(run_mod, cfg):
    assert run_mod.build_args({**cfg, "signal_window": (None, None)}).signal_window is None


def test_signal_window_tuple_round_trip(run_mod, cfg):
    ns = run_mod.build_args({**cfg, "signal_window": ("2015-01-01", "2020-12-31")})
    assert ns.signal_window == "2015-01-01:2020-12-31"
    # 命令层真正消费它，必须能解析回同一对日期
    assert parse_signal_window(ns.signal_window) == ("2015-01-01", "2020-12-31")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ((None, None), None),
        (("2015-01-01", None), "2015-01-01:"),
        ((None, "2020-12-31"), ":2020-12-31"),
    ],
)
def test_signal_window_partial(run_mod, value, expected):
    assert run_mod._signal_window_arg(value) == expected


def test_signal_window_one_sided_parses(run_mod):
    assert parse_signal_window(run_mod._signal_window_arg(("2015-01-01", None))) == (
        "2015-01-01",
        None,
    )
    assert parse_signal_window(None) == (None, None)


# --------------------------------------------------------------------------- #
# initial_capital：曾经在桥接层被整个丢掉，配置写了也不生效
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("capital", [1.0, 100.0, 1_000_000.0])
def test_initial_capital_reaches_backtest_config(run_mod, cfg, capital):
    ns = run_mod.build_args({**cfg, "initial_capital": capital})
    assert make_backtest_config(ns).initial_capital == capital


def test_missing_initial_capital_falls_back(run_mod):
    """旧版 Namespace 没有该字段时应回落 1.0，而不是抛 AttributeError。"""
    bare = argparse.Namespace(cost_bps=2.0, cost_mode="per_side", annualization=252, risk_free=0.0)
    assert make_backtest_config(bare).initial_capital == 1.0


def test_every_config_key_is_consumed(run_mod, cfg):
    """配置项必须真的被桥接层读走，防止再出现「写了没人用」的静默空转。"""
    ns = run_mod.build_args(cfg)
    for key in (
        "short",
        "long",
        "tol_mode",
        "tol",
        "std_window",
        "cost_bps",
        "cost_mode",
        "direction",
        "initial_capital",
        "annualization",
        "risk_free",
        "outdir",
        "mode",
        "vote_source",
        "target",
    ):
        assert hasattr(ns, key), f"build_args 漏掉配置项: {key}"


def test_config_values_survive_bridging(run_mod, cfg):
    """抽查若干个值：改配置就必须改到实际生效的参数。"""
    patched = {
        **cfg,
        "short_window": 5,
        "long_window": 60,
        "tol_mode": "bp",
        "tol": 3.0,
        "direction": "long_short",
        "cost_bps": 5.0,
        "sweep_target": "composite",
    }
    ns = run_mod.build_args(patched)
    assert (ns.short, ns.long) == (5, 60)
    assert (ns.tol_mode, ns.tol) == ("bp", 3.0)
    assert ns.direction == "long_short"
    assert make_backtest_config(ns).cost_bps == 5.0
    assert ns.target == "composite"


def test_validate_rejects_bad_direction(run_mod, cfg):
    with pytest.raises(SystemExit):
        run_mod.validate({**cfg, "direction": "both"})


def test_validate_rejects_bad_tol_mode(run_mod, cfg):
    with pytest.raises(SystemExit):
        run_mod.validate({**cfg, "tol_mode": "ppt"})


def test_validate_rejects_rel_tol_as_percent(run_mod, cfg):
    """rel 模式下把比例写成百分数（0.03 写成 3）必须被拦下。"""
    run_mod.validate({**cfg, "tol_mode": "rel", "tol": 0.03})  # 合法
    with pytest.raises(SystemExit):
        run_mod.validate({**cfg, "tol_mode": "rel", "tol": 3.0})


def test_validate_rejects_nonpositive_capital(run_mod, cfg):
    with pytest.raises(SystemExit):
        run_mod.validate({**cfg, "initial_capital": 0.0})


def test_validate_rejects_all_steps_off(run_mod, cfg):
    off = dict.fromkeys(cfg["steps"], False)
    with pytest.raises(SystemExit):
        run_mod.validate({**cfg, "steps": off})


def test_direction_is_a_known_constant(run_mod, cfg):
    """配置里的方向必须是回测层认识的常量。

    这里刻意**不**断言它等于 ``BacktestConfig`` 的默认值 —— 使用者会按需把
    CONFIG['direction'] 改成 short_only / long_short，那是正常用法，
    测试锁死具体取值只会让人改配置就红。
    """
    assert cfg["direction"] in DIRECTIONS
    assert BacktestConfig().direction in DIRECTIONS


def test_shipped_config_is_valid(run_mod, cfg):
    """随附的 CONFIG 必须自洽：跑之前不该被自己的校验拦下。

    若使用者正在改配置、暂时填了个非法值，这条会红 —— 那是预期的，
    因为此时 run.py 本来也跑不起来。
    """
    run_mod.validate(cfg)
