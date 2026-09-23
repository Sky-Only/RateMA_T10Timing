"""测试夹具。

受限环境下（系统临时目录不可写、且 pytest 自建的临时目录可能无法清理），
把 ``tmp_path`` 覆盖为项目内的 ``build/pytest-tmp``，用例结束后自行删除。
这样 pytest 的临时目录机制完全不会被触发。
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from _data import make_dataset, make_panel, write_workbook
from ratema.io_utils import Dataset

_TMP_ROOT = Path(__file__).resolve().parent.parent / "build" / "pytest-tmp"


@pytest.fixture
def tmp_path():
    """与 pytest 内置同名，但落在项目目录内。"""
    path = _TMP_ROOT / uuid.uuid4().hex[:12]
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def pytest_sessionfinish(session, exitstatus):
    """收尾：清掉空的临时根目录。"""
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 共享数据夹具
# --------------------------------------------------------------------------- #
@pytest.fixture()
def dataset() -> Dataset:
    """默认合成数据集（400 个交易日，2 个利率指标 + 标的指数）。"""
    return make_dataset()


@pytest.fixture()
def sample_workbook(tmp_path: Path) -> Path:
    return write_workbook(tmp_path / "sample.xlsx")


@pytest.fixture()
def sample_panel_csv(tmp_path: Path) -> Path:
    """落盘的合成 panel.csv（含 columns.json，便于校验中文名回读）。"""
    path = tmp_path / "panel.csv"
    make_panel().to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    (tmp_path / "columns.json").write_text(
        json.dumps(
            {
                "DR001": "银存间质押1日",
                "R001": "银行间质押1日",
                "CBA04502.CS": "中债-10年期国债净价(总值)指数",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path
