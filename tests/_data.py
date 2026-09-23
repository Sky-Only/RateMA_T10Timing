"""测试用合成数据构造器（供 conftest 与各测试模块共用）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from ratema.io_utils import Dataset

NAME_MAP = {
    "DR001": "银存间质押1日",
    "R001": "银行间质押1日",
    "CBA04502.CS": "中债-10年期国债净价(总值)指数",
}


def make_panel(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """生成足够长的合成面板数据（含标的指数与两个利率指标）。

    利率用较有趋势的随机游走，保证 MA20 与 MA120 的偏离达到 bp 量级，
    这样 2bp 之类的阈值才有意义。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    rate_a = 2.0 + rng.normal(0, 0.10, n).cumsum() * 0.12
    rate_b = 1.8 + rng.normal(0, 0.10, n).cumsum() * 0.12
    index = 100.0 * np.exp(np.cumsum(-0.03 * (rate_a - rate_a.mean()) + rng.normal(0, 0.001, n)))
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "DR001": rate_a,
            "R001": rate_b,
            "CBA04502.CS": index,
        }
    )


def make_dataset(n: int = 400, seed: int = 42) -> Dataset:
    """内存中的 Dataset（不落盘）。"""
    return Dataset(frame=make_panel(n=n, seed=seed), name_map=dict(NAME_MAP), source="synthetic")


def write_workbook(path: Path) -> Path:
    """构造一个与真实文件同结构的小 Excel（两行表头：中文名 + 代码）。"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["\ufeff日期", "银存间质押1日", "银行间质押1日", "中债-10年期国债净价(总值)指数"])
    ws.append([None, "DR001", "R001", "CBA04502.CS"])
    rows = [
        ("2024-01-02", 1.60, 1.70, 100.0),
        ("2024-01-03", 1.62, 1.72, 100.5),
        ("2024-01-04", 1.58, 1.68, 100.2),
        ("2024-01-05", 1.55, 1.65, 100.8),
    ]
    for date, a, b, c in rows:
        ws.append([pd.Timestamp(date).to_pydatetime(), a, b, c])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
