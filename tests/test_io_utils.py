"""Excel -> CSV 转换与数据集加载测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ratema.io_utils import (
    convert_excel_to_csv,
    frame_from_rows,
    load_dataset,
    resolve_roles,
)


def test_frame_from_rows_uses_codes_and_parses_dates(sample_workbook: Path):
    from ratema.io_utils import read_workbook_rows

    rows = read_workbook_rows(sample_workbook)["Sheet1"]
    df, names = frame_from_rows(rows, header_rows=2)

    assert list(df.columns) == ["date", "DR001", "R001", "CBA04502.CS"]
    assert str(df["date"].dtype).startswith("datetime64")
    assert len(df) == 4
    assert df["DR001"].iloc[0] == pytest.approx(1.60)
    assert names["DR001"] == "银存间质押1日"
    # 表头里的 BOM 必须被清掉
    assert names["CBA04502.CS"].startswith("中债")


def test_convert_writes_all_expected_files(sample_workbook: Path, tmp_path: Path):
    outdir = tmp_path / "csv"
    report = convert_excel_to_csv(sample_workbook, outdir)

    expected = [
        outdir / "Sheet1_raw.csv",
        outdir / "panel.csv",
        outdir / "long.csv",
        outdir / "series" / "DR001.csv",
        outdir / "series" / "R001.csv",
        outdir / "series" / "CBA04502.CS.csv",
        outdir / "columns.json",
    ]
    for path in expected:
        assert path.exists(), f"缺少产出文件 {path}"

    assert report.rows == 4
    assert report.columns == ["DR001", "R001", "CBA04502.CS"]
    assert report.date_min == "2024-01-02"
    assert report.date_max == "2024-01-05"

    # UTF-8 BOM，方便 Excel 直接打开中文不乱码
    raw_bytes = (outdir / "panel.csv").read_bytes()
    assert raw_bytes.startswith(b"\xef\xbb\xbf")

    name_map = json.loads((outdir / "columns.json").read_text(encoding="utf-8"))
    assert name_map["DR001"] == "银存间质押1日"

    long = pd.read_csv(outdir / "long.csv", encoding="utf-8-sig")
    assert set(long.columns) == {"date", "series_code", "series_name", "value"}
    assert len(long) == 4 * 3

    raw = pd.read_csv(outdir / "Sheet1_raw.csv", encoding="utf-8-sig")
    assert raw.iloc[0].tolist()[1:] == ["DR001", "R001", "CBA04502.CS"]


def test_load_dataset_from_excel_and_csv_agree(sample_workbook: Path, tmp_path: Path):
    outdir = tmp_path / "csv"
    convert_excel_to_csv(sample_workbook, outdir)

    from_excel = load_dataset(sample_workbook)
    from_csv = load_dataset(outdir / "panel.csv")

    assert from_excel.series == from_csv.series
    pd.testing.assert_frame_equal(
        from_excel.frame.reset_index(drop=True),
        from_csv.frame[from_excel.frame.columns].reset_index(drop=True),
        check_dtype=False,
    )
    # CSV 带来的 columns.json 让中文名一起回来了
    assert from_csv.display_name("DR001") == "银存间质押1日"


def test_load_dataset_from_long_format(sample_workbook: Path, tmp_path: Path):
    outdir = tmp_path / "csv"
    convert_excel_to_csv(sample_workbook, outdir)
    ds = load_dataset(outdir / "long.csv")
    assert set(ds.series) == {"DR001", "R001", "CBA04502.CS"}
    assert len(ds.frame) == 4
    assert ds.display_name("R001") == "银行间质押1日"


def test_load_dataset_from_single_series_csv(sample_workbook: Path, tmp_path: Path):
    outdir = tmp_path / "csv"
    convert_excel_to_csv(sample_workbook, outdir)
    ds = load_dataset(outdir / "series" / "DR001.csv")
    assert ds.series == ["DR001"]
    assert len(ds.frame) == 4


def test_resolve_roles_auto_detects_index_column(sample_workbook: Path):
    ds = load_dataset(sample_workbook)
    index_col, rates = resolve_roles(ds)
    assert index_col == "CBA04502.CS"
    assert rates == ["DR001", "R001"]


def test_resolve_roles_honours_explicit_columns(sample_workbook: Path):
    ds = load_dataset(sample_workbook)
    index_col, rates = resolve_roles(ds, index_col="R001", rate_cols=["DR001"])
    assert index_col == "R001"
    assert rates == ["DR001"]


def test_resolve_roles_rejects_missing_column(sample_workbook: Path):
    ds = load_dataset(sample_workbook)
    with pytest.raises(ValueError, match="不存在"):
        resolve_roles(ds, rate_cols=["NOT_A_COLUMN"])


def test_unsupported_extension_raises(tmp_path: Path):
    path = tmp_path / "x.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持"):
        load_dataset(path)


def test_frame_from_rows_handles_ragged_rows():
    rows = [
        ["日期", "A", "B"],
        [None, "A1", "B1"],
        [pd.Timestamp("2024-01-02").to_pydatetime(), 1.0],  # 少一列
        [pd.Timestamp("2024-01-03").to_pydatetime(), 2.0, 3.0],
    ]
    df, _ = frame_from_rows(rows, header_rows=2)
    assert list(df.columns) == ["date", "A1", "B1"]
    assert pd.isna(df["B1"].iloc[0])
    assert df["B1"].iloc[1] == pytest.approx(3.0)
