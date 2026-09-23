"""数据读取与格式转换：Excel -> CSV，以及统一的数据集加载入口。

约定
----
原始 Excel 每个 sheet 的前两行是表头：

    第 1 行：中文名称（如「银存间质押1日」「中债-10年期国债净价(总值)指数」）
    第 2 行：指标代码（如 DR001、R001、CBA04502.CS）

第 1 列固定为日期。代码（第 2 行）被用作规范列名，中文名称被保留为展示名。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

DATE_COL = "date"
COLUMNS_META_FILE = "columns.json"

#: 支持的 Excel 扩展名（小写）
EXCEL_SUFFIXES = frozenset({".xlsx", ".xlsm", ".xls", ".xltx", ".xltm"})
_BOM = "\ufeff"


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def clean_label(value: Any) -> str:
    """去掉 BOM / 首尾空白 / 压缩内部空白。"""
    if value is None:
        return ""
    text = str(value).replace(_BOM, "").strip()
    return re.sub(r"\s+", " ", text)


def _dedupe(names: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in names:
        name = raw or "unnamed"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


# --------------------------------------------------------------------------- #
# 从 Excel 工作表切出行数据
# --------------------------------------------------------------------------- #
def read_workbook_rows(path: str | Path) -> dict[str, list[list[Any]]]:
    """读取 workbook 全部 sheet，返回 {sheet_name: rows}。"""
    import openpyxl

    path = Path(path)
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        sheets: dict[str, list[list[Any]]] = {}
        for ws in wb.worksheets:
            sheets[ws.title] = [list(row) for row in ws.iter_rows(values_only=True)]
        return sheets
    finally:
        wb.close()


def frame_from_rows(
    rows: list[list[Any]],
    header_rows: int = 2,
    *,
    prefer_code: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """把 sheet 的原始行组装成规范 DataFrame。

    Returns
    -------
    (df, name_map)
        df: 第 1 列为 ``date``（datetime64[ns]），其余列为数值型指标。
        name_map: {列名: 中文展示名}
    """
    if header_rows < 1:
        raise ValueError("header_rows 必须 >= 1")
    if len(rows) <= header_rows:
        raise ValueError("工作表为空或表头行数超过总行数")

    names_row = [clean_label(v) for v in rows[0]]
    codes_row = [clean_label(v) for v in rows[1]] if header_rows >= 2 else list(names_row)

    width = max(len(names_row), len(codes_row), *(len(r) for r in rows))
    names_row += [""] * (width - len(names_row))
    codes_row += [""] * (width - len(codes_row))

    if prefer_code:
        labels = [code or name for code, name in zip(codes_row, names_row)]
    else:
        labels = [name or code for code, name in zip(codes_row, names_row)]

    # 第 1 列永远是日期
    labels[0] = DATE_COL
    labels = _dedupe(labels)

    body = [list(r) + [None] * (width - len(r)) for r in rows[header_rows:]]
    df = pd.DataFrame(body, columns=labels)

    # 丢弃全空行（Excel 常见的尾部空行）
    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty:
        raise ValueError("工作表中没有数据行")

    # 日期
    dates = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[DATE_COL] = dates
    df = df.loc[df[DATE_COL].notna()].reset_index(drop=True)

    # 数值列
    for col in df.columns:
        if col == DATE_COL:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 丢弃整列全空的列（除日期外）
    keep = [DATE_COL] + [c for c in df.columns if c != DATE_COL and df[c].notna().any()]
    df = df[keep]

    name_map: dict[str, str] = {}
    for label, code, name in zip(labels, codes_row, names_row):
        if label == DATE_COL:
            continue
        name_map[label] = name or code or label

    df = df.sort_values(DATE_COL, kind="mergesort")
    df = df.drop_duplicates(subset=[DATE_COL], keep="last").reset_index(drop=True)
    return df, name_map


# --------------------------------------------------------------------------- #
# 写出 CSV
# --------------------------------------------------------------------------- #
def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig 让中文列名在 Excel 中直接双击打开不会乱码
    df.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")


def rows_to_raw_frame(rows: list[list[Any]], header_rows: int = 2) -> pd.DataFrame:
    """忠实还原：第 1 行中文名做表头，第 2 行代码作为首行数据。"""
    if header_rows >= 2:
        header = [clean_label(v) or f"col{i}" for i, v in enumerate(rows[0])]
        data = [list(rows[1])] + [list(r) for r in rows[2:]]
    else:
        header = [clean_label(v) or f"col{i}" for i, v in enumerate(rows[0])]
        data = [list(r) for r in rows[1:]]
    header = _dedupe(header)
    width = len(header)
    body = [r + [None] * (width - len(r)) for r in data]
    df = pd.DataFrame(body, columns=header)
    # 日期列格式化
    first = df.columns[0]
    for i, val in enumerate(df[first]):
        ts = pd.to_datetime(val, errors="coerce")
        if pd.notna(ts):
            df.iloc[i, 0] = ts.strftime("%Y-%m-%d")
    return df


@dataclass
class ConversionReport:
    """转换结果清单。"""

    input_file: str
    output_dir: str
    files: list[str] = field(default_factory=list)
    sheets: list[str] = field(default_factory=list)
    rows: int = 0
    columns: list[str] = field(default_factory=list)
    name_map: dict[str, str] = field(default_factory=dict)
    date_min: str = ""
    date_max: str = ""


def convert_excel_to_csv(
    xlsx_path: str | Path,
    outdir: str | Path = "data/csv",
    header_rows: int = 2,
) -> ConversionReport:
    """把文件夹内的 Excel 转成 CSV。

    产出：
      * ``<outdir>/<sheet>_raw.csv``   —— 忠实还原（表头=中文名，第 1 行数据=代码）
      * ``<outdir>/panel.csv``         —— 规范宽表（date + 指标代码列），回测直接使用
      * ``<outdir>/long.csv``          —— 长表（date, series_code, series_name, value）
      * ``<outdir>/series/<code>.csv`` —— 每个指标单独一个 CSV
      * ``<outdir>/columns.json``      —— 代码 -> 中文名 映射
    """
    xlsx_path = Path(xlsx_path)
    outdir = Path(outdir)
    sheets = read_workbook_rows(xlsx_path)

    report = ConversionReport(input_file=str(xlsx_path), output_dir=str(outdir))
    report.sheets = list(sheets.keys())

    panel_frames: list[pd.DataFrame] = []
    name_map: dict[str, str] = {}

    for sheet_name, rows in sheets.items():
        if not rows:
            continue
        raw_df = rows_to_raw_frame(rows, header_rows=header_rows)
        raw_path = outdir / f"{sheet_name}_raw.csv"
        _write_csv(raw_df, raw_path)
        report.files.append(str(raw_path))

        try:
            df, sheet_names = frame_from_rows(rows, header_rows=header_rows)
        except ValueError:
            continue
        name_map.update(sheet_names)
        panel_frames.append(df)

    if not panel_frames:
        raise ValueError(f"{xlsx_path} 中没有可用的数据表")

    panel = panel_frames[0]
    for extra in panel_frames[1:]:
        panel = panel.merge(extra, on=DATE_COL, how="outer")
    panel = panel.sort_values(DATE_COL).reset_index(drop=True)

    panel_path = outdir / "panel.csv"
    _write_csv(panel, panel_path)
    report.files.append(str(panel_path))

    long = panel.melt(id_vars=DATE_COL, var_name="series_code", value_name="value")
    long = long.dropna(subset=["value"]).reset_index(drop=True)
    long.insert(2, "series_name", long["series_code"].map(name_map).fillna(long["series_code"]))
    long_path = outdir / "long.csv"
    _write_csv(long, long_path)
    report.files.append(str(long_path))

    for col in panel.columns:
        if col == DATE_COL:
            continue
        series_path = outdir / "series" / f"{col}.csv"
        _write_csv(panel[[DATE_COL, col]], series_path)
        report.files.append(str(series_path))

    meta_path = outdir / COLUMNS_META_FILE
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(name_map, ensure_ascii=False, indent=2), encoding="utf-8")
    report.files.append(str(meta_path))

    report.rows = len(panel)
    report.columns = [c for c in panel.columns if c != DATE_COL]
    report.name_map = name_map
    report.date_min = panel[DATE_COL].min().strftime("%Y-%m-%d")
    report.date_max = panel[DATE_COL].max().strftime("%Y-%m-%d")
    return report


# --------------------------------------------------------------------------- #
# 统一加载入口
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    frame: pd.DataFrame
    name_map: dict[str, str]
    source: str

    @property
    def date(self) -> pd.Series:
        return self.frame[DATE_COL]

    @property
    def series(self) -> list[str]:
        return [c for c in self.frame.columns if c != DATE_COL]

    def display_name(self, code: str) -> str:
        return self.name_map.get(code, code)


def find_excel_input(base: str | Path = ".") -> Path:
    """在目录下查找 Excel 文件（忽略 Excel 的 ~$ 临时文件）。"""
    base = Path(base)
    candidates = sorted(
        (
            p
            for p in base.glob("*.xls*")
            if p.suffix.lower() in {".xlsx", ".xlsm", ".xls", ".xltx", ".xltm"}
            and not p.name.startswith("~$")
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"在 {base.resolve()} 下没有找到 Excel 文件")
    return candidates[0]


def find_default_input(base: str | Path = ".") -> Path:
    """回测默认输入：优先 data/csv/panel.csv，其次目录下的 Excel。"""
    base = Path(base)
    panel = base / "data" / "csv" / "panel.csv"
    if panel.exists():
        return panel
    return find_excel_input(base)


def load_dataset(path: str | Path) -> Dataset:
    """读取 Excel 或 CSV（panel / long / 单指标宽表）为统一数据集。"""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        sheets = read_workbook_rows(path)
        if not sheets:
            raise ValueError(f"{path} 中没有工作表")
        first_name, rows = next(iter(sheets.items()))
        frame, name_map = frame_from_rows(rows, header_rows=2)
        return Dataset(frame=frame, name_map=name_map, source=f"{path}[{first_name}]")

    if suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame.columns = [clean_label(c) for c in frame.columns]

        if DATE_COL not in frame.columns:
            # 允许首列是日期但列名不规范
            frame = frame.rename(columns={frame.columns[0]: DATE_COL})

        if "series_code" in frame.columns and "value" in frame.columns:
            # long 格式
            name_map = {}
            if "series_name" in frame.columns:
                name_map = (
                    frame.drop_duplicates("series_code")
                    .set_index("series_code")["series_name"]
                    .to_dict()
                )
            frame = frame.pivot_table(
                index=DATE_COL, columns="series_code", values="value", aggfunc="last"
            ).reset_index()
            frame.columns.name = None
        else:
            name_map = {}

        frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="coerce")
        frame = frame.loc[frame[DATE_COL].notna()]
        for col in frame.columns:
            if col != DATE_COL:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

        # 若同目录有 columns.json，补上中文名
        meta_path = path.parent / COLUMNS_META_FILE
        if meta_path.exists():
            try:
                stored = json.loads(meta_path.read_text(encoding="utf-8"))
                merged = {k: v for k, v in stored.items() if k in frame.columns}
                merged.update(name_map)
                name_map = merged or name_map
                if not name_map:
                    name_map = stored
            except (json.JSONDecodeError, OSError):
                pass

        frame = frame.sort_values(DATE_COL).reset_index(drop=True)
        frame = frame.drop_duplicates(subset=[DATE_COL], keep="last").reset_index(drop=True)
        if not name_map:
            name_map = {c: c for c in frame.columns if c != DATE_COL}
        return Dataset(frame=frame, name_map=name_map, source=str(path))

    raise ValueError(f"不支持的文件类型：{path.suffix}")


def resolve_roles(
    dataset: Dataset,
    index_col: str | None = None,
    rate_cols: list[str] | None = None,
) -> tuple[str, list[str]]:
    """确定「标的指数列」与「底层利率指标列」。"""
    available = dataset.series
    if not available:
        raise ValueError("数据集中没有可用的指标列")

    if index_col is None:
        index_col = next(
            (c for c in available if "CBA04502" in c.upper()),
            next((c for c in available if "指数" in dataset.display_name(c)), None),
        )
    if index_col is None or index_col not in available:
        raise ValueError(f"无法确定标的指数列，请用 --index-col 指定。可选：{available}")

    if rate_cols:
        missing = [c for c in rate_cols if c not in available]
        if missing:
            raise ValueError(f"指定的利率列不存在：{missing}；可选：{available}")
        resolved = list(rate_cols)
    else:
        resolved = [c for c in available if c != index_col]

    if not resolved:
        raise ValueError("没有可用的底层利率指标列")
    return index_col, resolved


__all__ = [
    "COLUMNS_META_FILE",
    "DATE_COL",
    "ConversionReport",
    "Dataset",
    "clean_label",
    "convert_excel_to_csv",
    "find_default_input",
    "find_excel_input",
    "frame_from_rows",
    "load_dataset",
    "read_workbook_rows",
    "resolve_roles",
]
