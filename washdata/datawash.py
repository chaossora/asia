#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APMCM 2025 Problem C - Unified Data Cleaning Script

本脚本读取：
- USITC Tariff Database 关税 Excel 文件（2015–2025）
- USITC DataWeb 导出文件：
    * DataWeb-Query-Export.xlsx  （FAS Value，出口额）
    * DataWeb-Query-Import.xlsx  （General Import Charges，关税收入）

输出清洗后的 CSV 文件：
- tariff_yearly.csv                : 年度 HTS8 关税面板
- tariff_hs2_panel.csv             : 年度 HS2 聚合 MFN 关税
- tariff_hs4_panel.csv             : 年度 HS4 聚合 MFN 关税
- trade_export_panel.csv           : 出口额 + HS2 关税
- trade_duty_panel.csv             : 关税收入 + HS2 关税
- exports_CN_sector.csv            : 对华出口（按年份 × sector_big）
- duty_total_year.csv              : 全部关税收入（按年）
- duty_by_sector_year.csv          : 关税收入（按年 × sector_big）

只需要根据你本地数据文件的位置改一下 CONFIG 部分（默认假设脚本和数据在同一目录）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# 可选依赖：用于将国家名映射为 ISO3 代码。如果没有安装 pycountry，则 partner_iso3 会是 None。
try:
    import pycountry  # type: ignore
except ImportError:  # pragma: no cover - safe fallback
    pycountry = None  # type: ignore


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def build_default_config() -> Dict[str, object]:
    """
    构建默认配置字典。

    默认优先使用脚本旁的 data/ 目录存放原始文件；若不存在则回退到脚本目录。
    如果你的路径不同，只需要在这里改文件名/路径即可。
    """
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        # 交互式环境下的回退
        base_dir = Path.cwd()

    data_dir = base_dir / "data"
    data_root = data_dir if data_dir.is_dir() else base_dir
    output_dir = base_dir / "output"

    # 关税 Excel 文件列表：(year, filename)
    tariff_file_info: List[Tuple[int, str]] = [
        (2015, "tariff_database_2015.xlsx"),
        (2016, "tariff_database_2016.xlsx"),
        (2017, "tariff_database_2017.xlsx"),
        (2018, "tariff_database_2018.xlsx"),
        (2019, "2019_Tariff_Database_v11.xlsx"),
        (2020, "tariff_database_202010.xlsx"),
        (2021, "tariff database_202106.xlsx"),
        (2022, "tariff database_202207.xlsx"),
        (2023, "tariff database_202307.xlsx"),
        (2024, "tariff_database_202405.xlsx"),
        (2025, "tariff_database_2025.xlsx"),
    ]

    config: Dict[str, object] = {
        "BASE_DIR": base_dir,
        "DATA_DIR": data_root,
        "OUTPUT_DIR": output_dir,
        "TARIFF_DIR": data_root,  # 关税 Excel 所在目录
        "TARIFF_FILE_INFO": tariff_file_info,
        "DATAWEB_EXPORT_XLSX": data_root / "DataWeb-Query-Export.xlsx",
        "DATAWEB_IMPORT_XLSX": data_root / "DataWeb-Query-Import.xlsx",
        "MID_MONTH_DAY": "-06-30",  # 年度中点
    }
    return config


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def ensure_output_dir(path: Path) -> None:
    """
    确保输出目录存在。
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def zero_pad_series(series: pd.Series, width: int) -> pd.Series:
    """
    将 Series 转为字符串并左侧补零到指定宽度。
    """
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(width)
    )


def parse_date_series(series: pd.Series) -> pd.Series:
    """
    将 Series 解析为 datetime，错误转为 NaT。
    """
    return pd.to_datetime(series, errors="coerce")


def map_country_to_iso3(name: Optional[str]) -> Optional[str]:
    """
    将国家名称映射为 ISO3 代码（如果安装了 pycountry）。

    未安装 pycountry 或匹配失败时返回 None。
    """
    if not isinstance(name, str):
        return None
    name_str = name.strip()
    if not name_str:
        return None
    if pycountry is None:
        return None
    try:
        country = pycountry.countries.lookup(name_str)
        return country.alpha_3  # type: ignore[no-any-return]
    except Exception:
        return None


def classify_hs2_sector_big(hs2_code: str) -> str:
    """
    将 HS2 章号映射为大类行业（sector_big），用于宏观分析。
    """
    try:
        code_int = int(str(hs2_code).strip())
    except (TypeError, ValueError):
        return "others"

    if 1 <= code_int <= 24:
        return "agriculture"
    if 25 <= code_int <= 27:
        return "mineral"
    if 28 <= code_int <= 38:
        return "chemical"
    if 39 <= code_int <= 40:
        return "plastic_rubber"
    if 41 <= code_int <= 43:
        return "hides_skins_leather"
    if 44 <= code_int <= 49:
        return "wood_paper"
    if 50 <= code_int <= 63:
        return "textiles"
    if 64 <= code_int <= 67:
        return "footwear_headgear"
    if 68 <= code_int <= 71:
        return "stone_glass_jewelry"
    if 72 <= code_int <= 83:
        return "base_metals"
    if code_int == 84:
        return "machinery"
    if code_int == 85:
        return "electrical_equipment"
    if 86 <= code_int <= 89:
        return "transport_equipment"
    if 90 <= code_int <= 92:
        return "precision_instruments"
    if code_int == 93:
        return "arms_ammunition"
    if 94 <= code_int <= 96:
        return "misc_manufactures"
    if code_int == 97:
        return "art_collectors_pieces"
    return "others"


def classify_hs4_sector_specific(hs4_code: str) -> str:
    """
    对关键行业的 HS4 做更细标签：大豆、汽车、半导体等。

    未列出的 HS4 返回 "other"。
    """
    code_str = str(hs4_code).strip()
    if code_str.isdigit() and len(code_str) < 4:
        code_str = code_str.zfill(4)

    mapping: Dict[str, str] = {
        "1201": "soybean",
        "8703": "auto_passenger",
        "8704": "auto_truck",
        "8541": "semiconductor_diode",
        "8542": "semiconductor_ic",
    }
    return mapping.get(code_str, "other")


# ---------------------------------------------------------------------------
# 1. 关税库：读取并年化 HTS8 面板
# ---------------------------------------------------------------------------

def read_single_tariff_file(
    year: int,
    file_path: Path,
) -> pd.DataFrame:
    """
    读取单个年度关税 Excel，并做基础清洗：
    - hts8 补零为 8 位字符串
    - 生成 hs2、hs4、hs6
    - 解析 begin_effect_date / end_effective_date
    - 转换主要税率字段为数值
    - 生成 has_additional_duty 标志
    - 添加 year 列
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Tariff file for year {year} not found: {file_path}")

    df = pd.read_excel(file_path)

    # 确保 hts8 存在并补零
    if "hts8" not in df.columns:
        raise KeyError(f"'hts8' column not found in tariff file: {file_path}")

    df["hts8"] = zero_pad_series(df["hts8"], 8)
    df["hs2"] = df["hts8"].str[:2]
    df["hs4"] = df["hts8"].str[:4]
    df["hs6"] = df["hts8"].str[:6]

    df["year"] = int(year)

    # 日期字段
    if "begin_effect_date" in df.columns:
        df["begin_effect_date"] = parse_date_series(df["begin_effect_date"])
    else:
        df["begin_effect_date"] = pd.NaT

    if "end_effective_date" in df.columns:
        df["end_effective_date"] = parse_date_series(df["end_effective_date"])
    else:
        df["end_effective_date"] = pd.NaT

    df["end_effective_date"] = df["end_effective_date"].fillna(pd.Timestamp("2050-12-31"))

    # 税率字段
    float_cols = [
        "mfn_ad_val_rate",
        "mfn_specific_rate",
        "mfn_other_rate",
        "col2_ad_val_rate",
        "col2_specific_rate",
        "col2_other_rate",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    int_cols = [
        "mfn_rate_type_code",
        "col2_rate_type_code",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # additional_duty Yes/No -> has_additional_duty
    if "additional_duty" in df.columns:
        df["has_additional_duty"] = (
            df["additional_duty"].astype(str).str.strip().str.lower() == "yes"
        ).astype(int)
    else:
        df["has_additional_duty"] = 0

    return df


def align_tariff_columns(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    多个年度关税 DataFrame 列对齐：取所有列的并集，然后重索引。
    """
    all_cols: List[str] = sorted({col for df in dfs for col in df.columns})
    aligned = [df.reindex(columns=all_cols) for df in dfs]
    combined = pd.concat(aligned, ignore_index=True)
    return combined


def annualize_tariff_by_middate(tariff_raw_allyears: pd.DataFrame) -> pd.DataFrame:
    """
    对 (year, hts8) 做年度选择，形成唯一记录，规则：

    1. 对每一行计算 mid_date = year-06-30，year_end = year-12-31。
    2. 首选满足 begin_effect_date <= mid_date <= end_effective_date 的记录。
    3. 若没有覆盖 mid_date 的记录，则在 begin_effect_date <= year_end 的记录中选。
    4. 在候选记录中，按 priority（覆盖 mid_date 优先）和 begin_effect_date
      （越新越优先）选一条。
    5. 如果没有任何符合条件的记录，该 (year, hts8) 会被丢弃。
    """
    df = tariff_raw_allyears.copy()

    # 确保日期为 datetime
    df["begin_effect_date"] = parse_date_series(df["begin_effect_date"])
    df["end_effective_date"] = parse_date_series(df["end_effective_date"])
    df["end_effective_date"] = df["end_effective_date"].fillna(pd.Timestamp("2050-12-31"))

    df["year"] = df["year"].astype(int)
    df["mid_date"] = pd.to_datetime(df["year"].astype(str) + "-06-30")
    df["year_end"] = pd.to_datetime(df["year"].astype(str) + "-12-31")

    covers_mid = (df["begin_effect_date"] <= df["mid_date"]) & (df["end_effective_date"] >= df["mid_date"])
    before_end = df["begin_effect_date"] <= df["year_end"]

    # priority: 2 = 覆盖 mid_date；1 = 只在 year_end 之前有效；0 = 无效
    df["priority"] = np.select(
        condlist=[covers_mid, before_end],
        choicelist=[2, 1],
        default=0,
    )

    df_valid = df[df["priority"] > 0].copy()
    if df_valid.empty:
        raise ValueError("No valid tariff records found after applying mid-date/year-end selection rules.")

    # 排序，使得每个 (year, hts8) 中最后一条是我们需要的
    df_valid = df_valid.sort_values(
        ["year", "hts8", "priority", "begin_effect_date"],
        ascending=[True, True, True, True],
    )

    # 每组取最后一条
    tariff_yearly = (
        df_valid
        .groupby(["year", "hts8"], as_index=False)
        .tail(1)
        .drop(columns=["mid_date", "year_end", "priority"])
        .sort_values(["year", "hts8"])
        .reset_index(drop=True)
    )

    return tariff_yearly


def build_tariff_yearly_panel(config: Dict[str, object]) -> pd.DataFrame:
    """
    读取所有年度关税 Excel，列对齐后按年度中点规则生成年度 HTS8 面板。
    """
    tariff_dir: Path = config["TARIFF_DIR"]  # type: ignore[assignment]
    tariff_file_info: List[Tuple[int, str]] = config["TARIFF_FILE_INFO"]  # type: ignore[assignment]

    all_year_dfs: List[pd.DataFrame] = []

    for year, filename in tariff_file_info:
        file_path = tariff_dir / filename
        print(f"[Tariff] Reading {file_path} for year {year} ...")
        df = read_single_tariff_file(year, file_path)
        all_year_dfs.append(df)

    print("[Tariff] Aligning columns across years ...")
    tariff_raw_allyears = align_tariff_columns(all_year_dfs)

    print("[Tariff] Annualizing by mid-date selection ...")
    tariff_yearly = annualize_tariff_by_middate(tariff_raw_allyears)

    print(f"[Tariff] Completed annual panel with {len(tariff_yearly)} rows.")
    return tariff_yearly


# ---------------------------------------------------------------------------
# 2. 将 HTS8 聚合到 HS2 / HS4
# ---------------------------------------------------------------------------

def build_tariff_aggregates(
    tariff_yearly: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从 HTS8 年度面板构造 HS2 / HS4 聚合 MFN 关税。

    返回
    -------
    tariff_hs2_panel : DataFrame
        列：year, hs2, mfn_adval_hs2, mfn_spec_q1_hs2, mfn_other_hs2,
             has_additional_duty_hs2
    tariff_hs4_panel : DataFrame
        列：year, hs4, mfn_adval_hs4, mfn_spec_q1_hs4, mfn_other_hs4,
             has_additional_duty_hs4, sector
    """
    core_cols = [
        "year",
        "hts8",
        "hs2",
        "hs4",
        "hs6",
        "mfn_ad_val_rate",
        "mfn_specific_rate",
        "mfn_other_rate",
        "has_additional_duty",
    ]
    missing = [c for c in core_cols if c not in tariff_yearly.columns]
    if missing:
        raise KeyError(f"Missing expected columns in tariff_yearly: {missing}")

    t_core = tariff_yearly[core_cols].copy()

    # HS2 聚合
    hs2_group = t_core.groupby(["year", "hs2"], as_index=False)
    tariff_hs2_panel = hs2_group.agg(
        mfn_adval_hs2=("mfn_ad_val_rate", "mean"),
        mfn_spec_q1_hs2=("mfn_specific_rate", "mean"),
        mfn_other_hs2=("mfn_other_rate", "mean"),
        has_additional_duty_hs2=("has_additional_duty", "max"),
    )

    # HS4 聚合
    hs4_group = t_core.groupby(["year", "hs4"], as_index=False)
    tariff_hs4_panel = hs4_group.agg(
        mfn_adval_hs4=("mfn_ad_val_rate", "mean"),
        mfn_spec_q1_hs4=("mfn_specific_rate", "mean"),
        mfn_other_hs4=("mfn_other_rate", "mean"),
        has_additional_duty_hs4=("has_additional_duty", "max"),
    )

    # HS4 加具体行业标签
    tariff_hs4_panel["sector"] = tariff_hs4_panel["hs4"].map(classify_hs4_sector_specific)

    return tariff_hs2_panel, tariff_hs4_panel


# ---------------------------------------------------------------------------
# 3. DataWeb 贸易数据：宽表转长表
# ---------------------------------------------------------------------------

def read_dataweb_metric(
    xlsx_path: Path,
    sheet_name: str,
    metric_name: str,
) -> pd.DataFrame:
    """
    读取一个 DataWeb 导出表并转成长表。

    假定 sheet 结构（与你的样例一致）：
    - 第 0 行： "Total Exports|Annual Data" / "General Import Charges|Annual Data"
    - 第 1 行： "Data Row Count" 元信息
    - 第 2 行： 表头： "Data Type", "HTS Number", "Description", "Country", "2020", ... "2025"
    - 第 3 行起：真实数据

    参数
    ----------
    xlsx_path : Path
    sheet_name : str
        "FAS Value" 或 "General Import Charges"
    metric_name : str
        内部指标名："export_fas" 或 "import_duty"

    返回
    -------
    长表 DataFrame，列：
        year, hs2, description, partner_name, partner_iso3, metric, value
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"DataWeb file not found: {xlsx_path}")

    print(f"[DataWeb] Reading {xlsx_path} sheet '{sheet_name}' ...")

    # header=2 直接用第三行作为列名
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=2)

    expected_cols = ["Data Type", "HTS Number", "Description", "Country"]
    for col in expected_cols:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in sheet '{sheet_name}' of {xlsx_path}")

    # HTS Number -> HS2
    df["HTS Number"] = pd.to_numeric(df["HTS Number"], errors="coerce")
    df["hs2"] = zero_pad_series(df["HTS Number"], 2)

    # 自动识别年份列（2020–2025）
    year_cols: List[str] = []
    for col in df.columns:
        try:
            year_int = int(str(col))
        except (TypeError, ValueError):
            continue
        if 2020 <= year_int <= 2100:
            year_cols.append(str(col))

    if not year_cols:
        raise ValueError(f"No year columns (like 2020, 2021, ...) found in sheet '{sheet_name}'")

    id_vars = ["hs2", "Description", "Country"]
    long = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )

    long["year"] = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce").fillna(0.0)

    long["metric"] = metric_name
    long["partner_name"] = long["Country"].astype(str)
    long["description"] = long["Description"].astype(str)

    # ISO3 代码（可选）
    long["partner_iso3"] = long["partner_name"].apply(map_country_to_iso3)

    cols = [
        "year",
        "hs2",
        "description",
        "partner_name",
        "partner_iso3",
        "metric",
        "value",
    ]
    long = long[cols].copy()

    print(f"[DataWeb] Loaded {len(long)} rows for metric '{metric_name}'.")
    return long


def build_trade_long_tables(
    config: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从 DataWeb 文件构建：
    - exports_long (FAS Value)
    - duties_long (General Import Charges)
    """
    export_path: Path = config["DATAWEB_EXPORT_XLSX"]  # type: ignore[assignment]
    import_path: Path = config["DATAWEB_IMPORT_XLSX"]  # type: ignore[assignment]

    exports_long = read_dataweb_metric(
        export_path,
        sheet_name="FAS Value",
        metric_name="export_fas",
    )
    duties_long = read_dataweb_metric(
        import_path,
        sheet_name="General Import Charges",
        metric_name="import_duty",
    )

    return exports_long, duties_long


# ---------------------------------------------------------------------------
# 4. 将关税并入贸易数据
# ---------------------------------------------------------------------------

def merge_tariff_trade(
    tariff_hs2_panel: pd.DataFrame,
    exports_long: pd.DataFrame,
    duties_long: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    将 HS2 聚合关税合并到出口 & 关税收入面板里。

    返回
    -------
    trade_export_panel : year, hs2, partner_name, partner_iso3, export_fas, mfn_*_hs2...
    trade_duty_panel   : year, hs2, partner_name, partner_iso3, import_duty, mfn_*_hs2...
    """
    # 出口面板
    e = exports_long.copy()
    e = e.rename(columns={"value": "export_fas"})
    e = e[["year", "hs2", "partner_name", "partner_iso3", "export_fas"]]

    trade_export_panel = e.merge(
        tariff_hs2_panel,
        how="left",
        on=["year", "hs2"],
        validate="m:1",
    )

    # 关税收入面板
    d = duties_long.copy()
    d = d.rename(columns={"value": "import_duty"})
    d = d[["year", "hs2", "partner_name", "partner_iso3", "import_duty"]]

    trade_duty_panel = d.merge(
        tariff_hs2_panel,
        how="left",
        on=["year", "hs2"],
        validate="m:1",
    )

    print(f"[Merge] trade_export_panel rows: {len(trade_export_panel)}")
    print(f"[Merge] trade_duty_panel rows: {len(trade_duty_panel)}")

    return trade_export_panel, trade_duty_panel


# ---------------------------------------------------------------------------
# 5. 五题共用的派生变量
# ---------------------------------------------------------------------------

def add_sector_labels(
    tariff_yearly: pd.DataFrame,
    tariff_hs2_panel: pd.DataFrame,
    tariff_hs4_panel: pd.DataFrame,
    trade_export_panel: pd.DataFrame,
    trade_duty_panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    给关税和贸易面板加上行业标签：HS4 的具体 sector，HS2 的 sector_big。
    """
    # HTS8 年度面板：用 hs4 映射具体行业
    if "sector" not in tariff_yearly.columns:
        tariff_yearly = tariff_yearly.copy()
        tariff_yearly["sector"] = tariff_yearly["hs4"].map(classify_hs4_sector_specific)

    # HS2 大类行业标签
    for df, name in [
        (tariff_hs2_panel, "tariff_hs2_panel"),
        (trade_export_panel, "trade_export_panel"),
        (trade_duty_panel, "trade_duty_panel"),
    ]:
        if "hs2" not in df.columns:
            raise KeyError(f"'hs2' column not found in {name}")
        df["sector_big"] = df["hs2"].astype(str).map(classify_hs2_sector_big)

    return tariff_yearly, tariff_hs2_panel, tariff_hs4_panel, trade_export_panel, trade_duty_panel


def build_common_features(
    tariff_yearly: pd.DataFrame,
    tariff_hs2_panel: pd.DataFrame,
    tariff_hs4_panel: pd.DataFrame,
    trade_export_panel: pd.DataFrame,
    trade_duty_panel: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    构建五道题都要用到的统一派生数据集。
    """
    (
        tariff_yearly,
        tariff_hs2_panel,
        tariff_hs4_panel,
        trade_export_panel,
        trade_duty_panel,
    ) = add_sector_labels(
        tariff_yearly=tariff_yearly,
        tariff_hs2_panel=tariff_hs2_panel,
        tariff_hs4_panel=tariff_hs4_panel,
        trade_export_panel=trade_export_panel,
        trade_duty_panel=trade_duty_panel,
    )

    # (1) 对中国出口（按年 × sector_big）
    exports_CN_sector = (
        trade_export_panel.loc[trade_export_panel["partner_name"] == "China"]
        .groupby(["year", "sector_big"], as_index=False)["export_fas"]
        .sum()
        .rename(columns={"export_fas": "export_US_to_CN_by_sector"})
    )

    # (2) 全部关税收入（按年）
    duty_total_year = (
        trade_duty_panel
        .groupby("year", as_index=False)["import_duty"]
        .sum()
        .rename(columns={"import_duty": "duty_total"})
    )

    # (3) 按年 × sector_big 的关税收入
    duty_by_sector_year = (
        trade_duty_panel
        .groupby(["year", "sector_big"], as_index=False)["import_duty"]
        .sum()
        .rename(columns={"import_duty": "duty_by_sector"})
    )

    outputs: Dict[str, pd.DataFrame] = {
        "tariff_yearly": tariff_yearly,
        "tariff_hs2_panel": tariff_hs2_panel,
        "tariff_hs4_panel": tariff_hs4_panel,
        "trade_export_panel": trade_export_panel,
        "trade_duty_panel": trade_duty_panel,
        "exports_CN_sector": exports_CN_sector,
        "duty_total_year": duty_total_year,
        "duty_by_sector_year": duty_by_sector_year,
    }
    return outputs


# ---------------------------------------------------------------------------
# 6. 保存结果 & 基础质量检查
# ---------------------------------------------------------------------------

def save_all_outputs(
    outputs: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    将所有 DataFrame 保存为 CSV。
    """
    ensure_output_dir(output_dir)
    for name, df in outputs.items():
        file_path = output_dir / f"{name}.csv"
        print(f"[Save] Writing {file_path} ({len(df)} rows, {len(df.columns)} columns)")
        df.to_csv(file_path, index=False)


def run_basic_quality_checks(
    outputs: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    做一些基础检查和简单图表：
    - MFN 从价税率是否在 [0,1]
    - 关税收入是否为非负
    - 按年关税收入时间序列折线图
    """
    ensure_output_dir(output_dir)

    tariff_yearly = outputs["tariff_yearly"]
    trade_duty_panel = outputs["trade_duty_panel"]
    duty_total_year = outputs["duty_total_year"]

    # 6.1 MFN 税率范围检查
    if "mfn_ad_val_rate" in tariff_yearly.columns:
        bad_mfn = tariff_yearly[
            (tariff_yearly["mfn_ad_val_rate"] < 0)
            | (tariff_yearly["mfn_ad_val_rate"] > 1)
        ]
        bad_mfn_path = output_dir / "check_bad_mfn_rates.csv"
        print(f"[Check] MFN rates outside [0,1]: {len(bad_mfn)} rows -> {bad_mfn_path}")
        bad_mfn.to_csv(bad_mfn_path, index=False)

    # 6.2 关税收入非负检查
    bad_duty = trade_duty_panel[trade_duty_panel["import_duty"] < 0]
    bad_duty_path = output_dir / "check_negative_duty.csv"
    print(f"[Check] Negative import duties: {len(bad_duty)} rows -> {bad_duty_path}")
    bad_duty.to_csv(bad_duty_path, index=False)

    # 6.3 按年关税收入时间序列图
    if not duty_total_year.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(duty_total_year["year"], duty_total_year["duty_total"], marker="o")
        ax.set_xlabel("Year")
        ax.set_ylabel("Total Import Duty (General Import Charges)")
        ax.set_title("US Import Duty Revenue by Year")
        ax.grid(True, linestyle="--", alpha=0.5)
        plot_path = output_dir / "duty_total_by_year.png"
        print(f"[Plot] Saving duty_total_by_year plot to {plot_path}")
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 7. 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    """
    主流程：
    1. 构建年度 HTS8 关税面板
    2. 聚合到 HS2 / HS4
    3. 从 DataWeb 构建贸易长表
    4. 合并关税与贸易
    5. 构建五题共用的派生数据
    6. 保存结果并做基础质量检查
    """
    config = build_default_config()

    output_dir: Path = config["OUTPUT_DIR"]  # type: ignore[assignment]
    ensure_output_dir(output_dir)

    # 1. 关税面板
    tariff_yearly = build_tariff_yearly_panel(config)

    # 2. HS2 / HS4 聚合
    tariff_hs2_panel, tariff_hs4_panel = build_tariff_aggregates(tariff_yearly)

    # 3. 贸易长表
    exports_long, duties_long = build_trade_long_tables(config)

    # 4. 合并关税与贸易
    trade_export_panel, trade_duty_panel = merge_tariff_trade(
        tariff_hs2_panel=tariff_hs2_panel,
        exports_long=exports_long,
        duties_long=duties_long,
    )

    # 5. 构建统一派生数据
    outputs = build_common_features(
        tariff_yearly=tariff_yearly,
        tariff_hs2_panel=tariff_hs2_panel,
        tariff_hs4_panel=tariff_hs4_panel,
        trade_export_panel=trade_export_panel,
        trade_duty_panel=trade_duty_panel,
    )

    # 6. 保存并检查
    save_all_outputs(outputs, output_dir=output_dir)
    run_basic_quality_checks(outputs, output_dir=output_dir)

    print("[Done] All data cleaned and saved.")


if __name__ == "__main__":
    main()
