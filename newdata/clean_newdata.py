#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lightweight cleaning utilities for the supplementary newdata/ files used in
Question 2.

Outputs go to newdata/output/ and aim to be ready for the Armington model:
  - auto_imports_partner_cmd_annual.csv : 8703/8704 imports by partner
  - auto_imports_partner_total.csv      : imports summed across 8703/8704
  - fred_auto_sales_monthly.csv         : FRED monthly series (ALTSALES, LAUTONSA, FAUTONSA, FAUTOSAAR)
  - fred_auto_sales_annual.csv          : annual aggregates + import share
  - auto_import_sales_bea_quarterly.csv : BEA imported autos sales (quarterly)
  - auto_import_sales_bea_annual.csv    : annual mean of BEA series
  - auto_employment_monthly.csv         : BLS CES3133600101 series
  - auto_employment_annual.csv          : annual averages of employment
  - oica_production_country_year.csv    : vehicle production by country-year
  - io_total_requirements.csv           : BEA total requirements matrix (as-is)

Run from project root:
    python newdata/clean_newdata.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def build_config() -> Dict[str, Path]:
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()

    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "BASE_DIR": base_dir,
        "OUTPUT_DIR": output_dir,
    }


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------------------------------
# 1) Customs imports: 8703/8704 value + quantity
# ---------------------------------------------------------------------------

CMD_MAP = {
    8703: "passenger_vehicles",
    8704: "goods_vehicles",
}

# EU 成员 ISO3，用于汇总为 EU
EU_ISO = {
    "AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST",
    "FIN","FRA","DEU","GRC","HUN","IRL","ITA","LVA",
    "LTU","LUX","MLT","NLD","POL","PRT","ROU","SVK",
    "SVN","ESP","SWE",
}

PARTNER_ISO_MAP = {
    "CAN": "Canada",
    "124": "Canada",
    "JPN": "Japan",
    "392": "Japan",
    "MEX": "Mexico",
    "484": "Mexico",
    "KOR": "Korea",
    "CHN": "China",
}


def map_partner(iso: str, desc: str) -> str:
    iso_u = str(iso).upper()
    if iso_u in PARTNER_ISO_MAP:
        return PARTNER_ISO_MAP[iso_u]
    if iso_u in EU_ISO:
        return "EU"
    # Fallback to description
    name = str(desc).strip()
    return name if name else iso_u


def clean_auto_imports(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    # 兼容扩展文件列名
    if "partnerCode" in df.columns:
        df["partner"] = df["partnerCode"].map(lambda x: PARTNER_ISO_MAP.get(str(x), str(x)))
    else:
        df["partner"] = df.apply(lambda r: map_partner(r.get("partnerISO", ""), r.get("partnerDesc", "")), axis=1)
    cmd_col = "cmdCode"
    val_col = "cifvalue" if "cifvalue" in df.columns else "import_value_usd"
    qty_col = "qty" if "qty" in df.columns else "quantity"
    df["cmd"] = df[cmd_col].map(CMD_MAP).fillna(df[cmd_col].astype(str))
    df["value"] = to_numeric(df[val_col])
    df["qty"] = to_numeric(df[qty_col])
    df["unit_price"] = df["value"] / df["qty"]

    grouped = (
        df.groupby(["refYear", "partner", "cmd"], as_index=False)
        .agg(
            value=("value", "sum"),
            qty=("qty", "sum"),
        )
    )
    grouped["unit_price"] = grouped["value"] / grouped["qty"]

    # Shares by year relative to total partners for each cmd and for all cmd combined.
    yearly_totals = (
        grouped.groupby(["refYear", "cmd"], as_index=False)[["value", "qty"]]
        .sum()
        .rename(columns={"value": "value_total_cmd", "qty": "qty_total_cmd"})
    )
    grouped = grouped.merge(yearly_totals, on=["refYear", "cmd"], how="left", validate="m:1")
    grouped["value_share_cmd"] = grouped["value"] / grouped["value_total_cmd"]
    grouped["qty_share_cmd"] = grouped["qty"] / grouped["qty_total_cmd"]

    # Total across cmd
    total_by_partner = (
        grouped.groupby(["refYear", "partner"], as_index=False)[["value", "qty"]]
        .sum()
        .rename(columns={"value": "value_total", "qty": "qty_total"})
    )

    grand_totals = total_by_partner.groupby("refYear", as_index=False)[["value_total", "qty_total"]].sum().rename(
        columns={"value_total": "value_total_all", "qty_total": "qty_total_all"}
    )
    total_by_partner = total_by_partner.merge(grand_totals, on="refYear", how="left", validate="m:1")
    total_by_partner["value_share_all"] = total_by_partner["value_total"] / total_by_partner["value_total_all"]
    total_by_partner["qty_share_all"] = total_by_partner["qty_total"] / total_by_partner["qty_total_all"]

    return grouped, total_by_partner


def add_rest_from_world_total(total_by_partner: pd.DataFrame, world_path: Path) -> pd.DataFrame:
    """
    将“世界合计”减去已列伙伴，补一行 Rest。
    world_total csv: refYear, import_value_usd, quantity
    """
    if not world_path.exists():
        return total_by_partner
    world = pd.read_csv(world_path)
    world = world.rename(columns={"import_value_usd": "world_value", "quantity": "world_qty"})
    total = total_by_partner.copy()

    rows = []
    for _, row in world.iterrows():
        year = int(row["refYear"])
        w_val = float(row["world_value"])
        w_qty = float(row["world_qty"])
        sub = total[total["refYear"] == year]
        used_val = sub["value_total"].sum()
        used_qty = sub["qty_total"].sum()
        rest_val = max(0.0, w_val - used_val)
        rest_qty = max(0.0, w_qty - used_qty)
        if rest_val > 0 or rest_qty > 0:
            rows.append({"refYear": year, "partner": "Rest", "value_total": rest_val, "qty_total": rest_qty})

    if rows:
        rest_df = pd.DataFrame(rows)
        total = pd.concat([total, rest_df], ignore_index=True)
    # 重新计算 shares
    for col in ["value_total_all", "qty_total_all", "value_share_all", "qty_share_all"]:
        if col in total.columns:
            total = total.drop(columns=[col])
    totals_year = total.groupby("refYear", as_index=False)[["value_total", "qty_total"]].sum().rename(
        columns={"value_total": "value_total_all", "qty_total": "qty_total_all"}
    )
    total = total.merge(totals_year, on="refYear", how="left", validate="m:1")
    total["value_share_all"] = total["value_total"] / total["value_total_all"]
    total["qty_share_all"] = total["qty_total"] / total["qty_total_all"]
    return total


# ---------------------------------------------------------------------------
# 2) FRED auto sales series (monthly + annual)
# ---------------------------------------------------------------------------

def read_fred_series(path: Path, value_col: str, series_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = to_datetime(df["observation_date"])
    df["value"] = to_numeric(df[value_col])
    df = df.dropna(subset=["date", "value"])
    df["series"] = series_name
    return df[["date", "series", "value"]]


def annualize_fred(df: pd.DataFrame, agg: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["year"] = tmp["date"].dt.year
    if agg == "sum":
        out = tmp.groupby(["series", "year"], as_index=False)["value"].sum()
    else:
        out = tmp.groupby(["series", "year"], as_index=False)["value"].mean()
    out = out.rename(columns={"value": "value_annual"})
    out["aggregation"] = agg
    return out


def build_fred_outputs(base_dir: Path) -> Dict[str, pd.DataFrame]:
    alt = read_fred_series(base_dir / "ALTSALES.csv", "ALTSALES", "ALTSALES")  # Light vehicle sales, SAAR (millions)
    laut = read_fred_series(base_dir / "LAUTONSA.csv", "LAUTONSA", "LAUTONSA")  # Retail auto sales total (thousands)
    faut = read_fred_series(base_dir / "FAUTONSA.csv", "FAUTONSA", "FAUTONSA")  # Retail auto sales foreign (thousands)
    faut_saar = read_fred_series(base_dir / "FAUTOSAAR.csv", "FAUTOSAAR", "FAUTOSAAR")  # Foreign autos, SAAR

    monthly = pd.concat([alt, laut, faut, faut_saar], ignore_index=True)

    annual_parts = [
        annualize_fred(alt, agg="mean"),      # SAAR -> use mean of monthly SAAR as annual level
        annualize_fred(faut_saar, agg="mean"),
        annualize_fred(laut, agg="sum"),      # non-SAAR -> sum monthly levels
        annualize_fred(faut, agg="sum"),
    ]
    annual = pd.concat(annual_parts, ignore_index=True)

    # Import share from retail autos (foreign vs total, non-SAAR).
    share = (
        laut.rename(columns={"value": "laut_value"})
        .merge(faut.rename(columns={"value": "faut_value"}), on="date", how="inner")
    )
    share["import_share"] = share["faut_value"] / share["laut_value"]
    share["year"] = share["date"].dt.year

    share_annual = (
        share.groupby("year", as_index=False)[["import_share"]]
        .mean()
        .rename(columns={"import_share": "import_share_avg"})
    )

    return {
        "monthly": monthly,
        "annual": annual,
        "import_share_monthly": share[["date", "import_share"]],
        "import_share_annual": share_annual,
    }


# ---------------------------------------------------------------------------
# 3) BEA imported auto sales (quarterly)
# ---------------------------------------------------------------------------

def clean_bea_import_sales(path: Path) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    df["date"] = to_datetime(df["observation_date"])
    df["value"] = to_numeric(df.iloc[:, 1])  # second column is the series
    df = df.dropna(subset=["date", "value"])

    annual = df.copy()
    annual["year"] = annual["date"].dt.year
    annual = (
        annual.groupby("year", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "value_mean"})
    )

    return {
        "quarterly": df[["date", "value"]],
        "annual_mean": annual,
    }


# ---------------------------------------------------------------------------
# 4) Employment series (BLS CES3133600101)
# ---------------------------------------------------------------------------

def clean_employment(path: Path) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    df["date"] = to_datetime(df["observation_date"])
    df["employment_thousands"] = to_numeric(df.iloc[:, 1])
    df = df.dropna(subset=["date", "employment_thousands"])

    annual = df.copy()
    annual["year"] = annual["date"].dt.year
    annual = (
        annual.groupby("year", as_index=False)["employment_thousands"]
        .mean()
        .rename(columns={"employment_thousands": "employment_thousands_avg"})
    )

    return {
        "monthly": df[["date", "employment_thousands"]],
        "annual": annual,
    }


# ---------------------------------------------------------------------------
# 5) OICA country production tables
# ---------------------------------------------------------------------------

def clean_oica_file(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=3)
    df = df.rename(columns={df.columns[0]: "country"})

    value_cols: List[tuple[str, int]] = []
    year_pattern = re.compile(r"YTD\s*(\d{4})", flags=re.IGNORECASE)
    for col in df.columns[1:]:
        m = year_pattern.search(str(col))
        if m:
            value_cols.append((col, int(m.group(1))))

    parts: List[pd.DataFrame] = []
    for col, year in value_cols:
        tmp = df[["country", col]].copy()
        tmp["year"] = year
        tmp["production"] = to_numeric(tmp[col])
        parts.append(tmp[["country", "year", "production"]])

    if not parts:
        return pd.DataFrame(columns=["country", "year", "production"])

    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["production"])
    out = out.dropna(subset=["country"])
    out["country"] = out["country"].astype(str).str.strip()

    drop_patterns = [
        r"^UNITS$",
        r"VARIATION",
        r"Double Counts",
        r"WORLD MOTOR",
        r"World Total",
    ]
    drop_re = re.compile("|".join(drop_patterns), flags=re.IGNORECASE)
    out = out[~out["country"].str.contains(drop_re)]
    out = out[out["country"].notna()]
    out = out[~out["country"].str.fullmatch(r"\s*", na=False)]
    out = out[out["country"].str.upper() != "ALL VEHICLES"]

    return out


def build_oica_outputs(base_dir: Path) -> pd.DataFrame:
    files = sorted(base_dir.glob("By-country*20*.xlsx"))
    frames: List[pd.DataFrame] = []
    for f in files:
        frames.append(clean_oica_file(f))
    if not frames:
        return pd.DataFrame(columns=["country", "year", "production"])
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["country", "year"])
    return out


# ---------------------------------------------------------------------------
# 6) BEA total requirements matrix (Table.csv)
# ---------------------------------------------------------------------------

def clean_io_table(path: Path) -> pd.DataFrame:
    # skip first two title rows, keep matrix as-is
    return pd.read_csv(path, skiprows=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = build_config()
    base_dir: Path = cfg["BASE_DIR"]
    out_dir: Path = cfg["OUTPUT_DIR"]

    # 1) Imports 8703/8704
    # 优先使用扩展版本（含韩/欧/中）
    ext_path = base_dir / "us_auto_imports_8703_8704_extended_kor_chn_eu.csv"
    imports_raw_path = ext_path if ext_path.exists() else base_dir / "us_auto_imports_8703_8704.csv"
    world_total_path = base_dir / "us_auto_imports_8703_8704_world_total.csv"
    imports_by_cmd, imports_total = clean_auto_imports(imports_raw_path)
    imports_total = add_rest_from_world_total(imports_total, world_total_path)
    imports_by_cmd.to_csv(out_dir / "auto_imports_partner_cmd_annual.csv", index=False)
    imports_total.to_csv(out_dir / "auto_imports_partner_total.csv", index=False)

    # 2) FRED auto sales
    fred = build_fred_outputs(base_dir)
    fred["monthly"].to_csv(out_dir / "fred_auto_sales_monthly.csv", index=False)
    fred["annual"].to_csv(out_dir / "fred_auto_sales_annual.csv", index=False)
    fred["import_share_monthly"].to_csv(out_dir / "fred_import_share_monthly.csv", index=False)
    fred["import_share_annual"].to_csv(out_dir / "fred_import_share_annual.csv", index=False)

    # 3) BEA imported autos sales (quarterly)
    bea = clean_bea_import_sales(base_dir / "B149RC1Q027SBEA.csv")
    bea["quarterly"].to_csv(out_dir / "auto_import_sales_bea_quarterly.csv", index=False)
    bea["annual_mean"].to_csv(out_dir / "auto_import_sales_bea_annual.csv", index=False)

    # 4) Employment
    emp = clean_employment(base_dir / "CES3133600101.csv")
    emp["monthly"].to_csv(out_dir / "auto_employment_monthly.csv", index=False)
    emp["annual"].to_csv(out_dir / "auto_employment_annual.csv", index=False)

    # 5) OICA production by country/year
    oica = build_oica_outputs(base_dir)
    oica.to_csv(out_dir / "oica_production_country_year.csv", index=False)

    # 6) IO total requirements matrix
    io_df = clean_io_table(base_dir / "Table.csv")
    io_df.to_csv(out_dir / "io_total_requirements.csv", index=False)

    print(f"[Done] Cleaned outputs written to {out_dir}")


if __name__ == "__main__":
    main()
