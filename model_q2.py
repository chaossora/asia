#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APMCM Problem C - Question 2

完全版可运行代码：读取 washdata 和 newdata 的清洗结果，标定 Armington 模型，
构造情景（基准、互惠关税无应对、互惠关税+日本应对），输出模拟指标。

输入（已清洗好的文件）：
  - washdata/output/tariff_yearly.csv      : HTS8 年度面板（含关税字段）
  - washdata/output/trade_duty_panel.csv   : HS2 关税收入（此处备用，不强依赖）
  - newdata/output/auto_imports_partner_total.csv   : 8703/8704 对加/日/墨 进口额/数量/份额
  - newdata/output/auto_imports_partner_cmd_annual.csv : 分 8703/8704 的细分
  - newdata/output/fred_auto_sales_annual.csv         : ALTSALES、FAUTONSA 等年度汇总
  - newdata/output/fred_import_share_annual.csv       : 乘用车“外国生产”销量占比
  - newdata/output/oica_production_country_year.csv   : 各国整车产量（用作渠道参考）
  - newdata/output/io_total_requirements.csv          : BEA 总需求系数表（简化乘数）

输出：
  - output_q2/base_df.csv
  - output_q2/scenario_results.csv        : 各情景关键指标
  - output_q2/scenario_shares.csv         : 各情景份额
  - output_q2/scenario_sales_by_partner.csv : 各情景销量（辆）

运行方法（项目根目录）：
    python model_q2.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

CONFIG = {
    "base_year": 2024,
    # 加入韩 / 欧 / 中
    "partners": ["Japan", "Mexico", "Canada", "Korea", "EU", "China", "Rest"],
    "sigma": 3.0,            # Armington 替代弹性（单值备选）
    "sigma_grid": [3.0, 4.0, 5.0],  # 敏感性分析区间
    "eta": 1.0,              # 总需求对均价弹性
    "reciprocal_tariff_japan": 0.25,  # 互惠关税示例：日本 25%
    # 日系三大在北美的营业利润率（可按实际披露替换），用于估算关税吸收比例 theta_JP
    # 数值含义：base=冲击前，shock=冲击后；加权由 brand_weights 决定。
    "jp_brand_margins": {
        "Toyota": {"base": 0.028, "shock": 0.006},  # FY2024≈2.8%，FY2025≈0.6%
        "Honda": {"base": 0.05, "shock": 0.03},     # 示例：5%→3%，可替换为最新分部数据
        "Nissan": {"base": 0.01, "shock": -0.01},   # 示例：1%→-1%，反映北美盈利承压
    },
    # 日系品牌在美销量权重（只用于 theta_JP 计算，可据真实份额调整）
    "jp_brand_weights": {
        "Toyota": 0.4,
        "Honda": 0.35,
        "Nissan": 0.25,
    },
}

ROOT = Path(__file__).resolve().parent
WASH_OUT = ROOT / "washdata" / "output"
NEWDATA_OUT = ROOT / "newdata" / "output"
OUTPUT_DIR = ROOT / "output_q2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, **kwargs)


def zero_pad(series: pd.Series, width: int) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(width)


# ---------------------------------------------------------------------------
# 1. 读取数据
# ---------------------------------------------------------------------------

def load_tariff_yearly() -> pd.DataFrame:
    df = safe_read_csv(WASH_OUT / "tariff_yearly.csv")
    df["hts8"] = zero_pad(df["hts8"], 8)
    df["hs2"] = df["hts8"].str[:2]
    df["hs4"] = df["hts8"].str[:4]
    return df


def load_imports_partner() -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_cmd = safe_read_csv(NEWDATA_OUT / "auto_imports_partner_cmd_annual.csv")
    total = safe_read_csv(NEWDATA_OUT / "auto_imports_partner_total.csv")
    return by_cmd, total


def load_fred() -> Tuple[pd.DataFrame, pd.DataFrame]:
    sales_annual = safe_read_csv(NEWDATA_OUT / "fred_auto_sales_annual.csv")
    import_share_annual = safe_read_csv(NEWDATA_OUT / "fred_import_share_annual.csv")
    return sales_annual, import_share_annual


def bea_import_penetration_override() -> pd.DataFrame:
    """
    手工补入 BEA 基于价值的进口渗透率（2019-2023），单位：份额。
    数据源：B149RX/B707RX/AB75RX 组合估算。
    """
    data = [
        {"year": 2019, "import_share_avg": 0.237},
        {"year": 2020, "import_share_avg": 0.208},
        {"year": 2021, "import_share_avg": 0.206},
        {"year": 2022, "import_share_avg": 0.177},
        {"year": 2023, "import_share_avg": 0.203},
    ]
    return pd.DataFrame(data)


def load_oica() -> pd.DataFrame:
    path = NEWDATA_OUT / "oica_production_country_year.csv"
    if not path.exists():
        return pd.DataFrame(columns=["country", "year", "production"])
    df = pd.read_csv(path)
    return df


def load_io_table() -> pd.DataFrame:
    path = NEWDATA_OUT / "io_total_requirements.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_employment_annual() -> pd.DataFrame:
    path = NEWDATA_OUT / "auto_employment_annual.csv"
    if not path.exists():
        return pd.DataFrame(columns=["year", "employment_thousands_avg"])
    return pd.read_csv(path)


def derive_lambda_jp_from_toyota() -> Dict[str, float] | None:
    """
    用 newdata/production_sales_figures_en_202412.xls 的
    “Production by country・region”表估算日系（以丰田近似）在美/墨/日的生产占比。
    使用 2024 Cumulative Total 列的产量。
    """
    xls_path = ROOT / "newdata" / "production_sales_figures_en_202412.xls"
    if not xls_path.exists():
        return None
    try:
        xls = pd.ExcelFile(xls_path)
        sheet = [n for n in xls.sheet_names if "Production by country" in n][0]
        df = pd.read_excel(xls_path, sheet_name=sheet, header=None)
        # 找到“2024 Cumulative Total”列
        target_cols = [c for c in df.columns if str(df.iloc[2, c]).strip() == "2024 Cumulative Total"]
        if not target_cols:
            return None
        col_ct = target_cols[0]
        name_cols = [1, 2]

        def get_prod(keyword: str) -> float | None:
            for nc in name_cols:
                sub = df[df[nc].astype(str).str.contains(keyword, case=False, na=False)]
                if not sub.empty:
                    vals = pd.to_numeric(sub[col_ct], errors="coerce").dropna()
                    if not vals.empty:
                        return float(vals.iloc[0])
            return None

        prod_us = get_prod("U.S.")
        prod_mx = get_prod("Mexico")
        prod_jp = get_prod("Japan")
        if prod_us is None or prod_mx is None or prod_jp is None:
            return None
        total = prod_us + prod_mx + prod_jp
        if total <= 0:
            return None
        return {
            "JP": prod_jp / total,
            "US": prod_us / total,
            "MX": prod_mx / total,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2. 基期关税：简单平均法 + 优惠列优先
# ---------------------------------------------------------------------------

def filter_auto_tariff_rows(tariff_df: pd.DataFrame) -> pd.DataFrame:
    prefixes = ("8703", "8704")
    return tariff_df[tariff_df["hs4"].astype(str).str.startswith(prefixes)].copy()


def mean_if_exists(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else np.nan


def build_tau0_dict(tariff_auto: pd.DataFrame) -> Dict[str, float]:
    """
    尝试优先使用协定税率；否则使用 MFN。
    """
    out: Dict[str, float] = {}
    out["Mexico"] = mean_if_exists(tariff_auto, "usmca_ad_val_rate")
    out["Canada"] = mean_if_exists(tariff_auto, "usmca_ad_val_rate")
    out["Japan"] = mean_if_exists(tariff_auto, "japan_ad_val_rate")
    out["Korea"] = mean_if_exists(tariff_auto, "korea_ad_val_rate")
    out["EU"] = mean_if_exists(tariff_auto, "mfn_ad_val_rate")
    out["China"] = mean_if_exists(tariff_auto, "mfn_ad_val_rate")
    # 回填缺失为 MFN 均值
    mfn_mean = mean_if_exists(tariff_auto, "mfn_ad_val_rate")
    for k in list(out.keys()):
        if np.isnan(out[k]):
            out[k] = mfn_mean
    # Rest 用 MFN
    out.setdefault("Rest", mfn_mean if not np.isnan(mfn_mean) else 0.0)
    return out


def compute_trade_weighted_tariffs(
    tariff_yearly: pd.DataFrame,
    imports_by_cmd: pd.DataFrame,
    year: int,
) -> Dict[str, float]:
    """
    用 HS4=8703/8704 的进口额做权重，对 partner 的关税列做加权平均。
    imports_by_cmd 来自 auto_imports_partner_cmd_annual.csv。
    """
    # 映射 cmd -> hs4
    hs4_map = {
        "passenger_vehicles": "8703",
        "goods_vehicles": "8704",
        "8703": "8703",
        "8704": "8704",
    }
    # 准备进口权重
    imp = imports_by_cmd.copy()
    imp = imp[imp["refYear"] == year]
    imp["hs4"] = imp["cmd"].map(hs4_map)
    imp = imp.dropna(subset=["hs4"])
    imp["value"] = pd.to_numeric(imp["value"], errors="coerce")
    # 只保留有价值的记录
    imp = imp[imp["value"] > 0]
    if imp.empty:
        return {}

    # 取关税表中相应年份 8703/8704
    t = tariff_yearly[tariff_yearly["year"] == year].copy()
    t["hs4"] = t["hts8"].astype(str).str.zfill(8).str[:4]
    t = t[t["hs4"].isin(["8703", "8704"])]
    if t.empty:
        return {}

    def rate_for_partner(df: pd.DataFrame, partner: str) -> pd.Series:
        """
        先取专用列，若列不存在或全为空则回退 MFN；符合“无协定值=用 MFN”而非免税。
        """
        if df.empty:
            return pd.Series(dtype=float)
        if partner in ["Mexico", "Canada"] and "usmca_ad_val_rate" in df.columns:
            s = pd.to_numeric(df["usmca_ad_val_rate"], errors="coerce")
            if s.dropna().empty and "mfn_ad_val_rate" in df.columns:
                s = pd.to_numeric(df["mfn_ad_val_rate"], errors="coerce")
            return s
        if partner == "Japan" and "japan_ad_val_rate" in df.columns:
            s = pd.to_numeric(df["japan_ad_val_rate"], errors="coerce")
            if s.dropna().empty and "mfn_ad_val_rate" in df.columns:
                s = pd.to_numeric(df["mfn_ad_val_rate"], errors="coerce")
            return s
        if partner == "Korea" and "korea_ad_val_rate" in df.columns:
            s = pd.to_numeric(df["korea_ad_val_rate"], errors="coerce")
            if s.dropna().empty and "mfn_ad_val_rate" in df.columns:
                s = pd.to_numeric(df["mfn_ad_val_rate"], errors="coerce")
            return s
        return pd.to_numeric(df["mfn_ad_val_rate"], errors="coerce") if "mfn_ad_val_rate" in df.columns else pd.Series(dtype=float)

    results: Dict[str, float] = {}
    for partner in CONFIG["partners"]:
        # join 关税率和进口额按 hs4
        rates_hs4 = []
        weights_hs4 = []
        for hs4 in ["8703", "8704"]:
            t_h = t[t["hs4"] == hs4]
            rate_col = rate_for_partner(t_h, partner)
            rate_mean = pd.to_numeric(rate_col, errors="coerce").dropna().mean() if not rate_col.empty else np.nan
            if np.isnan(rate_mean):
                continue
            imp_val = imp[(imp["hs4"] == hs4) & (imp["partner"] == partner)]["value"].sum()
            if imp_val > 0 and not np.isnan(rate_mean):
                rates_hs4.append(rate_mean)
                weights_hs4.append(imp_val)
        if weights_hs4:
            results[partner] = float(np.average(rates_hs4, weights=weights_hs4))
    return results


def extract_partner_tariffs_by_year(tariff_yearly: pd.DataFrame, year: int) -> Dict[str, float]:
    df = tariff_yearly[tariff_yearly["year"] == year].copy()
    df = filter_auto_tariff_rows(df)
    if df.empty:
        return {}
    return build_tau0_dict(df)


# ---------------------------------------------------------------------------
# 3. 基期市场：份额、均价、总销量
# ---------------------------------------------------------------------------

def build_market_inputs(
    imports_total: pd.DataFrame,
    sales_annual: pd.DataFrame,
    import_share_annual: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    base_year = CONFIG["base_year"]

    # 3.1 总销量（ALTSALES：百万辆，SAAR 平均值）
    alt_row = sales_annual[(sales_annual["series"] == "ALTSALES") & (sales_annual["year"] == base_year)]
    if alt_row.empty:
        raise ValueError("ALTSALES base year not found")
    total_sales_units = float(alt_row["value_annual"].iloc[0]) * 1_000_000

    # 3.2 零售进口占比（乘用车外国生产占总零售销量）
    imp_share_row = import_share_annual[import_share_annual["year"] == base_year]
    import_penetration = float(imp_share_row["import_share_avg"].iloc[0]) if not imp_share_row.empty else 0.3

    # 3.3 进口市场份额（在进口内部的份额） -> 乘以进口渗透率，得到全市场份额
    imports_2024 = imports_total[imports_total["refYear"] == base_year].copy()
    partner_shares_import = dict(zip(imports_2024["partner"], imports_2024["value_share_all"]))

    market_shares: Dict[str, float] = {}
    for p in CONFIG["partners"]:
        if p in partner_shares_import:
            market_shares[p] = partner_shares_import[p] * import_penetration
        else:
            market_shares[p] = 0.0

    # 进口覆盖的总份额
    used = sum(market_shares.values())
    # Rest = 美国本土 + 未列出的进口
    market_shares["Rest"] = max(0.0, 1.0 - used)

    # 3.4 平均价格 = 进口额/数量（仅对有数量的国家）
    avg_price_by_partner: Dict[str, float] = {}
    for _, row in imports_2024.iterrows():
        partner = row["partner"]
        if row["qty_total"] > 0:
            avg_price_by_partner[partner] = float(row["value_total"] / row["qty_total"])

    return market_shares, avg_price_by_partner, total_sales_units


# ---------------------------------------------------------------------------
# 4. Armington 核心函数（与伪代码一致）
# ---------------------------------------------------------------------------

def build_base_market_data(
    tau0_dict: Dict[str, float],
    market_shares: Dict[str, float],
    avg_price_by_partner: Dict[str, float],
    total_sales_units: float,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for p in CONFIG["partners"]:
        s0 = market_shares.get(p, 0.0)
        Q0 = s0 * total_sales_units
        P_obs0 = avg_price_by_partner.get(p, 30000.0)  # 缺失给默认 3 万
        tau0 = tau0_dict.get(p, 0.0)
        c0 = P_obs0 / (1.0 + tau0) if (1.0 + tau0) != 0 else P_obs0
        rows.append(
            {
                "partner": p,
                "share0": s0,
                "Q0": Q0,
                "P_obs0": P_obs0,
                "tau0": tau0,
                "c0": c0,
            }
        )
    return pd.DataFrame(rows)


def calibrate_armington_parameters(base_df: pd.DataFrame, sigma: float) -> Dict[str, float]:
    A = {}
    for _, row in base_df.iterrows():
        p = row["partner"]
        P0 = row["P_obs0"]
        s0 = row["share0"]
        A[p] = s0 * (P0 ** sigma)
    return A


def compute_base_price_index(base_df: pd.DataFrame) -> float:
    return float((base_df["share0"] * base_df["P_obs0"]).sum())


def effective_tariff_japan(lambda_dict: Dict[str, float], tau_direct: float, tau_mx: float, theta_absorb: float) -> float:
    return lambda_dict["JP"] * (1.0 - theta_absorb) * tau_direct + lambda_dict["MX"] * tau_mx  # lambda_US *0


def build_effective_price_dict(base_df: pd.DataFrame, scenario_params: Dict) -> Dict[str, float]:
    P_new: Dict[str, float] = {}
    base_dict = {row["partner"]: row for _, row in base_df.iterrows()}
    for partner in CONFIG["partners"]:
        if partner not in base_dict:
            continue
        row = base_dict[partner]
        c0 = row["c0"]
        if partner == "Japan":
            lambda_JP = scenario_params["lambda_JP"]
            tau_JP_direct = scenario_params["new_tariffs"]["Japan_direct"]
            tau_MX = scenario_params["new_tariffs"].get("Mexico", row["tau0"])
            theta = scenario_params["theta_JP"]
            tau_eff = effective_tariff_japan(lambda_JP, tau_JP_direct, tau_MX, theta)
            P_new[partner] = c0 * (1.0 + tau_eff)
        else:
            tau1 = scenario_params["new_tariffs"].get(partner, row["tau0"])
            P_new[partner] = c0 * (1.0 + tau1)
    return P_new


def simulate_armington_scenario(
    base_df: pd.DataFrame,
    A_params: Dict[str, float],
    P_new_dict: Dict[str, float],
    Q_total_base: float,
    P_bar0: float,
    sigma: float,
    eta: float,
) -> Dict[str, object]:
    denom = 0.0
    for p, A_r in A_params.items():
        P_r1 = P_new_dict.get(p)
        if P_r1 is None:
            continue
        denom += A_r * (P_r1 ** (-sigma))

    shares_new: Dict[str, float] = {}
    for p, A_r in A_params.items():
        P_r1 = P_new_dict.get(p)
        if P_r1 is None:
            continue
        shares_new[p] = A_r * (P_r1 ** (-sigma)) / denom

    P_bar_new = sum(shares_new[p] * P_new_dict[p] for p in shares_new)
    Q_total_new = Q_total_base * (P_bar_new / P_bar0) ** (-eta)
    Q_by_partner_new = {p: shares_new[p] * Q_total_new for p in shares_new}

    return {
        "shares_new": shares_new,
        "Q_total_new": Q_total_new,
        "Q_by_partner_new": Q_by_partner_new,
        "P_bar_new": P_bar_new,
    }


def decompose_japan_channels(Q_Japan_total: float, lambda_JP: Dict[str, float]) -> Dict[str, float]:
    return {
        "Q_JP_direct": lambda_JP["JP"] * Q_Japan_total,
        "Q_JP_USprod": lambda_JP["US"] * Q_Japan_total,
        "Q_JP_MXprod": lambda_JP["MX"] * Q_Japan_total,
    }


def compute_io_impact(delta_US_auto_output: float, multiplier: float, emp_per_output: float) -> Dict[str, float]:
    total_output_change = delta_US_auto_output * multiplier
    employment_change = total_output_change * emp_per_output
    return {"output_change": total_output_change, "employment_change": employment_change}


def derive_io_params(io_df: pd.DataFrame, emp_df: pd.DataFrame, oica_df: pd.DataFrame) -> Tuple[float, float]:
    """
    从 BEA 总需求系数和就业/产量近似推导乘数与就业系数。
    - multiplier: 取 total requirements 表中汽车列（3361MV）的列和作为总产出乘数。
    - emp_per_output: 用 2024 年美国汽车产量（OICA）* 3 万美元 近似为行业产出，
      再用就业人数 / 产出美元得到就业系数。
    """
    # 默认值
    multiplier = 1.5
    emp_per_output = 5e-6

    # 乘数
    if not io_df.empty and "3361MV" in io_df.columns:
        try:
            col_vals = pd.to_numeric(io_df["3361MV"], errors="coerce").dropna()
            if not col_vals.empty:
                multiplier = float(col_vals.sum())
        except Exception:
            pass

    # 就业系数
    emp_row = emp_df[emp_df["year"] == CONFIG["base_year"]] if not emp_df.empty else pd.DataFrame()
    prod_row = oica_df[(oica_df["country"].str.upper() == "USA") & (oica_df["year"] == CONFIG["base_year"])] if not oica_df.empty else pd.DataFrame()
    if not emp_row.empty and not prod_row.empty:
        jobs = float(emp_row["employment_thousands_avg"].iloc[0]) * 1_000
        prod_units = float(prod_row["production"].iloc[0])
        output_value = prod_units * 30_000.0  # 假定未税均价 3 万美元
        if output_value > 0:
            emp_per_output = jobs / output_value

    return multiplier, emp_per_output


# ---------------------------------------------------------------------------
# 估计 Armington σ（份额-价格回归）与 η（总量-均价回归）
# ---------------------------------------------------------------------------

def estimate_sigma(
    imports_total: pd.DataFrame,
    import_share_annual: pd.DataFrame,
    partners: List[str],
    rest_price: float = 30000.0,
) -> float:
    """
    使用多年份（2019-2024）的份额和均价估计 Armington σ：
      ln(s_r) - ln(s_rest) = α_r - σ ln P_r
    用伙伴内去均值的 OLS，估计斜率 b，σ = -b。
    """
    df = imports_total.copy()
    df["year"] = df["refYear"]
    df["price"] = df["value_total"] / df["qty_total"]
    df = df[["year", "partner", "value_share_all", "price"]]
    imp_share = import_share_annual.rename(columns={"import_share_avg": "import_penetration"})
    # 用最新年份的 import_penetration 回填缺失（近似法）
    if not imp_share.empty:
        latest = imp_share.sort_values("year").iloc[-1]["import_penetration"]
        imp_share["import_penetration"] = imp_share["import_penetration"].fillna(latest)
    else:
        latest = None
    df = df.merge(imp_share[["year", "import_penetration"]], on="year", how="left")
    if latest is not None:
        df["import_penetration"] = df["import_penetration"].fillna(latest)
    df["share_total"] = df["value_share_all"] * df["import_penetration"]

    rows = []
    for y, g in df.groupby("year"):
        # 只保留模型中的伙伴
        g = g[g["partner"].isin(partners)]
        s_sum = g["share_total"].sum()
        rest_share = max(1.0 - s_sum, 1e-6)
        for _, r in g.iterrows():
            if r["share_total"] <= 0 or r["price"] <= 0:
                continue
            rows.append(
                {
                    "year": y,
                    "partner": r["partner"],
                    "share": r["share_total"],
                    "price": r["price"],
                    "rest_share": rest_share,
                }
            )
    if len(rows) < 4:  # 太少数据不估
        return CONFIG["sigma"]

    df2 = pd.DataFrame(rows)
    df2["y"] = np.log(df2["share"]) - np.log(df2["rest_share"])
    df2["x"] = np.log(df2["price"])

    # 去除伙伴固定项：按伙伴去均值
    df2["y_dm"] = df2["y"] - df2.groupby("partner")["y"].transform("mean")
    df2["x_dm"] = df2["x"] - df2.groupby("partner")["x"].transform("mean")

    num = (df2["x_dm"] * df2["y_dm"]).sum()
    den = (df2["x_dm"] ** 2).sum()
    if den <= 0:
        return CONFIG["sigma"]
    b_hat = num / den
    sigma_hat = -b_hat
    if sigma_hat <= 0:
        return CONFIG["sigma"]
    return float(sigma_hat)


def estimate_eta(
    imports_total: pd.DataFrame,
    import_share_annual: pd.DataFrame,
    sales_annual: pd.DataFrame,
    avg_price_by_partner_year: Dict[Tuple[int, str], float],
    partners: List[str],
    rest_price: float = 30000.0,
) -> float:
    """
    使用年度总销量 Q_total 与均价指数 P_bar 加法回归 log(Q)=c-η log(P_bar)，估计 η。
    均价指数按 (伙伴份额*价格 + Rest*rest_price) 计算。
    """
    imp = imports_total.copy()
    imp["year"] = imp["refYear"]
    imp["price"] = imp.apply(lambda r: avg_price_by_partner_year.get((r["refYear"], r["partner"]), np.nan), axis=1)
    imp = imp[["year", "partner", "value_share_all", "price"]]
    imp_share = import_share_annual.rename(columns={"import_share_avg": "import_penetration"})
    if not imp_share.empty:
        latest = imp_share.sort_values("year").iloc[-1]["import_penetration"]
        imp_share["import_penetration"] = imp_share["import_penetration"].fillna(latest)
    else:
        latest = None
    imp = imp.merge(imp_share[["year", "import_penetration"]], on="year", how="left")
    if latest is not None:
        imp["import_penetration"] = imp["import_penetration"].fillna(latest)
    imp["share_total"] = imp["value_share_all"] * imp["import_penetration"]

    # 总销量（ALTSALES）— 使用 mean SAAR * 1e6
    alt = sales_annual[(sales_annual["series"] == "ALTSALES")][["year", "value_annual"]]
    alt["Q_total"] = alt["value_annual"] * 1_000_000

    P_list = []
    for y, g in imp.groupby("year"):
        g = g[g["partner"].isin(partners)]
        s_sum = g["share_total"].sum()
        rest_share = max(1.0 - s_sum, 0.0)
        P_bar = (g["share_total"] * g["price"]).sum() + rest_share * rest_price
        P_list.append({"year": y, "P_bar": P_bar})

    P_df = pd.DataFrame(P_list)
    merged = alt.merge(P_df, on="year", how="inner")
    merged = merged[(merged["Q_total"] > 0) & (merged["P_bar"] > 0)]
    if len(merged) < 3:
        return CONFIG["eta"]
    y = np.log(merged["Q_total"])
    x = np.log(merged["P_bar"])
    x_mean = x.mean()
    y_mean = y.mean()
    b_hat = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
    eta_hat = -b_hat
    if eta_hat <= 0:
        return CONFIG["eta"]
    return float(eta_hat)


# ---------------------------------------------------------------------------
# 5. 情景
# ---------------------------------------------------------------------------

def normalize_lambda(lambda_dict: Dict[str, float]) -> Dict[str, float]:
    s = sum(lambda_dict.values())
    if s <= 0:
        return lambda_dict
    return {k: v / s for k, v in lambda_dict.items()}


def compute_theta_from_brand_margins() -> Tuple[float, float, float]:
    """
    根据 CONFIG 中的日系品牌营业利润率（base/shock）以及权重，计算吸收比例的高/中/低。
    吸收比例 = max(0, min(1, (base - shock)/base))，再按品牌权重加权。
    返回：(low, mid, high) 三档。
    """
    margins = CONFIG.get("jp_brand_margins", {})
    weights = CONFIG.get("jp_brand_weights", {})
    num = 0.0
    den = 0.0
    for brand, w in weights.items():
        m = margins.get(brand, {})
        base = m.get("base", None)
        shock = m.get("shock", None)
        if base is None or shock is None or base <= 0:
            continue
        absorb = (base - shock) / base
        absorb = max(0.0, min(1.0, absorb))
        num += w * absorb
        den += w
    theta_high = num / den if den > 0 else 0.0
    theta_mid = theta_high * 0.5
    theta_low = 0.0
    return theta_low, theta_mid, theta_high


def build_scenarios(
    tau0_dict: Dict[str, float],
    tau_next_dict: Dict[str, float],
    lambda_base: Dict[str, float] | None,
) -> Dict[str, Dict]:
    base_tau = {p: tau0_dict.get(p, 0.0) for p in CONFIG["partners"]}
    next_tau = {p: tau_next_dict.get(p, base_tau.get(p, 0.0)) for p in CONFIG["partners"]}
    japan_recip = next_tau.get("Japan", CONFIG["reciprocal_tariff_japan"])

    lambda_baseline = lambda_base or {"JP": 0.4, "US": 0.4, "MX": 0.2}
    lambda_baseline = normalize_lambda(lambda_baseline)

    # 应对情景：增加 US/MX 比例，减少 JP，简单加 0.1/0.05 后归一
    lambda_response = {
        "JP": max(0.0, lambda_baseline.get("JP", 0.4) - 0.15),
        "US": lambda_baseline.get("US", 0.4) + 0.1,
        "MX": lambda_baseline.get("MX", 0.2) + 0.05,
    }
    lambda_response = normalize_lambda(lambda_response)

    # 基于日系品牌营业利润率压缩估算关税吸收
    theta_absorb_low, theta_absorb_mid, theta_absorb_high = compute_theta_from_brand_margins()

    scen = {
        "baseline": {
            "new_tariffs": {
                "Japan_direct": base_tau["Japan"],
                **base_tau,
            },
            "lambda_JP": lambda_baseline,
            "theta_JP": theta_absorb_low,
        },
        "reciprocal_no_response": {
            "new_tariffs": {
                "Japan_direct": japan_recip,
                "Japan": japan_recip,
                **{k: next_tau.get(k, base_tau.get(k, 0.0)) for k in CONFIG["partners"] if k != "Japan"},
            },
            "lambda_JP": lambda_baseline,
            "theta_JP": theta_absorb_low,
        },
        "reciprocal_with_response": {
            "new_tariffs": {
                "Japan_direct": japan_recip,
                "Japan": japan_recip,
                **{k: next_tau.get(k, base_tau.get(k, 0.0)) for k in CONFIG["partners"] if k != "Japan"},
            },
            "lambda_JP": lambda_response,
            "theta_JP": theta_absorb_mid,
        },
        "reciprocal_strong_absorb": {
            "new_tariffs": {
                "Japan_direct": japan_recip,
                "Japan": japan_recip,
                **{k: next_tau.get(k, base_tau.get(k, 0.0)) for k in CONFIG["partners"] if k != "Japan"},
            },
            "lambda_JP": lambda_response,
            "theta_JP": theta_absorb_high,
        },
    }
    return scen


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    # 读取数据
    tariff_yearly = load_tariff_yearly()
    tariff_auto = filter_auto_tariff_rows(tariff_yearly[tariff_yearly["year"] == CONFIG["base_year"]])
    by_cmd, imports_total = load_imports_partner()
    sales_annual, import_share_annual = load_fred()
    # 用 BEA 2019-2023 渗透率覆盖缺失年份
    bea_imp = bea_import_penetration_override()
    import_share_annual = (
        pd.concat([import_share_annual, bea_imp], ignore_index=True)
        .sort_values("year")
        .drop_duplicates(subset="year", keep="last")
    )
    oica_df = load_oica()
    io_df = load_io_table()
    emp_df = load_employment_annual()
    lambda_jp_data = derive_lambda_jp_from_toyota()

    # 关税字典（贸易加权若可计算则优先）
    tau_weighted = compute_trade_weighted_tariffs(tariff_yearly, by_cmd, CONFIG["base_year"])
    tau0_dict = tau_weighted if tau_weighted else build_tau0_dict(tariff_auto)
    tau_next_weighted = compute_trade_weighted_tariffs(tariff_yearly, by_cmd, CONFIG["base_year"] + 1)
    tau_next_dict = tau_next_weighted if tau_next_weighted else extract_partner_tariffs_by_year(tariff_yearly, CONFIG["base_year"] + 1)

    # 市场输入
    market_shares, avg_price_by_partner, total_sales_units = build_market_inputs(
        imports_total, sales_annual, import_share_annual
    )

    # 基期数据
    base_df = build_base_market_data(
        tau0_dict=tau0_dict,
        market_shares=market_shares,
        avg_price_by_partner=avg_price_by_partner,
        total_sales_units=total_sales_units,
    )
    base_df.to_csv(OUTPUT_DIR / "base_df.csv", index=False)

    # 估计 σ、η（用多年份份额/价格/销量）
    # 构建年度伙伴均价字典
    avg_price_year: Dict[Tuple[int, str], float] = {}
    for _, row in imports_total.iterrows():
        if row.get("qty_total", 0) and row.get("value_total", 0):
            avg_price_year[(int(row["refYear"]), row["partner"])] = float(row["value_total"] / row["qty_total"])
    sigma_hat = estimate_sigma(imports_total, import_share_annual, CONFIG["partners"])
    eta_hat = estimate_eta(imports_total, import_share_annual, sales_annual, avg_price_year, CONFIG["partners"])

    # 更新配置（打印/写入 summary，计算仍使用估计值）
    sigma = sigma_hat
    eta = eta_hat

    # Armington 标定
    A_params = calibrate_armington_parameters(base_df, sigma)
    P_bar0 = compute_base_price_index(base_df)
    Q_total_base = base_df["Q0"].sum()

    # IO 乘数与就业系数
    multiplier, emp_per_output = derive_io_params(io_df, emp_df, oica_df)

    # 情景
    scenarios = build_scenarios(tau0_dict, tau_next_dict, lambda_jp_data)

    results_rows: List[Dict[str, object]] = []
    shares_rows: List[Dict[str, object]] = []
    sales_rows: List[Dict[str, object]] = []

    for name, scen in scenarios.items():
        P_new = build_effective_price_dict(base_df, scen)
        sim = simulate_armington_scenario(
            base_df=base_df,
            A_params=A_params,
            P_new_dict=P_new,
            Q_total_base=Q_total_base,
            P_bar0=P_bar0,
            sigma=sigma,
            eta=eta,
        )

        Q_Japan_total = sim["Q_by_partner_new"].get("Japan", 0.0)
        jp_channels = decompose_japan_channels(Q_Japan_total, scen["lambda_JP"])

        # 简化 IO：进口减少的一半替代为美产，乘数 1.5，就业 5e-6/美元
        Q_Japan_base_units = base_df[base_df["partner"] == "Japan"]["Q0"].iloc[0]
        delta_import_units = Q_Japan_base_units - jp_channels["Q_JP_direct"]
        avg_price_jp = base_df[base_df["partner"] == "Japan"]["P_obs0"].iloc[0]
        delta_import_value = delta_import_units * avg_price_jp
        delta_US_output = delta_import_value * 0.5
        io_impact = compute_io_impact(delta_US_output, multiplier=multiplier, emp_per_output=emp_per_output)

        results_rows.append(
            {
                "scenario": name,
                "P_bar_new": sim["P_bar_new"],
                "Q_total_new": sim["Q_total_new"],
                "Q_total_base": Q_total_base,
                "jp_direct_units": jp_channels["Q_JP_direct"],
                "jp_usprod_units": jp_channels["Q_JP_USprod"],
                "jp_mxprod_units": jp_channels["Q_JP_MXprod"],
                "delta_US_output": io_impact["output_change"],
                "delta_US_employment": io_impact["employment_change"],
            }
        )

        for p, s in sim["shares_new"].items():
            shares_rows.append({"scenario": name, "partner": p, "share_new": s})
        for p, q in sim["Q_by_partner_new"].items():
            sales_rows.append({"scenario": name, "partner": p, "Q_units": q})

    pd.DataFrame(results_rows).to_csv(OUTPUT_DIR / "scenario_results.csv", index=False)
    pd.DataFrame(shares_rows).to_csv(OUTPUT_DIR / "scenario_shares.csv", index=False)
    pd.DataFrame(sales_rows).to_csv(OUTPUT_DIR / "scenario_sales_by_partner.csv", index=False)

    # Sigma grid 敏感性分析（额外输出）
    sigma_grid = CONFIG.get("sigma_grid", [CONFIG["sigma"]])
    if sigma_grid:
        res_g: List[Dict[str, object]] = []
        shares_g: List[Dict[str, object]] = []
        sales_g: List[Dict[str, object]] = []
        for sigma in sigma_grid:
            eta = eta_hat
            A_params = calibrate_armington_parameters(base_df, sigma)
            P_bar0 = compute_base_price_index(base_df)
            Q_total_base = base_df["Q0"].sum()
            for name, scen in scenarios.items():
                P_new = build_effective_price_dict(base_df, scen)
                sim = simulate_armington_scenario(
                    base_df=base_df,
                    A_params=A_params,
                    P_new_dict=P_new,
                    Q_total_base=Q_total_base,
                    P_bar0=P_bar0,
                    sigma=sigma,
                    eta=eta,
                )
                Q_Japan_total = sim["Q_by_partner_new"].get("Japan", 0.0)
                jp_channels = decompose_japan_channels(Q_Japan_total, scen["lambda_JP"])
                Q_Japan_base_units = base_df[base_df["partner"] == "Japan"]["Q0"].iloc[0]
                delta_import_units = Q_Japan_base_units - jp_channels["Q_JP_direct"]
                avg_price_jp = base_df[base_df["partner"] == "Japan"]["P_obs0"].iloc[0]
                delta_import_value = delta_import_units * avg_price_jp
                delta_US_output = delta_import_value * 0.5
                io_impact = compute_io_impact(delta_US_output, multiplier=multiplier, emp_per_output=emp_per_output)
                res_g.append(
                    {
                        "sigma": sigma,
                        "scenario": name,
                        "P_bar_new": sim["P_bar_new"],
                        "Q_total_new": sim["Q_total_new"],
                        "Q_total_base": Q_total_base,
                        "jp_direct_units": jp_channels["Q_JP_direct"],
                        "jp_usprod_units": jp_channels["Q_JP_USprod"],
                        "jp_mxprod_units": jp_channels["Q_JP_MXprod"],
                        "delta_US_output": io_impact["output_change"],
                        "delta_US_employment": io_impact["employment_change"],
                    }
                )
                for p, s in sim["shares_new"].items():
                    shares_g.append({"sigma": sigma, "scenario": name, "partner": p, "share_new": s})
                for p, q in sim["Q_by_partner_new"].items():
                    sales_g.append({"sigma": sigma, "scenario": name, "partner": p, "Q_units": q})
        pd.DataFrame(res_g).to_csv(OUTPUT_DIR / "scenario_results_sigma_grid.csv", index=False)
        pd.DataFrame(shares_g).to_csv(OUTPUT_DIR / "scenario_shares_sigma_grid.csv", index=False)
        pd.DataFrame(sales_g).to_csv(OUTPUT_DIR / "scenario_sales_sigma_grid.csv", index=False)

    summary = {
        "config": CONFIG,
        "tau0_dict": tau0_dict,
        "tau_next_dict": tau_next_dict,
        "market_shares": market_shares,
        "avg_price_by_partner": avg_price_by_partner,
        "io_multiplier": multiplier,
        "io_emp_per_output": emp_per_output,
        "lambda_jp_from_toyota": lambda_jp_data,
        "sigma_estimated": sigma,
        "eta_estimated": eta,
    }
    with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Done] Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
