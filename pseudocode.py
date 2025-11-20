"""
APMCM Problem C - Question 2
基于你项目中真实数据文件的 Python 伪代码框架

需要的库（真实代码时）：
- pandas as pd
- numpy as np

真实文件（已经在你的项目中）：
- DataWeb-Query-Import.xlsx         （General Import Charges）
- DataWeb-Query-Export.xlsx         （FAS Value）
- tariff_database_202405.xlsx       （2024 年关税）
- tariff_database_2025.xlsx         （2025 年关税）
- td-codes.pdf / td-fields.pdf      （字段说明，用来看，不在代码里直接用）
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# 0. 全局配置
# =========================

CONFIG = {
    "base_year": 2024,  # 问题二基准年
    # 汽车相关 HS8 前缀（你可以按题意调整，比如只看 8703 乘用车）
    "auto_hs8_prefixes": ["8703", "8704"],
    # 重点国家 / 地区（可按题意修改）
    "partners": ["Japan", "Mexico", "Canada", "Korea", "EU", "China", "Rest"],
    # Armington 替代弹性（可以做敏感性分析）
    "sigma": 3.0,
    # 美国汽车总需求对平均车价的价格弹性
    "eta": 1.0,
}

# 假设所有数据文件都在项目根目录的 data/ 下，你可以改成 "."
DATA_DIR = Path("data")  # 如果文件在当前目录，改成 Path(".")


TARIFF_FILE_INFO = {
    # year: (文件名, sheet 名)
    2024: ("tariff_database_202405.xlsx", "trade_tariff_database_202405"),
    2025: ("tariff_database_2025.xlsx",  "trade_tariff_database_2025"),
}


# =========================
# 1. DataWeb 进口 / 出口数据读取与整理
# =========================

def load_import_duties(path: Path) -> pd.DataFrame:
    """
    读取 DataWeb-Query-Import.xlsx 中的 General Import Charges sheet，
    返回清洗后的宽表 DataFrame，列：
        ['Data Type', 'HTS Number', 'Description', 'Country', '2020', ... , '2025']
    其中年份列转为 float，缺失填 0。
    """
    raw = pd.read_excel(path, sheet_name="General Import Charges", header=None)

    # 第 2 行是表头
    df = raw.copy()
    df.columns = df.iloc[2]
    df = df.iloc[3:].reset_index(drop=True)

    # 把 2025.0 列名改成字符串 '2025'
    df = df.rename(columns={2025.0: "2025"})

    # 只保留 General Import Charges 这一种 Data Type
    df = df[df["Data Type"] == "General Import Charges"].copy()

    # 统一年份列
    year_cols = [str(y) for y in range(2020, 2026)]
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def import_duties_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    把 General Import Charges 宽表变成长表：
        输入列：['Data Type','HTS Number','Description','Country','2020',...,'2025']
        输出列：['hts2','country','year','duty']
    """
    year_cols = [str(y) for y in range(2020, 2026)]
    long_df = df.melt(
        id_vars=["HTS Number", "Description", "Country"],
        value_vars=year_cols,
        var_name="year",
        value_name="duty"
    )

    long_df = long_df.rename(
        columns={
            "HTS Number": "hts2",
            "Country": "country",
            "Description": "description",
        }
    )

    long_df["year"] = long_df["year"].astype(int)
    long_df["duty"] = pd.to_numeric(long_df["duty"], errors="coerce").fillna(0.0)

    return long_df


def load_export_values(path: Path) -> pd.DataFrame:
    """
    读取 DataWeb-Query-Export.xlsx 中的 FAS Value sheet，
    返回清洗后的宽表 DataFrame：
        ['Data Type', 'HTS Number', 'Description', 'Country', '2020', ... , '2025']
    """
    raw = pd.read_excel(path, sheet_name="FAS Value", header=None)

    df = raw.copy()
    df.columns = df.iloc[2]
    df = df.iloc[3:].reset_index(drop=True)

    df = df.rename(columns={2025.0: "2025"})
    df = df[df["Data Type"] == "FAS Value"].copy()

    year_cols = [str(y) for y in range(2020, 2026)]
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def export_values_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    把 FAS Value 宽表变成长表：
        输出列：['hts2','country','year','export_val']
    """
    year_cols = [str(y) for y in range(2020, 2026)]
    long_df = df.melt(
        id_vars=["HTS Number", "Description", "Country"],
        value_vars=year_cols,
        var_name="year",
        value_name="export_val"
    )

    long_df = long_df.rename(
        columns={
            "HTS Number": "hts2",
            "Country": "country",
            "Description": "description",
        }
    )

    long_df["year"] = long_df["year"].astype(int)
    long_df["export_val"] = pd.to_numeric(long_df["export_val"], errors="coerce").fillna(0.0)

    return long_df


# =========================
# 2. 关税数据库读取与筛选汽车 HS8
# =========================

def load_tariff_schedule(year: int, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    读取指定年份的关税数据库（xlsx 版本），返回完整 DataFrame。

    例如：
        2024 -> tariff_database_202405.xlsx, sheet trade_tariff_database_202405
        2025 -> tariff_database_2025.xlsx,  sheet trade_tariff_database_2025
    """
    fname, sheet_name = TARIFF_FILE_INFO[year]
    path = data_dir / fname
    df = pd.read_excel(path, sheet_name=sheet_name)
    return df


def filter_auto_tariff_rows(tariff_df: pd.DataFrame) -> pd.DataFrame:
    """
    在关税表中筛选汽车相关的 hts8：
    - 以 8703 / 8704 为前缀（乘用车 + 卡车，按需要可扩展到 8708 零部件）
    """
    prefixes = tuple(CONFIG["auto_hs8_prefixes"])
    auto_df = tariff_df[tariff_df["hts8"].astype(str).str.startswith(prefixes)].copy()
    return auto_df


def get_auto_tariff_by_partner(auto_df: pd.DataFrame, partner: str) -> float:
    """
    根据关税数据库，给某个 partner（Japan / Mexico / Canada / Korea / 其他）
    提供 2024 年汽车（HS8 in 87xx）平均从价税率（简单平均）。

    注意：这里只是“法定税率”的简单平均，不是贸易加权平均。
          如果你将来获得按 HS8 + 国家分解的进口额，可以做加权平均。
    """
    # 优先使用双边协定字段；否则退回 MFN
    if partner in ["Mexico", "Canada"]:
        # USMCA
        col = "usmca_ad_val_rate"
        if col in auto_df.columns:
            vals = auto_df[col].dropna()
            if not vals.empty:
                return float(vals.mean())

    if partner == "Japan":
        col = "japan_ad_val_rate"
        if col in auto_df.columns:
            vals = auto_df[col].dropna()
            if not vals.empty:
                return float(vals.mean())

    if partner == "Korea":
        col = "korea_ad_val_rate"
        if col in auto_df.columns:
            vals = auto_df[col].dropna()
            if not vals.empty:
                return float(vals.mean())

    # 默认使用 MFN 从价税率（汽车大多是类型 7：纯从价）
    vals = auto_df["mfn_ad_val_rate"].dropna()
    if vals.empty:
        return 0.0
    return float(vals.mean())


def build_partner_auto_tariff_dict(auto_df: pd.DataFrame) -> dict:
    """
    对 CONFIG['partners'] 中的每个国家，生成一个基准（2024）关税率字典：
        {'Japan': tau_JP_2024, 'Mexico': tau_MX_2024, ...}
    """
    tau0 = {}
    for p in CONFIG["partners"]:
        tau0[p] = get_auto_tariff_by_partner(auto_df, p)
    return tau0


# =========================
# 3. 构造基期美国汽车市场数据（按来源国）
# =========================

def build_base_market_data(
    tau0_dict: dict,
    market_shares: dict,
    avg_price_by_partner: dict,
    total_sales_units: float,
) -> pd.DataFrame:
    """
    使用（外生给定的）市场份额、平均价格和总销量，构造基期汽车市场数据表 base_df，列：
      - partner: 国家 / 地区
      - share0: 基期市场份额（按销量）
      - Q0: 基期销量（辆）
      - P_obs0: 基期含税平均车价
      - tau0: 基期从价税率
      - c0: 基期未含税出厂价，c0 = P_obs0 / (1 + tau0)

    说明：
      - market_shares 可以根据赛题正文给的“美国汽车进口结构”数值手动填；
      - avg_price_by_partner 可以用行业统计或统一假设（例如都 35000 美元）；
      - total_sales_units 可以用题目中给的“美国轻型车销量”。
    """
    rows = []
    for p in CONFIG["partners"]:
        s0 = market_shares.get(p, 0.0)
        Q0 = s0 * total_sales_units

        P_obs0 = avg_price_by_partner.get(p, 30000.0)  # 默认 3 万给个数
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

    base_df = pd.DataFrame(rows)
    return base_df


# =========================
# 4. Armington（logit）模型标定
# =========================

def calibrate_armington_parameters(base_df: pd.DataFrame, sigma: float = None) -> dict:
    """
    基于基期数据标定 Armington 口味参数 A_r：
      s_r0 = A_r * P_r0^{-sigma} / sum_k A_k P_k0^{-sigma}

    若只在一个基期标定，可令 denom = 1，则有：
      A_r = s_r0 * P_r0^{sigma}
    """
    if sigma is None:
        sigma = CONFIG["sigma"]

    A = {}
    for _, row in base_df.iterrows():
        p = row["partner"]
        P0 = row["P_obs0"]
        s0 = row["share0"]
        A[p] = s0 * (P0 ** sigma)

    return A


def compute_base_price_index(base_df: pd.DataFrame) -> float:
    """
    计算基期平均车价指数：
      P_bar0 = sum_r s_r0 * P_r0
    """
    P_bar0 = float((base_df["share0"] * base_df["P_obs0"]).sum())
    return P_bar0


# =========================
# 5. 日本非关税应对：有效关税率 & 新价格
# =========================

def effective_tariff_japan(
    lambda_JP: float,
    lambda_US: float,
    lambda_MX: float,
    tau_JP_direct: float,
    tau_MX: float,
    theta_absorb: float,
) -> float:
    """
    日本品牌在美国市场的平均有效关税率：
      τ_eff^JP = λ_JP * (1 - θ) * τ_JP_direct + λ_MX * τ_MX + λ_US * 0

    其中：
      λ_JP + λ_US + λ_MX = 1
      θ 为日本吸收关税的比例（0~1）
    """
    tau_eff = lambda_JP * (1.0 - theta_absorb) * tau_JP_direct + lambda_MX * tau_MX
    return tau_eff


def build_effective_price_dict(base_df: pd.DataFrame, scenario_params: dict) -> dict:
    """
    在某个情景下，根据：
      - 新关税 new_tariffs
      - 日本渠道结构 λ_JP（JP/US/MX）
      - 日本吸收关税比例 theta_JP
    计算各来源国新的含税价格 P_r1。
    """
    P_new = {}
    base_dict = {row["partner"]: row for _, row in base_df.iterrows()}

    for partner in CONFIG["partners"]:
        if partner not in base_dict:
            continue

        row = base_dict[partner]
        c0 = row["c0"]

        if partner == "Japan":
            lambda_JP = scenario_params["lambda_JP"]["JP"]
            lambda_US = scenario_params["lambda_JP"]["US"]
            lambda_MX = scenario_params["lambda_JP"]["MX"]

            tau_JP_direct = scenario_params["new_tariffs"]["Japan_direct"]
            tau_MX = scenario_params["new_tariffs"].get("Mexico", row["tau0"])
            theta = scenario_params["theta_JP"]

            tau_eff = effective_tariff_japan(
                lambda_JP, lambda_US, lambda_MX, tau_JP_direct, tau_MX, theta
            )
            P_new[partner] = c0 * (1.0 + tau_eff)

        else:
            tau1 = scenario_params["new_tariffs"].get(partner, row["tau0"])
            P_new[partner] = c0 * (1.0 + tau1)

    return P_new


# =========================
# 6. Armington 均衡求解：份额 + 总需求 + 各国销量
# =========================

def simulate_armington_scenario(
    base_df: pd.DataFrame,
    A_params: dict,
    P_new_dict: dict,
    Q_total_base: float,
    P_bar0: float,
    eta: float = None,
) -> dict:
    """
    输入：
      - base_df: 基期数据
      - A_params: Armington A_r
      - P_new_dict: 情景下各 partner 的价格 {partner: P_r1}
      - Q_total_base: 基期美国总销量
      - P_bar0: 基期平均价格
      - eta: 总体需求价格弹性

    输出：
      - shares_new: 新份额 s_r1
      - Q_total_new: 新总销量
      - Q_by_partner_new: 各国销量
      - P_bar_new: 新平均价
    """
    if eta is None:
        eta = CONFIG["eta"]
    sigma = CONFIG["sigma"]

    # 1) Armington 分母
    denom = 0.0
    for p, A_r in A_params.items():
        P_r1 = P_new_dict.get(p, None)
        if P_r1 is None:
            continue
        denom += A_r * (P_r1 ** (-sigma))

    # 2) 新份额
    s1 = {}
    for p, A_r in A_params.items():
        P_r1 = P_new_dict.get(p, None)
        if P_r1 is None:
            continue
        s1[p] = A_r * (P_r1 ** (-sigma)) / denom

    # 3) 新平均价
    P_bar1 = 0.0
    for p, share in s1.items():
        P_bar1 += share * P_new_dict[p]

    # 4) 总需求（价格弹性）
    Q_total1 = Q_total_base * (P_bar1 / P_bar0) ** (-eta)

    # 5) 各国销量
    Q1 = {p: share * Q_total1 for p, share in s1.items()}

    return {
        "shares_new": s1,
        "Q_total_new": Q_total1,
        "Q_by_partner_new": Q1,
        "P_bar_new": P_bar1,
    }


# =========================
# 7. 日本渠道分解（进口 vs 本地生产）
# =========================

def decompose_japan_channels(Q_Japan_total: float, scenario_params: dict) -> dict:
    """
    按 λ_JP 的三种渠道比例，把日本品牌在美销售分解为：
      - Q_JP_direct: 日本本土生产，直接出口美国
      - Q_JP_USprod: 日本品牌在美国本土生产
      - Q_JP_MXprod: 日本品牌在墨西哥生产再出口美国
    """
    lambda_JP = scenario_params["lambda_JP"]
    lam_JP = lambda_JP["JP"]
    lam_US = lambda_JP["US"]
    lam_MX = lambda_JP["MX"]

    Q_JP_direct = lam_JP * Q_Japan_total
    Q_JP_USprod = lam_US * Q_Japan_total
    Q_JP_MXprod = lam_MX * Q_Japan_total

    return {
        "Q_JP_direct": Q_JP_direct,
        "Q_JP_USprod": Q_JP_USprod,
        "Q_JP_MXprod": Q_JP_MXprod,
    }


# =========================
# 8. （可选）投入产出传导框架（简化版）
# =========================

def compute_io_impact(delta_US_auto_output: float, io_params: dict) -> dict:
    """
    简化 IO 传导：
      输入：
        - delta_US_auto_output: 美国汽车本土产出增加量（美元）
        - io_params: {
              "multiplier_auto_output_to_total": 总产出乘数,
              "employment_per_output": 每百万美元产出对应就业数
          }

      返回：
        - output_change: 总产出变化
        - employment_change: 就业变化
    """
    m = io_params.get("multiplier_auto_output_to_total", 1.5)
    e = io_params.get("employment_per_output", 5e-6)  # 例：每 1 美元 5e-6 个就业

    total_output_change = delta_US_auto_output * m
    employment_change = total_output_change * e

    return {
        "output_change": total_output_change,
        "employment_change": employment_change,
    }


# =========================
# 9. 主程序：串起来跑几个情景
# =========================

def main():
    # ---------- 1. 读取 & 清洗 DataWeb 数据 ----------
    imp_path = DATA_DIR / "DataWeb-Query-Import.xlsx"
    exp_path = DATA_DIR / "DataWeb-Query-Export.xlsx"

    import_df_wide = load_import_duties(imp_path)
    import_long = import_duties_to_long(import_df_wide)

    export_df_wide = load_export_values(exp_path)
    export_long = export_values_to_long(export_df_wide)

    # 示例：查看 87 章（车辆）在 2024 年的关税收入（按国家）
    base_year = CONFIG["base_year"]
    duties_87_2024 = import_long[
        (import_long["hts2"] == "87") & (import_long["year"] == base_year)
    ].copy()

    # 这里的 duties_87_2024['duty'] 只是“关税收入”，不是进口额。
    # 由于题目没有给 import value，后面关税率主要用 Tariff DB 的法定税率来提供。

    # ---------- 2. 读取 & 筛选关税表 ----------
    tariff_2024 = load_tariff_schedule(2024, DATA_DIR)
    auto_2024 = filter_auto_tariff_rows(tariff_2024)

    tau0_dict = build_partner_auto_tariff_dict(auto_2024)
    # tau0_dict 里大概会是：
    # {"Japan": 0.025, "Mexico": 0.0, "Canada": 0.0, "Korea": 0.0, "EU": 0.025, ...}

    # ---------- 3. 构造基期美国汽车市场数据 ----------
    # 下面这三个参数需要你根据赛题正文或行业数据手动填：
    # 1) market_shares: 美国新车市场中，各来源国的份额（可以是“日本品牌 + 其他进口” 等）
    market_shares = {
        "Japan": 0.15,    # 示例：日本品牌占 15%
        "Mexico": 0.20,
        "Canada": 0.10,
        "Korea": 0.10,
        "EU": 0.10,
        "China": 0.05,
        "Rest": 0.30,
    }

    # 2) avg_price_by_partner: 各来源国在美国市场上的平均车价（含税），用一个合理的假设
    avg_price_by_partner = {
        "Japan": 35000,
        "Mexico": 32000,
        "Canada": 33000,
        "Korea": 31000,
        "EU": 38000,
        "China": 28000,
        "Rest": 30000,
    }

    # 3) total_sales_units: 美国轻型车总销量（单位：辆）
    total_sales_units = 16_000_000  # 例如 1600 万辆，你可以按题目实际值改

    base_df = build_base_market_data(
        tau0_dict=tau0_dict,
        market_shares=market_shares,
        avg_price_by_partner=avg_price_by_partner,
        total_sales_units=total_sales_units,
    )

    # Armington 标定
    A_params = calibrate_armington_parameters(base_df, sigma=CONFIG["sigma"])
    P_bar0 = compute_base_price_index(base_df)
    Q_total_base = base_df["Q0"].sum()

    # ---------- 4. 定义政策情景 ----------
    # 注意：下面 new_tariffs 中的 0.25、0.10 等具体数值，需要你根据
    #       “互惠关税”机制（10% 基础 + 额外 11-50%）和赛题设定自己算出来。
    scenarios = {
        "baseline": {
            "new_tariffs": {
                # 保持基期（用 tau0）
                "Japan_direct": tau0_dict["Japan"],
                "Japan": tau0_dict["Japan"],
                "Mexico": tau0_dict["Mexico"],
                "Canada": tau0_dict["Canada"],
                "Korea": tau0_dict["Korea"],
                "EU": tau0_dict["EU"],
                "China": tau0_dict["China"],
                "Rest": tau0_dict["Rest"],
            },
            "lambda_JP": {"JP": 0.4, "US": 0.4, "MX": 0.2},
            "theta_JP": 0.0,
        },
        "reciprocal_no_response": {
            "new_tariffs": {
                # 示例：日本汽车对美关税从 2.5% 提到 25%
                "Japan_direct": 0.25,
                "Japan": 0.25,
                # 其他国家暂不变化（可按题目进一步设定）
                "Mexico": tau0_dict["Mexico"],
                "Canada": tau0_dict["Canada"],
                "Korea": tau0_dict["Korea"],
                "EU": tau0_dict["EU"],
                "China": tau0_dict["China"],
                "Rest": tau0_dict["Rest"],
            },
            "lambda_JP": {"JP": 0.4, "US": 0.4, "MX": 0.2},
            "theta_JP": 0.0,  # 日本不吸收关税，全部转嫁给消费者
        },
        "reciprocal_with_response": {
            "new_tariffs": {
                "Japan_direct": 0.25,
                "Japan": 0.25,
                "Mexico": tau0_dict["Mexico"],
                "Canada": tau0_dict["Canada"],
                "Korea": tau0_dict["Korea"],
                "EU": tau0_dict["EU"],
                "China": tau0_dict["China"],
                "Rest": tau0_dict["Rest"],
            },
            # 日本增加在美/墨本地生产
            "lambda_JP": {"JP": 0.2, "US": 0.5, "MX": 0.3},
            # 日本吸收一部分关税
            "theta_JP": 0.5,
        },
    }

    # ---------- 5. 逐情景模拟 ----------
    all_results = {}

    for name, scen in scenarios.items():
        # 5.1 计算新价格（关税 + 非关税应对）
        P_new = build_effective_price_dict(base_df, scen)

        # 5.2 Armington 均衡
        sim = simulate_armington_scenario(
            base_df=base_df,
            A_params=A_params,
            P_new_dict=P_new,
            Q_total_base=Q_total_base,
            P_bar0=P_bar0,
            eta=CONFIG["eta"],
        )

        # 5.3 日本渠道分解
        Q_Japan_total = sim["Q_by_partner_new"].get("Japan", 0.0)
        jp_channels = decompose_japan_channels(Q_Japan_total, scen)

        # 5.4 计算美国汽车本土产出变化（非常简化的近似）
        #   基于假设：进口减少的一部分被美国本土产出替代：
        #   delta_import_value ≈ (baseline 下从日本进口的价值 - 当前情景下从日本进口的价值)
        #   这里我们先用数量近似（真实模型中你可以乘以价格）
        Q_Japan_base = scenarios["baseline"]["lambda_JP"]["JP"] * \
                       base_df[base_df["partner"] == "Japan"]["Q0"].iloc[0]
        # 这里只是示意，严格来说应该用 baseline 模拟结果中的 Q_Japan_total

        Q_Japan_direct_new = jp_channels["Q_JP_direct"]
        delta_import_units = Q_Japan_base - Q_Japan_direct_new

        # 把单位数量转换成金额（乘日本车平均价格）
        avg_price_JP = avg_price_by_partner["Japan"]
        delta_import_value = delta_import_units * avg_price_JP

        # 假设其中 k 比例转换为美国本土产出增加
        k = 0.5
        delta_US_auto_output = delta_import_value * k

        # 5.5 IO 传导（这里用非常简单的乘数结构）
        io_params = {
            "multiplier_auto_output_to_total": 1.5,
            "employment_per_output": 5e-6,
        }
        io_impact = compute_io_impact(delta_US_auto_output, io_params)

        all_results[name] = {
            "armington": sim,
            "japan_channels": jp_channels,
            "io_impact": io_impact,
        }

    return all_results


if __name__ == "__main__":
    results = main()
    # TODO: 你可以把 results 存成 Excel / CSV，或者画图，用于写论文。
