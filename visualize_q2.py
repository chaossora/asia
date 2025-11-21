#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization script for APMCM Problem C - Question 2 Results.
Refactored for comprehensive analysis with Clean Academic Style.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Academic Style Settings
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun', 'STSong', 'SimHei', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['axes.unicode_minus'] = False

sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks", {"xtick.direction": "in", "ytick.direction": "in"})

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output_q2"
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load all result CSVs."""
    data = {}
    files = {
        "base": "base_df.csv",
        "results": "scenario_results.csv",
        "shares": "scenario_shares.csv",
        "sales": "scenario_sales_by_partner.csv",
        "results_sigma": "scenario_results_sigma_grid.csv",
        "shares_sigma": "scenario_shares_sigma_grid.csv",
        "sales_sigma": "scenario_sales_sigma_grid.csv"
    }
    
    for key, fname in files.items():
        path = OUTPUT_DIR / fname
        if path.exists():
            data[key] = pd.read_csv(path)
        else:
            print(f"Warning: {fname} not found.")
            data[key] = None
    return data

# ---------------------------------------------------------------------------
# 1. Base Data Visualization
# ---------------------------------------------------------------------------
def plot_base_data(df):
    """
    base_df.csv: Compare shares, prices, and tariffs.
    """
    if df is None: return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Share
    sns.barplot(data=df, x="partner", y="share0", ax=axes[0], palette="Set2", edgecolor="black", linewidth=0.5)
    axes[0].set_title("Base Market Share")
    axes[0].set_ylabel("Share")
    axes[0].tick_params(axis='x', rotation=30)
    
    # Price
    sns.barplot(data=df, x="partner", y="P_obs0", ax=axes[1], palette="Set2", edgecolor="black", linewidth=0.5)
    axes[1].set_title("Base Average Price ($)")
    axes[1].set_ylabel("Price ($)")
    axes[1].tick_params(axis='x', rotation=30)
    
    # Tariff
    sns.barplot(data=df, x="partner", y="tau0", ax=axes[2], palette="Set2", edgecolor="black", linewidth=0.5)
    axes[2].set_title("Base Tariff Rate")
    axes[2].set_ylabel("Tariff Rate")
    axes[2].tick_params(axis='x', rotation=30)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "base_overview.png")
    plt.close()
    print("Generated base_overview.png")

# ---------------------------------------------------------------------------
# 2. Scenario Results Visualization
# ---------------------------------------------------------------------------
def plot_scenario_results(df):
    """
    scenario_results.csv: Grouped Bar for metrics & Line for P/Q.
    """
    if df is None: return
    
    # Grouped Bar for US Impact
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # US Output Change
    sns.barplot(data=df, x="scenario", y="delta_US_output", ax=axes[0], palette="Blues_d", edgecolor="black", linewidth=0.5)
    axes[0].set_title("Change in US Output ($)")
    axes[0].set_ylabel("Delta Output ($)")
    axes[0].tick_params(axis='x', rotation=20)
    
    # US Employment Change
    sns.barplot(data=df, x="scenario", y="delta_US_employment", ax=axes[1], palette="Greens_d", edgecolor="black", linewidth=0.5)
    axes[1].set_title("Change in US Employment (Jobs)")
    axes[1].set_ylabel("Delta Employment")
    axes[1].tick_params(axis='x', rotation=20)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "scenario_results_us_impact.png")
    plt.close()
    print("Generated scenario_results_us_impact.png")
    
    # Line Chart for P_bar and Q_total (Dual Axis)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Price Index ($)', color=color)
    ax1.plot(df['scenario'], df['P_bar_new'], color=color, marker='o', linewidth=2, label='Price Index')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', rotation=20)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Total Sales (Units)', color=color)
    ax2.plot(df['scenario'], df['Q_total_new'], color=color, marker='s', linewidth=2, linestyle='--', label='Total Sales')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Macro Metrics: Price Index vs Total Sales")
    fig.tight_layout()
    plt.savefig(PLOT_DIR / "scenario_results_macro_line.png")
    plt.close()
    print("Generated scenario_results_macro_line.png")

# ---------------------------------------------------------------------------
# 3. Scenario Shares Visualization
# ---------------------------------------------------------------------------
def plot_scenario_shares(df):
    """
    scenario_shares.csv: 100% Stacked Bar.
    """
    if df is None: return
    
    pivot_df = df.pivot(index="scenario", columns="partner", values="share_new")
    # Sort scenarios if needed
    order = ["baseline", "reciprocal_no_response", "reciprocal_with_response", "reciprocal_strong_absorb"]
    pivot_df = pivot_df.reindex([x for x in order if x in pivot_df.index])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("Set2", n_colors=len(pivot_df.columns))
    
    pivot_df.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="black", width=0.7, linewidth=0.5)
    
    ax.set_title("Market Share Redistribution", pad=20)
    ax.set_ylabel("Share")
    ax.set_xlabel("Scenario")
    plt.xticks(rotation=20, ha='right')
    ax.legend(title="Partner", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "scenario_shares_stacked.png")
    plt.close()
    print("Generated scenario_shares_stacked.png")

# ---------------------------------------------------------------------------
# 4. Scenario Sales Visualization
# ---------------------------------------------------------------------------
def plot_scenario_sales(df):
    """
    scenario_sales_by_partner.csv: Grouped Bar.
    """
    if df is None: return
    
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df, 
        x="scenario", 
        y="Q_units", 
        hue="partner", 
        palette="Set2", 
        edgecolor="black",
        linewidth=0.5
    )
    
    plt.title("Sales Volume by Partner", pad=20)
    plt.ylabel("Sales (Units)")
    plt.xlabel("Scenario")
    plt.xticks(rotation=20, ha='right')
    plt.legend(title="Partner", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "scenario_sales_grouped.png")
    plt.close()
    print("Generated scenario_sales_grouped.png")

def plot_sales_radar(df):
    """
    scenario_sales_by_partner.csv: Radar Chart (% change from baseline).
    """
    if df is None: return

    # 1. Pivot to get partners as columns
    pivot = df.pivot(index="scenario", columns="partner", values="Q_units")
    
    if "baseline" not in pivot.index:
        print("Baseline scenario not found for radar chart.")
        return

    # 2. Calculate % Change
    baseline = pivot.loc["baseline"]
    pct_change = (pivot - baseline) / baseline * 100
    
    # Drop baseline row (all 0s)
    pct_change = pct_change.drop("baseline")
    
    # 3. Prepare Radar Data
    labels = pct_change.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Helper to plot each scenario
    colors = sns.color_palette("deep", n_colors=len(pct_change))
    line_styles = ['-', '--', '-.', ':'] # Different styles for different scenarios
    markers = ['o', 's', '^', 'D'] # Different markers
    
    for i, (scenario, row) in enumerate(pct_change.iterrows()):
        values = row.tolist()
        values += values[:1] # Close the loop
        
        style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]
        
        ax.plot(angles, values, color=colors[i], linewidth=2, linestyle=style, marker=marker, markersize=6, label=scenario)
        ax.fill(angles, values, color=colors[i], alpha=0.05) # Very light fill to avoid clutter
    
    # Fix axis labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.title("Sales Volume % Change vs Baseline", y=1.1, fontsize=16)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "scenario_sales_radar.png")
    plt.close()
    print("Generated scenario_sales_radar.png")

def plot_sigma_results(df):
    """
    scenario_results_sigma_grid.csv: Heatmap & Line.
    """
    if df is None: return
    
    # Heatmap for Q_total_new
    pivot = df.pivot(index="scenario", columns="sigma", values="Q_total_new")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=False, cmap="YlGnBu", cbar_kws={'label': 'Total Sales'})
    plt.title("Sensitivity Heatmap: Total Sales", pad=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "sigma_results_heatmap_sales.png")
    plt.close()
    print("Generated sigma_results_heatmap_sales.png")
    
    # Line Chart for US Employment
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, x="sigma", y="delta_US_employment", hue="scenario", style="scenario",
        markers=True, dashes=False, palette="deep", linewidth=2.5, markersize=8
    )
    plt.title("Sensitivity: US Employment Change vs Sigma")
    plt.ylabel("Delta Employment")
    plt.grid(True, linestyle=":", alpha=0.6)
    sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "sigma_results_line_employment.png")
    plt.close()
    print("Generated sigma_results_line_employment.png")

# ---------------------------------------------------------------------------
# 6. Sigma Facets Visualization
# ---------------------------------------------------------------------------
def plot_sigma_facets(df_shares, df_sales):
    """
    scenario_shares_sigma_grid.csv / scenario_sales_sigma_grid.csv: Facet Grids.
    """
    # Shares Facet
    if df_shares is not None:
        g = sns.relplot(
            data=df_shares,
            x="sigma", y="share_new", hue="scenario", col="partner",
            col_wrap=3, kind="line", marker="o", height=4, aspect=1.2,
            palette="deep", facet_kws={'sharey': False}
        )
        g.set_titles("{col_name}")
        g.set_axis_labels("Sigma", "Market Share")
        sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False, title=None)
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle("Sensitivity: Market Share by Partner", fontsize=16)
        plt.savefig(PLOT_DIR / "sigma_shares_facet.png")
        plt.close()
        print("Generated sigma_shares_facet.png")
        
    # Sales Facet
    if df_sales is not None:
        g = sns.relplot(
            data=df_sales,
            x="sigma", y="Q_units", hue="scenario", col="partner",
            col_wrap=3, kind="line", marker="o", height=4, aspect=1.2,
            palette="deep", facet_kws={'sharey': False}
        )
        g.set_titles("{col_name}")
        g.set_axis_labels("Sigma", "Sales Volume")
        sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False, title=None)
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle("Sensitivity: Sales Volume by Partner", fontsize=16)
        plt.savefig(PLOT_DIR / "sigma_sales_facet.png")
        plt.close()
        print("Generated sigma_sales_facet.png")

def main():
    print("Starting comprehensive visualization...")
    data = load_data()
    
    plot_base_data(data.get("base"))
    plot_scenario_results(data.get("results"))
    plot_scenario_shares(data.get("shares"))
    plot_scenario_sales(data.get("sales"))
    plot_sales_radar(data.get("sales")) # Add radar chart
    plot_sigma_results(data.get("results_sigma"))
    plot_sigma_facets(data.get("shares_sigma"), data.get("sales_sigma"))
    
    print(f"All plots saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()
