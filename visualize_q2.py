#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization script for APMCM Problem C - Question 2 Results.
Clean Academic Style: Serif fonts, high DPI, no hatching, distinct colors.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Academic Style Settings
# ---------------------------------------------------------------------------
# Attempt to use Times New Roman for English/Numbers and SimSun (Songti) for Chinese
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

# Seaborn style
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
        "shares": "scenario_shares.csv",
        "sales": "scenario_sales_by_partner.csv",
        "results": "scenario_results.csv",
        "sigma_sales": "scenario_sales_sigma_grid.csv"
    }
    
    for key, fname in files.items():
        path = OUTPUT_DIR / fname
        if path.exists():
            data[key] = pd.read_csv(path)
        else:
            print(f"Warning: {fname} not found.")
            data[key] = None
    return data

def plot_market_shares(df):
    """
    1. Market Share Stacked Bar (Clean Academic)
    """
    if df is None: return
    
    pivot_df = df.pivot(index="scenario", columns="partner", values="share_new")
    order = ["baseline", "reciprocal_no_response", "reciprocal_with_response", "reciprocal_strong_absorb"]
    pivot_df = pivot_df.reindex([x for x in order if x in pivot_df.index])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Use a professional palette (e.g., Set2 or deep)
    # Set2 is good for qualitative data and print-friendly
    colors = sns.color_palette("Set2", n_colors=len(pivot_df.columns))
    
    pivot_df.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="black", width=0.7, linewidth=0.5)
            
    ax.set_title("Market Share by Scenario", pad=20)
    ax.set_ylabel("Share")
    ax.set_xlabel("Scenario")
    plt.xticks(rotation=20, ha='right')
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Partner", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "market_shares_clean.png")
    plt.close()
    print("Generated market_shares_clean.png")

def plot_sales_volume(df):
    """
    2. Sales Volume Grouped Bar (Clean Academic)
    """
    if df is None: return
    
    plt.figure(figsize=(12, 7))
    
    # Use hue for partners
    ax = sns.barplot(
        data=df, 
        x="scenario", 
        y="Q_units", 
        hue="partner", 
        palette="Set2", 
        edgecolor="black",
        linewidth=0.5
    )
    
    ax.set_title("Sales Volume by Scenario", pad=20)
    ax.set_ylabel("Sales Volume (Units)")
    ax.set_xlabel("Scenario")
    plt.xticks(rotation=20, ha='right')
    
    ax.legend(title="Partner", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "sales_volume_clean.png")
    plt.close()
    print("Generated sales_volume_clean.png")

def plot_macro_metrics(df):
    """
    3. Macro Metrics (Clean Academic)
    """
    if df is None: return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Price Index
    # Use a single distinct color
    sns.barplot(data=df, x="scenario", y="P_bar_new", ax=axes[0], color="#4c72b0", edgecolor="black", linewidth=0.5)
        
    axes[0].set_title("Price Index ($)", pad=15)
    axes[0].set_ylabel("Price Index")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=20)
    
    # Total Sales
    sns.barplot(data=df, x="scenario", y="Q_total_new", ax=axes[1], color="#dd8452", edgecolor="black", linewidth=0.5)
        
    axes[1].set_title("Total Sales (Units)", pad=15)
    axes[1].set_ylabel("Units")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='x', rotation=20)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "macro_metrics_clean.png")
    plt.close()
    print("Generated macro_metrics_clean.png")

def plot_sensitivity(df):
    """
    4. Sensitivity Analysis (Clean Academic Line Plot)
    """
    if df is None: return
    
    jp_df = df[df["partner"] == "Japan"]
    
    plt.figure(figsize=(10, 7))
    
    # Line plot with markers
    sns.lineplot(
        data=jp_df, 
        x="sigma", 
        y="Q_units", 
        hue="scenario", 
        style="scenario", 
        markers=True, 
        dashes=False,
        palette="deep",
        linewidth=2.5,
        markersize=10
    )
    
    plt.title("Sensitivity: Elasticity ($\sigma$) vs Japan Sales", pad=20)
    plt.ylabel("Japan Sales (Units)")
    plt.xlabel("Armington Elasticity ($\sigma$)")
    
    # Add grid manually if needed, but 'ticks' style removes it. 
    # Academic plots often have no grid or very light grid.
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.legend(title="Scenario", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "sensitivity_clean.png")
    plt.close()
    print("Generated sensitivity_clean.png")

def main():
    print("Starting clean academic visualization...")
    data = load_data()
    
    plot_market_shares(data.get("shares"))
    plot_sales_volume(data.get("sales"))
    plot_macro_metrics(data.get("results"))
    plot_sensitivity(data.get("sigma_sales"))
    
    print(f"All plots saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()
