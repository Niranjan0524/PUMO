"""Generates evaluation figures and plots for the Paratransit Routing module."""
from __future__ import annotations

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

# Ensure directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
results_dir = os.path.join(parent_dir, "results")
figures_dir = os.path.join(results_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)


def plot_service_rate():
    csv_file = os.path.join(results_dir, "benchmark_summary.csv")
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)

    scenarios = ["Scenario_Low", "Scenario_Medium", "Scenario_High"]
    scenario_labels = ["Low Demand (20 req)", "Medium Demand (60 req)", "High Demand (150 req)"]
    algos = df["algorithm"].unique()

    # Short display names
    name_map = {
        "A2_NearestAvailable": "A2: Nearest Avail (Solo)",
        "A3_NearestFeasible": "A3: Nearest Feas (Append)",
        "A4_GreedyInsertion": "A4: Greedy Insertion",
        "A5_RegretInsertion": "A5: Regret Insertion",
        "A6_Pareto_TimeFirst": "A6: Pareto (Time-1st)",
        "A6_Pareto_FareFirst": "A6: Pareto (Fare-1st)",
        "A6_Pareto_Balanced": "A6: Pareto (Balanced)",
    }

    colors = ["#7f7f7f", "#aec7e8", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scenarios))
    width = 0.11

    for i, algo in enumerate(algos):
        sub_df = df[df["algorithm"] == algo]
        rates = [sub_df[sub_df["scenario"] == s]["service_rate_pct"].values[0] for s in scenarios]
        offset = (i - len(algos) / 2 + 0.5) * width
        rects = ax.bar(x + offset, rates, width, label=name_map.get(algo, algo), color=colors[i % len(colors)], edgecolor="black", alpha=0.88)

    ax.set_ylabel("Service Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Paratransit Service Rate Across Demand Levels", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "service_rate_by_demand.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_waiting_and_journey_time():
    csv_file = os.path.join(results_dir, "benchmark_summary.csv")
    df = pd.read_csv(csv_file)
    med_df = df[df["scenario"] == "Scenario_Medium"].copy()

    name_map = {
        "A2_NearestAvailable": "A2: NAV\n(Solo)",
        "A3_NearestFeasible": "A3: NFV\n(Append)",
        "A4_GreedyInsertion": "A4: Greedy\nInsertion",
        "A5_RegretInsertion": "A5: Regret\nInsertion",
        "A6_Pareto_TimeFirst": "A6: Pareto\n(Time)",
        "A6_Pareto_FareFirst": "A6: Pareto\n(Fare)",
        "A6_Pareto_Balanced": "A6: Pareto\n(Balanced)",
    }
    med_df["display_name"] = med_df["algorithm"].map(name_map)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Average wait time
    bars1 = ax1.bar(
        med_df["display_name"],
        med_df["avg_wait_time_min"],
        color="#3498db",
        edgecolor="black",
        alpha=0.85
    )
    ax1.set_ylabel("Average Wait Time (minutes)", fontsize=11, fontweight="bold")
    ax1.set_title("Passenger Waiting Time (Medium Demand)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"{yval:.1f}m", ha="center", va="bottom", fontsize=9)

    # Average total journey time
    bars2 = ax2.bar(
        med_df["display_name"],
        med_df["avg_journey_time_min"],
        color="#e67e22",
        edgecolor="black",
        alpha=0.85
    )
    ax2.set_ylabel("Average Journey Time (minutes)", fontsize=11, fontweight="bold")
    ax2.set_title("Total Passenger Journey Time (Medium Demand)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{yval:.1f}m", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "waiting_and_journey_time.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_optimality_gap():
    gap_file = os.path.join(results_dir, "exact_optimality_gap.csv")
    if not os.path.exists(gap_file):
        print(f"File not found: {gap_file}")
        return

    df = pd.read_csv(gap_file)
    # Filter to non-exact algorithms
    heuristics = df[df["algorithm"] != "A7_ExactSolver (Global Optimum)"].copy()

    name_map = {
        "A2_NearestAvailable": "A2: NAV\n(Solo)",
        "A3_NearestFeasible": "A3: NFV\n(Append)",
        "A4_GreedyInsertion": "A4: Greedy\nInsertion",
        "A5_RegretInsertion": "A5: Regret\nInsertion",
        "A6_Pareto_TimeFirst": "A6: Pareto\n(Time)",
        "A6_Pareto_FareFirst": "A6: Pareto\n(Fare)",
        "A6_Pareto_Balanced": "A6: Pareto\n(Balanced)",
    }
    heuristics["display_name"] = heuristics["algorithm"].map(name_map)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        heuristics["display_name"],
        heuristics["optimality_gap_pct"],
        color=["#c0392b", "#d35400", "#27ae60", "#2ecc71", "#8e44ad", "#9b59b6", "#16a085"],
        edgecolor="black",
        alpha=0.88,
    )
    ax.set_ylabel("Optimality Gap vs Exact A7 (%)", fontsize=11, fontweight="bold")
    ax.set_title("Algorithm Optimality Gap Relative to Exact Global Optimum (A7)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "optimality_gap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Generating benchmark figures...")
    plot_service_rate()
    plot_waiting_and_journey_time()
    plot_optimality_gap()
    print("All figures successfully created.")


if __name__ == "__main__":
    main()
