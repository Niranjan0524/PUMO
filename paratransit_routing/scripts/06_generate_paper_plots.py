"""Generates visualizations grounded in Papers 1, 2, and 3."""
from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
results_dir = os.path.join(parent_dir, "results")
figures_dir = os.path.join(results_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pumo_paratransit.passenger_preference import (
    PassengerUtility,
    PassengerSegment,
    crowding_discomfort,
)
from pumo_paratransit.gtfs_model import create_bengaluru_shared_auto_network


def plot_wtp_by_segment():
    """Willingness-to-Pay (INR/min) across passenger segments (Paper 2 — Das et al. 2023)."""
    utility = PassengerUtility()
    segments = [
        PassengerSegment.REGULAR_COMMUTER,
        PassengerSegment.OCCASIONAL_USER,
        PassengerSegment.TIME_SENSITIVE,
        PassengerSegment.SAFETY_CONSCIOUS,
    ]
    seg_labels = ["Regular Commuter", "Occasional User", "Time-Sensitive", "Safety-Conscious"]

    wtp_tt = []
    wtp_wt = []
    wtp_at = []

    for seg in segments:
        summ = utility.wtp_summary(seg)
        wtp_tt.append(summ["wtp_in_vehicle_time_per_min_inr"])
        wtp_wt.append(summ["wtp_waiting_time_per_min_inr"])
        wtp_at.append(summ["wtp_access_time_per_min_inr"])

    x = np.arange(len(segments))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = ax.bar(x - width, wtp_tt, width, label="In-Vehicle Time (VoTT)", color="#3498db", edgecolor="black")
    r2 = ax.bar(x, wtp_wt, width, label="Waiting Time (VoWT)", color="#e74c3c", edgecolor="black")
    r3 = ax.bar(x + width, wtp_at, width, label="Access Time (VoAT)", color="#2ecc71", edgecolor="black")

    ax.set_ylabel("Willingness-to-Pay (₹ / minute)", fontsize=12, fontweight="bold")
    ax.set_title("Willingness-to-Pay Across Commuter Segments\n(Grounded in Das, Bhaduri & Velaga 2023 Mixed Logit Model)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(seg_labels, fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate heights
    for rects in [r1, r2, r3]:
        for bar in rects:
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, y + 0.05, f"₹{y:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "wtp_by_segment.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_crowding_discomfort_curve():
    """Crowding Discomfort vs. Occupancy (Paper 3 — Das & Mandal 2021)."""
    occupancies = np.linspace(0.0, 1.0, 200)
    discomforts = [crowding_discomfort(occ, threshold=0.75) for occ in occupancies]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(occupancies * 100, discomforts, color="#d35400", linewidth=3, label="Discomfort Level $D(occ)$")
    ax.axvline(75, color="#7f8c8d", linestyle="--", label="Comfort Threshold (75% Occupancy)")
    ax.fill_between(occupancies * 100, discomforts, where=(occupancies >= 0.75), color="#e67e22", alpha=0.25, label="High Discomfort Zone")

    ax.set_xlabel("Vehicle Occupancy Ratio (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Crowding Discomfort", fontsize=11, fontweight="bold")
    ax.set_title("Crowding Discomfort Penalty Model\n(Grounded in Das & Mandal 2021 Passenger Expectations)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "crowding_discomfort_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_generalised_cost_comparison():
    """Comparison of Generalised Cost across algorithms (Paper 2 & 3 utility)."""
    csv_path = os.path.join(results_dir, "paper_preference_summary.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        df["strategy"],
        df["avg_generalised_cost_inr"],
        color=["#7f8c8d", "#95a5a6", "#34495e", "#2980b9", "#e74c3c", "#27ae60", "#8e44ad"],
        edgecolor="black",
        alpha=0.88,
    )

    ax.set_ylabel("Average Generalised Cost (₹)", fontsize=11, fontweight="bold")
    ax.set_title("Passenger Generalised Cost Comparison Across Dispatch Strategies\n(Fare + VoTT×Time + VoWT×Wait + VoAT×Access + Crowding Penalty)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=25, ha="right", fontsize=9)

    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, y + 3.0, f"₹{y:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "generalised_cost_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_shared_auto_corridor_network():
    """Digital Matatu-style shared-auto corridor map for Bengaluru (Paper 1)."""
    net = create_bengaluru_shared_auto_network()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Node positions (lon, lat)
    coords = {sid: (s.longitude, s.latitude) for sid, s in net.stops.items()}

    # Plot routes
    route_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    for i, (rid, route) in enumerate(net.routes.items()):
        stop_ids = route.ordered_stop_ids()
        rx = [coords[s][0] for s in stop_ids if s in coords]
        ry = [coords[s][1] for s in stop_ids if s in coords]
        ax.plot(rx, ry, marker="o", linewidth=2.8, markersize=7, color=route_colors[i % len(route_colors)], label=f"{route.route_id}: {route.route_name}", alpha=0.85)

    # Annotate stops
    for sid, stop in net.stops.items():
        x, y = coords[sid]
        marker_style = "s" if stop.zone_type == "terminus" else "^"
        color = "darkred" if stop.zone_type == "terminus" else "navy"
        ax.scatter(x, y, s=80, marker=marker_style, color=color, zorder=5)
        ax.annotate(
            stop.name.replace(" Bus Stand", "").replace(" Phase 1", ""),
            (x, y),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        )

    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.set_title("Bengaluru Shared-Auto Corridors & Stops\n(Digital Matatu GTFS-Paratransit Methodology — Williams et al. 2015)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "shared_auto_corridor_map.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Generating figures grounded in Papers 1, 2, and 3...")
    plot_wtp_by_segment()
    plot_crowding_discomfort_curve()
    plot_generalised_cost_comparison()
    plot_shared_auto_corridor_network()
    print("All paper figures successfully generated.")


if __name__ == "__main__":
    main()
