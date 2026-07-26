"""Render paper-ready closed-loop delay robustness panels from aggregate data."""

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ["OpenVLA", "OpenVLA-OFT", "UniVLA", "CloudEdgeVLA"]
COLORS = {
    "OpenVLA": "#94A3B8",
    "OpenVLA-OFT": "#2A9D8F",
    "UniVLA": "#E76F51",
    "CloudEdgeVLA": "#2563EB",
}
MARKERS = {"OpenVLA": "D", "OpenVLA-OFT": "s", "UniVLA": "^", "CloudEdgeVLA": "o"}
LINESTYLES = {"OpenVLA": "--", "OpenVLA-OFT": "--", "UniVLA": "--", "CloudEdgeVLA": "-"}


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _draw_curve(ax, delays: Sequence[int], success: Dict[str, Sequence[float]]) -> None:
    ax.axvspan(20, max(delays), color="#DBEAFE", alpha=0.55, zorder=0)
    ax.axvline(20, color="#64748B", lw=1.2, linestyle=(0, (4, 3)))
    for method in METHODS:
        ax.plot(
            delays,
            success[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            lw=3.0 if method == "CloudEdgeVLA" else 2.2,
            ms=6.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=method,
        )
    ax.set_title("Closed-loop success under delay", loc="left", fontweight="bold")
    ax.set_xlabel("Delay window $d$")
    ax.set_ylabel("Task success rate (%)")
    ax.set_xticks(delays)
    ax.set_ylim(-2, 103)
    ax.grid(axis="y", color="#CBD5E1", alpha=0.6, lw=0.8)
    ax.legend(loc="center left")
    ax.text(
        0.97,
        0.08,
        "Beyond training window ($d>20$)",
        transform=ax.transAxes,
        ha="right",
        color="#64748B",
        fontsize=9,
    )


def _draw_metric_bars(
    ax,
    metrics: Dict,
    key: str,
    title: str,
    *,
    show_method_labels: bool = True,
    xlabel: str = "Retention score (%)",
) -> None:
    values = np.asarray([metrics[method][key] for method in METHODS])
    positions = np.arange(len(METHODS))
    bars = ax.barh(positions, values, color=[COLORS[method] for method in METHODS], alpha=0.92)
    if show_method_labels:
        ax.set_yticks(positions, METHODS)
    else:
        ax.set_yticks(positions)
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xlim(0, 105)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", color="#CBD5E1", alpha=0.6, lw=0.8)
    for bar, value in zip(bars, values):
        inside = value >= 60
        ax.text(
            value - 2.0 if inside else value + 2.0,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            ha="right" if inside else "left",
            va="center",
            color="white" if inside else "#0F172A",
            fontweight="bold" if inside else "normal",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--prefix", default="fig_closed_loop_delay_robustness")
    args = parser.parse_args()

    data = json.loads(args.data_path.read_text())
    delays = data["delay_window"]
    success = data["success_rate_percent"]
    metrics = data["derived_metrics"]
    _paper_style()

    fig, ax = plt.subplots(figsize=(6.7, 4.8))
    _draw_curve(ax, delays, success)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.15, top=0.91)
    _save(fig, args.output_dir / f"{args.prefix}_curve.png")

    bar_specs = (
        ("auc", "normalized_delay_aurc_percent", "Normalized delay AUC"),
        ("retention_d40", "retention_at_d40_percent", "Success retained at $d=40$"),
    )
    for suffix, key, title in bar_specs:
        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        _draw_metric_bars(ax, metrics, key, title)
        fig.subplots_adjust(left=0.25, right=0.97, bottom=0.16, top=0.90)
        _save(fig, args.output_dir / f"{args.prefix}_{suffix}.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7), sharey=True)
    _draw_metric_bars(
        axes[0],
        metrics,
        "normalized_delay_aurc_percent",
        "Delay AUC",
        xlabel="Score (%)",
    )
    _draw_metric_bars(
        axes[1],
        metrics,
        "retention_at_d40_percent",
        "Retention at $d=40$",
        show_method_labels=False,
        xlabel="Score (%)",
    )
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.20, top=0.88, wspace=0.16)
    _save(fig, args.output_dir / f"{args.prefix}_retention_summary.png")


if __name__ == "__main__":
    main()
