"""Plot closed-loop delay robustness from the four LIBERO suites.

The aggregate success rates in this file are the plotting source of truth.  The
paper visualizes the macro-average across all four suites, while the JSON
sidecar also exports each suite so that the manuscript table can be checked
against the same data.
"""

import argparse
import json
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# Keep Matplotlib's writable cache inside the repository.  Some container images
# mount ~/.config with a host UID that differs from the runtime user.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MPL_CONFIG_DIR = _PROJECT_ROOT / ".cache" / "matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DELAYS = np.array([0, 5, 10, 15, 20, 25, 30, 40], dtype=float)
METHOD_ORDER = ("OpenVLA", "OpenVLA-OFT", "UniVLA", "VLASH", "CloudEdgeVLA")
BASELINE_METHODS = METHOD_ORDER[:-1]
PLOT_SUITE = "libero_mean"
CLOUDEDGE_RANDOM_SEEDS = (7, 8, 9)

SUCCESS_RATES_BY_SUITE = {
    "libero_spatial": {
        "OpenVLA": [84.6, 35.2, 6.8, 1.0, 0.2, 0.0, 0.0, 0.0],
        "OpenVLA-OFT": [98.4, 67.0, 10.6, 4.8, 0.0, 0.0, 0.0, 0.0],
        "UniVLA": [96.0, 73.6, 27.4, 4.4, 0.4, 0.0, 0.0, 0.0],
        "VLASH": [97.3, 92.2, 60.0, 22.8, 5.0, 0.4, 0.0, 0.0],
        "CloudEdgeVLA": [97.9, 94.8, 93.6, 92.3, 89.6, 89.3, 88.9, 76.4],
    },
    "libero_object": {
        "OpenVLA": [71.2, 35.8, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        "OpenVLA-OFT": [98.6, 83.2, 3.6, 0.2, 0.0, 0.0, 0.0, 0.0],
        "UniVLA": [96.6, 77.4, 34.8, 7.8, 2.6, 0.2, 0.0, 0.0],
        "VLASH": [99.6, 94.6, 62.8, 29.2, 8.2, 5.4, 3.0, 0.4],
        "CloudEdgeVLA": [98.40, 95.07, 94.40, 94.00, 92.07, 89.73, 88.13, 77.80],
    },
    "libero_goal": {
        "OpenVLA": [77.0, 36.2, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        "OpenVLA-OFT": [97.2, 76.2, 26.2, 15.2, 4.0, 2.4, 0.0, 0.0],
        "UniVLA": [94.6, 87.8, 48.2, 31.4, 21.8, 16.0, 11.8, 3.0],
        "VLASH": [96.7, 92.6, 56.8, 39.2, 28.2, 20.0, 13.6, 6.4],
        "CloudEdgeVLA": [96.53, 93.60, 92.87, 93.27, 90.87, 89.47, 87.53, 78.00],
    },
    "libero_long": {
        "OpenVLA": [56.2, 20.8, 5.2, 0.4, 0.0, 0.0, 0.0, 0.0],
        "OpenVLA-OFT": [93.4, 72.6, 8.4, 2.2, 1.4, 0.0, 0.0, 0.0],
        "UniVLA": [93.2, 70.0, 35.0, 8.0, 2.2, 0.6, 0.0, 0.5],
        "VLASH": [93.5, 85.8, 45.6, 17.2, 9.2, 3.0, 0.2, 0.0],
        "CloudEdgeVLA": [91.73, 87.30, 83.20, 80.13, 78.07, 76.60, 76.27, 63.80],
    },
}

# Standard deviations across three CloudEdgeVLA random seeds.  The provided
# statistics are suite-level marginals; without cross-suite covariances they
# cannot be combined into a statistically valid macro-average error band.
CLOUDEDGE_STD_BY_SUITE = {
    "libero_spatial": [0.23, 0.53, 0.53, 0.46, 1.20, 1.01, 0.81, 0.92],
    "libero_object": [0.40, 0.31, 0.53, 0.40, 0.42, 0.83, 0.64, 0.92],
    "libero_goal": [0.31, 0.72, 0.83, 0.64, 0.81, 0.76, 0.70, 1.20],
    "libero_long": [0.42, 0.42, 0.53, 0.50, 0.42, 0.92, 1.01, 1.51],
}

# Tol-inspired categorical palette.  Markers and line styles duplicate the
# categorical encoding so that the plot remains legible in grayscale.
COLORS = {
    "OpenVLA": "#666666",
    "OpenVLA-OFT": "#33BBEE",
    "UniVLA": "#EE7733",
    "VLASH": "#009988",
    "CloudEdgeVLA": "#0077BB",
}
MARKERS = {
    "OpenVLA": "D",
    "OpenVLA-OFT": "s",
    "UniVLA": "^",
    "VLASH": "P",
    "CloudEdgeVLA": "o",
}
LINESTYLES = {
    "OpenVLA": (0, (1, 2)),
    "OpenVLA-OFT": (0, (4, 2)),
    "UniVLA": (0, (6, 2, 1, 2)),
    "VLASH": (0, (2, 1)),
    "CloudEdgeVLA": "-",
}


def format_one_decimal(value: float) -> str:
    """Format reported values with conventional round-half-up behavior."""
    decimal_value = Decimal(f"{float(value):.10f}")
    return format(decimal_value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP), "f")


def _as_arrays(suite: str) -> dict[str, np.ndarray]:
    if suite == "libero_mean":
        return {
            method: np.mean(
                [SUCCESS_RATES_BY_SUITE[name][method] for name in SUCCESS_RATES_BY_SUITE],
                axis=0,
            )
            for method in METHOD_ORDER
        }
    return {
        method: np.asarray(values, dtype=float)
        for method, values in SUCCESS_RATES_BY_SUITE[suite].items()
    }


def validate_data() -> None:
    """Fail early if a suite is missing a method or a delay value."""
    for suite, methods in SUCCESS_RATES_BY_SUITE.items():
        if tuple(methods) != METHOD_ORDER:
            raise ValueError(f"{suite}: methods must follow METHOD_ORDER")
        for method, values in methods.items():
            array = np.asarray(values, dtype=float)
            if array.shape != DELAYS.shape:
                raise ValueError(
                    f"{suite}/{method}: expected {len(DELAYS)} values, got {len(array)}"
                )
            if not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 100)):
                raise ValueError(f"{suite}/{method}: success rates must be finite in [0, 100]")
            if array[0] <= 0:
                raise ValueError(f"{suite}/{method}: d=0 must be positive for retention")

    if tuple(CLOUDEDGE_STD_BY_SUITE) != tuple(SUCCESS_RATES_BY_SUITE):
        raise ValueError("CloudEdgeVLA standard deviations must cover every suite")
    for suite, values in CLOUDEDGE_STD_BY_SUITE.items():
        array = np.asarray(values, dtype=float)
        if array.shape != DELAYS.shape:
            raise ValueError(
                f"{suite}/CloudEdgeVLA std: expected {len(DELAYS)} values, "
                f"got {len(array)}"
            )
        if not np.all(np.isfinite(array)) or np.any(array < 0):
            raise ValueError(
                f"{suite}/CloudEdgeVLA std: values must be finite and nonnegative"
            )


def normalized_delay_auc(values: np.ndarray) -> float:
    """Trapezoidal area under the retention curve, normalized to [0, 100]."""
    retention = values / values[0]
    return float(np.trapz(retention, DELAYS) / (DELAYS[-1] - DELAYS[0]) * 100.0)


def retention_at_d40(values: np.ndarray) -> float:
    """Percentage of synchronous success retained at d=40."""
    return float(values[-1] / values[0] * 100.0)


def _set_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _draw_curve(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    annotate: bool = True,
    ylabel: str = "Task success rate (%)",
) -> None:
    ax.axvspan(20, 40, color="#E5E7EB", alpha=0.65, zorder=0)
    ax.axvline(20, color="#6B7280", lw=0.8, ls=(0, (3, 2)), zorder=1)

    best_baseline = np.maximum.reduce([data[method] for method in BASELINE_METHODS])
    ours = data["CloudEdgeVLA"]
    ax.fill_between(
        DELAYS,
        best_baseline,
        ours,
        where=ours >= best_baseline,
        interpolate=True,
        color=COLORS["CloudEdgeVLA"],
        alpha=0.10,
        zorder=1,
    )

    for method in METHOD_ORDER:
        is_ours = method == "CloudEdgeVLA"
        ax.plot(
            DELAYS,
            data[method],
            label=method,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=4.8 if is_ours else 3.8,
            linewidth=2.2 if is_ours else 1.25,
            linestyle=LINESTYLES[method],
            markeredgecolor="white" if is_ours else COLORS[method],
            markeredgewidth=0.6,
            zorder=4 if is_ours else 3,
        )

    if annotate:
        best_at_d40 = max(data[method][-1] for method in BASELINE_METHODS)
        gap = ours[-1] - best_at_d40
        ax.annotate(
            f"{format_one_decimal(ours[-1])}%",
            xy=(40, ours[-1]),
            xytext=(36.0, ours[-1] + 8.0),
            color=COLORS["CloudEdgeVLA"],
            fontweight="bold",
            fontsize=7.5,
            arrowprops={"arrowstyle": "-", "color": COLORS["CloudEdgeVLA"], "lw": 0.8},
        )
        ax.text(
            29.0,
            54.0,
            f"+{format_one_decimal(gap)} pp at $d_{{\\max}}=40$\nvs. best baseline",
            ha="center",
            va="center",
            color="#0C4A6E",
            fontsize=7.2,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#7DD3FC",
                "alpha": 0.92,
            },
            zorder=5,
        )
        ax.text(
            30.0,
            18.0,
            "Beyond training window",
            ha="center",
            va="bottom",
            color="#6B7280",
            fontsize=6.8,
        )

    ax.set_xlabel("Delay window $d_{\\max}$ (steps)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1, 42)
    ax.set_ylim(0, 103)
    ax.set_xticks(DELAYS)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#D1D5DB", alpha=0.75, lw=0.6)
    ax.set_axisbelow(True)


def _summary_values(data: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray, np.ndarray]:
    display_order = ["CloudEdgeVLA", "VLASH", "UniVLA", "OpenVLA-OFT", "OpenVLA"]
    auc = np.array([normalized_delay_auc(data[method]) for method in display_order])
    d40 = np.array([retention_at_d40(data[method]) for method in display_order])
    return display_order, auc, d40


def _draw_bar_summary(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    values: np.ndarray,
    display_order: list[str],
    panel_label: str | None,
) -> None:
    y = np.arange(len(display_order))
    bars = ax.barh(
        y,
        values,
        color=[COLORS[method] for method in display_order],
        height=0.62,
        alpha=0.95,
    )
    ax.set_yticks(y, labels=display_order)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    if panel_label:
        ax.set_title(panel_label, loc="left", fontweight="bold", pad=3)
    ax.grid(axis="x", color="#D1D5DB", alpha=0.65, lw=0.6)
    ax.set_axisbelow(True)
    for bar, value, method in zip(bars, values, display_order):
        inside = value >= 72
        ax.text(
            value - 2.2 if inside else value + 1.5,
            bar.get_y() + bar.get_height() / 2,
            format_one_decimal(value),
            ha="right" if inside else "left",
            va="center",
            color="white" if inside else "#111827",
            fontweight="bold" if method == "CloudEdgeVLA" else "normal",
            fontsize=7.2,
        )


def plot_curve(output_path: Path, suite: str = PLOT_SUITE) -> None:
    data = _as_arrays(suite)
    fig, ax = plt.subplots(figsize=(3.35, 2.85), facecolor="white")
    ylabel = (
        "Macro-average success rate (%)"
        if suite == "libero_mean"
        else "Task success rate (%)"
    )
    _draw_curve(ax, data, ylabel=ylabel)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        handlelength=2.3,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.29)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def plot_summary(output_path: Path, suite: str = PLOT_SUITE) -> None:
    data = _as_arrays(suite)
    display_order, auc, d40 = _summary_values(data)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 1.5), facecolor="white")
    _draw_bar_summary(axes[0], data, auc, display_order, "(a)")
    _draw_bar_summary(axes[1], data, d40, display_order, "(b)")
    fig.supxlabel("Retention score (%)", y=0.025, fontsize=8.5)
    fig.subplots_adjust(left=0.135, right=0.99, top=0.84, bottom=0.31, wspace=0.47)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white", bbox_inches=None)
    plt.close(fig)


def plot_single_summary(
    output_path: Path, metric: str, suite: str = PLOT_SUITE
) -> None:
    data = _as_arrays(suite)
    display_order, auc, d40 = _summary_values(data)
    values = auc if metric == "auc" else d40
    fig, ax = plt.subplots(figsize=(3.35, 2.2), facecolor="white")
    _draw_bar_summary(ax, data, values, display_order, None)
    ax.set_xlabel("Retention score (%)")
    fig.subplots_adjust(left=0.31, right=0.98, top=0.91, bottom=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def plot_composite(output_path: Path, suite: str = PLOT_SUITE) -> None:
    data = _as_arrays(suite)
    display_order, auc, d40 = _summary_values(data)
    fig = plt.figure(figsize=(6.9, 3.35), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.55, 1.0), wspace=0.52, hspace=0.52)
    ax_curve = fig.add_subplot(grid[:, 0])
    ax_auc = fig.add_subplot(grid[0, 1])
    ax_d40 = fig.add_subplot(grid[1, 1])
    ylabel = (
        "Macro-average success rate (%)"
        if suite == "libero_mean"
        else "Task success rate (%)"
    )
    _draw_curve(ax_curve, data, ylabel=ylabel)
    _draw_bar_summary(ax_auc, data, auc, display_order, "(b)")
    _draw_bar_summary(ax_d40, data, d40, display_order, "(c)")
    ax_curve.set_title("(a)", loc="left", fontweight="bold")
    ax_d40.set_xlabel("Retention score (%)")
    ax_curve.legend(loc="lower left", frameon=False, ncol=2, columnspacing=0.8)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def build_export(plotted_suite: str) -> dict:
    all_metrics = {}
    all_rates = {}
    for suite in SUCCESS_RATES_BY_SUITE:
        arrays = _as_arrays(suite)
        all_rates[suite] = {method: values.tolist() for method, values in arrays.items()}
        all_metrics[suite] = {
            method: {
                "normalized_delay_auc_percent": normalized_delay_auc(values),
                "retention_at_d40_percent": retention_at_d40(values),
                "success_at_d40_percent": float(values[-1]),
                "drop_from_d0_to_d40_percentage_points": float(values[0] - values[-1]),
            }
            for method, values in arrays.items()
        }

    macro_arrays = _as_arrays("libero_mean")
    all_rates["libero_mean"] = {
        method: values.tolist() for method, values in macro_arrays.items()
    }
    all_metrics["libero_mean"] = {
        method: {
            "normalized_delay_auc_percent": normalized_delay_auc(values),
            "retention_at_d40_percent": retention_at_d40(values),
            "success_at_d40_percent": float(values[-1]),
            "drop_from_d0_to_d40_percentage_points": float(values[0] - values[-1]),
        }
        for method, values in macro_arrays.items()
    }

    return {
        "plotted_benchmark": plotted_suite,
        "delay_window": DELAYS.astype(int).tolist(),
        # Backward-compatible view of the suite shown in the figure.
        "success_rate_percent": all_rates[plotted_suite],
        "derived_metrics": all_metrics[plotted_suite],
        "success_rate_percent_by_suite": all_rates,
        "success_rate_std_percent_by_suite": {
            suite: {"CloudEdgeVLA": values}
            for suite, values in CLOUDEDGE_STD_BY_SUITE.items()
        },
        "random_seed_count": {"CloudEdgeVLA": len(CLOUDEDGE_RANDOM_SEEDS)},
        "random_seed_values": {"CloudEdgeVLA": list(CLOUDEDGE_RANDOM_SEEDS)},
        "derived_metrics_by_suite": all_metrics,
        "uncertainty": (
            "Suite-level CloudEdgeVLA values are means and standard deviations "
            "over random seeds 7, 8, and 9; standard deviations are exported in "
            "success_rate_std_percent_by_suite. Baseline uncertainty and "
            "cross-suite covariances were not provided, so no macro-average "
            "error band is plotted."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("results/fig_closed_loop_delay_robustness.png"),
        help="Composite preview path (PNG or PDF).",
    )
    parser.add_argument(
        "--paper_figure_dir",
        type=Path,
        default=None,
        help="Directory for the paper-ready curve and summary PDFs.",
    )
    parser.add_argument(
        "--suite",
        choices=("libero_mean", *SUCCESS_RATES_BY_SUITE),
        default=PLOT_SUITE,
        help="Suite to visualize; all suites are always exported to JSON.",
    )
    args = parser.parse_args()

    validate_data()
    _set_style()

    plot_composite(args.output_path, args.suite)
    if args.output_path.suffix.lower() != ".pdf":
        plot_composite(args.output_path.with_suffix(".pdf"), args.suite)

    figure_dir = args.paper_figure_dir or args.output_path.parent
    curve_path = figure_dir / "fig_closed_loop_delay_robustness_curve.pdf"
    summary_path = figure_dir / "fig_closed_loop_delay_robustness_retention_summary.pdf"
    auc_path = figure_dir / "fig_closed_loop_delay_robustness_auc.pdf"
    d40_path = figure_dir / "fig_closed_loop_delay_robustness_retention_d40.pdf"
    plot_curve(curve_path, args.suite)
    plot_summary(summary_path, args.suite)
    plot_single_summary(auc_path, "auc", args.suite)
    plot_single_summary(d40_path, "d40", args.suite)

    export = build_export(args.suite)
    json_path = args.output_path.with_name(f"{args.output_path.stem}_data.json")
    json_path.write_text(json.dumps(export, indent=2, allow_nan=False) + "\n")

    print(f"Saved composite: {args.output_path}")
    print(f"Saved curve:     {curve_path}")
    print(f"Saved summary:   {summary_path}")
    print(f"Saved data:      {json_path}")


if __name__ == "__main__":
    main()
