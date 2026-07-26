"""Compare fresh-vs-stale action consistency across OFT and CloudEdge models.

For every current observation t, this script replaces the cloud planning feature
h_t with h_{t-d}. For CloudEdge models, the current edge feature z_t is held
fixed. Standard OFT has no edge feature and therefore uses g(h) directly. The
metric is the mean absolute action drift in normalized action space:

    D(d) = mean |g(h_t, z_t) - g(h_{t-d}, z_t)|.

Lower drift means that the policy is less sensitive to cloud-feature staleness.
Cloud hidden states are extracted with the same normalized proprioceptive input
used during training and normal LIBERO evaluation.
"""

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import scripts.visualize_action_head_attribution as attr
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


DEFAULT_DELAYS = [0, 1, 3, 5, 8, 10, 15, 20]
MODEL_ORDER = ("OpenVLA-OFT", "CloudEdge baseline", "CloudEdgeVLA")
COLORS = {
    "OpenVLA-OFT": "#64748B",
    "CloudEdge baseline": "#E76F51",
    "CloudEdgeVLA": "#2563EB",
}
MARKERS = {"OpenVLA-OFT": "^", "CloudEdge baseline": "s", "CloudEdgeVLA": "o"}
LINESTYLES = {"OpenVLA-OFT": ":", "CloudEdge baseline": "--", "CloudEdgeVLA": "-"}


@torch.inference_mode()
def predict_from_cached_features(
    action_head,
    hidden_state: torch.Tensor,
    vision_feature: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run the action head from cached features, with edge vision when supported."""
    if vision_feature is None:
        return action_head.predict_action(hidden_state).float()

    batch_size = hidden_state.shape[0]
    llm_feature = hidden_state.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
    vision_projected = action_head.vision_projector(vision_feature)
    vision_projected = vision_projected.unsqueeze(1).expand(-1, NUM_ACTIONS_CHUNK, -1)
    fused = torch.cat([llm_feature, vision_projected], dim=-1)
    return action_head.fusion_mlp(fused).float()


def extract_cached_features(
    checkpoint: str,
    observations: Dict[str, List[dict]],
    lora_rank: int,
    num_views: int,
    uses_edge_vision: bool,
) -> Tuple[object, Dict[str, List[torch.Tensor]], Dict[str, List[Optional[torch.Tensor]]]]:
    """Load one checkpoint and extract cloud hidden states and edge features."""
    print(f"[INFO] Loading {checkpoint}", flush=True)
    vla, action_head, processor, cfg = attr.load_model(
        checkpoint,
        lora_rank=lora_rank,
        action_head_vision_encoder="siglip-base",
        num_views=num_views,
        use_vision_action_head=uses_edge_vision,
    )
    from experiments.robot.openvla_utils import get_proprio_projector, normalize_proprio

    action_head.eval()
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    proprio_projector.eval()
    proprio_norm_stats = vla.norm_stats["libero_goal_no_noops"]["proprio"]
    image_cfg = SimpleNamespace(num_images_in_input=num_views, center_crop=cfg.center_crop)
    tokenizer = processor.tokenizer
    hidden_by_task: Dict[str, List[torch.Tensor]] = {}
    vision_by_task: Dict[str, List[Optional[torch.Tensor]]] = {}

    for task_index, (task_description, frames) in enumerate(observations.items(), start=1):
        hidden_states: List[torch.Tensor] = []
        vision_features: List[Optional[torch.Tensor]] = []
        for frame in frames:
            proprio = normalize_proprio(
                np.asarray(frame["state"], dtype=np.float32), proprio_norm_stats
            )
            pixels = attr.obs_to_pixel_values(frame, processor, image_cfg).to(
                attr.DEVICE, dtype=torch.bfloat16
            )
            prompt = (
                "In: What action should the robot take to "
                f"{frame['task_label'].lower()}?\nOut:"
            )
            input_ids = tokenizer(
                prompt, truncation=True, return_tensors="pt"
            ).input_ids.to(attr.DEVICE)
            with attr._HiddenStateCapture(action_head) as capture:
                vla.predict_action(
                    input_ids=input_ids,
                    pixel_values=pixels,
                    attention_mask=torch.ones_like(input_ids),
                    unnorm_key="libero_goal_no_noops",
                    proprio=proprio,
                    proprio_projector=proprio_projector,
                    action_head=action_head,
                )
                vision = action_head.encode_vision(pixels) if uses_edge_vision else None
            if capture.hidden_states is None:
                raise RuntimeError("Failed to capture cloud hidden state")
            hidden_states.append(capture.hidden_states.detach().clone())
            vision_features.append(vision.detach().clone() if vision is not None else None)
        hidden_by_task[task_description] = hidden_states
        vision_by_task[task_description] = vision_features
        print(
            f"  [FEATURES] {task_index}/{len(observations)}: "
            f"{task_description} ({len(frames)} frames)",
            flush=True,
        )

    del vla, processor, proprio_projector
    gc.collect()
    torch.cuda.empty_cache()
    return action_head, hidden_by_task, vision_by_task


def compute_action_drift(
    action_head,
    observations: Dict[str, List[dict]],
    hidden_by_task: Dict[str, List[torch.Tensor]],
    vision_by_task: Dict[str, List[Optional[torch.Tensor]]],
    delays: List[int],
    max_samples_per_task: int,
) -> Tuple[Dict[int, List[float]], Dict[str, List[float]]]:
    results: Dict[int, List[float]] = defaultdict(list)
    diagnostics: Dict[str, List[float]] = defaultdict(list)
    for task_description, frames in observations.items():
        hidden = hidden_by_task[task_description]
        vision = vision_by_task[task_description]
        fresh_actions = [
            predict_from_cached_features(action_head, h, z)
            for h, z in zip(hidden, vision)
        ]
        for action, h, z in zip(fresh_actions, hidden, vision):
            diagnostics["fresh_action_magnitude"].append(
                float(torch.mean(torch.abs(action)).item())
            )
            planning_ablated = predict_from_cached_features(
                action_head, torch.zeros_like(h), z
            )
            planning_effect = torch.mean(torch.abs(action - planning_ablated)).item()
            diagnostics["planning_ablation_effect"].append(float(planning_effect))
            if z is not None:
                vision_ablated = predict_from_cached_features(
                    action_head, h, torch.zeros_like(z)
                )
                vision_effect = torch.mean(torch.abs(action - vision_ablated)).item()
                diagnostics["vision_ablation_effect"].append(float(vision_effect))
                diagnostics["planning_ablation_share"].append(
                    float(planning_effect / (planning_effect + vision_effect + 1e-8))
                )

        for t in range(1, len(frames)):
            if frames[t]["episode_id"] == frames[t - 1]["episode_id"]:
                diagnostics["fresh_temporal_change"].append(
                    float(torch.mean(torch.abs(fresh_actions[t] - fresh_actions[t - 1])).item())
                )

        for delay in delays:
            eligible = [
                t
                for t in range(delay, len(frames))
                if frames[t]["episode_id"] == frames[t - delay]["episode_id"]
            ]
            if len(eligible) > max_samples_per_task:
                selected = np.linspace(
                    0, len(eligible) - 1, max_samples_per_task, dtype=int
                )
                eligible = [eligible[index] for index in selected]
            for t in eligible:
                stale_action = predict_from_cached_features(
                    action_head, hidden[t - delay], vision[t]
                )
                drift = torch.mean(torch.abs(fresh_actions[t] - stale_action)).item()
                results[delay].append(float(drift))
    return dict(results), dict(diagnostics)


def plot_results(
    results: Dict[str, Dict[int, List[float]]],
    output_path: Path,
    checkpoint_steps: Dict[str, int],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, (ax_curve, ax_distribution) = plt.subplots(
        1, 2, figsize=(14.2, 5.2), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )

    labels = [label for label in MODEL_ORDER if label in results]
    for label in labels:
        delays = sorted(results[label])
        means = np.array([np.mean(results[label][delay]) for delay in delays])
        stds = np.array([np.std(results[label][delay]) for delay in delays])
        ax_curve.plot(
            delays,
            means,
            color=COLORS[label],
            marker=MARKERS[label],
            ms=7,
            lw=3.0 if label == "CloudEdgeVLA" else 2.3,
            ls=LINESTYLES[label],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=f"{label} ({checkpoint_steps[label] // 1000}k)",
            zorder=3,
        )
        ax_curve.fill_between(
            delays,
            np.maximum(0.0, means - stds),
            means + stds,
            color=COLORS[label],
            alpha=0.13,
            linewidth=0,
            zorder=2,
        )

    ax_curve.axvspan(0, 20, color="#DBEAFE", alpha=0.22, zorder=0)
    ax_curve.text(
        10,
        ax_curve.get_ylim()[1] * 0.94,
        "Training delay support",
        color="#475569",
        ha="center",
        va="top",
        fontsize=10,
    )
    ax_curve.set_title("(a) Action drift vs. cloud-feature age", loc="left", fontweight="bold")
    ax_curve.set_xlabel("Cloud-feature delay $d$")
    ax_curve.set_ylabel(
        r"Normalized action drift "
        r"$\mathbb{E}|\hat{a}^{\mathrm{fresh}}_t-\hat{a}^{\mathrm{stale}(d)}_t|$"
    )
    ax_curve.set_xticks(sorted(next(iter(results.values())).keys()))
    ax_curve.set_xlim(-0.5, 20.5)
    ax_curve.set_ylim(bottom=0)
    ax_curve.grid(axis="y", color="#CBD5E1", alpha=0.65, lw=0.8)
    ax_curve.legend(frameon=False, loc="upper left")

    # Raincloud-style summary at the largest tested delay.
    max_delay = max(next(iter(results.values())).keys())
    distributions = [results[label][max_delay] for label in labels]
    positions = list(range(len(labels)))
    violins = ax_distribution.violinplot(
        distributions,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, label in zip(violins["bodies"], labels):
        body.set_facecolor(COLORS[label])
        body.set_edgecolor("none")
        body.set_alpha(0.25)

    rng = np.random.default_rng(42)
    for position, label, values in zip(
        positions, labels, distributions
    ):
        values_array = np.asarray(values)
        jitter = rng.normal(0.0, 0.045, size=len(values_array))
        ax_distribution.scatter(
            position + jitter,
            values_array,
            s=13,
            alpha=0.28,
            color=COLORS[label],
            edgecolors="none",
            zorder=2,
        )
        quartiles = np.percentile(values_array, [25, 50, 75])
        ax_distribution.plot(
            [position, position],
            [quartiles[0], quartiles[2]],
            color=COLORS[label],
            lw=7,
            solid_capstyle="round",
            zorder=4,
        )
        ax_distribution.scatter(
            [position], [quartiles[1]], s=42, color="white", edgecolor=COLORS[label], zorder=5
        )

    ours_mean = float(np.mean(results["CloudEdgeVLA"][max_delay]))
    comparisons = []
    for label in labels:
        if label == "CloudEdgeVLA":
            continue
        reference_mean = float(np.mean(results[label][max_delay]))
        reduction = (reference_mean - ours_mean) / reference_mean * 100.0 if reference_mean else 0.0
        direction = "lower" if reduction >= 0 else "higher"
        comparisons.append(f"vs. {label}: {abs(reduction):.1f}% {direction}")

    top = max(max(values) for values in distributions)
    ax_distribution.text(
        0.98,
        0.97,
        "CloudEdgeVLA\n" + "\n".join(comparisons),
        transform=ax_distribution.transAxes,
        ha="right",
        va="top",
        color="#1E3A8A",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BFDBFE", "alpha": 0.9},
    )
    ax_distribution.set_title(
        f"(b) Distribution at $d={max_delay}$", loc="left", fontweight="bold"
    )
    ax_distribution.set_xticks(positions)
    ax_distribution.set_xticklabels([label.replace(" ", "\n") for label in labels])
    ax_distribution.set_ylabel("Per-sample normalized action drift")
    ax_distribution.set_ylim(0, top * 1.16 if top > 0 else 1.0)
    ax_distribution.grid(axis="y", color="#CBD5E1", alpha=0.55, lw=0.8)

    fig.suptitle(
        "Fresh–Stale Action Consistency on LIBERO-Goal (latest available checkpoints)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.035,
        "CloudEdge: current edge vision is fixed while cloud features are delayed  •  OFT: cloud-only action head  •  Mean ± 1 SD",
        ha="center",
        color="#64748B",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.19, top=0.82, wspace=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_oft", required=True)
    parser.add_argument("--checkpoint_baseline", required=True)
    parser.add_argument("--checkpoint_ours", required=True)
    parser.add_argument("--checkpoint_oft_step", type=int, required=True)
    parser.add_argument("--checkpoint_baseline_step", type=int, required=True)
    parser.add_argument("--checkpoint_ours_step", type=int, required=True)
    parser.add_argument("--delays", type=int, nargs="+", default=DEFAULT_DELAYS)
    parser.add_argument("--num_episodes", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=35)
    parser.add_argument("--max_tasks", type=int, default=5)
    parser.add_argument("--max_samples_per_task", type=int, default=20)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("results/fig_action_consistency_latest.png"),
    )
    args = parser.parse_args()

    attr.configure_runtime(args.device)
    print("[INFO] Collecting shared LIBERO-Goal trajectories", flush=True)
    observations = attr.collect_observations(
        "libero_goal",
        num_episodes=args.num_episodes,
        num_frames=args.num_frames,
        max_tasks=args.max_tasks,
        rollout_mode="random",
    )

    results: Dict[str, Dict[int, List[float]]] = {}
    diagnostics: Dict[str, Dict[str, List[float]]] = {}
    model_specs = (
        ("OpenVLA-OFT", args.checkpoint_oft, False),
        ("CloudEdge baseline", args.checkpoint_baseline, True),
        ("CloudEdgeVLA", args.checkpoint_ours, True),
    )
    checkpoint_steps = {
        "OpenVLA-OFT": args.checkpoint_oft_step,
        "CloudEdge baseline": args.checkpoint_baseline_step,
        "CloudEdgeVLA": args.checkpoint_ours_step,
    }
    for label, checkpoint, uses_edge_vision in model_specs:
        action_head, hidden, vision = extract_cached_features(
            checkpoint,
            observations,
            args.lora_rank,
            args.num_views,
            uses_edge_vision,
        )
        results[label], diagnostics[label] = compute_action_drift(
            action_head,
            observations,
            hidden,
            vision,
            args.delays,
            args.max_samples_per_task,
        )
        diagnostic_summary = {
            name: float(np.mean(values)) for name, values in diagnostics[label].items()
        }
        print(f"[DIAGNOSTICS] {label}: {diagnostic_summary}", flush=True)
        del action_head, hidden, vision
        gc.collect()
        torch.cuda.empty_cache()

    plot_results(results, args.output_path, checkpoint_steps)
    export = {
        "config": {
            "checkpoint_oft": args.checkpoint_oft,
            "checkpoint_baseline": args.checkpoint_baseline,
            "checkpoint_ours": args.checkpoint_ours,
            "checkpoint_steps": checkpoint_steps,
            "task_suite": "libero_goal",
            "delays": args.delays,
            "num_episodes": args.num_episodes,
            "num_frames": args.num_frames,
            "max_tasks": args.max_tasks,
            "max_samples_per_task": args.max_samples_per_task,
            "device": args.device,
            "metric": "mean_absolute_normalized_action_drift",
            "metric_note": (
                "CloudEdge models hold current edge vision fixed; OpenVLA-OFT has no edge-vision action-head input. "
                "All cloud hidden states include normalized current proprioception."
            ),
        },
        "models": {
            label: {
                str(delay): {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "n": len(values),
                    "values": values,
                }
                for delay, values in model_results.items()
            }
            for label, model_results in results.items()
        },
        "diagnostics": {
            label: {
                name: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "n": len(values),
                    "values": values,
                }
                for name, values in model_diagnostics.items()
            }
            for label, model_diagnostics in diagnostics.items()
        },
    }
    json_path = args.output_path.with_name(f"{args.output_path.stem}_data.json")
    json_path.write_text(json.dumps(export, indent=2, allow_nan=False) + "\n")
    print(f"[INFO] Saved figure: {args.output_path}")
    print(f"[INFO] Saved data:   {json_path}")


if __name__ == "__main__":
    main()
