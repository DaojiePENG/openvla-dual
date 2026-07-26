"""Paper-oriented diagnostics for CloudEdgeVLA's backbone/head delay mechanism.

The script uses real LIBERO demonstration trajectories and independently ages
the cloud-backbone observation and the edge-head observation.  It produces:

1. an overview figure with offline action error, a two-dimensional delay
   response surface, edge-rescue curves, and correction-vector alignment;
2. a backbone/head decomposition exported as the original four-panel overview,
   two paper-friendly panel pairs, and four standalone panels;
3. a geometry/task figure plus three standalone geometry/task panels; and
4. an action-chunk heatmap plus three standalone chunk panels showing where
   delay training suppresses stale-backbone drift.

Unlike action-consistency-only plots, the overview also reports normalized MAE
to demonstration action chunks, so an insensitive or collapsed policy is not
automatically rewarded.
"""

import argparse
import gc
import json
import math
import os
import textwrap
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
import torch.nn.functional as F

try:
    import scripts.visualize_action_head_attribution as attr
except ModuleNotFoundError as error:
    # Some server images install an unrelated top-level ``scripts`` package.
    # Direct execution still has this file's directory on sys.path, so fall
    # back to the sibling module without changing the repository package tree.
    if error.name not in {"scripts", "scripts.visualize_action_head_attribution"}:
        raise
    import visualize_action_head_attribution as attr
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


DEFAULT_DELAYS = [0, 1, 3, 5, 8, 10, 15, 20]
OFT_LABEL = "OpenVLA-OFT"
CLOUD_LABEL = "CloudEdgeVLA"
COLORS = {OFT_LABEL: "#64748B", CLOUD_LABEL: "#2563EB"}
ACTION_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]


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


def _normalize(values: np.ndarray, stats: Dict) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    normalized = np.where(
        mask,
        np.clip(2.0 * (values - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0),
        values,
    )
    return normalized.astype(np.float32)


def _transform_libero_actions(raw_actions: np.ndarray) -> np.ndarray:
    """Match libero_dataset_transform: -1=open/+1=close -> 1=open/0=close."""
    actions = np.asarray(raw_actions, dtype=np.float32).copy()
    actions[..., -1] = 1.0 - np.clip(actions[..., -1], 0.0, 1.0)
    return actions


def load_demonstration_samples(
    dataset_dir: Path,
    action_stats: Dict,
    delays: Sequence[int],
    max_tasks: int,
    episodes_per_task: int,
    max_frames_per_episode: int,
    samples_per_task: int,
) -> Tuple[List[dict], List[dict]]:
    """Load shared, temporally aligned samples from local TFDS demonstrations."""
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(dataset_dir))
    dataset = builder.as_dataset(split="train", shuffle_files=False)
    max_delay = max(delays)
    task_counts: Dict[str, int] = defaultdict(int)
    task_order: List[str] = []
    episodes: List[dict] = []

    for raw_episode in dataset:
        steps = list(raw_episode["steps"].as_numpy_iterator())
        if not steps:
            continue
        task = steps[0]["language_instruction"].decode("utf-8").strip()
        is_new_task = task not in task_counts
        if is_new_task and len(task_order) >= max_tasks:
            continue
        if not is_new_task and task_counts[task] >= episodes_per_task:
            continue
        if len(steps) < max_delay + NUM_ACTIONS_CHUNK + 2:
            continue

        if is_new_task:
            task_order.append(task)
        steps = steps[:max_frames_per_episode]
        raw_actions = np.stack([step["action"] for step in steps])
        actions = _normalize(_transform_libero_actions(raw_actions), action_stats)
        frames = [
            {
                "full_image": step["observation"]["image"],
                "wrist_image": step["observation"]["wrist_image"],
                "state": step["observation"]["state"].astype(np.float32),
                "task_label": task,
                "action": actions[index],
            }
            for index, step in enumerate(steps)
        ]
        episode_index = len(episodes)
        episodes.append({"task": task, "frames": frames, "actions": actions})
        task_counts[task] += 1
        if len(task_order) >= max_tasks and all(
            task_counts[name] >= episodes_per_task for name in task_order
        ):
            break

    samples: List[dict] = []
    samples_by_task: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for episode_index, episode in enumerate(episodes):
        last_t = len(episode["frames"]) - NUM_ACTIONS_CHUNK
        eligible = np.arange(max_delay, last_t + 1, dtype=int)
        if len(eligible) == 0:
            continue
        take_count = min(samples_per_task, len(eligible))
        selected = eligible[np.linspace(0, len(eligible) - 1, take_count, dtype=int)]
        samples_by_task[episode["task"]].extend((episode_index, int(t)) for t in selected)

    # Keep a fixed number per task even if multiple episodes were requested.
    for task in task_order:
        candidates = samples_by_task[task]
        if len(candidates) > samples_per_task:
            selected_ids = np.linspace(0, len(candidates) - 1, samples_per_task, dtype=int)
            candidates = [candidates[index] for index in selected_ids]
        for episode_index, t in candidates:
            episode = episodes[episode_index]
            samples.append(
                {
                    "episode_index": episode_index,
                    "t": t,
                    "task": task,
                    "action_chunk": episode["actions"][t : t + NUM_ACTIONS_CHUNK],
                }
            )

    if not samples:
        raise RuntimeError("No valid demonstration samples were found.")
    print(
        f"[DATA] {len(task_order)} tasks, {len(episodes)} episodes, "
        f"{len(samples)} shared current timesteps",
        flush=True,
    )
    for task in task_order:
        count = sum(sample["task"] == task for sample in samples)
        print(f"  [TASK] {task} ({count} samples)", flush=True)
    return episodes, samples


class _CaptureOnlyHead:
    """Capture action-token hidden states without running the real Action Head."""

    def __init__(self):
        self.hidden: Optional[torch.Tensor] = None

    def predict_action(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self.hidden = hidden_states.detach().clone()
        return torch.zeros(
            (hidden_states.shape[0], NUM_ACTIONS_CHUNK, ACTION_DIM),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )


@torch.inference_mode()
def _predict_from_features(
    action_head,
    hidden: torch.Tensor,
    vision: Optional[torch.Tensor],
) -> torch.Tensor:
    if vision is None:
        return action_head.predict_action(hidden).float()
    batch_size = hidden.shape[0]
    llm_feature = hidden.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
    vision_projected = action_head.vision_projector(vision)
    vision_projected = vision_projected.unsqueeze(1).expand(-1, NUM_ACTIONS_CHUNK, -1)
    fused = torch.cat([llm_feature, vision_projected], dim=-1)
    return action_head.fusion_mlp(fused).float()


@torch.inference_mode()
def _run_action_head_batched(
    action_head,
    hidden: List[torch.Tensor],
    vision: Optional[List[torch.Tensor]],
    batch_size: int = 24,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(hidden), batch_size):
        h_batch = torch.cat(hidden[start : start + batch_size], dim=0)
        z_batch = None
        if vision is not None:
            z_batch = torch.cat(vision[start : start + batch_size], dim=0)
        outputs.append(_predict_from_features(action_head, h_batch, z_batch).cpu())
    return torch.cat(outputs, dim=0)


def extract_model_features(
    label: str,
    checkpoint: str,
    episodes: List[dict],
    samples: List[dict],
    delays: Sequence[int],
    lora_rank: int,
    num_views: int,
    unnorm_key: str,
    use_vision_action_head: bool,
) -> Tuple[object, Dict[int, List[torch.Tensor]], Optional[Dict[int, List[torch.Tensor]]]]:
    """Extract h(image[t-d], proprio[t]) and z(image[t-d]) for every sample."""
    print(f"[MODEL] Loading {label}: {checkpoint}", flush=True)
    vla, action_head, processor, cfg = attr.load_model(
        checkpoint,
        lora_rank=lora_rank,
        action_head_vision_encoder="siglip-base",
        num_views=num_views,
        use_vision_action_head=use_vision_action_head,
    )
    from experiments.robot.openvla_utils import get_proprio_projector, normalize_proprio

    action_head.eval()
    # The 7B backbone and the large regression head do not need to be resident on
    # the GPU at the same time.  Keep the head on CPU while extracting h; after
    # unloading the backbone, move it back for edge encoding and metric heads.
    action_head = action_head.cpu()
    torch.cuda.empty_cache()
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    proprio_projector.eval()
    proprio_stats = vla.norm_stats[unnorm_key]["proprio"]
    image_cfg = SimpleNamespace(num_images_in_input=num_views, center_crop=cfg.center_crop)
    tokenizer = processor.tokenizer
    prompt_cache: Dict[str, torch.Tensor] = {}
    hidden_by_delay: Dict[int, List[torch.Tensor]] = {delay: [] for delay in delays}
    vision_cache: Dict[Tuple[int, int], torch.Tensor] = {}
    capture_head = _CaptureOnlyHead()

    for sample_index, sample in enumerate(samples, start=1):
        episode = episodes[sample["episode_index"]]
        current_frame = episode["frames"][sample["t"]]
        proprio = normalize_proprio(current_frame["state"], proprio_stats)
        task = sample["task"]
        if task not in prompt_cache:
            prompt = f"In: What action should the robot take to {task.lower()}?\nOut:"
            prompt_cache[task] = tokenizer(
                prompt, truncation=True, return_tensors="pt"
            ).input_ids.to(attr.DEVICE)
        input_ids = prompt_cache[task]

        for delay in delays:
            frame_index = sample["t"] - delay
            frame = episode["frames"][frame_index]
            pixels = attr.obs_to_pixel_values(frame, processor, image_cfg).to(
                attr.DEVICE, dtype=torch.bfloat16
            )
            capture_head.hidden = None
            vla.predict_action(
                input_ids=input_ids,
                pixel_values=pixels,
                attention_mask=torch.ones_like(input_ids),
                unnorm_key=unnorm_key,
                proprio=proprio,
                proprio_projector=proprio_projector,
                action_head=capture_head,
            )
            if capture_head.hidden is None:
                raise RuntimeError("Failed to capture action-token hidden states.")
            hidden_by_delay[delay].append(capture_head.hidden)
            del pixels

        if sample_index % 10 == 0 or sample_index == len(samples):
            print(f"  [FEATURES] {label}: {sample_index}/{len(samples)}", flush=True)

    del vla, proprio_projector, capture_head, prompt_cache
    gc.collect()
    torch.cuda.empty_cache()
    action_head = action_head.to(attr.DEVICE)
    action_head.eval()

    vision_by_delay = None
    if use_vision_action_head:
        print(f"  [EDGE] {label}: extracting current/stale Head features", flush=True)
        for sample in samples:
            episode = episodes[sample["episode_index"]]
            for delay in delays:
                frame_index = sample["t"] - delay
                cache_key = (sample["episode_index"], frame_index)
                if cache_key in vision_cache:
                    continue
                frame = episode["frames"][frame_index]
                pixels = attr.obs_to_pixel_values(frame, processor, image_cfg).to(
                    attr.DEVICE, dtype=torch.bfloat16
                )
                vision_cache[cache_key] = action_head.encode_vision(pixels).detach().clone()
                del pixels
        vision_by_delay = {delay: [] for delay in delays}
        for sample in samples:
            for delay in delays:
                key = (sample["episode_index"], sample["t"] - delay)
                vision_by_delay[delay].append(vision_cache[key])

    del processor, vision_cache
    gc.collect()
    torch.cuda.empty_cache()
    return action_head, hidden_by_delay, vision_by_delay


def _hidden_cosine_distance(
    fresh: List[torch.Tensor], stale: List[torch.Tensor], batch_size: int = 24
) -> np.ndarray:
    values = []
    for start in range(0, len(fresh), batch_size):
        h0 = torch.cat(fresh[start : start + batch_size], dim=0).float()
        hd = torch.cat(stale[start : start + batch_size], dim=0).float()
        distance = 1.0 - F.cosine_similarity(h0.flatten(1), hd.flatten(1), dim=1)
        values.append(distance.cpu().numpy())
    return np.concatenate(values)


def evaluate_oft(
    action_head,
    hidden: Dict[int, List[torch.Tensor]],
    ground_truth: torch.Tensor,
    delays: Sequence[int],
) -> Dict:
    predictions = {
        delay: _run_action_head_batched(action_head, hidden[delay], None) for delay in delays
    }
    fresh = predictions[0]
    fresh_demo_mae = torch.mean(torch.abs(fresh - ground_truth), dim=(1, 2)).numpy()
    per_delay = {}
    for delay in delays:
        prediction = predictions[delay]
        drift = torch.mean(torch.abs(prediction - fresh), dim=(1, 2)).numpy()
        demo_mae = torch.mean(torch.abs(prediction - ground_truth), dim=(1, 2)).numpy()
        per_delay[delay] = {
            "action_drift": drift,
            "demo_mae": demo_mae,
            "excess_demo_mae": demo_mae - fresh_demo_mae,
            "hidden_cosine_distance": _hidden_cosine_distance(hidden[0], hidden[delay]),
        }
    return {"predictions": predictions, "per_delay": per_delay}


def evaluate_cloudedge(
    action_head,
    hidden: Dict[int, List[torch.Tensor]],
    vision: Dict[int, List[torch.Tensor]],
    ground_truth: torch.Tensor,
    delays: Sequence[int],
) -> Dict:
    grid_predictions = {}
    for cloud_delay in delays:
        for edge_delay in delays:
            grid_predictions[(cloud_delay, edge_delay)] = _run_action_head_batched(
                action_head, hidden[cloud_delay], vision[edge_delay]
            )
    fresh = grid_predictions[(0, 0)]
    fresh_demo_mae = torch.mean(torch.abs(fresh - ground_truth), dim=(1, 2)).numpy()
    per_delay = {}
    for delay in delays:
        current_edge = grid_predictions[(delay, 0)]
        both_stale = grid_predictions[(delay, delay)]
        current_drift = torch.mean(torch.abs(current_edge - fresh), dim=(1, 2)).numpy()
        both_drift = torch.mean(torch.abs(both_stale - fresh), dim=(1, 2)).numpy()
        demo_mae = torch.mean(torch.abs(current_edge - ground_truth), dim=(1, 2)).numpy()
        needed = (fresh - both_stale).flatten(1)
        correction = (current_edge - both_stale).flatten(1)
        correction_cosine = F.cosine_similarity(correction, needed, dim=1, eps=1e-8).numpy()
        projected_recovery = (
            torch.sum(correction * needed, dim=1)
            / (torch.sum(needed * needed, dim=1) + 1e-8)
        ).numpy()
        sample_rescue = 1.0 - current_drift / (both_drift + 1e-8)
        per_delay[delay] = {
            "action_drift": current_drift,
            "both_stale_action_drift": both_drift,
            "demo_mae": demo_mae,
            "excess_demo_mae": demo_mae - fresh_demo_mae,
            "hidden_cosine_distance": _hidden_cosine_distance(hidden[0], hidden[delay]),
            "edge_rescue_fraction": sample_rescue,
            "correction_cosine": correction_cosine,
            "projected_recovery": projected_recovery,
        }

    drift_surface = np.zeros((len(delays), len(delays)), dtype=np.float32)
    for row, cloud_delay in enumerate(delays):
        for col, edge_delay in enumerate(delays):
            values = torch.mean(
                torch.abs(grid_predictions[(cloud_delay, edge_delay)] - fresh), dim=(1, 2)
            )
            drift_surface[row, col] = float(values.mean())

    return {
        "predictions": grid_predictions,
        "per_delay": per_delay,
        "drift_surface": drift_surface,
    }


def _mean_ci(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    ci = 1.96 * float(np.std(values)) / math.sqrt(max(len(values), 1))
    return mean, ci


def _task_means(values: np.ndarray, samples: List[dict]) -> np.ndarray:
    values = np.asarray(values)
    tasks = list(dict.fromkeys(sample["task"] for sample in samples))
    return np.asarray(
        [
            np.mean(values[[index for index, sample in enumerate(samples) if sample["task"] == task]])
            for task in tasks
        ],
        dtype=np.float64,
    )


def _line_with_ci(ax, x, values_by_x, label, color, marker="o", linestyle="-"):
    means, cis = zip(*[_mean_ci(values_by_x[value]) for value in x])
    means = np.asarray(means)
    cis = np.asarray(cis)
    ax.plot(
        x,
        means,
        color=color,
        marker=marker,
        lw=2.6,
        ms=6.5,
        linestyle=linestyle,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label=label,
        zorder=3,
    )
    ax.fill_between(x, means - cis, means + cis, color=color, alpha=0.14, linewidth=0)
    return means


def plot_overview(
    output_path: Path,
    delays: Sequence[int],
    oft: Dict,
    cloud: Dict,
    suite_label: str,
) -> None:
    _paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.4))
    ax_error, ax_surface, ax_rescue, ax_geometry = axes.flatten()

    _line_with_ci(
        ax_error,
        delays,
        {d: oft["per_delay"][d]["excess_demo_mae"] for d in delays},
        OFT_LABEL,
        COLORS[OFT_LABEL],
        marker="^",
        linestyle=":",
    )
    _line_with_ci(
        ax_error,
        delays,
        {d: cloud["per_delay"][d]["excess_demo_mae"] for d in delays},
        CLOUD_LABEL + " (current edge)",
        COLORS[CLOUD_LABEL],
    )
    ax_error.axhline(0, color="#94A3B8", lw=0.8)
    ax_error.set_title("(a) Delay-induced error to demonstrations", loc="left", fontweight="bold")
    ax_error.set_xlabel("Cloud-backbone delay $d_h$")
    ax_error.set_ylabel("Excess normalized action MAE")
    ax_error.set_xticks(delays)
    ax_error.grid(axis="y", color="#CBD5E1", alpha=0.65, lw=0.8)
    ax_error.legend(loc="upper left")

    surface = cloud["drift_surface"]
    image = ax_surface.imshow(surface, cmap="Blues", origin="lower", aspect="auto")
    threshold = 0.62 * float(surface.max())
    for row in range(len(delays)):
        for col in range(len(delays)):
            ax_surface.text(
                col,
                row,
                f"{surface[row, col]:.3f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if surface[row, col] > threshold else "#1E293B",
            )
    ax_surface.plot(range(len(delays)), range(len(delays)), "--", color="#F97316", lw=1.8)
    ax_surface.set_xticks(range(len(delays)), delays)
    ax_surface.set_yticks(range(len(delays)), delays)
    ax_surface.set_xlabel("Edge-head observation delay $d_z$")
    ax_surface.set_ylabel("Cloud-backbone delay $d_h$")
    ax_surface.set_title("(b) Backbone–Head counterfactual delay surface", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax_surface, fraction=0.046, pad=0.035)
    colorbar.set_label("Drift from fresh CloudEdge action")

    _line_with_ci(
        ax_rescue,
        delays,
        {d: cloud["per_delay"][d]["both_stale_action_drift"] for d in delays},
        "Backbone + Head stale",
        "#F97316",
        marker="s",
        linestyle="--",
    )
    _line_with_ci(
        ax_rescue,
        delays,
        {d: cloud["per_delay"][d]["action_drift"] for d in delays},
        "Backbone stale, Head current",
        COLORS[CLOUD_LABEL],
    )
    ax_rescue.set_title("(c) Counterfactual edge-age ablation", loc="left", fontweight="bold")
    ax_rescue.set_xlabel("Delay $d$")
    ax_rescue.set_ylabel("Normalized action drift")
    ax_rescue.set_xticks(delays)
    ax_rescue.grid(axis="y", color="#CBD5E1", alpha=0.65, lw=0.8)
    ax_rescue.legend(loc="upper left")

    nonzero = [delay for delay in delays if delay > 0]
    rescue = [np.mean(cloud["per_delay"][d]["edge_rescue_fraction"]) for d in nonzero]
    alignment = [np.mean(cloud["per_delay"][d]["correction_cosine"]) for d in nonzero]
    projected = [np.mean(cloud["per_delay"][d]["projected_recovery"]) for d in nonzero]
    ax_geometry.plot(nonzero, rescue, "o-", color="#0EA5E9", lw=2.5, label="Edge rescue fraction")
    ax_geometry.plot(nonzero, alignment, "s-", color="#7C3AED", lw=2.5, label="Correction alignment")
    ax_geometry.plot(nonzero, projected, "^-", color="#16A34A", lw=2.2, label="Projected recovery")
    ax_geometry.axhline(0, color="#94A3B8", lw=0.8)
    ax_geometry.set_ylim(-0.1, 1.05)
    ax_geometry.set_xticks(nonzero)
    ax_geometry.set_xlabel("Delay $d$")
    ax_geometry.set_ylabel("Fraction / cosine similarity")
    ax_geometry.set_title("(d) Measured edge-correction signal", loc="left", fontweight="bold")
    ax_geometry.grid(axis="y", color="#CBD5E1", alpha=0.65, lw=0.8)
    ax_geometry.legend(loc="best")

    fig.suptitle(
        f"Backbone–Head Counterfactual Delay Audit · {suite_label}",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "Shared demonstration states  •  current proprio for every cloud query  •  mean ± 95% CI  •  mechanism diagnostic, not closed-loop success",
        ha="center",
        color="#64748B",
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.09, top=0.92, hspace=0.34, wspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    seed: int,
    repeats: int = 1000,
) -> Tuple[float, float, float]:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    estimate = float(np.mean(numerator) / (np.mean(denominator) + 1e-8))
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        selected = rng.integers(0, len(numerator), size=len(numerator))
        draws[index] = np.mean(numerator[selected]) / (np.mean(denominator[selected]) + 1e-8)
    low, high = np.percentile(draws, [2.5, 97.5])
    return estimate, float(low), float(high)


def _annotate_reduction(
    ax,
    reference: float,
    ours: float,
    delay: int,
    suffix: str = " lower",
) -> None:
    reduction = (reference - ours) / (abs(reference) + 1e-8) * 100.0
    ax.text(
        0.97,
        0.07,
        f"at $d={delay}$: {reduction:.1f}%{suffix}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#1E3A8A",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#EFF6FF", "edgecolor": "#BFDBFE"},
    )


def _prepare_decomposition_series(
    delays: Sequence[int],
    samples: List[dict],
    oft: Dict,
    cloud: Dict,
) -> Dict:
    """Compute every decomposition curve once for all requested layouts."""
    results = {OFT_LABEL: oft, CLOUD_LABEL: cloud}
    series = {"hidden": {}, "transfer": {}, "action": {}, "demo": {}}
    nonzero = [delay for delay in delays if delay > 0]

    for model_index, (label, result) in enumerate(results.items()):
        series["hidden"][label] = {
            delay: _task_means(
                result["per_delay"][delay]["hidden_cosine_distance"], samples
            )
            for delay in delays
        }
        series["action"][label] = {
            delay: _task_means(result["per_delay"][delay]["action_drift"], samples)
            for delay in delays
        }
        series["demo"][label] = {
            delay: _task_means(result["per_delay"][delay]["demo_mae"], samples)
            for delay in delays
        }

        estimates, lows, highs = [], [], []
        for delay in nonzero:
            estimate, low, high = _bootstrap_ratio(
                series["action"][label][delay],
                series["hidden"][label][delay],
                seed=1000 * model_index + delay,
            )
            estimates.append(estimate)
            lows.append(low)
            highs.append(high)
        series["transfer"][label] = {
            "x": np.asarray(nonzero),
            "estimate": np.asarray(estimates),
            "low": np.asarray(lows),
            "high": np.asarray(highs),
        }
    return series


def _panel_title(letter: Optional[str], title: str) -> str:
    return f"({letter}) {title}" if letter else title


def _draw_hidden_panel(
    ax,
    delays: Sequence[int],
    series: Dict,
    letter: Optional[str],
) -> None:
    means = {}
    for label, marker, linestyle in (
        (OFT_LABEL, "^", ":"),
        (CLOUD_LABEL, "o", "-"),
    ):
        means[label] = _line_with_ci(
            ax,
            delays,
            series["hidden"][label],
            label,
            COLORS[label],
            marker=marker,
            linestyle=linestyle,
        )
    ax.set_title(
        _panel_title(letter, "Backbone: representation staleness"),
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("Backbone image delay $d_h$")
    ax.set_ylabel("Hidden-state cosine distance")
    ax.set_xticks(delays)
    ax.grid(axis="y", color="#CBD5E1", alpha=0.65, lw=0.8)
    ax.legend(loc="upper left")
    _annotate_reduction(
        ax,
        means[OFT_LABEL][-1],
        means[CLOUD_LABEL][-1],
        delay=max(delays),
    )


def _draw_transfer_panel(
    ax,
    delays: Sequence[int],
    series: Dict,
    letter: Optional[str],
) -> None:
    for label, marker, linestyle in (
        (OFT_LABEL, "^", ":"),
        (CLOUD_LABEL, "o", "-"),
    ):
        curve = series["transfer"][label]
        ax.plot(
            curve["x"],
            curve["estimate"],
            color=COLORS[label],
            marker=marker,
            linestyle=linestyle,
            lw=2.6,
            ms=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
        ax.fill_between(
            curve["x"],
            curve["low"],
            curve["high"],
            color=COLORS[label],
            alpha=0.14,
            linewidth=0,
        )
    nonzero = [delay for delay in delays if delay > 0]
    ax.set_title(
        _panel_title(letter, "Head: staleness transfer gain"),
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("Delay $d$")
    ax.set_ylabel(r"$\kappa(d)=D_{action}(d)\,/\,D_{hidden}(d)$")
    ax.set_xticks(nonzero)
    ax.grid(axis="y", color="#CBD5E1", alpha=0.65, lw=0.8)
    ax.legend(loc="upper left")
    _annotate_reduction(
        ax,
        series["transfer"][OFT_LABEL]["estimate"][-1],
        series["transfer"][CLOUD_LABEL]["estimate"][-1],
        delay=max(delays),
        suffix=" lower transfer",
    )


def _draw_action_panel(
    ax,
    delays: Sequence[int],
    series: Dict,
    letter: Optional[str],
) -> None:
    means = {}
    for label, marker, linestyle in (
        (OFT_LABEL, "^", ":"),
        (CLOUD_LABEL, "o", "-"),
    ):
        means[label] = _line_with_ci(
            ax,
            delays,
            series["action"][label],
            label,
            COLORS[label],
            marker=marker,
            linestyle=linestyle,
        )
    ax.set_title(
        _panel_title(letter, "End-to-end: action drift"),
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("Delay $d$")
    ax.set_ylabel("Normalized action drift")
    ax.set_xticks(delays)
    ax.grid(axis="y", color="#CBD5E1", alpha=0.65, lw=0.8)
    ax.legend(loc="upper left")
    _annotate_reduction(
        ax,
        means[OFT_LABEL][-1],
        means[CLOUD_LABEL][-1],
        delay=max(delays),
    )


def _draw_demo_panel(
    ax,
    delays: Sequence[int],
    series: Dict,
    letter: Optional[str],
) -> None:
    means = {}
    for label, marker, linestyle in (
        (OFT_LABEL, "^", ":"),
        (CLOUD_LABEL, "o", "-"),
    ):
        means[label] = _line_with_ci(
            ax,
            delays,
            series["demo"][label],
            label,
            COLORS[label],
            marker=marker,
            linestyle=linestyle,
        )
    ax.set_title(
        _panel_title(letter, "Offline evidence: action error to demos"),
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("Delay $d$")
    ax.set_ylabel("Normalized action MAE")
    ax.set_xticks(delays)
    ax.grid(axis="y", color="#CBD5E1", alpha=0.65, lw=0.8)
    ax.legend(loc="upper left")
    _annotate_reduction(
        ax,
        means[OFT_LABEL][-1],
        means[CLOUD_LABEL][-1],
        delay=max(delays),
    )


def _save_figure_pair(fig, output_path: Path) -> List[Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [output_path, pdf_path]


def _decomposition_variant_path(output_path: Path, variant: str) -> Path:
    stem = output_path.stem
    suffix = "_decomposition"
    prefix = stem[: -len(suffix)] if stem.endswith(suffix) else stem
    return output_path.with_name(f"{prefix}_decomposition_{variant}{output_path.suffix}")


def _artifact_variant_path(output_path: Path, artifact: str, variant: str) -> Path:
    """Name a standalone panel next to its backward-compatible composite."""
    stem = output_path.stem
    suffix = f"_{artifact}"
    prefix = stem[: -len(suffix)] if stem.endswith(suffix) else stem
    return output_path.with_name(f"{prefix}_{artifact}_{variant}{output_path.suffix}")


def plot_backbone_head_decomposition(
    output_path: Path,
    delays: Sequence[int],
    samples: List[dict],
    oft: Dict,
    cloud: Dict,
    suite_label: str,
) -> List[Path]:
    """Export a composite, two paper-friendly pairs, and four single panels."""
    _paper_style()
    series = _prepare_decomposition_series(delays, samples, oft, cloud)
    footer = (
        f"{len(samples)} shared demonstration timesteps across "
        f"{len(set(sample['task'] for sample in samples))} tasks  •  "
        "delayed image + current proprio  •  mean ± 95% CI across tasks"
    )
    saved_paths: List[Path] = []

    # Keep the original four-panel artifact for backward-compatible references.
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.0))
    _draw_hidden_panel(axes[0, 0], delays, series, "a")
    _draw_transfer_panel(axes[0, 1], delays, series, "b")
    _draw_action_panel(axes[1, 0], delays, series, "c")
    _draw_demo_panel(axes[1, 1], delays, series, "d")
    fig.suptitle(
        f"Delay Robustness Decomposes into a Stable Backbone and a Contractive Head · {suite_label}",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        footer,
        ha="center",
        color="#64748B",
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.92, hspace=0.34, wspace=0.23)
    saved_paths.extend(_save_figure_pair(fig, output_path))

    pair_specs = (
        (
            "backbone_head",
            "Backbone–Head Delay Mechanism",
            (_draw_hidden_panel, _draw_transfer_panel),
        ),
        (
            "action_quality",
            "End-to-End Delay Effects",
            (_draw_action_panel, _draw_demo_panel),
        ),
    )
    for variant, title, draw_functions in pair_specs:
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
        for letter, ax, draw_panel in zip(("a", "b"), axes, draw_functions):
            draw_panel(ax, delays, series, letter)
        fig.suptitle(
            f"{title} · {suite_label}",
            fontsize=15,
            fontweight="bold",
            y=0.985,
        )
        fig.text(0.5, 0.018, footer, ha="center", color="#64748B", fontsize=9.0)
        fig.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.84, wspace=0.24)
        saved_paths.extend(
            _save_figure_pair(fig, _decomposition_variant_path(output_path, variant))
        )

    single_specs = (
        ("backbone_staleness", _draw_hidden_panel),
        ("head_transfer", _draw_transfer_panel),
        ("action_drift", _draw_action_panel),
        ("demo_mae", _draw_demo_panel),
    )
    for variant, draw_panel in single_specs:
        fig, ax = plt.subplots(figsize=(6.6, 4.7))
        draw_panel(ax, delays, series, None)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.91)
        saved_paths.extend(
            _save_figure_pair(fig, _decomposition_variant_path(output_path, variant))
        )
    return saved_paths


def _binned_median(x: np.ndarray, y: np.ndarray, bins: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    centers, medians = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (x >= left) & (x <= right)
        if np.any(mask):
            centers.append(float(np.median(x[mask])))
            medians.append(float(np.median(y[mask])))
    return np.asarray(centers), np.asarray(medians)


def _short_task(task: str, width: int = 28) -> str:
    spatial_prefix = "pick up the black bowl "
    spatial_suffix = " and place it on the plate"
    if task.startswith(spatial_prefix) and task.endswith(spatial_suffix):
        task = task[len(spatial_prefix) : -len(spatial_suffix)]
    return textwrap.shorten(task, width=width, placeholder="…")


def plot_geometry_and_tasks(
    output_path: Path,
    delays: Sequence[int],
    samples: List[dict],
    oft: Dict,
    cloud: Dict,
    suite_label: str,
) -> List[Path]:
    """Export the original three-panel geometry figure and three singles."""
    _paper_style()
    nonzero = [delay for delay in delays if delay > 0]
    max_delay = max(delays)
    tasks = list(dict.fromkeys(sample["task"] for sample in samples))
    task_indices = {task: np.asarray([i for i, sample in enumerate(samples) if sample["task"] == task]) for task in tasks}
    task_error = []
    task_improvement = []
    for task in tasks:
        indices = task_indices[task]
        oft_error = float(np.mean(oft["per_delay"][max_delay]["demo_mae"][indices]))
        cloud_error = float(np.mean(cloud["per_delay"][max_delay]["demo_mae"][indices]))
        task_error.append((task, oft_error, cloud_error))
        task_improvement.append((task, oft_error - cloud_error))

    task_error.sort(key=lambda item: item[1] - item[2])
    task_improvement.sort(key=lambda item: item[1])

    def draw_scatter(ax, letter: Optional[str]) -> None:
        for label, result, marker in (
            (OFT_LABEL, oft, "^"),
            (CLOUD_LABEL, cloud, "o"),
        ):
            x = np.concatenate(
                [result["per_delay"][d]["hidden_cosine_distance"] for d in nonzero]
            )
            y = np.concatenate(
                [result["per_delay"][d]["action_drift"] for d in nonzero]
            )
            ax.scatter(
                x,
                y,
                s=13,
                alpha=0.13,
                color=COLORS[label],
                marker=marker,
                edgecolors="none",
            )
            bx, by = _binned_median(x, y)
            ax.plot(
                bx,
                by,
                color=COLORS[label],
                lw=3.0,
                marker=marker,
                ms=6,
                label=label,
            )
        ax.set_xlabel("Backbone hidden-state cosine distance")
        ax.set_ylabel("Normalized action drift")
        ax.set_title(
            _panel_title(letter, "Head attenuation of backbone drift"),
            loc="left",
            fontweight="bold",
        )
        ax.grid(color="#CBD5E1", alpha=0.55, lw=0.8)
        ax.legend()

    def draw_task_error(ax, letter: Optional[str]) -> None:
        y_positions = np.arange(len(task_error))
        for y, (_, oft_value, cloud_value) in zip(y_positions, task_error):
            ax.plot([oft_value, cloud_value], [y, y], color="#CBD5E1", lw=2)
            ax.scatter(
                oft_value,
                y,
                color=COLORS[OFT_LABEL],
                marker="^",
                s=48,
                zorder=3,
            )
            ax.scatter(
                cloud_value,
                y,
                color=COLORS[CLOUD_LABEL],
                marker="o",
                s=48,
                zorder=3,
            )
        ax.axvline(0, color="#94A3B8", lw=0.8)
        ax.set_yticks(
            y_positions,
            [_short_task(item[0]) for item in task_error],
            fontsize=8,
        )
        ax.set_xlabel(f"Demo action MAE at $d={max_delay}$")
        ax.set_title(
            _panel_title(letter, "Per-task delayed action error"),
            loc="left",
            fontweight="bold",
        )
        ax.grid(axis="x", color="#CBD5E1", alpha=0.55, lw=0.8)
        ax.scatter([], [], color=COLORS[OFT_LABEL], marker="^", label=OFT_LABEL)
        ax.scatter([], [], color=COLORS[CLOUD_LABEL], marker="o", label=CLOUD_LABEL)
        ax.legend(loc="upper right", fontsize=8.5)

    def draw_task_gain(ax, letter: Optional[str]) -> None:
        values = np.asarray([item[1] for item in task_improvement])
        colors = np.where(values >= 0, "#2563EB", "#EF4444")
        y_positions = np.arange(len(task_improvement))
        ax.barh(y_positions, values, color=colors, alpha=0.85)
        ax.axvline(0, color="#475569", lw=0.9)
        ax.set_yticks(
            y_positions,
            [_short_task(item[0]) for item in task_improvement],
            fontsize=8,
        )
        ax.set_xlabel("Demo action MAE reduced")
        ax.set_title(
            _panel_title(letter, f"Per-task robustness gain at $d={max_delay}$"),
            loc="left",
            fontweight="bold",
        )
        ax.grid(axis="x", color="#CBD5E1", alpha=0.55, lw=0.8)

    saved_paths: List[Path] = []
    fig, axes = plt.subplots(
        1, 3, figsize=(19.5, 5.8), gridspec_kw={"width_ratios": [1.25, 1.05, 1.0]}
    )
    draw_scatter(axes[0], "a")
    draw_task_error(axes[1], "b")
    draw_task_gain(axes[2], "c")

    fig.suptitle(
        f"Feature-to-Action Delay Geometry on {suite_label}",
        fontsize=15.5,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.13, top=0.88, wspace=0.53)
    saved_paths.extend(_save_figure_pair(fig, output_path))

    single_specs = (
        ("head_attenuation", draw_scatter, (6.6, 4.7), 0.16),
        ("task_error", draw_task_error, (7.0, 5.2), 0.40),
        ("task_gain", draw_task_gain, (7.0, 5.2), 0.40),
    )
    for variant, draw_panel, figsize, left_margin in single_specs:
        fig, ax = plt.subplots(figsize=figsize)
        draw_panel(ax, None)
        fig.subplots_adjust(left=left_margin, right=0.98, bottom=0.15, top=0.91)
        saved_paths.extend(
            _save_figure_pair(
                fig,
                _artifact_variant_path(output_path, "geometry", variant),
            )
        )
    return saved_paths


def plot_action_chunk_rescue(
    output_path: Path,
    delays: Sequence[int],
    oft: Dict,
    cloud: Dict,
    suite_label: str,
) -> Tuple[Dict[str, np.ndarray], List[Path]]:
    """Export the original chunk composite and three standalone heatmaps."""
    _paper_style()
    max_delay = max(delays)
    oft_error = torch.mean(
        torch.abs(oft["predictions"][max_delay] - oft["predictions"][0]), dim=0
    ).numpy()
    cloud_fresh = cloud["predictions"][(0, 0)]
    cloud_current_error = torch.mean(
        torch.abs(cloud["predictions"][(max_delay, 0)] - cloud_fresh), dim=0
    ).numpy()
    suppression = oft_error - cloud_current_error

    fig = plt.figure(figsize=(15.2, 5.0))
    grid = fig.add_gridspec(
        1,
        5,
        width_ratios=[1, 1, 0.055, 1.08, 0.055],
        left=0.055,
        right=0.975,
        bottom=0.17,
        top=0.78,
        wspace=0.28,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 3])]
    shared_colorbar_ax = fig.add_subplot(grid[0, 2])
    suppression_colorbar_ax = fig.add_subplot(grid[0, 4])
    shared_max = float(max(oft_error.max(), cloud_current_error.max()))
    titles = [
        "(a) OFT: stale backbone",
        "(b) CloudEdgeVLA: stale backbone",
    ]
    matrices = [oft_error, cloud_current_error]
    last_image = None
    for ax, title, matrix in zip(axes[:2], titles, matrices):
        last_image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=shared_max, aspect="auto", origin="upper")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(range(ACTION_DIM), ACTION_LABELS, rotation=35, ha="right")
        ax.set_yticks(range(NUM_ACTIONS_CHUNK), [f"t+{i}" for i in range(NUM_ACTIONS_CHUNK)])
        ax.set_xlabel("Action dimension")
    axes[1].set_yticklabels([])
    axes[0].set_ylabel("Action-chunk horizon")
    shared_colorbar = fig.colorbar(last_image, cax=shared_colorbar_ax)
    shared_colorbar.ax.set_title("Drift", fontsize=9.5, pad=8)

    suppression_limit = float(max(abs(suppression.min()), abs(suppression.max()), 1e-5))
    suppression_image = axes[2].imshow(
        suppression,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-suppression_limit, vcenter=0.0, vmax=suppression_limit),
        aspect="auto",
        origin="upper",
    )
    axes[2].set_title("(c) Drift suppressed by training", loc="left", fontweight="bold")
    axes[2].set_xticks(range(ACTION_DIM), ACTION_LABELS, rotation=35, ha="right")
    axes[2].set_yticks(range(NUM_ACTIONS_CHUNK), [f"t+{i}" for i in range(NUM_ACTIONS_CHUNK)])
    axes[2].set_yticklabels([])
    axes[2].set_xlabel("Action dimension")
    fig.colorbar(
        suppression_image,
        cax=suppression_colorbar_ax,
        label="OFT drift − CloudEdgeVLA drift",
    )

    fig.suptitle(
        f"Delay Training Suppresses {max_delay}-Step Staleness across the Action Chunk · {suite_label}",
        fontsize=15.5,
        fontweight="bold",
        y=0.97,
    )
    saved_paths = _save_figure_pair(fig, output_path)

    single_specs = (
        (
            "oft_stale",
            oft_error,
            "OpenVLA-OFT: stale-backbone action drift",
            "magma",
            {"vmin": 0, "vmax": shared_max},
            "Normalized action drift",
        ),
        (
            "cloudedge_stale",
            cloud_current_error,
            "CloudEdgeVLA: stale-backbone action drift",
            "magma",
            {"vmin": 0, "vmax": shared_max},
            "Normalized action drift",
        ),
        (
            "suppression",
            suppression,
            "Action drift suppressed by delay training",
            "RdBu",
            {
                "norm": TwoSlopeNorm(
                    vmin=-suppression_limit,
                    vcenter=0.0,
                    vmax=suppression_limit,
                )
            },
            "OFT drift − CloudEdgeVLA drift",
        ),
    )
    for variant, matrix, title, cmap, image_kwargs, colorbar_label in single_specs:
        fig, ax = plt.subplots(figsize=(6.2, 4.9))
        image = ax.imshow(
            matrix,
            cmap=cmap,
            aspect="auto",
            origin="upper",
            **image_kwargs,
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(range(ACTION_DIM), ACTION_LABELS, rotation=35, ha="right")
        ax.set_yticks(
            range(NUM_ACTIONS_CHUNK),
            [f"t+{i}" for i in range(NUM_ACTIONS_CHUNK)],
        )
        ax.set_xlabel("Action dimension")
        ax.set_ylabel("Action-chunk horizon")
        fig.colorbar(image, ax=ax, label=colorbar_label, fraction=0.05, pad=0.04)
        fig.subplots_adjust(left=0.14, right=0.91, bottom=0.17, top=0.89)
        saved_paths.extend(
            _save_figure_pair(
                fig,
                _artifact_variant_path(output_path, "action_chunk", variant),
            )
        )

    maps = {
        "oft_stale_error": oft_error,
        "cloudedge_stale_error": cloud_current_error,
        "training_suppression": suppression,
    }
    return maps, saved_paths


def _summarize_array(values: np.ndarray) -> Dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "n": int(values.size),
        "values": values.tolist(),
    }


def export_results(
    output_path: Path,
    args,
    samples: List[dict],
    oft: Dict,
    cloud: Dict,
    chunk_maps: Dict[str, np.ndarray],
) -> None:
    export = {
        "config": {
            "checkpoint_oft": args.checkpoint_oft,
            "checkpoint_cloudedge": args.checkpoint_cloudedge,
            "dataset_dir": str(args.dataset_dir),
            "unnorm_key": args.unnorm_key,
            "suite_label": args.suite_label,
            "delays": args.delays,
            "max_tasks": args.max_tasks,
            "episodes_per_task": args.episodes_per_task,
            "max_frames_per_episode": args.max_frames_per_episode,
            "samples_per_task": args.samples_per_task,
            "num_samples": len(samples),
            "device": args.device,
            "proprio_note": "Cloud features use delayed images with current normalized proprio, matching training/evaluation.",
        },
        "tasks": list(dict.fromkeys(sample["task"] for sample in samples)),
        "samples": [
            {"task": sample["task"], "episode_index": sample["episode_index"], "t": sample["t"]}
            for sample in samples
        ],
        "models": {},
        "cloudedge": {
            "drift_surface": cloud["drift_surface"].tolist(),
            "surface_rows_cloud_delay": args.delays,
            "surface_columns_edge_delay": args.delays,
        },
        "action_chunk_maps_at_max_delay": {name: values.tolist() for name, values in chunk_maps.items()},
    }
    for label, result in ((OFT_LABEL, oft), (CLOUD_LABEL, cloud)):
        export["models"][label] = {}
        for delay in args.delays:
            export["models"][label][str(delay)] = {
                name: _summarize_array(values)
                for name, values in result["per_delay"][delay].items()
            }
    output_path.write_text(json.dumps(export, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_oft", required=True)
    parser.add_argument("--checkpoint_cloudedge", required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--unnorm_key", default="libero_10_no_noops")
    parser.add_argument("--suite_label", default="LIBERO-10")
    parser.add_argument("--delays", type=int, nargs="+", default=DEFAULT_DELAYS)
    parser.add_argument("--max_tasks", type=int, default=10)
    parser.add_argument("--episodes_per_task", type=int, default=1)
    parser.add_argument("--max_frames_per_episode", type=int, default=64)
    parser.add_argument("--samples_per_task", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--prefix", default="fig_backbone_head_delay_libero10")
    parser.add_argument(
        "--stage",
        choices=("all", "oft", "cloud", "render"),
        default="all",
        help="Run both models, one cacheable model stage, or render cached stages.",
    )
    parser.add_argument(
        "--stage_cache_dir",
        type=Path,
        default=Path("logs/backbone_head_delay_cache"),
    )
    args = parser.parse_args()

    if sorted(set(args.delays)) != args.delays or args.delays[0] != 0:
        raise ValueError("--delays must be sorted, unique, and start with 0.")
    attr.configure_runtime(args.device)

    stats_path = Path(args.checkpoint_cloudedge) / "dataset_statistics.json"
    dataset_stats = json.loads(stats_path.read_text())[args.unnorm_key]
    episodes, samples = load_demonstration_samples(
        args.dataset_dir,
        dataset_stats["action"],
        args.delays,
        args.max_tasks,
        args.episodes_per_task,
        args.max_frames_per_episode,
        args.samples_per_task,
    )
    ground_truth = torch.from_numpy(np.stack([sample["action_chunk"] for sample in samples])).float()

    args.stage_cache_dir.mkdir(parents=True, exist_ok=True)
    oft_cache_path = args.stage_cache_dir / f"{args.prefix}_oft.pt"
    cloud_cache_path = args.stage_cache_dir / f"{args.prefix}_cloud.pt"

    oft_results = None
    if args.stage in ("all", "oft"):
        oft_head, oft_hidden, _ = extract_model_features(
            OFT_LABEL,
            args.checkpoint_oft,
            episodes,
            samples,
            args.delays,
            args.lora_rank,
            args.num_views,
            args.unnorm_key,
            use_vision_action_head=False,
        )
        oft_results = evaluate_oft(oft_head, oft_hidden, ground_truth, args.delays)
        torch.save(oft_results, oft_cache_path)
        print(f"[CACHE] {oft_cache_path}", flush=True)
        del oft_head, oft_hidden
        gc.collect()
        torch.cuda.empty_cache()
        if args.stage == "oft":
            return
    elif args.stage == "render":
        oft_results = torch.load(oft_cache_path, map_location="cpu")

    cloud_results = None
    if args.stage in ("all", "cloud"):
        cloud_head, cloud_hidden, cloud_vision = extract_model_features(
            CLOUD_LABEL,
            args.checkpoint_cloudedge,
            episodes,
            samples,
            args.delays,
            args.lora_rank,
            args.num_views,
            args.unnorm_key,
            use_vision_action_head=True,
        )
        cloud_results = evaluate_cloudedge(
            cloud_head, cloud_hidden, cloud_vision, ground_truth, args.delays
        )
        torch.save(cloud_results, cloud_cache_path)
        print(f"[CACHE] {cloud_cache_path}", flush=True)
        del cloud_head, cloud_hidden, cloud_vision
        gc.collect()
        torch.cuda.empty_cache()
        if args.stage == "cloud":
            return
    elif args.stage == "render":
        cloud_results = torch.load(cloud_cache_path, map_location="cpu")

    overview_path = args.output_dir / f"{args.prefix}_overview.png"
    decomposition_path = args.output_dir / f"{args.prefix}_decomposition.png"
    geometry_path = args.output_dir / f"{args.prefix}_geometry.png"
    chunk_path = args.output_dir / f"{args.prefix}_action_chunk.png"
    data_path = args.output_dir / f"{args.prefix}_data.json"
    plot_overview(
        overview_path, args.delays, oft_results, cloud_results, args.suite_label
    )
    decomposition_paths = plot_backbone_head_decomposition(
        decomposition_path,
        args.delays,
        samples,
        oft_results,
        cloud_results,
        args.suite_label,
    )
    geometry_paths = plot_geometry_and_tasks(
        geometry_path,
        args.delays,
        samples,
        oft_results,
        cloud_results,
        args.suite_label,
    )
    chunk_maps, chunk_paths = plot_action_chunk_rescue(
        chunk_path, args.delays, oft_results, cloud_results, args.suite_label
    )
    export_results(data_path, args, samples, oft_results, cloud_results, chunk_maps)

    max_delay = max(args.delays)
    oft_excess = np.mean(oft_results["per_delay"][max_delay]["excess_demo_mae"])
    cloud_excess = np.mean(cloud_results["per_delay"][max_delay]["excess_demo_mae"])
    rescue = np.mean(cloud_results["per_delay"][max_delay]["edge_rescue_fraction"])
    alignment = np.mean(cloud_results["per_delay"][max_delay]["correction_cosine"])
    print(
        f"[SUMMARY] d={max_delay}: OFT excess demo MAE={oft_excess:.5f}, "
        f"CloudEdge={cloud_excess:.5f}, edge rescue={rescue:.3f}, "
        f"correction alignment={alignment:.3f}",
        flush=True,
    )
    saved_paths = [
        overview_path,
        *decomposition_paths,
        *geometry_paths,
        *chunk_paths,
        data_path,
    ]
    for path in saved_paths:
        print(f"[SAVED] {path}", flush=True)


if __name__ == "__main__":
    main()
