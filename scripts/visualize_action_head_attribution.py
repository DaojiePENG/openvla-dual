"""
visualize_action_head_attribution.py

Computes and visualizes the effective input attribution α(d) for the VisionActionHead
under varying delay conditions.

For a concat + MLP architecture (no explicit attention), we measure how much the
action prediction relies on cloud planning features vs. edge vision features using
two complementary methods:

  Method A (default): Gradient × input attribution in the fusion space
      α(d) = RMS((∂â/∂h)⊙h) / [RMS((∂â/∂h)⊙h) + RMS((∂â/∂z)⊙z)]

  Method B: Symmetric input ablation
      α(d) = Δh / (Δh + Δz), where
      Δh = mean|â(h,z) - â(0,z)| and Δz = mean|â(h,z) - â(h,0)|

Usage (single suite, gradient method):
    python scripts/visualize_action_head_attribution.py \
        --pretrained_checkpoint /path/to/checkpoint \
        --task_suite_name libero_spatial \
        --method gradient \
        --output_path results/fig6_attribution.pdf

Usage (all 4 suites, grouped bar chart):
    python scripts/visualize_action_head_attribution.py \
        --pretrained_checkpoint /path/to/checkpoint \
        --task_suite_name all \
        --method gradient \
        --output_path results/fig6_attribution_all_suites.pdf
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as _TVF

# Importing prismatic.vla pulls in TensorFlow through the RLDS package. Disable
# TensorFlow GPU discovery before that import so it cannot reserve H800 memory.
import tensorflow as _tf
try:
    _tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

# Monkey-patch to_tensor to work with numpy 1.26 + torch 2.3 ABI mismatch
_orig_to_tensor = _TVF.to_tensor
def _patched_to_tensor(pic):
    import numpy as np
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    nptype = mode_to_nptype.get(pic.mode, np.uint8)
    img = np.asarray(pic, dtype=nptype)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    img = img.transpose((2, 0, 1))
    t = torch.tensor(img.tolist(), dtype=torch.float32 if nptype == np.uint8 else None)
    if nptype == np.uint8:
        t = t.div_(255)
    return t
_TVF.to_tensor = _patched_to_tensor

# Append project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from prismatic.models.action_heads import VisionActionHead
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

TASK_SUITE_NAMES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
TASK_SUITE_LABELS = {
    "libero_spatial":  "Spatial",
    "libero_object":   "Object",
    "libero_goal":     "Goal",
    "libero_10":       "Long",
}

DEFAULT_DELAYS = [0, 1, 3, 5, 8, 10, 15, 20]


def configure_runtime(device: str) -> None:
    """Select the torch device and prevent TensorFlow from reserving GPU VRAM."""
    global DEVICE
    DEVICE = torch.device(device)
    if DEVICE.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        torch.cuda.set_device(DEVICE)

    try:
        _tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass

    import experiments.robot.openvla_utils as openvla_utils
    openvla_utils.DEVICE = DEVICE
    print(f"[INFO] Torch device: {DEVICE}")


# ---------------------------------------------------------------------------
# Forward-hook helper to capture the LLM hidden states that flow into the
# VisionActionHead during a normal VLA forward pass.
# ---------------------------------------------------------------------------
class _HiddenStateCapture:
    """
    Context manager that registers a forward hook on ``action_head.predict_action``
    (the Python method, not an nn.Module) to intercept the ``llm_hidden_states``
    argument each time the action head is called.

    Usage::

        with _HiddenStateCapture(action_head) as cap:
            vla.predict_action(input_ids=..., pixel_values=..., action_head=action_head, ...)
        h_captured = cap.hidden_states  # tensor on DEVICE, detached
    """

    def __init__(self, action_head: VisionActionHead):
        self._action_head = action_head
        self._original_predict = action_head.__class__.predict_action
        self.hidden_states: Optional[torch.Tensor] = None

    def __enter__(self):
        _outer = self  # capture for closure

        def _intercepting_predict(self_head, *args, **kwargs):
            # First positional arg is always llm_hidden_states for both
            # L1RegressionActionHead(self, llm_hidden_states) and
            # VisionActionHead(self, llm_hidden_states, pixel_values=None)
            _outer.hidden_states = args[0].detach().clone()
            # Run the real action head so the VLA can finish its forward pass
            return _outer._original_predict(self_head, *args, **kwargs)

        # Monkey-patch at class level (affects all instances, but we only have one)
        self._action_head.__class__.predict_action = _intercepting_predict
        return self

    def __exit__(self, *exc):
        self._action_head.__class__.predict_action = self._original_predict
        return False


# ---------------------------------------------------------------------------
# Model loading (reuses existing infrastructure)
# ---------------------------------------------------------------------------
def load_model(
    pretrained_checkpoint: str,
    lora_rank: int = 32,
    action_head_vision_encoder: str = "siglip-base",
    num_views: int = 2,
    use_vision_action_head: bool = True,
) -> Tuple[nn.Module, nn.Module, any, "argparse.Namespace"]:
    """
    Returns:
        vla, action_head, processor, cfg
    """
    # Build minimal config namespace
    import types
    cfg = types.SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=pretrained_checkpoint,
        lora_rank=lora_rank,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=num_views,
        use_proprio=True,
        center_crop=True,
        load_in_8bit=False,
        load_in_4bit=False,
        use_vision_action_head=use_vision_action_head,
        action_head_vision_encoder=action_head_vision_encoder,
        # The checkpoint contains the complete VisionActionHead state, including
        # its frozen SigLIP encoder. Avoid a redundant Hub download before loading it.
        action_head_vision_pretrained=False,
        freeze_action_head_vision=True,
        action_head_num_views=num_views,
    )

    from experiments.robot.openvla_utils import get_action_head, get_processor, get_vla
    from experiments.robot.robot_utils import get_model

    checkpoint_path = Path(pretrained_checkpoint)
    adapter_dir = checkpoint_path / "lora_adapter"
    adapter_only_checkpoint = (
        checkpoint_path.is_dir()
        and not (checkpoint_path / "config.json").is_file()
        and (adapter_dir / "adapter_config.json").is_file()
    )

    if adapter_only_checkpoint:
        # Pruned training checkpoints retain only the LoRA adapter and auxiliary heads. Reconstruct the VLA from the
        # clean base recorded by PEFT instead of requiring the large merged model shards to be kept for every step.
        adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
        base_model_path = adapter_config["base_model_name_or_path"]
        model_cfg = types.SimpleNamespace(**vars(cfg))
        model_cfg.pretrained_checkpoint = base_model_path
        print(f"[INFO] Reconstructing pruned checkpoint from base `{base_model_path}` + adapter `{adapter_dir}`")
        vla = get_vla(model_cfg)
        from peft import PeftModel

        vla = PeftModel.from_pretrained(vla, adapter_dir).merge_and_unload()
        processor = get_processor(model_cfg)
    else:
        vla = get_vla(cfg)
        processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)

    # Cast action_head to bfloat16 to match VLA dtype
    action_head = action_head.to(torch.bfloat16)

    # Monkey-patch _unnormalize_actions to use pure torch (bypass numpy ABI issues)
    def _torch_unnormalize_actions(self, normalized_actions, unnorm_key):
        action_norm_stats = self.get_action_stats(unnorm_key)
        device = normalized_actions.device if hasattr(normalized_actions, 'device') else torch.device("cuda:0")
        na = torch.tensor(normalized_actions.tolist(), dtype=torch.float32, device=device) if hasattr(normalized_actions, 'tolist') else torch.tensor(normalized_actions, dtype=torch.float32, device=device)
        action_low = torch.as_tensor(list(action_norm_stats["q01"]), dtype=torch.float32, device=device)
        action_high = torch.as_tensor(list(action_norm_stats["q99"]), dtype=torch.float32, device=device)
        mask = torch.as_tensor(list(action_norm_stats.get("mask", [1]*len(action_low))), dtype=torch.bool, device=device)
        actions = torch.where(
            mask,
            0.5 * (na + 1) * (action_high - action_low + 1e-8) + action_low,
            na,
        )
        return actions
    vla._unnormalize_actions = types.MethodType(_torch_unnormalize_actions, vla)

    return vla, action_head, processor, cfg


# ---------------------------------------------------------------------------
# Observation collection from LIBERO
# ---------------------------------------------------------------------------
def collect_observations(
    task_suite_name: str,
    num_episodes: int = 3,
    num_frames: int = 40,
    seed: int = 42,
    max_tasks: int = 5,
    rollout_mode: str = "random",
) -> Dict[str, List[dict]]:
    """
    Collect raw observations from LIBERO environments.

    Returns:
        dict  task_description -> list of obs dicts
            Each dict: full_image (H,W,3), wrist_image (H,W,3), state (8,), task_label
    """
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action, get_libero_env, get_libero_image,
        get_libero_wrist_image, quat2axisangle,
    )
    from experiments.robot.openvla_utils import resize_image_for_policy

    rng = np.random.default_rng(seed)

    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    all_obs = {}

    for task_id in range(min(task_suite.n_tasks, max_tasks)):
        task = task_suite.get_task(task_id)
        env, task_desc = get_libero_env(task, "openvla", resolution=256)
        init_states = task_suite.get_task_init_states(task_id)

        task_obs = []
        for ep in range(min(num_episodes, len(init_states))):
            env.reset()
            obs = env.set_init_state(init_states[ep])
            random_action = np.zeros(ACTION_DIM, dtype=np.float32)

            for step in range(num_frames):
                if step < 10:  # let objects stabilise
                    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                    continue

                img  = get_libero_image(obs)
                wimg = get_libero_wrist_image(obs)
                img_r  = resize_image_for_policy(img,  224)
                wimg_r = resize_image_for_policy(wimg, 224)
                state  = np.concatenate((
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ))

                task_obs.append({
                    "full_image":   img_r,
                    "wrist_image":  wimg_r,
                    "state":        state,
                    "task_label":   task_desc,
                    "task_id":      task_id,
                    "episode_id":   ep,
                    "timestep":     step,
                })

                if rollout_mode == "random":
                    target = rng.normal(0.0, 0.35, size=ACTION_DIM)
                    target[3:6] *= 0.6
                    random_action = np.clip(0.75 * random_action + 0.25 * target, -1.0, 1.0)
                    random_action[-1] = -1.0 if ((step - 10) // 6) % 2 == 0 else 1.0
                    action = random_action
                else:
                    action = get_libero_dummy_action("openvla")
                obs, _, done, _ = env.step(action)
                if done:
                    break
        all_obs[task_desc] = task_obs
        env.close()

    return all_obs


# ---------------------------------------------------------------------------
# Build pixel_values from raw observation (reuses VLA processor)
# ---------------------------------------------------------------------------
def obs_to_pixel_values(
    obs: dict,
    processor,
    cfg,
) -> torch.Tensor:
    """
    Convert a raw obs dict into the pixel_values tensor expected by the VLA.

    Returns:
        pixel_values: (1, C, H, W) tensor on CPU (will be moved to DEVICE later)
    """
    from PIL import Image
    from experiments.robot.openvla_utils import center_crop_image

    images = [obs["full_image"]]
    if cfg.num_images_in_input > 1 and "wrist_image" in obs:
        images.append(obs["wrist_image"])

    processed = []
    for img in images:
        pil = Image.fromarray(img).convert("RGB")
        if cfg.center_crop:
            pil = center_crop_image(pil)
        processed.append(pil)

    primary = processed.pop(0)
    prompt = f"In: What action should the robot take to {obs['task_label'].lower()}?\nOut:"
    inputs = processor(prompt, primary)
    pv = inputs["pixel_values"]  # (1, C, H, W)

    if processed:
        wrist_pvs = [processor(prompt, wimg)["pixel_values"] for wimg in processed]
        pv = torch.cat([pv] + wrist_pvs, dim=1)

    return pv  # (1, num_views*C, H, W) on CPU


# ---------------------------------------------------------------------------
# Attribution computation
# ---------------------------------------------------------------------------
def compute_gradient_attribution(
    action_head: VisionActionHead,
    h_stale: torch.Tensor,
    vision_features_current: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    Gradient-times-input effective attribution in the fusion space:

        α(d) = RMS((∂â/∂h)⊙h) / [RMS((∂â/∂h)⊙h) + RMS((∂â/∂z)⊙z)]

    High α means the planning branch has the larger local saliency score;
    low α means the vision branch has the larger score. No delay trend is
    assumed by the definition.

    Returns:
        (alpha, planning_score, vision_score)
    """
    was_training = action_head.training
    action_head.eval()

    B = h_stale.shape[0]

    # --- LLM path: reshape to preserve per-action-dimension information ---
    # (B, chunk_len * action_dim, hidden_dim) -> (B, chunk_len, action_dim * hidden_dim)
    # h_stale may be an inference tensor (captured under torch.inference_mode),
    # so we must clone into a fresh tensor before requiring grad.
    llm_features = h_stale.reshape(B, NUM_ACTIONS_CHUNK, -1).clone().detach().requires_grad_(True)

    # Compare both operands at the fusion-MLP input. Detaching after the vision
    # projector avoids mixing the projector Jacobian and raw encoder scale into
    # the branch comparison.
    with torch.no_grad():
        vision_proj = action_head.vision_projector(vision_features_current)
    vision_proj = (
        vision_proj.unsqueeze(1)
        .expand(-1, NUM_ACTIONS_CHUNK, -1)
        .clone()
        .detach()
        .requires_grad_(True)
    )

    # --- Fusion + prediction ---
    fused = torch.cat([llm_features, vision_proj], dim=-1)            # (B, chunk, action_dim*D + D)
    actions = action_head.fusion_mlp(fused)                           # (B, chunk, act_dim)

    # Scalar target for backward
    action_head.zero_grad(set_to_none=True)
    actions.abs().sum().backward()

    g_h = llm_features.grad
    g_z = vision_proj.grad

    planning_score = (
        torch.sqrt(torch.mean((g_h.float() * llm_features.detach().float()) ** 2)).item()
        if g_h is not None else 0.0
    )
    vision_score = (
        torch.sqrt(torch.mean((g_z.float() * vision_proj.detach().float()) ** 2)).item()
        if g_z is not None else 0.0
    )

    alpha = planning_score / (planning_score + vision_score + 1e-8)

    action_head.train(was_training)
    return alpha, planning_score, vision_score


def compute_ablation_attribution(
    action_head: VisionActionHead,
    h_stale: torch.Tensor,
    vision_features_current: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    Symmetric input-ablation effective attribution:

        α(d) = Δh / (Δh + Δz)

    where Δh measures the output change after removing planning features and
    Δz measures the output change after removing vision features.

    Returns:
        (alpha, planning_effect, vision_effect)
    """
    was_training = action_head.training
    action_head.eval()
    with torch.no_grad():
        batch_size = h_stale.shape[0]
        llm_features = h_stale.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        vision_proj = action_head.vision_projector(vision_features_current)
        vision_proj = vision_proj.unsqueeze(1).expand(-1, NUM_ACTIONS_CHUNK, -1)
        full = action_head.fusion_mlp(torch.cat([llm_features, vision_proj], dim=-1))
        no_planning = action_head.fusion_mlp(
            torch.cat([torch.zeros_like(llm_features), vision_proj], dim=-1)
        )
        no_vision = action_head.fusion_mlp(
            torch.cat([llm_features, torch.zeros_like(vision_proj)], dim=-1)
        )
        planning_effect = (full - no_planning).abs().mean().item()
        vision_effect = (full - no_vision).abs().mean().item()
        alpha = planning_effect / (planning_effect + vision_effect + 1e-8)
    action_head.train(was_training)
    return alpha, planning_effect, vision_effect


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_attribution(
    pretrained_checkpoint: str,
    task_suite_name: str = "libero_spatial",
    delays: Optional[List[int]] = None,
    method: str = "gradient",
    num_episodes: int = 3,
    num_frames: int = 40,
    max_samples_per_delay: int = 30,
    lora_rank: int = 32,
    action_head_vision_encoder: str = "siglip-base",
    num_views: int = 2,
    max_tasks: int = 5,
    rollout_mode: str = "random",
) -> Tuple[Dict[int, List[float]], Dict[str, any]]:
    """
    Run the full attribution sweep over delays for one task suite.

    Returns:
        results:  dict  delay -> list of α values
        raw_info: dict with per-sample branch scores and per-task breakdown
                  for saving to JSON
    """
    if delays is None:
        delays = DEFAULT_DELAYS

    print(f"[INFO] Loading model from {pretrained_checkpoint} ...")
    vla, action_head, processor, cfg = load_model(
        pretrained_checkpoint, lora_rank, action_head_vision_encoder, num_views,
    )

    print(f"[INFO] Collecting LIBERO observations ({task_suite_name}) ...")
    all_obs = collect_observations(
        task_suite_name, num_episodes, num_frames, max_tasks=max_tasks,
        rollout_mode=rollout_mode,
    )

    # Pre-build a simple cfg namespace for obs_to_pixel_values
    img_cfg = types.SimpleNamespace(num_images_in_input=num_views, center_crop=cfg.center_crop)

    unnorm_key = f"{task_suite_name}_no_noops"

    methods = ["gradient", "ablation"] if method == "both" else [method]
    results_by_method = {name: defaultdict(list) for name in methods}
    # Per-task breakdown and branch-score details for raw data export.
    per_task_by_method: Dict[str, Dict[str, Dict[int, List[float]]]] = {
        name: {} for name in methods
    }
    branch_scores_by_method: Dict[str, Dict[int, List[Dict[str, float]]]] = {
        name: defaultdict(list) for name in methods
    }

    from transformers import LlamaTokenizerFast
    tokenizer = processor.tokenizer

    for task_desc, obs_list in all_obs.items():
        print(f"  [TASK] {task_desc}  ({len(obs_list)} frames)")
        n = len(obs_list)
        for attr_method in methods:
            per_task_by_method[attr_method][task_desc] = defaultdict(list)

        # Precompute pixels, backbone hidden states, and edge-vision features once
        # per frame. The old loop reran the 7B backbone for every delay.
        pvs = [obs_to_pixel_values(o, processor, img_cfg) for o in obs_list]
        hidden_states = []
        vision_features = []
        for frame_idx, (obs, pv_cpu) in enumerate(zip(obs_list, pvs)):
            pv = pv_cpu.to(DEVICE, dtype=torch.bfloat16)
            prompt = f"In: What action should the robot take to {obs['task_label'].lower()}?\nOut:"
            input_ids = tokenizer(prompt, truncation=True, return_tensors="pt").input_ids.to(DEVICE)
            with _HiddenStateCapture(action_head) as cap:
                with torch.inference_mode():
                    vla.predict_action(
                        input_ids=input_ids,
                        pixel_values=pv,
                        attention_mask=torch.ones_like(input_ids),
                        unnorm_key=unnorm_key,
                        action_head=action_head,
                    )
                    z = action_head.encode_vision(pv)
            if cap.hidden_states is None:
                raise RuntimeError(f"Failed to capture hidden state for frame {frame_idx}")
            hidden_states.append(cap.hidden_states)
            vision_features.append(z.detach().clone())

        for delay in delays:
            eligible = [
                t for t in range(delay, n)
                if obs_list[t]["episode_id"] == obs_list[t - delay]["episode_id"]
            ]
            if len(eligible) > max_samples_per_delay:
                take = np.linspace(0, len(eligible) - 1, max_samples_per_delay, dtype=int)
                eligible = [eligible[i] for i in take]

            for t in eligible:
                h_stale = hidden_states[t - delay]
                z_current = vision_features[t]

                # --- Compute one or both attribution variants ---
                for attr_method in methods:
                    if attr_method == "gradient":
                        alpha, planning_score, vision_score = compute_gradient_attribution(
                            action_head, h_stale, z_current
                        )
                    else:
                        alpha, planning_score, vision_score = compute_ablation_attribution(
                            action_head, h_stale, z_current
                        )

                    branch_scores_by_method[attr_method][delay].append({
                        "alpha": alpha,
                        "planning_score": planning_score,
                        "vision_score": vision_score,
                    })

                    results_by_method[attr_method][delay].append(alpha)
                    per_task_by_method[attr_method][task_desc][delay].append(alpha)

    # Convert defaultdicts to regular dicts for JSON serialization.
    raw_by_method = {}
    for attr_method in methods:
        raw = {
            "per_task": {
                task: dict(delay_values)
                for task, delay_values in per_task_by_method[attr_method].items()
            },
            "branch_scores": {
                int(d): values
                for d, values in branch_scores_by_method[attr_method].items()
            },
        }
        raw_by_method[attr_method] = raw

    if method == "both":
        return ({name: dict(values) for name, values in results_by_method.items()}, raw_by_method)
    return dict(results_by_method[method]), raw_by_method[method]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curves(
    results: Dict[int, List[float]],
    output_path: str,
    method: str = "gradient",
    title_suffix: str = "",
) -> None:
    """Line plot: α(d) and 1-α(d) vs. delay with ±1 std shading."""
    delays = sorted(results.keys())
    means  = [np.mean(results[d]) for d in delays]
    stds   = [np.std(results[d])  for d in delays]

    fig, ax = plt.subplots(figsize=(8, 5))

    if method == "gradient":
        planning_label = r"Planning saliency $\alpha_{\mathrm{grad}}(d)$"
        vision_label = r"Vision saliency $1 - \alpha_{\mathrm{grad}}(d)$"
        title = "Local Gradient × Input Saliency vs. Delay"
    else:
        planning_label = r"Planning ablation share $\alpha_{\mathrm{abl}}(d)$"
        vision_label = r"Vision ablation share $1 - \alpha_{\mathrm{abl}}(d)$"
        title = "Zero-Baseline Branch Ablation vs. Delay"

    # Planning attribution
    ax.plot(delays, means, "o-", color="#2563EB", lw=2.5, ms=8,
            label=planning_label, zorder=3)
    ax.fill_between(delays,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.20, color="#2563EB", zorder=2)

    # Vision attribution (complement)
    vmeans = [1.0 - m for m in means]
    ax.plot(delays, vmeans, "s--", color="#DC2626", lw=2.0, ms=7,
            label=vision_label, zorder=3)
    ax.fill_between(delays,
                    [m - s for m, s in zip(vmeans, stds)],
                    [m + s for m, s in zip(vmeans, stds)],
                    alpha=0.15, color="#DC2626", zorder=2)

    ax.set_xlabel("Delay $d$ (environment steps)", fontsize=13)
    if method == "gradient":
        ax.set_ylabel(
            r"$\alpha_{\mathrm{grad}} = S_h\,/[S_h + S_z]$"
            "\n" r"$S_x=\operatorname{RMS}(\nabla_x\hat{a}\odot x)$",
            fontsize=12,
        )
    else:
        ax.set_ylabel(
            r"$\alpha_{\mathrm{abl}} = \Delta_h\,/\,(\Delta_h + \Delta_z)$",
            fontsize=12,
        )

    ax.set_title(f"{title}{title_suffix}", fontsize=14)
    ax.legend(fontsize=12, loc="center right")
    all_values = [v for values in results.values() for v in values]
    lower = min(-0.05, min(all_values) - 0.05) if all_values else -0.05
    upper = max(1.05, max(all_values) + 0.05) if all_values else 1.05
    ax.set_ylim(lower, upper)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if Path(output_path).suffix.lower() != ".png":
        png_path = str(Path(output_path).with_suffix(".png"))
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved → {png_path}")
    print(f"[INFO] Saved → {output_path}")
    plt.close(fig)


def plot_grouped_bars(
    all_results: Dict[str, Dict[int, List[float]]],
    output_path: str,
    method: str = "gradient",
) -> None:
    """Grouped bar chart: one group per delay, bars coloured by task suite."""
    suites  = list(all_results.keys())
    delays  = sorted({d for r in all_results.values() for d in r})
    n_suites = len(suites)
    bw = 0.8 / n_suites
    colors = ["#2563EB", "#DC2626", "#16A34A", "#F59E0B"]

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, suite in enumerate(suites):
        r = all_results[suite]
        means = [np.mean(r.get(d, [0])) for d in delays]
        stds  = [np.std(r.get(d, [0]))  for d in delays]
        x = np.arange(len(delays)) + i * bw
        ax.bar(x, means, bw, yerr=stds,
               label=TASK_SUITE_LABELS.get(suite, suite),
               color=colors[i % len(colors)], alpha=0.85, capsize=3)

    ax.set_xlabel("Delay $d$ (environment steps)", fontsize=13)
    ax.set_ylabel(r"$\alpha_{\mathrm{eff}}(d)$ (planning attribution)", fontsize=12)
    ax.set_title("Effective Planning Attribution by Task Suite and Delay", fontsize=14)
    ax.set_xticks(np.arange(len(delays)) + bw * (n_suites - 1) / 2)
    ax.set_xticklabels(delays)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"[INFO] Saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Raw data export
# ---------------------------------------------------------------------------
def save_raw_data(
    output_path: str,
    all_results: Dict[str, Dict[int, List[float]]],
    all_raw_info: Dict[str, Dict[str, any]],
    args: argparse.Namespace,
) -> None:
    """
    Save raw attribution data and configuration alongside the figure.

    Saves a JSON file with the same stem as the PDF output, e.g.
        results/fig6_attribution_goal.pdf  →  results/fig6_attribution_goal_data.json
    """
    stem, _ = os.path.splitext(output_path)
    json_path = stem + "_data.json"

    export: Dict[str, any] = {
        "config": {
            "pretrained_checkpoint": args.pretrained_checkpoint,
            "task_suite_name": args.task_suite_name,
            "method": args.method,
            "delays": args.delays if args.delays is not None else DEFAULT_DELAYS,
            "num_episodes": args.num_episodes,
            "num_frames": args.num_frames,
            "max_samples_per_delay": args.max_samples_per_delay,
            "lora_rank": args.lora_rank,
            "action_head_vision_encoder": args.action_head_vision_encoder,
            "num_views": args.num_views,
            "max_tasks": args.max_tasks,
            "rollout_mode": args.rollout_mode,
            "device": args.device,
            "attribution_definition": (
                "fusion_grad_x_input_rms"
                if args.method == "gradient"
                else "symmetric_branch_ablation"
            ),
        },
        "suites": {},
    }

    for suite_key, results in all_results.items():
        suite_data: Dict[str, any] = {
            "summary": {},
            "per_task": {},
        }
        # Per-delay summary
        for d in sorted(results):
            v = results[d]
            suite_data["summary"][str(d)] = {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "n": len(v),
                "values": v,
            }
        # Per-task breakdown (if available)
        raw = all_raw_info.get(suite_key, {})
        if "per_task" in raw:
            for task, task_res in raw["per_task"].items():
                suite_data["per_task"][task] = {
                    str(d): {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "n": len(vals),
                        "values": vals,
                    }
                    for d, vals in task_res.items()
                }
        if "branch_scores" in raw:
            suite_data["branch_scores"] = {}
            for d, entries in raw["branch_scores"].items():
                suite_data["branch_scores"][str(d)] = {
                    "mean_alpha": float(np.mean([e["alpha"] for e in entries])),
                    "mean_planning_score": float(np.mean([
                        e["planning_score"] for e in entries
                    ])),
                    "mean_vision_score": float(np.mean([
                        e["vision_score"] for e in entries
                    ])),
                    "per_sample": entries,
                }
        export["suites"][suite_key] = suite_data

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(
            export,
            f,
            indent=2,
            allow_nan=False,
            default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
        )
    print(f"[INFO] Raw data saved → {json_path}")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(results: Dict[int, List[float]], label: str = "") -> None:
    if label:
        print(f"\n{label}:")
    for d in sorted(results):
        v = results[d]
        print(f"  delay={d:3d}  α = {np.mean(v):.4f} ± {np.std(v):.4f}  (n={len(v)})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize effective input attribution α(d) for VisionActionHead",
    )
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial",
                        choices=TASK_SUITE_NAMES + ["all"])
    parser.add_argument("--method", type=str, default="gradient",
                        choices=["gradient", "ablation", "both"])
    parser.add_argument("--delays", type=int, nargs="+", default=None)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--num_frames", type=int, default=40)
    parser.add_argument("--max_samples_per_delay", type=int, default=30)
    parser.add_argument("--output_path", type=str, default="results/fig6_attribution.pdf")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--action_head_vision_encoder", type=str, default="siglip-base")
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--max_tasks", type=int, default=5)
    parser.add_argument("--rollout_mode", choices=["random", "noop"], default="random")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    configure_runtime(args.device)

    delays = args.delays if args.delays is not None else DEFAULT_DELAYS

    if args.method == "both" and args.task_suite_name == "all":
        parser.error("--method both currently supports one task suite at a time")

    if args.task_suite_name == "all":
        all_res = {}
        all_raw_info = {}
        for suite in TASK_SUITE_NAMES:
            print(f"\n{'='*60}\n[SUITE] {suite}\n{'='*60}")
            results, raw_info = run_attribution(
                pretrained_checkpoint=args.pretrained_checkpoint,
                task_suite_name=suite,
                delays=delays,
                method=args.method,
                num_episodes=args.num_episodes,
                num_frames=args.num_frames,
                max_samples_per_delay=args.max_samples_per_delay,
                lora_rank=args.lora_rank,
                action_head_vision_encoder=args.action_head_vision_encoder,
                num_views=args.num_views,
                max_tasks=args.max_tasks,
                rollout_mode=args.rollout_mode,
            )
            all_res[suite] = results
            all_raw_info[suite] = raw_info
        plot_grouped_bars(all_res, args.output_path, args.method)
        save_raw_data(args.output_path, all_res, all_raw_info, args)
        for suite, r in all_res.items():
            print_summary(r, TASK_SUITE_LABELS.get(suite, suite))
    else:
        results, raw_info = run_attribution(
            pretrained_checkpoint=args.pretrained_checkpoint,
            task_suite_name=args.task_suite_name,
            delays=delays,
            method=args.method,
            num_episodes=args.num_episodes,
            num_frames=args.num_frames,
            max_samples_per_delay=args.max_samples_per_delay,
            lora_rank=args.lora_rank,
            action_head_vision_encoder=args.action_head_vision_encoder,
            num_views=args.num_views,
            max_tasks=args.max_tasks,
            rollout_mode=args.rollout_mode,
        )
        suite_label = TASK_SUITE_LABELS.get(args.task_suite_name, args.task_suite_name)
        if args.method == "both":
            output = Path(args.output_path)
            for attr_method in ("gradient", "ablation"):
                method_path = str(output.with_name(f"{output.stem}_{attr_method}{output.suffix}"))
                method_args = argparse.Namespace(**vars(args))
                method_args.method = attr_method
                plot_curves(results[attr_method], method_path, attr_method,
                            title_suffix=f" ({suite_label})")
                save_raw_data(method_path,
                              {suite_label: results[attr_method]},
                              {suite_label: raw_info[attr_method]},
                              method_args)
                print_summary(results[attr_method], attr_method.capitalize())
        else:
            plot_curves(results, args.output_path, args.method,
                        title_suffix=f" ({suite_label})")
            save_raw_data(args.output_path,
                          {suite_label: results},
                          {suite_label: raw_info},
                          args)
            print_summary(results)


# Avoid name collision with the types import inside load_model
import types  # noqa: E402  (needed at module level for img_cfg in run_attribution)


if __name__ == "__main__":
    main()
