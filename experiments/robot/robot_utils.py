"""Utils for evaluating robot policies in various environments."""

import os
import random
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

import tensorflow as tf
import logging
from prismatic.vla.constants import DELAY_KWARGS

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
)

# Initialize important constants
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Model image size configuration
MODEL_IMAGE_SIZES = {
    "openvla": 224,
    # Add other models as needed
}


def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg: Any, wrap_diffusion_policy_for_droid: bool = False) -> torch.nn.Module:
    """
    Load and initialize model for evaluation based on configuration.

    Args:
        cfg: Configuration object with model parameters
        wrap_diffusion_policy_for_droid: Whether to wrap diffusion policy for DROID

    Returns:
        torch.nn.Module: The loaded model

    Raises:
        ValueError: If model family is not supported
    """
    if cfg.model_family == "openvla":
        model = get_vla(cfg)
    else:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg: Any) -> Union[int, tuple]:
    """
    Get image resize dimensions for a specific model.

    If returned value is an int, the resized image will be a square.
    If returned value is a tuple, the resized image will be a rectangle.

    Args:
        cfg: Configuration object with model parameters

    Returns:
        Union[int, tuple]: Image resize dimensions

    Raises:
        ValueError: If model family is not supported
    """
    if cfg.model_family not in MODEL_IMAGE_SIZES:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    return MODEL_IMAGE_SIZES[cfg.model_family]


def get_action(
    cfg: Any,
    model: torch.nn.Module,
    obs: Dict[str, Any],
    obs_real: Dict[str, Any],
    task_label: str,
    processor: Optional[Any] = None,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
) -> Union[List[np.ndarray], np.ndarray]:
    """
    Query the model to get action predictions.

    Args:
        cfg: Configuration object with model parameters
        model: The loaded model
        obs: Observation dictionary
        task_label: Text description of the task
        processor: Model processor for inputs
        action_head: Optional action head for continuous actions
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        use_film: Whether to use FiLM

    Returns:
        Union[List[np.ndarray], np.ndarray]: Predicted actions

    Raises:
        ValueError: If model family is not supported
    """
    with torch.no_grad():
        if cfg.model_family == "openvla":
            action = get_vla_action(
                cfg=cfg,
                vla=model,
                processor=processor,
                obs=obs,
                obs_real=obs_real,
                task_label=task_label,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=use_film,
            )
        else:
            raise ValueError(f"Unsupported model family: {cfg.model_family}")

    return action


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action


def apply_evaluation_obs_delay_v1(
    observation: Dict[str, Any],
    timestep: int,
    delay_kwargs: Dict = DELAY_KWARGS,
    delay_state: Dict = None  # 用于跟踪主视图历史的状态
) -> Dict[str, Any]:
    """
    仅对观测中的主视图full_image（训练时是image_primary）应用随机延迟，其他观测部分保持不变。
    需维护评测时使用，需维护主视图的历史队列以支持延迟查询。
    
    Args:
        observation: 单步观测字典（需包含"full_image"键）
        timestep: 当前时间步（用于计算最大可能延迟）
        delay_kwargs: 延迟配置参数（来自constants.DELAY_KWARGS）
        delay_state: 用于存储主视图历史的状态字典（首次调用时自动初始化）
    
    Returns:
        主视图应用延迟后的观测数据和更新后的状态
    """
    # 初始化延迟状态（仅存储主视图的历史）
    if delay_state is None:
        delay_state = {
            "full_image_history": [],  # 仅存储主视图历史
            "rng": tf.random.Generator.from_seed(delay_kwargs.get("random_seed", 42))
        }
    
    # 检查是否启用延迟及主视图是否存在
    use_random_obs = delay_kwargs.get("use_random_obs", False)
    max_delay_window = delay_kwargs.get("max_delay_window", 0)
    if not use_random_obs or max_delay_window <= 0:
        # 不启用延迟时，仅更新主视图历史并返回原始观测
        if "full_image" in observation:
            delay_state["full_image_history"].append(observation["full_image"])
        return observation, delay_state
    
    # 验证主视图是否存在
    if "full_image" not in observation:
        logging.warning("观测中不存在 'full_image'，不应用延迟")
        return observation, delay_state
    
    # 获取当前延迟分布参数
    delay_distribution = delay_kwargs.get("delay_distribution", "uniform")
    log_delay_info = delay_kwargs.get("log_delay_info", False)
    
    # 计算当前时间步允许的最大延迟（不能超过历史长度）
    current_max_delay = min(timestep, max_delay_window)
    current_max_delay = max(current_max_delay, 0)  # 确保非负，# 初始时间步（timestep=0）无历史，延迟为0
    
    # 生成当前时间步的延迟值
    if delay_distribution == "deterministic":
        delay_value = min(delay_kwargs.get("value", 0), current_max_delay)
    else:
        n_candidates = current_max_delay + 1  # 可能的延迟值数量（0到current_max_delay）
        if delay_distribution == "uniform":
            # 均匀分布采样
            random_normed = delay_state["rng"].uniform(shape=(), dtype=tf.float32)
            delay_value = tf.cast(tf.floor(random_normed * n_candidates), tf.int32)
        elif delay_distribution == "trunc_normal":
            # 截断正态分布采样
            mean = delay_kwargs.get("mean", 0.0)
            std = delay_kwargs.get("std", 1.0)
            z = delay_state["rng"].normal(shape=(), mean=mean, stddev=std, dtype=tf.float32)
            z_clamped = tf.clip_by_value(z, 0.0, tf.cast(current_max_delay, tf.float32))
            random_normed = z_clamped / tf.cast(n_candidates, tf.float32)
            delay_value = tf.cast(tf.floor(random_normed * n_candidates), tf.int32)
        elif delay_distribution == "exponential":
            # 指数分布采样
            lambd = delay_kwargs.get("lambda_", 1.0)
            u = delay_state["rng"].uniform(shape=(), dtype=tf.float32)
            exp_sample = -tf.math.log(1 - u) / lambd
            exp_clamped = tf.minimum(exp_sample, tf.cast(current_max_delay, tf.float32))
            random_normed = exp_clamped / tf.cast(n_candidates, tf.float32)
            delay_value = tf.cast(tf.floor(random_normed * n_candidates), tf.int32)
        else:
            raise ValueError(f"未知延迟分布: {delay_distribution}")
        
        delay_value = tf.clip_by_value(delay_value, 0, current_max_delay)  # 确保延迟合法
    
    # 记录延迟信息（首次时间步或日志开启时）
    if log_delay_info and (timestep == 0 or delay_value > 0):
        logging.info(f"[评测延迟 v1] 时间步 {timestep}，主视图延迟 {delay_value} 步，分布 {delay_distribution}")
    
    # 更新主视图历史队列（只保留最近max_delay_window步，节省内存）
    delay_state["full_image_history"].append(observation["full_image"])
    if len(delay_state["full_image_history"]) > max_delay_window + 1:
        delay_state["full_image_history"].pop(0)  # 移除过旧的主视图
    
    # 计算历史索引并获取延迟后的主视图（延迟0则为当前观测）
    history_idx = len(delay_state["full_image_history"]) - 1 - delay_value
    delayed_full_image = delay_state["full_image_history"][history_idx]
    
    # 仅替换主视图，保持其他观测部分不变
    observation_with_full_image_delay = observation.copy()
    observation_with_full_image_delay["full_image"] = delayed_full_image
    
    return observation_with_full_image_delay, delay_state


def apply_evaluation_obs_delay_v2(
    observation: Dict[str, Any],
    timestep: int,
    delay_kwargs: Dict = DELAY_KWARGS,
    delay_state: Dict = None  # 用于跟踪历史观测的状态
) -> Dict[str, Any]:
    """
    为评测时的单步观测数据应用随机延迟（基于配置的DELAY_KWARGS）。
    需维护历史观测队列以支持延迟查询。
    
    Args:
        observation: 单步观测字典（包含图像、状态等）
        timestep: 当前时间步（用于计算最大可能延迟）
        delay_kwargs: 延迟配置参数（来自constants.DELAY_KWARGS）
        delay_state: 用于存储历史观测的状态字典（首次调用时自动初始化）
    
    Returns:
        延迟后的观测数据
    """
    # 初始化延迟状态（存储历史观测队列和随机种子）
    if delay_state is None:
        delay_state = {
            "history": [],  # 存储历史观测的队列
            "rng": tf.random.Generator.from_seed(delay_kwargs.get("random_seed", 42))
        }
    
    # 检查是否启用延迟
    use_random_obs = delay_kwargs.get("use_random_obs", False)
    max_delay_window = delay_kwargs.get("max_delay_window", 0)
    if not use_random_obs or max_delay_window <= 0:
        # 不启用延迟时，直接返回当前观测并更新历史
        delay_state["history"].append(observation)
        return observation, delay_state
    
    # 获取当前延迟分布参数
    delay_distribution = delay_kwargs.get("delay_distribution", "uniform")
    log_delay_info = delay_kwargs.get("log_delay_info", False)
    
    # 计算当前时间步允许的最大延迟（不能超过历史长度）
    current_max_delay = min(timestep, max_delay_window)
    if current_max_delay < 0:
        current_max_delay = 0  # 初始时间步（timestep=0）无历史，延迟为0
    
    # 生成当前时间步的延迟值
    if delay_distribution == "deterministic":
        delay_value = min(delay_kwargs.get("value", 0), current_max_delay)
    else:
        n_candidates = current_max_delay + 1  # 可能的延迟值数量（0到current_max_delay）
        if delay_distribution == "uniform":
            # 均匀分布采样
            random_normed = delay_state["rng"].uniform(shape=(), dtype=tf.float32)
            delay_value = tf.cast(tf.floor(random_normed * n_candidates), tf.int32)
        elif delay_distribution == "trunc_normal":
            # 截断正态分布采样
            mean = delay_kwargs.get("mean", 0.0)
            std = delay_kwargs.get("std", 1.0)
            z = delay_state["rng"].normal(shape=(), mean=mean, stddev=std, dtype=tf.float32)
            z_clamped = tf.clip_by_value(z, 0.0, tf.cast(current_max_delay, tf.float32))
            random_normed = z_clamped / tf.cast(n_candidates, tf.float32)
            delay_value = tf.cast(tf.floor(random_normed * n_candidates), tf.int32)
        elif delay_distribution == "exponential":
            # 指数分布采样
            lambd = delay_kwargs.get("lambda_", 1.0)
            u = delay_state["rng"].uniform(shape=(), dtype=tf.float32)
            exp_sample = -tf.math.log(1 - u) / lambd
            exp_clamped = tf.minimum(exp_sample, tf.cast(current_max_delay, tf.float32))
            random_normed = exp_clamped / tf.cast(n_candidates, tf.float32)
            delay_value = tf.cast(tf.floor(random_normed * n_candidates), tf.int32)
        else:
            raise ValueError(f"未知延迟分布: {delay_distribution}")
        
        delay_value = tf.clip_by_value(delay_value, 0, current_max_delay)  # 确保延迟合法
    
    # 记录延迟信息（首次时间步或日志开启时）
    if log_delay_info and (timestep == 0 or delay_value > 0):
        logging.info(f"[评测延迟] 时间步 {timestep}，延迟 {delay_value} 步，分布 {delay_distribution}")
    
    # 更新历史观测队列（只保留最近max_delay_window步，节省内存）
    delay_state["history"].append(observation)
    if len(delay_state["history"]) > max_delay_window + 1:
        delay_state["history"].pop(0)  # 移除过旧的观测
    
    # 根据延迟值获取历史观测（延迟0则为当前观测）
    history_idx = len(delay_state["history"]) - 1 - delay_value
    delayed_observation = delay_state["history"][history_idx]
    
    return delayed_observation, delay_state
