"""Utils for evaluating robot policies in various environments."""

import os
import random
import time

import numpy as np
import torch

from typing import Any, Dict, List, Optional, Union
import tensorflow as tf
import logging

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
)

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg, wrap_diffusion_policy_for_droid=False):
    """Load model for evaluation."""
    if cfg.model_family == "openvla":
        model = get_vla(cfg)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "openvla":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def get_action(cfg, model, obs, task_label, processor=None):
    """Queries the model to get an action."""
    if cfg.model_family == "openvla":
        action = get_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop
        )
        assert action.shape == (ACTION_DIM,)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action


def apply_evaluation_obs_delay_v1(
    observation: Dict[str, Any],
    timestep: int,
    delay_kwargs: Optional[Dict] = None,
    # delay_kwargs: Dict = DELAY_KWARGS,
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
        
        # print("[评测延迟 v1] 未启用延迟，直接返回当前观测")
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
    delay_kwargs: Optional[Dict] = None,
    # delay_kwargs: Dict = DELAY_KWARGS,
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
