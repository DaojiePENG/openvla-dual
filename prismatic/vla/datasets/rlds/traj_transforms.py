"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

import logging
from typing import Dict

import tensorflow as tf


def chunk_act_obs(traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    effective_traj_len = traj_len - future_action_window_size
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [effective_traj_len, window_size]) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None], [effective_traj_len, window_size]
    )

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [effective_traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None],
        [effective_traj_len, window_size + future_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    goal_timestep = tf.fill([effective_traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])

    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # Truncate other elements of the trajectory dict
    traj["task"] = tf.nest.map_structure(lambda x: tf.gather(x, tf.range(effective_traj_len)), traj["task"])
    traj["dataset_name"] = tf.gather(traj["dataset_name"], tf.range(effective_traj_len))
    traj["absolute_action_mask"] = tf.gather(traj["absolute_action_mask"], tf.range(effective_traj_len))

    return traj


def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj


def apply_random_observation_delay_v1(
    traj: Dict,
    delay_kwargs: Dict,  # 配置随机延迟的参数，在 prismatic/vla/constants.py 中定义
) -> Dict:
    """
    仅对 traj["observation"]["image_primary"] 应用随机延迟，其他观测部分保持不变，支持多种分布，并可选输出日志。
    Args:
        traj: 包含 "observation" 键的轨迹字典，需包含 "image_primary" 键。
        delay_kwargs: 控制延迟行为的参数字典，包含以下键：
            - use_random_obs (bool): 是否启用随机延迟。
            - max_delay_window (int): 最大延迟步数。
            - random_seed (int): 随机种子，确保可复现。
            - delay_distribution (str): 延迟分布类型，可选 "uniform", "deterministic", "trunc_normal", "exponential"。
            - log_delay_info (bool): 是否打印延迟信息。
            - 其他分布特定参数，如 "value"（deterministic），"mean" 和 "std"（trunc_normal），"lambda"（exponential）。
    Returns:
        主视图应用随机延迟后的轨迹字典。
    """
    use_random_obs = delay_kwargs.get("use_random_obs", False)
    max_delay_window = delay_kwargs.get("max_delay_window", 0)
    random_seed = delay_kwargs.get("random_seed", 42)
    delay_distribution = delay_kwargs.get("delay_distribution", "uniform")
    log_delay_info = delay_kwargs.get("log_delay_info", False)

    # 设置全局随机种子，确保可复现（可能还需要设置 num_parallel_calls=1或None来保证完全复现）
    tf.random.set_seed(random_seed)
    if not use_random_obs or max_delay_window <= 0:
        return traj

    # 验证主视图是否存在
    if "image_primary" not in traj["observation"]:
        logging.warning("轨迹观测中不存在 'image_primary'，不应用延迟")
        return traj

    logging.info("=" * 88)
    logging.info(f"对 image_primary 应用随机延迟，参数: {delay_kwargs}")
    logging.info("=" * 88)

    # 获取轨迹长度（基于主视图）
    primary_image = traj["observation"]["image_primary"]
    T = tf.shape(primary_image)[0]

    # ======================
    # 日志：构建 info 字符串
    # ======================
    if log_delay_info:
        info_parts = [f"[Delay v1] max_delay_window={max_delay_window}, delay_distribution={delay_distribution}"]

        # 根据分布类型，决定显示哪些参数
        dist_params = {
            "uniform": [],
            "deterministic": ["value"],
            "trunc_normal": ["mean", "std"],
            "exponential": ["lambda_"]
        }

        param_keys = dist_params.get(delay_distribution, [])

        for k in param_keys:
            if k in delay_kwargs:
                v = delay_kwargs[k]
                info_parts.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
        
        # 添加控制类参数（可选）
        if "use_random_obs" in delay_kwargs:
            info_parts.append(f"use_obs={delay_kwargs['use_random_obs']}")
        if "random_seed" in delay_kwargs:
            info_parts.append(f"seed={delay_kwargs['random_seed']}")
        logging.info(f"[Random Delay v1 Applied] {' '.join(info_parts)}")

    # ======================
    # 延迟逻辑
    # ======================
    if delay_distribution == "deterministic":
        delay_value = min(delay_kwargs.get("value", 0), max_delay_window)
        random_delays = tf.fill([T], delay_value)
    else:
        max_delays = tf.minimum(tf.range(T), max_delay_window)
        n_candidates = max_delays + 1

        if delay_distribution == "uniform":
            random_normed = tf.random.uniform(tf.shape(max_delays), dtype=tf.float32)
            random_delays = tf.cast(tf.floor(random_normed * tf.cast(n_candidates, tf.float32)), tf.int32)
        elif delay_distribution == "trunc_normal":
            mean = delay_kwargs.get("mean", 0.0)
            std = delay_kwargs.get("std", 1.0)
            z = tf.random.normal(tf.shape(max_delays), mean=mean, stddev=std, dtype=tf.float32)
            z_clamped = tf.clip_by_value(z, 0.0, tf.cast(max_delays, tf.float32))
            random_normed = z_clamped / tf.cast(n_candidates, tf.float32)
            random_delays = tf.cast(tf.floor(random_normed * tf.cast(n_candidates, tf.float32)), tf.int32)
        elif delay_distribution == "exponential":
            lambd = delay_kwargs.get("lambda", 1.0)
            u = tf.random.uniform(tf.shape(max_delays), dtype=tf.float32)
            exp_sample = -tf.math.log(1 - u) / lambd
            exp_clamped = tf.minimum(exp_sample, tf.cast(max_delays, tf.float32))
            random_normed = exp_clamped / tf.cast(n_candidates, tf.float32)
            random_delays = tf.cast(tf.floor(random_normed * tf.cast(n_candidates, tf.float32)), tf.int32)
        else:
            raise ValueError(f"未知分布: {delay_distribution}")

    # 确保索引合法
    gather_indices = tf.maximum(tf.range(T) - random_delays, 0)

    # ======================
    # 日志：打印采样的延迟值（可选）
    # ======================
    if log_delay_info:
        # 打印前 20 个延迟值，避免输出太长
        max_print = tf.minimum(T, 20)
        logging.info(f"[Sampled Delays v1] {random_delays[:max_print]}")

    # 仅对主视图应用延迟，其他观测保持不变
    traj["observation"]["image_primary"] = tf.gather(primary_image, gather_indices)

    return traj


def apply_random_observation_delay_v2(
    traj: Dict,
    delay_kwargs: Dict, # 配置随机延迟的参数，在 prismatic/vla/constants.py 中定义
) -> Dict:
    """
    对 traj["observation"] 应用随机延迟，支持多种分布，并可选输出日志。
    Args:
        traj: 包含 "observation" 键的轨迹字典。
        delay_kwargs: 控制延迟行为的参数字典，包含以下键：
            - use_random_obs (bool): 是否启用随机延迟。
            - max_delay_window (int): 最大延迟步数。
            - random_seed (int): 随机种子，确保可复现。
            - delay_distribution (str): 延迟分布类型，可选 "uniform", "deterministic", "trunc_normal", "exponential"。
            - log_delay_info (bool): 是否打印延迟信息。
            - 其他分布特定参数，如 "value"（deterministic），"mean" 和 "std"（trunc_normal），"lambda"（exponential）。
    Returns:
        应用随机延迟后的轨迹字典。
    """
    
    use_random_obs = delay_kwargs.get("use_random_obs", False)
    max_delay_window = delay_kwargs.get("max_delay_window", 0)
    random_seed = delay_kwargs.get("random_seed", 42)
    delay_distribution = delay_kwargs.get("delay_distribution", "uniform")
    log_delay_info = delay_kwargs.get("log_delay_info", False)

    # 设置全局随机种子，确保可复现（可能还需要设置 num_parallel_calls=1或None来保证完全复现）
    tf.random.set_seed(random_seed)
    if not use_random_obs or max_delay_window <= 0:
        return traj

    logging.info("=" * 88)
    logging.info(f"Applying random observation delay with parameters: {delay_kwargs}")
    logging.info("=" * 88)
    
    # 获取轨迹长度
    obs = traj["observation"]
    T = tf.shape(list(obs.values())[0])[0]

    # ======================
    # 日志：构建 info 字符串
    # ======================
    if log_delay_info:
        # 基础信息
        info_parts = [f"[Delay] max_delay_window={max_delay_window}, delay_distribution={delay_distribution}"]

        # 根据分布类型，决定显示哪些参数
        dist_params = {
            "uniform": [],
            "deterministic": ["value"],
            "trunc_normal": ["mean", "std"],
            "exponential": ["lambda_"]
        }

        param_keys = dist_params.get(delay_distribution, [])

        for k in param_keys:
            if k in delay_kwargs:
                v = delay_kwargs[k]
                if isinstance(v, float):
                    info_parts.append(f"{k}={v:.3g}")
                else:
                    info_parts.append(f"{k}={v}")

        # 添加控制类参数（可选）
        if "use_random_obs" in delay_kwargs:
            info_parts.append(f"use_obs={delay_kwargs['use_random_obs']}")

        if "random_seed" in delay_kwargs:
            info_parts.append(f"seed={delay_kwargs['random_seed']}")

        info_str = " ".join(info_parts)
        logging.info(f"[Random Delay Applied] {info_str}")

    # ======================
    # 延迟逻辑
    # ======================
    if delay_distribution == "deterministic":
        delay_value = (delay_kwargs or {}).get("value", 0)
        delay_value = min(delay_value, max_delay_window)
        random_delays = tf.fill([T], delay_value)

    else:
        max_delays = tf.minimum(tf.range(T), max_delay_window)
        n_candidates = max_delays + 1

        if delay_distribution == "uniform":
            random_normed = tf.random.uniform(tf.shape(max_delays), dtype=tf.float32)
            random_delays = tf.cast(tf.floor(random_normed * tf.cast(n_candidates, tf.float32)), tf.int32)

        elif delay_distribution == "trunc_normal":
            kw = delay_kwargs or {}
            mean = kw.get("mean", 0.0)
            std = kw.get("std", 1.0)
            z = tf.random.normal(tf.shape(max_delays), mean=mean, stddev=std, dtype=tf.float32)
            z_clamped = tf.clip_by_value(z, 0.0, tf.cast(max_delays, tf.float32))
            random_normed = z_clamped / tf.cast(n_candidates, tf.float32)
            random_delays = tf.cast(tf.floor(random_normed * tf.cast(n_candidates, tf.float32)), tf.int32)

        elif delay_distribution == "exponential":
            kw = delay_kwargs or {}
            lambd = kw.get("lambda", 1.0)
            u = tf.random.uniform(tf.shape(max_delays), dtype=tf.float32)
            exp_sample = -tf.math.log(1 - u) / lambd
            exp_clamped = tf.minimum(exp_sample, tf.cast(max_delays, tf.float32))
            random_normed = exp_clamped / tf.cast(n_candidates, tf.float32)
            random_delays = tf.cast(tf.floor(random_normed * tf.cast(n_candidates, tf.float32)), tf.int32)

        else:
            raise ValueError(f"Unknown distribution: {delay_distribution}")

    # 确保索引合法
    gather_indices = tf.maximum(tf.range(T) - random_delays, 0)

    # ======================
    # 日志：打印采样的延迟值（可选）
    # ======================
    if log_delay_info:
        # 打印前 20 个延迟值，避免输出太长
        max_print = tf.minimum(T, 20)
        delays_to_print = random_delays[:max_print]
        logging.info(f"[Sampled Delays] {delays_to_print}")

    # 应用延迟
    traj["observation"] = tf.nest.map_structure(
        lambda x: tf.gather(x, gather_indices),
        obs
    )

    return traj

