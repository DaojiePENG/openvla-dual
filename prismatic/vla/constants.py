"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict
import json
import os
import logging

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on

@dataclass
class DelayKwargs():
    """配置随机延迟的参数
    """
    use_random_obs: bool = True             # 是否使用随机延迟
    max_delay_window: int = 20               # 最大延迟步数
    random_seed: int = 42                   # 随机种子
    delay_distribution: str = "uniform"     # 延迟分布类型
    log_delay_info: bool = False            # 是否打印延迟信息

    value: int = 0                          # 用于 deterministic
    mean: float = 0.0                       # 用于 trunc_normal
    std: float = 1.0                        # 用于 trunc_normal
    lambda_: float = 1.0                    # 用于 exponential

    def to_dict(self) -> Dict:
        return asdict(self)

    def __call__(self) -> Dict:
        return self.to_dict()

# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
    "DELAY_KWARGS": DelayKwargs().to_dict(),
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
    "DELAY_KWARGS": DelayKwargs().to_dict(),
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
    "DELAY_KWARGS": DelayKwargs().to_dict(),
}


# Function to detect robot platform from command line arguments
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()

    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    else:
        # Default to LIBERO if unclear
        return "LIBERO"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]
DELAY_KWARGS = constants["DELAY_KWARGS"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print(f"  DELAY_KWARGS = {DELAY_KWARGS}")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")

# 保存所有的常量到字典中，方便调试
ALL_CONSTANTS = {
    "ROBOT_PLATFORM": ROBOT_PLATFORM,
    "NUM_ACTIONS_CHUNK": NUM_ACTIONS_CHUNK,
    "ACTION_DIM": ACTION_DIM,
    "PROPRIO_DIM": PROPRIO_DIM,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": ACTION_PROPRIO_NORMALIZATION_TYPE,
    "IGNORE_INDEX": IGNORE_INDEX,
    "ACTION_TOKEN_BEGIN_IDX": ACTION_TOKEN_BEGIN_IDX,
    "STOP_INDEX": STOP_INDEX,
    "DELAY_KWARGS": DELAY_KWARGS,
}

# 将所有常量保存到.json文件
def save_constants(dir, all_constants=ALL_CONSTANTS):
    """
    将所有常量保存到指定目录的 constants.json 文件中。

    Args:
        dir (str): 目标保存目录路径。
    """
    # 确保目标目录存在，如果不存在则创建
    os.makedirs(dir, exist_ok=True)
    
    # 构建文件路径
    file_path = os.path.join(dir, 'constants.json')
    
    # 将常量字典写入 JSON 文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(all_constants, f, indent=4, ensure_ascii=False)
    
    logging.info(f"Constants have been saved to: {file_path}")
