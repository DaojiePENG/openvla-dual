"""Utils for training/fine-tuning scripts."""

import torch

from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def get_current_action_mask(token_ids):
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)

    # Extract the action part only
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def get_next_actions_mask(token_ids):
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = cumsum > ACTION_DIM

    # Extract the action part only
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy


def compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask):
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy())
    )
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy())
    )
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss

def compute_weighted_l1_loss(
    ground_truth_actions, 
    predicted_actions, 
    weight_strategy: str = "inverse", 
    clip_max_weight: float = 10.0,  # 权重裁剪上限
    epsilon: float = 1e-3,
    alpha: float = 2.0,  # 指数衰减参数
    normalize_weights: bool = False  # 是否归一化权重
):
    """
    带权重裁剪和多策略选择的加权L1损失计算，仅使用前6个关节速度维度计算权重。
    
    Args:
        ground_truth_actions: 归一化的真实动作 (B, T, D)，D=7（前6维为关节速度）
        predicted_actions: 预测的动作 (B, T, D)
        weight_strategy: 权重计算策略，可选 "inverse"|"inverse_squared"|"exp_decay"|"log"
        clip_max_weight: 权重裁剪的最大值，避免极端权重
        epsilon: 避免除零的小常数
        alpha: 指数衰减的控制参数（仅对"exp_decay"策略有效）
        normalize_weights: 是否对权重进行归一化处理，确保平均权重为1
    
    Returns:
        weighted_loss: 加权L1损失
    """
    # 1. 提取前6个关节维度计算速度（忽略第7维夹爪）
    joint_gt = ground_truth_actions[..., :6]  # (B, T, 6)
    speed = torch.norm(joint_gt, dim=-1, keepdim=True)  # 关节速度大小 (B, T, 1)
    speed = torch.clamp(speed, min=epsilon)  # 避免零除

    # 2. 多策略计算权重（均基于速度反向映射，低速高权重）
    if weight_strategy == "inverse":
        # 基础反向比例：1/(speed)
        weights = 1.0 / speed
    elif weight_strategy == "inverse_squared":
        # 平方反向比例：放大低速动作的权重差异
        weights = 1.0 / (speed ** 2)
    elif weight_strategy == "exp_decay":
        # 指数衰减：权重随速度增长快速衰减（alpha控制衰减速率）
        weights = torch.exp(-alpha * speed)
    elif weight_strategy == "log":
        # 对数映射：缓解极端速度的权重波动
        weights = 1.0 / torch.log1p(speed)  # log1p = log(1+speed)
    else:
        raise ValueError(f"不支持的权重策略: {weight_strategy}")

    # 3. 权重裁剪：归一化前先限制最大值，避免个别样本权重过大
    weights = torch.clamp(weights, max=clip_max_weight)

    # 4. 权重归一化：确保平均权重为1，稳定训练尺度(可选，视情况使用，如果训练不稳定可启用)
    if normalize_weights:
        weights = weights / torch.mean(weights)

    # 5. 计算加权损失（对所有7维动作应用相同权重）
    l1_errors = torch.abs(ground_truth_actions - predicted_actions)
    weighted_errors = l1_errors * weights
    return torch.mean(weighted_errors)