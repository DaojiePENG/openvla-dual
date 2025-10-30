"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        return action


class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output


class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_diffusion_steps_train=50,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        )
        self.num_diffusion_steps_train = num_diffusion_steps_train
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps_train, beta_schedule="squaredcos_cap_v2")
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred


class VisionActionHead(nn.Module):
    """
    MLP-based action head that fuses vision hidden states and action hidden states for continuous action prediction.
    """
    def __init__(
        self,
        input_dim=4096,          # 大模型骨架输出的维度
        vision_dim=1024,         # 修正为DINOv2-L的实际输出维度（1024）
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = NUM_ACTIONS_CHUNK  # 从constants导入
        
        # 视觉特征处理分支（输入维度修正为1024）
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 动作特征处理分支
        self.action_proj = nn.Sequential(
            nn.Linear(input_dim * ACTION_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 融合后的预测头
        self.fusion_head = MLPResNet(
            num_blocks=2,
            input_dim=hidden_dim * 2,  # 融合视觉和动作特征
            hidden_dim=hidden_dim,
            output_dim=action_dim
        )

    def forward(self, actions_hidden_states, vision_hidden_states):
        """
        Args:
            actions_hidden_states: 大模型输出的动作相关隐藏状态 
                shape: (batch_size, chunk_len * action_dim, input_dim)
            vision_hidden_states: 视觉模型输出的隐藏状态
                shape: (batch_size, vision_seq_len, vision_dim) -> 例如(1, 151, 1024)
        Returns:
            预测的连续动作 shape: (batch_size, chunk_len, action_dim)
        """
        batch_size = actions_hidden_states.shape[0]
        
        # 调试：打印输入特征形状（确认问题时使用）
        # print(f"actions_hidden_states shape: {actions_hidden_states.shape}")
        # print(f"vision_hidden_states shape: {vision_hidden_states.shape}")
        
        # 处理动作特征
        rearranged_actions = actions_hidden_states.reshape(
            batch_size, self.num_actions_chunk, -1  # (batch_size, chunk_len, input_dim*ACTION_DIM)
        )
        action_features = self.action_proj(rearranged_actions)  # (batch_size, chunk_len, hidden_dim)
        
        # 处理视觉特征（关键修正）
        # 1. 确保视觉特征维度正确（排除异常情况）
        if vision_hidden_states.dim() != 3:
            # 紧急修复：当视觉特征维度异常时，强制reshape（适用于单样本情况）
            vision_hidden_states = vision_hidden_states.view(batch_size, -1, self.vision_proj[0].in_features)
        
        # 2. 提取DINOv2的分类token（通常是第一个token，比全局平均更有效）
        vision_global = vision_hidden_states[:, 0, :]  # (batch_size, 1024)
        # 如果分类token无效，使用全局平均作为备选
        # vision_global = torch.mean(vision_hidden_states, dim=1)  # (batch_size, 1024)
        
        # 3. 视觉特征投影
        vision_features = self.vision_proj(vision_global)  # (batch_size, hidden_dim)
        
        # 4. 扩展到与动作序列长度匹配
        vision_features = vision_features.unsqueeze(1).repeat(
            1, self.num_actions_chunk, 1  # (batch_size, chunk_len, hidden_dim)
        )
        
        # 融合特征并预测动作
        fused_features = torch.cat([action_features, vision_features], dim=-1)  # (batch_size, chunk_len, 2*hidden_dim)
        action_pred = self.fusion_head(fused_features)  # (batch_size, chunk_len, action_dim)
        
        return action_pred
    def predict_action(self, actions_hidden_states, vision_hidden_states):
        return self.forward(actions_hidden_states, vision_hidden_states)

class VisionActionHead_V2(nn.Module):
    """
    MLP-based action head that fuses vision hidden states and action hidden states for continuous action prediction.
    """
    def __init__(
        self,
        input_dim=4096,          # 大模型骨架输出的维度
        vision_dim=1024,         # 修正为DINOv2-L的实际输出维度（1024）
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = NUM_ACTIONS_CHUNK  # 从constants导入
        
        # 视觉特征处理分支（输入维度修正为1024）
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 动作特征处理分支
        self.action_proj = nn.Sequential(
            nn.Linear(input_dim * ACTION_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 融合后的预测头
        self.fusion_head = MLPResNet(
            num_blocks=2,
            input_dim=hidden_dim * 3,  # 融合视觉*2和动作特征
            hidden_dim=hidden_dim,
            output_dim=action_dim
        )

    def forward(self, actions_hidden_states, vision_hidden_states, vision_hidden_states_v2):
        """
        Args:
            actions_hidden_states: 大模型输出的动作相关隐藏状态 
                shape: (batch_size, chunk_len * action_dim, input_dim)
            vision_hidden_states: 视觉模型输出的隐藏状态，手腕视图特征输入
                shape: (batch_size, vision_seq_len, vision_dim) -> 例如(1, 151, 1024)
            vision_hidden_states_v2: 视觉模型输出的隐藏状态（版本2），主视图特征输入
                shape: (batch_size, vision_seq_len, vision_dim) -> 例如(1, 151, 1024)
        Returns:
            预测的连续动作 shape: (batch_size, chunk_len, action_dim)
        """
        batch_size = actions_hidden_states.shape[0]
        
        # 处理动作特征
        rearranged_actions = actions_hidden_states.reshape(
            batch_size, self.num_actions_chunk, -1  # (batch_size, chunk_len, input_dim*ACTION_DIM)
        )
        action_features = self.action_proj(rearranged_actions)  # (batch_size, chunk_len, hidden_dim)
        
        # 处理视觉特征（关键修正）
        # 1. 确保视觉特征维度正确（排除异常情况）
        if vision_hidden_states.dim() != 3:
            # 紧急修复：当视觉特征维度异常时，强制reshape（适用于单样本情况）
            vision_hidden_states = vision_hidden_states.view(batch_size, -1, self.vision_proj[0].in_features)
        
        # 2. 提取DINOv2的分类token（通常是第一个token，比全局平均更有效）
        vision_global = vision_hidden_states[:, 0, :]  # (batch_size, 1024)
        # 如果分类token无效，使用全局平均作为备选
        # vision_global = torch.mean(vision_hidden_states, dim=1)  # (batch_size, 1024)
        
        # 3. 视觉特征投影
        vision_features = self.vision_proj(vision_global)  # (batch_size, hidden_dim)
        
        # 4. 扩展到与动作序列长度匹配
        vision_features = vision_features.unsqueeze(1).repeat(
            1, self.num_actions_chunk, 1  # (batch_size, chunk_len, hidden_dim)
        )

        # -----------------------处理第二组主视图的视觉特征----------------------------
        # 处理视觉特征（关键修正）
        # 1. 确保视觉特征维度正确（排除异常情况）
        if vision_hidden_states_v2.dim() != 3:
            # 紧急修复：当视觉特征维度异常时，强制reshape（适用于单样本情况）
            vision_hidden_states_v2 = vision_hidden_states_v2.view(batch_size, -1, self.vision_proj[0].in_features)

        # 2. 提取DINOv2的分类token（通常是第一个token，比全局平均更有效）
        vision_global_v2 = vision_hidden_states_v2[:, 0, :]  # (batch_size, 1024)
        # 如果分类token无效，使用全局平均作为备选
        # vision_global = torch.mean(vision_hidden_states, dim=1)  # (batch_size, 1024)
        
        # 3. 视觉特征投影
        vision_features_v2 = self.vision_proj(vision_global_v2)  # (batch_size, hidden_dim)
        
        # 4. 扩展到与动作序列长度匹配
        vision_features_v2 = vision_features_v2.unsqueeze(1).repeat(
            1, self.num_actions_chunk, 1  # (batch_size, chunk_len, hidden_dim)
        )
        
        # 融合特征并预测动作
        fused_features = torch.cat([action_features, vision_features, vision_features_v2], dim=-1)  # (batch_size, chunk_len, 3*hidden_dim)
        action_pred = self.fusion_head(fused_features)  # (batch_size, chunk_len, action_dim)
        
        return action_pred
    def predict_action(self, actions_hidden_states, vision_hidden_states, vision_hidden_states_v2):
        return self.forward(actions_hidden_states, vision_hidden_states, vision_hidden_states_v2)


class VisionActionHead_E1(nn.Module):
    def __init__(
        self,
        input_dim=4096,          # 大模型骨架输出的维度
        vision_dim=1024,         # DINOv2-L输出维度
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = NUM_ACTIONS_CHUNK  # 动作序列长度（如5）
        self.vision_dim = vision_dim  # 新增：保存视觉特征维度，用于后续校验
        
        # 视觉特征处理分支
        self.vision_proj = MLPResNet(
            num_blocks=2,
            input_dim=vision_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # 动作特征处理分支 
        self.action_proj = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # 融合后的预测头（保持不变）
        self.fusion_head = MLPResNet(
            num_blocks=2,
            input_dim=hidden_dim * 2,  # 融合视觉和动作特征
            hidden_dim=hidden_dim,
            output_dim=action_dim
        )

    def forward(self, actions_hidden_states, vision_hidden_states):
        batch_size = actions_hidden_states.shape[0]
        vision_seq_len = vision_hidden_states.shape[1]  # 视觉token数量（如DINOv2的151）
        
        # 处理动作特征（保持不变）
        rearranged_actions = actions_hidden_states.reshape(
            batch_size, self.num_actions_chunk, -1  # (batch_size, chunk_len, input_dim*ACTION_DIM)
        )
        action_features = self.action_proj(rearranged_actions)  # (batch_size, chunk_len, hidden_dim)
        
        # 处理视觉特征（核心修改）
        # 1. 校验视觉特征维度
        if vision_hidden_states.dim() != 3:
            vision_hidden_states = vision_hidden_states.view(batch_size, -1, self.vision_dim)
        
        # 2. 选取与动作序列长度匹配的视觉token
        if vision_seq_len >= self.num_actions_chunk:
            # 策略：跳过[CLS] token，取后续有空间意义的token（更适合机器人任务）
            # 若视觉token充足，取前num_actions_chunk个非分类token
            selected_vision_tokens = vision_hidden_states[:, 1 : 1 + self.num_actions_chunk, :]  # (B, chunk_len, vision_dim)
        else:
            # 策略：若视觉token不足，重复最后一个token补齐（避免维度不匹配）
            pad_length = self.num_actions_chunk - vision_seq_len
            selected_vision_tokens = torch.cat([
                vision_hidden_states,  # 取全部视觉token
                vision_hidden_states[:, -1:, :].repeat(1, pad_length, 1)  # 补齐
            ], dim=1)
        
        # 3. 视觉特征投影（每个token单独投影，替代原全局投影）
        vision_features = self.vision_proj(selected_vision_tokens)  # (B, chunk_len, hidden_dim)
        # （无需再扩展维度，已与动作序列长度一致）
        
        # 融合特征并预测动作（保持不变）
        fused_features = torch.cat([action_features, vision_features], dim=-1)  # (B, chunk_len, 2*hidden_dim)
        action_pred = self.fusion_head(fused_features)  # (B, chunk_len, action_dim)
        
        return action_pred

    def predict_action(self, actions_hidden_states, vision_hidden_states):
        return self.forward(actions_hidden_states, vision_hidden_states)
    

class VisionActionHead_E2(nn.Module):
    def __init__(
        self,
        input_dim=4096,          # 大模型骨架输出的维度
        vision_dim=1024,         # DINOv2-L输出维度
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = NUM_ACTIONS_CHUNK  # 动作序列长度（如5）
        self.vision_dim = vision_dim  # 新增：保存视觉特征维度，用于后续校验
        
        # 视觉特征处理分支
        self.vision_proj = MLPResNet(
            num_blocks=2,
            input_dim=vision_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # 动作特征处理分支 
        self.action_proj = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # 融合后的预测头（保持不变）
        self.fusion_head = MLPResNet(
            num_blocks=2,
            input_dim=hidden_dim * 3,  # 融合视觉和动作特征
            hidden_dim=hidden_dim,
            output_dim=action_dim
        )

    def forward(self, actions_hidden_states, vision_hidden_states, vision_hidden_states_v2):
        batch_size = actions_hidden_states.shape[0]
        vision_seq_len = vision_hidden_states.shape[1]  # 视觉token数量（如DINOv2的151）
        vision_seq_len_v2 = vision_hidden_states_v2.shape[1]  # 视觉token数量（如DINOv2的151）
        
        # 处理动作特征（保持不变）
        rearranged_actions = actions_hidden_states.reshape(
            batch_size, self.num_actions_chunk, -1  # (batch_size, chunk_len, input_dim*ACTION_DIM)
        )
        action_features = self.action_proj(rearranged_actions)  # (batch_size, chunk_len, hidden_dim)
        
        # 处理视觉特征（核心修改）
        # 1. 校验视觉特征维度
        if vision_hidden_states.dim() != 3:
            vision_hidden_states = vision_hidden_states.view(batch_size, -1, self.vision_dim)
        
        # 2. 选取与动作序列长度匹配的视觉token
        if vision_seq_len >= self.num_actions_chunk:
            # 策略：跳过[CLS] token，取后续有空间意义的token（更适合机器人任务）
            # 若视觉token充足，取前num_actions_chunk个非分类token
            selected_vision_tokens = vision_hidden_states[:, 1 : 1 + self.num_actions_chunk, :]  # (B, chunk_len, vision_dim)
        else:
            # 策略：若视觉token不足，重复最后一个token补齐（避免维度不匹配）
            pad_length = self.num_actions_chunk - vision_seq_len
            selected_vision_tokens = torch.cat([
                vision_hidden_states,  # 取全部视觉token
                vision_hidden_states[:, -1:, :].repeat(1, pad_length, 1)  # 补齐
            ], dim=1)
        
        # 3. 视觉特征投影（每个token单独投影，替代原全局投影）
        vision_features = self.vision_proj(selected_vision_tokens)  # (B, chunk_len, hidden_dim)
        # （无需再扩展维度，已与动作序列长度一致）
        
        # -----------------------处理第二组主视图的视觉特征----------------------------
        # 处理视觉特征（核心修改）
        # 1. 校验视觉特征维度
        if vision_hidden_states_v2.dim() != 3:
            vision_hidden_states_v2 = vision_hidden_states_v2.view(batch_size, -1, self.vision_dim)

        # 2. 选取与动作序列长度匹配的视觉token
        if vision_seq_len_v2 >= self.num_actions_chunk:
            # 策略：跳过[CLS] token，取后续有空间意义的token（更适合机器人任务）
            # 若视觉token充足，取前num_actions_chunk个非分类token
            selected_vision_tokens_v2 = vision_hidden_states_v2[:, 1 : 1 + self.num_actions_chunk, :]  # (B, chunk_len, vision_dim)
        else:
            # 策略：若视觉token不足，重复最后一个token补齐（避免维度不匹配）
            pad_length = self.num_actions_chunk - vision_seq_len_v2
            selected_vision_tokens_v2 = torch.cat([
                vision_hidden_states_v2,  # 取全部视觉token
                vision_hidden_states_v2[:, -1:, :].repeat(1, pad_length, 1)  # 补齐
            ], dim=1)
        
        # 3. 视觉特征投影（每个token单独投影，替代原全局投影）
        vision_features_v2 = self.vision_proj(selected_vision_tokens_v2)  # (B, chunk_len, hidden_dim)
        # （无需再扩展维度，已与动作序列长度一致）


        # 融合特征并预测动作（保持不变）
        fused_features = torch.cat([action_features, vision_features, vision_features_v2], dim=-1)  # (B, chunk_len, 3*hidden_dim)
        action_pred = self.fusion_head(fused_features)  # (B, chunk_len, action_dim)
        
        return action_pred

    def predict_action(self, actions_hidden_states, vision_hidden_states):
        return self.forward(actions_hidden_states, vision_hidden_states)
    
