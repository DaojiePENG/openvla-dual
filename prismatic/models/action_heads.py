"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math
from typing import Optional

import numpy as np
import timm
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


# SigLIP model options for the VisionActionHead's vision encoder
SIGLIP_MODELS = {
    "siglip-base": "vit_base_patch16_siglip_224",
    "siglip-so400m": "vit_so400m_patch14_siglip_224",
}


class VisionEncoder(nn.Module):
    """
    Lightweight vision encoder for the VisionActionHead's real-time visual sensing.
    Uses SigLIP (frozen by default) to extract visual features from real-time images.
    """

    def __init__(
        self,
        model_name: str = "siglip-base",
        freeze: bool = True,
        num_views: int = 2,  # primary + wrist
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_views = num_views

        # Load SigLIP model from timm
        timm_model_name = SIGLIP_MODELS.get(model_name, model_name)
        self.vit = timm.create_model(timm_model_name, pretrained=pretrained, num_classes=0)

        # Get embedding dimension
        self.embed_dim = self.vit.embed_dim  # typically 768 for base, 1152 for so400m

        # Freeze vision encoder by default
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False
            self.vit.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Process real-time images and return pooled visual features.

        The input may come from:
        1. VLA processor: (B, num_views * 6, H, W) where 6 = 3 SigLIP + 3 DINOv2 channels per view
        2. Raw RGB: (B, num_views * 3, H, W) or (B, num_views, 3, H, W)

        We extract only the first 3 channels per view (SigLIP channels) for our own SigLIP encoder.

        Returns:
            vision_features: (B, num_views * embed_dim) pooled features
        """
        B = pixel_values.shape[0]

        # Handle 5D input: (B, num_views, C, H, W)
        if pixel_values.dim() == 5:
            V = pixel_values.shape[1]
            C = pixel_values.shape[2]
            # Take first 3 channels per view (SigLIP channels)
            pixel_values = pixel_values[:, :, :3, :, :]  # (B, V, 3, H, W)
            pixel_values = pixel_values.reshape(B * V, 3, pixel_values.shape[-2], pixel_values.shape[-1])
        else:
            # 4D input: (B, num_views * C, H, W)
            total_channels = pixel_values.shape[1]
            H, W = pixel_values.shape[2], pixel_values.shape[3]
            channels_per_view = total_channels // self.num_views

            if channels_per_view > 3:
                # VLA format: 6 channels per view (3 SigLIP + 3 DINOv2)
                # Reshape to (B, num_views, channels_per_view, H, W) and extract first 3
                pixel_values = pixel_values.reshape(B, self.num_views, channels_per_view, H, W)
                pixel_values = pixel_values[:, :, :3, :, :]  # (B, num_views, 3, H, W)
                pixel_values = pixel_values.reshape(B * self.num_views, 3, H, W)
            else:
                # Already 3 channels per view: (B, num_views * 3, H, W)
                pixel_values = pixel_values.reshape(B * self.num_views, 3, H, W)

        # Cast input to match vision encoder weight dtype (e.g., bfloat16)
        target_dtype = next(self.vit.parameters()).dtype
        pixel_values = pixel_values.to(target_dtype)

        # Extract features
        features = self.vit(pixel_values)  # (B*V, embed_dim)

        # Reshape to (B, V * embed_dim)
        features = features.reshape(B, self.num_views * self.embed_dim)

        return features


class VisionActionHead(nn.Module):
    """
    Dual-system ActionHead that combines:
    1. LLM hidden states — high-level planning (potentially stale from delayed frames)
    2. Real-time vision features (from own vision encoder) — fast reactive control

    During frame-delay training:
    - Fresh path: h_fresh from current frame + current vision → action_fresh
    - Stale path: h_stale from delayed frame + current vision → action_stale
    - Both are trained toward ground-truth actions.

    Uses its own frozen SigLIP encoder (same pretrained weights as VLA's base SigLIP).
    At deployment, copy the VLA's base SigLIP weights (without FiLM/LoRA) here.
    """

    def __init__(
        self,
        input_dim: int = 4096,  # LLM hidden dimension
        hidden_dim: int = 4096,
        action_dim: int = 7,
        vision_encoder_name: str = "siglip-base",
        freeze_vision_encoder: bool = True,
        num_views: int = 2,  # primary + wrist
        vision_encoder_pretrained: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_views = num_views

        # Vision encoder for real-time visual sensing
        self.vision_encoder = VisionEncoder(
            model_name=vision_encoder_name,
            freeze=freeze_vision_encoder,
            num_views=num_views,
            pretrained=vision_encoder_pretrained,
        )
        vision_embed_dim = self.vision_encoder.embed_dim * num_views

        # Project vision features to match LLM dimension
        self.vision_projector = nn.Sequential(
            nn.Linear(vision_embed_dim, input_dim),
            nn.GELU(),
        )

        # Fusion MLP: combines LLM hidden states + projected vision features
        # LLM side: per-chunk reshape preserves per-action-dimension information
        #   (B, chunk_len, action_dim * input_dim), same as L1RegressionActionHead
        # Vision side: (B, chunk_len, input_dim)
        fusion_input_dim = input_dim * action_dim + input_dim

        self.fusion_mlp = MLPResNet(
            num_blocks=2,
            input_dim=fusion_input_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Process real-time images through the vision encoder.

        Args:
            pixel_values: (B, num_views * C, H, W) or (B, num_views, C, H, W)

        Returns:
            vision_features: (B, vision_embed_dim) pooled features
        """
        return self.vision_encoder(pixel_values)

    def predict_action(
        self,
        llm_hidden_states: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict actions using LLM hidden states and real-time vision.

        Args:
            llm_hidden_states: (B, chunk_len * action_dim, hidden_dim)
                              Hidden states from LLM (potentially from delayed frame)
            pixel_values: (B, num_views * C, H, W) or (B, num_views, C, H, W)
                         Real-time images for the ActionHead's vision encoder

        Returns:
            actions: (B, chunk_len, action_dim) predicted actions
        """
        batch_size = llm_hidden_states.shape[0]
        device = llm_hidden_states.device

        # Reshape LLM hidden states to preserve per-action-dimension information
        # (B, chunk_len * action_dim, hidden_dim) -> (B, chunk_len, action_dim * hidden_dim)
        # Same reshape as L1RegressionActionHead — avoids lossy mean-pooling
        llm_features = llm_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)

        if pixel_values is not None:
            # Encode real-time vision
            vision_features = self.encode_vision(pixel_values)  # (B, vision_embed_dim)

            # Project vision features to LLM dimension
            vision_proj = self.vision_projector(vision_features)  # (B, input_dim)

            # Tile vision features to match chunk length
            vision_proj = vision_proj.unsqueeze(1).expand(-1, NUM_ACTIONS_CHUNK, -1)  # (B, chunk_len, input_dim)

            # Concatenate LLM and vision features
            fused = torch.cat([llm_features, vision_proj], dim=-1)  # (B, chunk_len, action_dim*D + D)
        else:
            # No vision input — pad with zeros (fallback mode)
            zeros = torch.zeros(
                batch_size, NUM_ACTIONS_CHUNK, self.vision_projector[0].out_features,
                device=device, dtype=llm_features.dtype,
            )
            fused = torch.cat([llm_features, zeros], dim=-1)

        # Predict actions
        actions = self.fusion_mlp(fused)  # (B, chunk_len, action_dim)
        return actions
