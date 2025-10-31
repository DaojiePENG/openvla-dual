import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import torch  # 用于速度计算

def visualize_trajectory_actions(traj_actions_path, save_fig_path=None, epsilon=1e-6):
    # 加载轨迹动作数据
    data = np.load(traj_actions_path)
    traj_indices = sorted([int(k.split("_")[1]) for k in data.keys()])
    num_trajectories = len(traj_indices)
    
    # 准备绘图数据（按轨迹索引和时间步排列）
    max_timesteps = max(data[f"traj_{i}"].shape[0] for i in traj_indices)
    action_dim = data[f"traj_{traj_indices[0]}"].shape[1]
    action_matrix = np.full((num_trajectories, max_timesteps, action_dim), np.nan)
    
    for traj_idx in traj_indices:
        acts = data[f"traj_{traj_idx}"]
        action_matrix[traj_idx, :acts.shape[0], :] = acts

    # ====================
    # 1. 原始动作热力热力图可视化
    # ====================
    fig, axes = plt.subplots(1, action_dim, figsize=(5*action_dim, num_trajectories//5), squeeze=False)
    for dim in range(action_dim):
        ax = axes[0, dim]
        im = ax.imshow(
            action_matrix[:, :, dim],
            aspect="auto",
            cmap="viridis",
            vmin=np.nanpercentile(action_matrix, 1),  # 去除极端值影响
            vmax=np.nanpercentile(action_matrix, 99)
        )
        ax.set_title(f"Action Dimension {dim}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Trajectory Index")
        fig.colorbar(im, ax=ax, shrink=0.5)

    plt.tight_layout()
    if save_fig_path:
        plt.savefig(f"{save_fig_path}_actions.png", dpi=300, bbox_inches="tight")
        print(f"动作热力图已保存至 {save_fig_path}_actions.png")
    else:
        plt.show()

    # ====================
    # 2. 关节速度计算与可视化
    # ====================
    if action_dim >= 6:  # 确保有至少6个关节维度
        # 提取前6个关节维度 (轨迹数, 最大时间步, 6)
        joint_actions = action_matrix[..., :6]
        
        # 计算速度：每个时间步的动作向量模长（模拟速度大小）
        # 转换为torch张量便于计算
        joint_tensor = torch.tensor(joint_actions, dtype=torch.float32)
        speed = torch.norm(joint_tensor, dim=-1, keepdim=False)  # (轨迹数, 最大时间步)
        speed = torch.clamp(speed, min=epsilon)  # 避免零值
        speed_np = speed.numpy()

        # 绘制速度热力图（轨迹索引 × 时间步）
        fig, ax = plt.subplots(figsize=(10, num_trajectories//5))
        im = ax.imshow(
            speed_np,
            aspect="auto",
            cmap="plasma",
            vmin=np.nanpercentile(speed_np, 1),
            vmax=np.nanpercentile(speed_np, 99)
        )
        ax.set_title("Joint Speed (Norm of First 6 Dimensions)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Trajectory Index")
        fig.colorbar(im, ax=ax, label="Speed Magnitude")
        plt.tight_layout()

        if save_fig_path:
            plt.savefig(f"{save_fig_path}_speed.png", dpi=300, bbox_inches="tight")
            print(f"关节速度图已保存至 {save_fig_path}_speed.png")
        else:
            plt.show()

        # 绘制速度分布直方图（所有轨迹的速度分布）
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            speed_np[~np.isnan(speed_np)].flatten(),
            bins=50,
            color="orange",
            alpha=0.7
        )
        ax.set_title("Distribution of Joint Speeds")
        ax.set_xlabel("Speed Magnitude")
        ax.set_ylabel("Frequency")
        plt.tight_layout()

        if save_fig_path:
            plt.savefig(f"{save_fig_path}_speed_dist.png", dpi=300, bbox_inches="tight")
            print(f"速度分布图已保存至 {save_fig_path}_speed_dist.png")
        else:
            plt.show()
    else:
        print("动作维度不足6，无法计算关节速度")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-actions-path", required=True, help="Path to trajectory_actions_xxx.npz")
    parser.add_argument("--save-fig", type=str, default="./data/action_vis",help="Path prefix to save visualization figures")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Minimum value for speed clamping")
    args = parser.parse_args()
    
    visualize_trajectory_actions(args.traj_actions_path, args.save_fig, args.epsilon)

# Example usage:
# python scripts/visualize_trajectory_actions_v1.py --traj-actions-path /mnt/slurmfs-3090node2/user_data/dpeng108/libero_spatial_no_noops/1.0.0/trajectory_actions_29e7fafa6c056eb48f40a23aa004fa135f6cb44282648763359ec81f932b8b35.npz --save-fig ./data/action_vis
