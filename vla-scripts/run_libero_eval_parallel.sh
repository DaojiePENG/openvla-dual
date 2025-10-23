#!/bin/bash
#SBATCH --job-name=libero_eval_mp      # 作业名称：使用多进程(multiprocessing)的评估任务
#SBATCH --nodes=1                      # 仅使用1个计算节点（所有GPU在同一节点）
#SBATCH --ntasks=1                     # 提交1个主任务（核心：规避AssocMaxSubmitJobLimit限制）
#SBATCH --cpus-per-task=32             # 分配32个CPU核心（4任务×8核心/任务，避免CPU资源竞争）
#SBATCH --output=libero_eval_mp_%j.out # 主任务输出日志，%j为作业ID
#SBATCH --error=libero_eval_mp_%j.err  # 主任务错误日志
#SBATCH --nodelist=3090node3           # 指定运行节点（根据集群配置修改）
#SBATCH --gres=gpu:4                   # 一次性申请4张GPU（供内部4个任务分配）

##########################################################
# 环境配置部分
##########################################################

# 设置NCCL网络接口（GPU间通信专用网卡）
# 需与集群实际网络适配器名称匹配（可通过ifconfig查看）
export NCCL_SOCKET_IFNAME=eno2

# 开启NCCL调试模式（便于排查分布式通信问题，生产环境可关闭）
export NCCL_DEBUG=INFO

# 加载Conda环境（确保包含OpenVLA及所有依赖库）
source /mnt/slurmfs-4090node1/homes/dpeng108/miniforge3/bin/activate openvla

# 指定Huggingface缓存目录
export HF_HOME="/mnt/slurmfs-3090node2/user_data/dpeng108/.cache/huggingface_pdj"

# 禁用Python输出缓冲，确保日志实时写入文件
# 避免程序崩溃时丢失缓存中的关键执行信息
export PYTHONUNBUFFERED=1

##########################################################
# 执行多进程评估脚本
##########################################################

# 运行封装好的多进程脚本，该脚本会在内部启动4个进程
# 分别绑定到4张GPU，并记录每个任务的执行时间
echo "===== 开始执行多进程评估任务 ====="
echo "主任务启动时间: $(date)"
python vla-scripts/run_libero_parallel_mp.py
echo "===== 所有评估任务执行完毕 ====="
echo "主任务结束时间: $(date)"
    