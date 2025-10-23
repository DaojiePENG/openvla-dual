import os
import time
import torch
import subprocess
from datetime import datetime
from torch.multiprocessing import Process

# --------------------------
# 1. 任务配置：定义4个评估任务的参数
#    每个任务包含：任务名称、模型路径、绑定的GPU编号和日志文件名
# --------------------------
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
# 确保日志目录存在（如果不存在则创建）
LOG_DIR = "experiments/logs_openvla_maxwindow20"
os.makedirs(LOG_DIR, exist_ok=True)  # exist_ok=True 避免目录已存在时报错

TASKS_OPENVLA_OFT = [
    {
        "task_suite": "libero_spatial",  # 任务套件名称
        "checkpoint": "moojink/openvla-7b-oft-finetuned-libero-spatial",  # 预训练模型路径
        "gpu_id": 0,  # 绑定到第0张GPU
        "log_file": f"{LOG_DIR}/libero_spatial_eval-{DATE_TIME}.log"  # 单独的日志文件（通过重定向实现）
    },
    {
        "task_suite": "libero_object",
        "checkpoint": "moojink/openvla-7b-oft-finetuned-libero-object",
        "gpu_id": 1,  # 绑定到第1张GPU
        "log_file": f"{LOG_DIR}/libero_object_eval-{DATE_TIME}.log"
    },
    {
        "task_suite": "libero_goal",
        "checkpoint": "moojink/openvla-7b-oft-finetuned-libero-goal",
        "gpu_id": 2,  # 绑定到第2张GPU
        "log_file": f"{LOG_DIR}/libero_goal_eval-{DATE_TIME}.log"
    },
    {
        "task_suite": "libero_10",
        "checkpoint": "moojink/openvla-7b-oft-finetuned-libero-10",
        "gpu_id": 3,  # 绑定到第3张GPU
        "log_file": f"{LOG_DIR}/libero_10_eval-{DATE_TIME}.log"
    }
]

TASKS_OPENVLA_ORIGINAL = [
    {
        "task_suite": "libero_spatial",  # 任务套件名称
        "checkpoint": "openvla/openvla-7b-finetuned-libero-spatial",  # 预训练模型路径
        "gpu_id": 0,  # 绑定到第0张GPU
        "log_file": f"{LOG_DIR}/libero_spatial_eval-{DATE_TIME}.log"  # 单独的日志文件（通过重定向实现）
    },
    {
        "task_suite": "libero_object",
        "checkpoint": "openvla/openvla-7b-finetuned-libero-object",
        "gpu_id": 1,  # 绑定到第1张GPU
        "log_file": f"{LOG_DIR}/libero_object_eval-{DATE_TIME}.log"
    },
    {
        "task_suite": "libero_goal",
        "checkpoint": "openvla/openvla-7b-finetuned-libero-goal",
        "gpu_id": 2,  # 绑定到第2张GPU
        "log_file": f"{LOG_DIR}/libero_goal_eval-{DATE_TIME}.log"
    },
    {
        "task_suite": "libero_10",
        "checkpoint": "openvla/openvla-7b-finetuned-libero-10",
        "gpu_id": 3,  # 绑定到第3张GPU
        "log_file": f"{LOG_DIR}/libero_10_eval-{DATE_TIME}.log"
    }
]

TASKS_UNIVLA = [
    {
        "task_suite": "libero_spatial",  # 任务套件名称
        "checkpoint": "/mnt/slurmfs-3090node2/user_data/dpeng108/.cache/huggingface/hub/models--qwbu--univla-7b-224-sft-libero/snapshots/0d1979ebd1b7ad00d6a8a02a98213a0835dc82a1/univla-libero-spatial",  # 预训练模型路径
        "gpu_id": 0,  # 绑定到第0张GPU
        "log_file": f"{LOG_DIR}/libero_spatial_eval-{DATE_TIME}.log"  # 单独的日志文件（通过重定向实现）
    },
    {
        "task_suite": "libero_object",
        "checkpoint": "/mnt/slurmfs-3090node2/user_data/dpeng108/.cache/huggingface/hub/models--qwbu--univla-7b-224-sft-libero/snapshots/0d1979ebd1b7ad00d6a8a02a98213a0835dc82a1/univla-libero-object",
        "gpu_id": 1,  # 绑定到第1张GPU
        "log_file": f"{LOG_DIR}/libero_object_eval-{DATE_TIME}.log"
    },
    {
        "task_suite": "libero_goal",
        "checkpoint": "/mnt/slurmfs-3090node2/user_data/dpeng108/.cache/huggingface/hub/models--qwbu--univla-7b-224-sft-libero/snapshots/0d1979ebd1b7ad00d6a8a02a98213a0835dc82a1/univla-libero-goal",
        "gpu_id": 2,  # 绑定到第2张GPU
        "log_file": f"{LOG_DIR}/libero_goal_eval-{DATE_TIME}.log"
    },
    {
        "task_suite": "libero_10",
        "checkpoint": "/mnt/slurmfs-3090node2/user_data/dpeng108/.cache/huggingface/hub/models--qwbu--univla-7b-224-sft-libero/snapshots/0d1979ebd1b7ad00d6a8a02a98213a0835dc82a1/univla-libero-10",
        "gpu_id": 3,  # 绑定到第3张GPU
        "log_file": f"{LOG_DIR}/libero_10_eval-{DATE_TIME}.log"
    }
]

TASKS = TASKS_OPENVLA_ORIGINAL
# --------------------------
# 2. 单个任务执行函数：使用输出重定向记录日志，不依赖--log_file参数
# --------------------------
def run_single_task(task):
    """
    执行单个评估任务，绑定到指定GPU，通过输出重定向记录日志并统计时间
    
    参数:
        task (dict): 任务配置字典，包含task_suite、checkpoint、gpu_id等信息
    """
    # 2.1 记录任务启动时间（精确到毫秒）
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 保留毫秒
    
    # 2.2 强制当前进程只可见指定的GPU（核心：避免GPU资源冲突）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(task["gpu_id"])
    
    # # 2.3 拼接评估命令（不含--log_file参数，保持与原脚本兼容）
    # base_cmd = [
    #     "python", "experiments/robot/libero/run_libero_eval.py",
    #     "--pretrained_checkpoint", task["checkpoint"],
    #     "--task_suite_name", task["task_suite"],
    #     "--local_log_dir", LOG_DIR,
    # ]
    # 2.3 拼接评估命令（不含--log_file参数，保持与原脚本兼容）
    if "univla" in task["checkpoint"]:
        # univla模型需要额外的action_decoder参数
        base_cmd = [
            "python", "experiments/robot/libero/run_libero_eval.py",
            "--action_decoder_path", task["checkpoint"] + "/action_decoder.pt",
            "--pretrained_checkpoint", task["checkpoint"],
            "--task_suite_name", task["task_suite"],
            "--local_log_dir", LOG_DIR,
        ]
    else:
        base_cmd = [
            "python", "experiments/robot/libero/run_libero_eval.py",
            "--pretrained_checkpoint", task["checkpoint"],
            "--task_suite_name", task["task_suite"],
            "--local_log_dir", LOG_DIR,
        ]
    
    # 2.4 准备日志文件：创建或清空日志文件，添加时间戳标记
    with open(task["log_file"], "w") as f:
        f.write(f"=== 任务 {task['task_suite']} 启动日志 ===\n")
        f.write(f"启动时间: {start_datetime}\n")
        f.write(f"绑定GPU: {task['gpu_id']}\n")
        f.write(f"执行命令: {' '.join(base_cmd)}\n")
        f.write(f"时间戳: {DATE_TIME}\n")  # 记录本次任务的统一时间戳
        f.write("=================================\n\n")
    
    # 2.5 执行命令并通过重定向捕获输出（替代--log_file参数）
    print(f"[GPU {task['gpu_id']}] 任务启动: {task['task_suite']}")
    print(f"[GPU {task['gpu_id']}] 启动时间: {start_datetime}")
    print(f"[GPU {task['gpu_id']}] 日志文件: {task['log_file']}")
    
    try:
        # 使用重定向将stdout和stderr都写入日志文件
        with open(task["log_file"], "a") as f:
            result = subprocess.run(
                base_cmd,
                stdout=f,       # 标准输出写入日志
                stderr=subprocess.STDOUT,  # 错误输出也写入日志（便于调试）
                text=True       # 输出为字符串格式
            )
    except Exception as e:
        # 捕获执行过程中的异常（如文件权限问题）
        with open(task["log_file"], "a") as f:
            f.write(f"\n执行过程中发生异常: {str(e)}\n")
        result = subprocess.CompletedProcess(args=base_cmd, returncode=1)
    
    # 2.6 记录任务结束时间并计算耗时
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60  # 转换为分钟便于阅读
    
    # 2.7 在日志文件末尾添加结束标记和时间统计
    with open(task["log_file"], "a") as f:
        f.write(f"\n=================================\n")
        f.write(f"=== 任务 {task['task_suite']} 结束日志 ===\n")
        f.write(f"结束时间: {end_datetime}\n")
        f.write(f"总耗时: {elapsed_seconds:.2f}秒 ({elapsed_minutes:.2f}分钟)\n")
        f.write(f"任务状态: {'成功' if result.returncode == 0 else f'失败（返回码: {result.returncode}）'}\n")
    
    # 2.8 在主日志中打印任务完成状态
    status = "成功" if result.returncode == 0 else f"失败（返回码: {result.returncode}）"
    print(f"\n[GPU {task['gpu_id']}] 任务结束: {task['task_suite']} ({status})")
    print(f"[GPU {task['gpu_id']}] 结束时间: {end_datetime}")
    print(f"[GPU {task['gpu_id']}] 总耗时: {elapsed_seconds:.2f}秒 ({elapsed_minutes:.2f}分钟)")
    print(f"[GPU {task['gpu_id']}] 详细日志: {task['log_file']}\n")

# --------------------------
# 3. 主函数：启动所有任务并等待完成
# --------------------------
if __name__ == "__main__":
    # 3.1 验证GPU数量是否满足需求
    available_gpus = torch.cuda.device_count()
    print(f"检测到可用GPU数量: {available_gpus}")
    
    if available_gpus < 4:
        raise RuntimeError(f"错误: 至少需要4张GPU，当前节点仅检测到{available_gpus}张！")
    
    # 3.2 创建并启动进程
    print("\n===== 开始启动所有评估任务 =====")
    main_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"主进程启动时间: {main_start_time}\n")
    
    processes = []
    for task in TASKS:
        p = Process(target=run_single_task, args=(task,))
        processes.append(p)
        p.start()  # 启动进程
    
    # 3.3 等待所有进程完成
    for p in processes:
        p.join()  # 等待进程结束
    
    # 3.4 打印所有任务完成信息
    main_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n===== 所有评估任务执行完毕 =====")
    print(f"主进程结束时间: {main_end_time}")
    print(f"各任务日志文件: {[t['log_file'] for t in TASKS]}")
    