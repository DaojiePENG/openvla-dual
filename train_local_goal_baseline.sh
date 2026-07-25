#!/bin/bash
set -euo pipefail

# ==============================================================================
# LIBERO-Goal single-frame baseline for comparison with CloudEdgeVLA.
# Trains to 200k total steps and saves every 10k, so matched intermediate
# checkpoints (including 80k) are available without stopping the run.
#
# 前台首次训练（默认从干净 Base 开始）：
#   bash ./train_local_goal_baseline.sh --gpus 0,1
#   bash ./train_local_goal_baseline.sh --num_gpus 2
#
# 前台续训（至少已有一个 checkpoint）：
#   bash ./train_local_goal_baseline.sh --gpus 0,1 --resume
#   bash ./train_local_goal_baseline.sh --gpus 0,1 --resume_from_checkpoint /path/to/checkpoint
#
# 后台首次训练（终端断开不影响）：
#   nohup bash ./train_local_goal_baseline.sh --gpus 0,1 &
#
# 后台自动续训：
#   nohup bash ./train_local_goal_baseline.sh --gpus 0,1 --resume &
#
# 自动日志（PID 会追加到文件名，同一命令多次运行不会覆盖）：
#   ls logs/train_goal_baseline_gpu2_*.log
#   tail -f logs/train_goal_baseline_gpu2_*.log
#
# 查看错误日志：
#   tail -f logs/train_goal_baseline_gpu2_*.err
#
# 中断训练：
#   kill $(cat logs/train_goal_baseline_gpu2_*.pid)
# ==============================================================================

GPU_IDS=""
NUM_GPUS=""
RUN_ID=""
RESUME=false
RESUME_FROM_CHECKPOINT="auto"
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus) GPU_IDS="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --run_id) RUN_ID="$2"; shift 2 ;;
        --resume) RESUME=true; RESUME_FROM_CHECKPOINT="auto"; shift ;;
        --resume_from_checkpoint) RESUME=true; RESUME_FROM_CHECKPOINT="$2"; shift 2 ;;
        --no_resume) RESUME=false; RESUME_FROM_CHECKPOINT=""; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
elif [ -n "$NUM_GPUS" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq 0 $((NUM_GPUS-1)) | tr '\n' ',' | sed 's/,$//')
else
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq 0 $((NUM_GPUS-1)) | tr '\n' ',' | sed 's/,$//')
fi

if [ -z "$RUN_ID" ]; then
    RUN_ID="goal_baseline_gpu${NUM_GPUS}"
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/sheng/workspace/huggingface

mkdir -p logs
LOG_PID=$$
echo ${LOG_PID} > logs/train_${RUN_ID}_${LOG_PID}.pid
exec > logs/train_${RUN_ID}_${LOG_PID}.log 2> logs/train_${RUN_ID}_${LOG_PID}.err

# Keep all architecture/data/optimizer settings matched with train_local_goal.sh.
VLA_PATH="/home/sheng/workspace/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
DATA_ROOT_DIR="/home/sheng/workspace/modified_libero_rlds"
DATASET_NAME="libero_goal_no_noops"
RUN_ROOT_DIR="/home/sheng/workspace/openvla-oft/runs_2/baseline-goal"
RESUME_ROOT_DIR="/home/sheng/workspace/openvla-oft/runs/baseline-goal"

RESUME_ARGS=(--resume "${RESUME}")
if [ "${RESUME}" = true ]; then
    RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}" --resume_root_dir "${RESUME_ROOT_DIR}")
fi

if [[ "$VLA_PATH" == "$RUN_ROOT_DIR/"* || "$VLA_PATH" == "$RESUME_ROOT_DIR/"* ]]; then
    echo "ERROR: Baseline training must start from the clean OFT base: $VLA_PATH"
    exit 1
fi

# Matched VisionActionHead architecture; only paired-frame training is disabled.
USE_VISION_ACTION_HEAD=true
ACTION_HEAD_VISION_ENCODER="siglip-base"
FREEZE_ACTION_HEAD_VISION=true
ACTION_HEAD_NUM_VIEWS=2
USE_FRAME_DELAY=false
WINDOW_SIZE=1

BATCH_SIZE=4
GRAD_ACCUM_STEPS=1
LEARNING_RATE=0.0005
LORA_RANK=16
MAX_STEPS=200000
NUM_STEPS_BEFORE_DECAY=100000
SAVE_FREQ=10000
NUM_IMAGES=2
USE_PROPRIO=true
RUN_ID_NOTE="single_frame_w${WINDOW_SIZE}_visionAH_baseline_goal"

echo "============================================================"
echo "LIBERO-Goal Single-Frame Baseline Training"
echo "============================================================"
echo "Run ID: ${RUN_ID}"
echo "GPUs: ${NUM_GPUS} (${CUDA_VISIBLE_DEVICES})"
echo "Log: logs/train_${RUN_ID}_${LOG_PID}.log"
echo "Err: logs/train_${RUN_ID}_${LOG_PID}.err"
echo "PID: logs/train_${RUN_ID}_${LOG_PID}.pid"
echo "Dataset: ${DATASET_NAME}"
echo "Resume training: ${RESUME}"
echo "Resume checkpoint: ${RESUME_FROM_CHECKPOINT:-disabled}"
echo "Run root: ${RUN_ROOT_DIR}"
echo "Batch Size: ${BATCH_SIZE} (effective: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_GPUS)))"
echo "Learning Rate: ${LEARNING_RATE}"
echo "LoRA Rank: ${LORA_RANK}"
echo "VisionActionHead: ${USE_VISION_ACTION_HEAD}"
echo "Frame Delay: ${USE_FRAME_DELAY}"
echo "Window Size: ${WINDOW_SIZE}"
echo "Target Step: ${MAX_STEPS}"
echo "============================================================"
echo "Started at: $(date)"

MASTER_PORT=$((20000 + RANDOM % 10000))
echo "Using master port: ${MASTER_PORT}"

torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} vla-scripts/finetune.py \
    --vla_path "${VLA_PATH}" \
    --data_root_dir "${DATA_ROOT_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --run_root_dir "${RUN_ROOT_DIR}" \
    "${RESUME_ARGS[@]}" \
    --batch_size ${BATCH_SIZE} \
    --grad_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --lora_rank ${LORA_RANK} \
    --max_steps ${MAX_STEPS} \
    --num_steps_before_decay ${NUM_STEPS_BEFORE_DECAY} \
    --save_freq ${SAVE_FREQ} \
    --num_images_in_input ${NUM_IMAGES} \
    --use_proprio ${USE_PROPRIO} \
    --use_l1_regression true \
    --image_aug true \
    --use_lora true \
    --lora_dropout 0.0 \
    --save_latest_checkpoint_only false \
    --wandb_entity "pengdaojie-the-hong-kong-university-of-science-and-techn" \
    --wandb_project "openvla-frame-delay" \
    --run_id_note "${RUN_ID_NOTE}" \
    --use_vision_action_head ${USE_VISION_ACTION_HEAD} \
    --action_head_vision_encoder ${ACTION_HEAD_VISION_ENCODER} \
    --freeze_action_head_vision ${FREEZE_ACTION_HEAD_VISION} \
    --action_head_num_views ${ACTION_HEAD_NUM_VIEWS} \
    --use_frame_delay ${USE_FRAME_DELAY} \
    --window_size ${WINDOW_SIZE}

echo "=============================================="
echo "Baseline training completed at $(date)!"
echo "=============================================="
