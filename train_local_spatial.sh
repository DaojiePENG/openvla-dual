#!/bin/bash
set -euo pipefail

# ==============================================================================
# Local Training: Frame Delay + VisionActionHead for async deployment
#
# Usage:
#   # 前台运行
#   ./train_local_spatial.sh --gpus 0,1,2,3       # Use specific GPU IDs
#   ./train_local_spatial.sh --num_gpus 4          # Use first 4 GPUs
#   ./train_local_spatial.sh                       # Use all GPUs; auto-resume latest checkpoint by default
#   ./train_local_spatial.sh --no_resume           # Explicitly start a fresh run
#   ./train_local_spatial.sh --resume_from_checkpoint /path/to/checkpoint
#
#   # 后台运行 (终端断开不影响)
#   nohup bash ./train_local_spatial.sh --num_gpus 2 &
#   nohup bash ./train_local_spatial.sh --gpus 4,5 &
#
#   # 自动日志: logs/train_gpu2_3850979.log (PID自动追加)
#   # 同一命令多次运行，日志不会覆盖
#
#   # 查看日志
#   ls logs/train_gpu2_*.log
#   tail -f logs/train_gpu2_*.log
#
#   # 中断训练
#   kill $(cat logs/train_gpu2_*.pid)
# ==============================================================================

# Parse arguments
GPU_IDS=""
NUM_GPUS=""
RUN_ID=""
RESUME=true
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

# Generate run ID: gpu count (自动覆盖，不手动指定)
if [ -z "$RUN_ID" ]; then
    RUN_ID="gpu${NUM_GPUS:-all}"
fi

# Configure GPUs
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
elif [ -n "$NUM_GPUS" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq 0 $((NUM_GPUS-1)) | tr '\n' ',' | sed 's/,$//')
else
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq 0 $((NUM_GPUS-1)) | tr '\n' ',' | sed 's/,$//')
fi

# ==============================================================================
# Training: Frame Delay + VisionActionHead for async deployment
#
# Method: VLA runs on both delayed and current frames each step.
#   - delayed frame → stale VLA latents (simulates cloud LLM processing old frame)
#   - current frame → fresh VLA latents
#   - VisionActionHead fuses either latent with real-time vision → both predict gt actions
#   - Loss = L1(action_fresh, gt) + L1(action_stale, gt)
#
# This teaches VisionActionHead to correct stale latents using real-time vision.
# ==============================================================================

# Environment setup
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com

export HF_HOME=/home/sheng/workspace/huggingface

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate PID for unique log files
LOG_PID=$$

# Save PID for process management
echo ${LOG_PID} > logs/train_${RUN_ID}_${LOG_PID}.pid

# Redirect output to log files (PID自动追加，同一命令不同次运行日志不同)
exec > logs/train_${RUN_ID}_${LOG_PID}.log 2> logs/train_${RUN_ID}_${LOG_PID}.err

# Training configuration
# Fresh training must start from the clean OFT base, not any checkpoint produced by the old windowed pipeline.
VLA_PATH="/home/sheng/workspace/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
DATA_ROOT_DIR="/home/sheng/workspace/modified_libero_rlds"
DATASET_NAME="libero_spatial_no_noops"
RUN_ROOT_DIR="/home/sheng/workspace/openvla-oft/runs_2"
# Auto-resume searches both RUN_ROOT_DIR (new checkpoints) and this legacy checkpoint root.
RESUME_ROOT_DIR="/home/sheng/workspace/openvla-oft/runs"

RESUME_ARGS=(--resume "${RESUME}")
if [ "${RESUME}" = true ]; then
    RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}" --resume_root_dir "${RESUME_ROOT_DIR}")
fi

if [[ "$VLA_PATH" == "$RUN_ROOT_DIR/"* || "$VLA_PATH" == "$RESUME_ROOT_DIR/"* ]]; then
    echo "ERROR: Fresh frame-delay training cannot start from an existing run checkpoint: $VLA_PATH"
    exit 1
fi

# VisionActionHead Configuration
USE_VISION_ACTION_HEAD=true
ACTION_HEAD_VISION_ENCODER="siglip-base"
FREEZE_ACTION_HEAD_VISION=true
ACTION_HEAD_NUM_VIEWS=2

# Frame Delay Configuration
# window_size controls how many past frames the dataset provides per sample.
# Set window_size = max_delay + 1 (e.g. 11 for max delay of 10 env steps).
USE_FRAME_DELAY=true
WINDOW_SIZE=21

# Curriculum Learning for Dual-Path Loss
# λ linearly ramps from 0 → STALE_LOSS_LAMBDA_MAX over WARMUP steps, then holds.
# Set WARMUP to -1 for auto (= max_steps/2).
STALE_LOSS_LAMBDA_MAX=0.5
STALE_LOSS_WARMUP_STEPS=80000

# Training hyperparameters
BATCH_SIZE=4                    # Halved from 8: frame_delay does 2 sequential forwards, each uses half the memory
GRAD_ACCUM_STEPS=1              # Effective batch = 4 * 1 * 2gpus = 8 (same as before)
LEARNING_RATE=0.0005
LORA_RANK=16
MAX_STEPS=200000
NUM_STEPS_BEFORE_DECAY=100000
SAVE_FREQ=10000
NUM_IMAGES=2
USE_PROPRIO=true

echo "============================================================"
echo "Frame Delay + VisionActionHead Training"
echo "============================================================"
echo "Run ID: ${RUN_ID}"
echo "GPUs: ${NUM_GPUS} (${CUDA_VISIBLE_DEVICES})"
echo "Log: logs/train_${RUN_ID}_${LOG_PID}.log"
echo "Err: logs/train_${RUN_ID}_${LOG_PID}.err"
echo "PID: logs/train_${RUN_ID}_${LOG_PID}.pid"
echo "Dataset: ${DATASET_NAME}"
echo "Resume training: ${RESUME}"
echo "Resume checkpoint: ${RESUME_FROM_CHECKPOINT:-disabled}"
echo "Resume search root: ${RESUME_ROOT_DIR}"
echo "Batch Size: ${BATCH_SIZE} (effective: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_GPUS)))"
echo "Grad Accum Steps: ${GRAD_ACCUM_STEPS}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "LoRA Rank: ${LORA_RANK}"
echo "VisionActionHead: ${USE_VISION_ACTION_HEAD}"
echo "Vision Encoder: ${ACTION_HEAD_VISION_ENCODER}"
echo "Num Views: ${ACTION_HEAD_NUM_VIEWS}"
echo "Frame Delay: ${USE_FRAME_DELAY}"
echo "Window Size: ${WINDOW_SIZE}"
echo "Action Target: current step + 7 future steps"
echo "Stale Loss λ_max: ${STALE_LOSS_LAMBDA_MAX}"
echo "Stale Loss Warmup: ${STALE_LOSS_WARMUP_STEPS} (-1 = auto)"
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
    --run_id_note "frame_delay_w${WINDOW_SIZE}_visionAH_aligned" \
    --use_vision_action_head ${USE_VISION_ACTION_HEAD} \
    --action_head_vision_encoder ${ACTION_HEAD_VISION_ENCODER} \
    --freeze_action_head_vision ${FREEZE_ACTION_HEAD_VISION} \
    --action_head_num_views ${ACTION_HEAD_NUM_VIEWS} \
    --use_frame_delay ${USE_FRAME_DELAY} \
    --window_size ${WINDOW_SIZE} \
    --stale_loss_lambda_max ${STALE_LOSS_LAMBDA_MAX} \
    --stale_loss_warmup_steps ${STALE_LOSS_WARMUP_STEPS}

echo "=============================================="
echo "Training completed at $(date)!"
echo "=============================================="
