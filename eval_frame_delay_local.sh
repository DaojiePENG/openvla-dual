#!/bin/bash
# Local evaluation for frame-delay VisionActionHead checkpoints on LIBERO suites.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

GPU_IDS="0,1"
NUM_TRIALS=50
STAGGER_DELAY=30
TASKS_PER_GPU=4
DELAYS_CSV="0,5,10,15,20"
CHECKPOINT="${EVAL_CHECKPOINT:-}"
TASK_SUITE_NAME="libero_goal"
SEED=7
MAX_TRAIN_DELAY=20
RESPECT_EXISTING_EVALUATORS=false
MIN_GPU_FREE_MIB=0
MIN_RAM_AVAILABLE_KIB=0
MIN_DISK_FREE_KIB=0
RESOURCE_POLL_SECONDS=30
WAIT_LOG_SECONDS=300
SAVE_ROLLOUTS=true
LOG_DIR_OVERRIDE=""

usage() {
    echo "Usage: $0 --checkpoint PATH [--gpus 0,1] [--num_trials 50] [--delays 0,5,10,15,20]"
    echo "          [--task_suite_name libero_goal] [--tasks_per_gpu 4] [--stagger 30] [--seed 7]"
    echo "          [--respect_existing_evaluators true] [--min_gpu_free_mib 20000]"
    echo "          [--min_ram_available_kib 67108864] [--min_disk_free_kib 10737418240]"
    echo "          [--resource_poll_seconds 30] [--wait_log_seconds 300]"
    echo "          [--save_rollouts true|false]"
    echo "          [--log_dir PATH]"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --gpus) GPU_IDS="$2"; shift 2 ;;
        --num_trials) NUM_TRIALS="$2"; shift 2 ;;
        --delays) DELAYS_CSV="$2"; shift 2 ;;
        --task_suite_name) TASK_SUITE_NAME="$2"; shift 2 ;;
        --tasks_per_gpu) TASKS_PER_GPU="$2"; shift 2 ;;
        --stagger) STAGGER_DELAY="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --respect_existing_evaluators) RESPECT_EXISTING_EVALUATORS="$2"; shift 2 ;;
        --min_gpu_free_mib) MIN_GPU_FREE_MIB="$2"; shift 2 ;;
        --min_ram_available_kib) MIN_RAM_AVAILABLE_KIB="$2"; shift 2 ;;
        --min_disk_free_kib) MIN_DISK_FREE_KIB="$2"; shift 2 ;;
        --resource_poll_seconds) RESOURCE_POLL_SECONDS="$2"; shift 2 ;;
        --wait_log_seconds) WAIT_LOG_SECONDS="$2"; shift 2 ;;
        --save_rollouts) SAVE_ROLLOUTS="$2"; shift 2 ;;
        --log_dir) LOG_DIR_OVERRIDE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: --checkpoint is required. Do not evaluate an old frame-delay checkpoint by accident."
    usage
    exit 1
fi
if [[ ! -d "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

shopt -s nullglob
ACTION_HEAD_FILES=("$CHECKPOINT"/action_head--*_checkpoint.pt)
PROPRIO_FILES=("$CHECKPOINT"/proprio_projector--*_checkpoint.pt)
if [[ ${#ACTION_HEAD_FILES[@]} -ne 1 || ${#PROPRIO_FILES[@]} -ne 1 ]]; then
    echo "ERROR: Expected exactly one action head and one proprio projector checkpoint in $CHECKPOINT"
    exit 1
fi
if [[ ! -f "$CHECKPOINT/model.safetensors.index.json" || ! -f "$CHECKPOINT/dataset_statistics.json" ]]; then
    echo "ERROR: Checkpoint is missing merged model weights or dataset statistics: $CHECKPOINT"
    exit 1
fi

source /home/sheng/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft-eval

export HF_HOME="/home/sheng/workspace/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL_SCRIPT="experiments/robot/libero/run_libero_eval.py"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
IFS=',' read -r -a DELAY_ARRAY <<< "$DELAYS_CSV"
NUM_GPUS=${#GPU_ARRAY[@]}

if [[ $NUM_GPUS -eq 0 || $TASKS_PER_GPU -lt 1 ]]; then
    echo "ERROR: At least one GPU and one task per GPU are required."
    exit 1
fi
if [[ ${#DELAY_ARRAY[@]} -eq 0 ]]; then
    echo "ERROR: At least one evaluation delay is required."
    exit 1
fi
if [[ "$RESPECT_EXISTING_EVALUATORS" != "true" && "$RESPECT_EXISTING_EVALUATORS" != "false" ]]; then
    echo "ERROR: --respect_existing_evaluators must be true or false."
    exit 1
fi
if [[ "$SAVE_ROLLOUTS" != "true" && "$SAVE_ROLLOUTS" != "false" ]]; then
    echo "ERROR: --save_rollouts must be true or false."
    exit 1
fi
for LIMIT in "$MIN_GPU_FREE_MIB" "$MIN_RAM_AVAILABLE_KIB" "$MIN_DISK_FREE_KIB" "$RESOURCE_POLL_SECONDS" "$WAIT_LOG_SECONDS"; do
    if [[ ! "$LIMIT" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Resource safety thresholds must be non-negative integers."
        exit 1
    fi
done
if (( RESOURCE_POLL_SECONDS < 1 || WAIT_LOG_SECONDS < 1 )); then
    echo "ERROR: Resource polling and wait logging intervals must be at least 1 second."
    exit 1
fi

for DELAY in "${DELAY_ARRAY[@]}"; do
    if [[ ! "$DELAY" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Invalid delay '$DELAY'; delays must be non-negative integers."
        exit 1
    fi
    if (( DELAY > MAX_TRAIN_DELAY )); then
        echo "WARNING: delay=$DELAY exceeds the training range 0-$MAX_TRAIN_DELAY environment steps."
    fi
done

COMMON_ARGS=(
    --use_l1_regression True
    --use_diffusion False
    --use_film False
    --num_images_in_input 2
    --use_proprio True
    --lora_rank 16
    --center_crop True
    --num_trials_per_task "$NUM_TRIALS"
    --num_open_loop_steps 8
    --use_vision_action_head True
    --action_head_vision_encoder siglip-base
    --freeze_action_head_vision True
    --action_head_num_views 2
    --task_suite_name "$TASK_SUITE_NAME"
    --seed "$SEED"
    --save_rollouts "$SAVE_ROLLOUTS"
)

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
if [[ -n "$LOG_DIR_OVERRIDE" ]]; then
    LOG_DIR="$LOG_DIR_OVERRIDE"
else
    LOG_DIR="logs/eval_frame_delay_local_${TIMESTAMP}"
fi
mkdir -p "$LOG_DIR"
START_SECONDS=$SECONDS

declare -a GPU_SLOTS=()
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_SLOTS+=(0)
done

declare -A PID_GPU_IDX=()
declare -A PID_LABEL=()
declare -A TASK_RESULTS=()
RUNNING=0
SUCCEEDED=0
FAILED=0
SELECTED_GPU_IDX=0
LAST_WAIT_LOG_SECONDS=0

stop_children() {
    trap - INT TERM
    echo "Stopping ${RUNNING} evaluation process(es)..."
    for PID in "${!PID_GPU_IDX[@]}"; do
        kill "$PID" 2>/dev/null || true
    done
    wait || true
    exit 130
}
trap stop_children INT TERM

reap_finished() {
    local PID GPU_IDX LABEL EXIT_CODE
    for PID in "${!PID_GPU_IDX[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            continue
        fi

        GPU_IDX=${PID_GPU_IDX[$PID]}
        LABEL=${PID_LABEL[$PID]}
        if wait "$PID"; then
            EXIT_CODE=0
        else
            EXIT_CODE=$?
        fi

        GPU_SLOTS[$GPU_IDX]=$((GPU_SLOTS[$GPU_IDX] - 1))
        RUNNING=$((RUNNING - 1))
        unset 'PID_GPU_IDX[$PID]'
        unset 'PID_LABEL[$PID]'

        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "[OK]     $LABEL"
            SUCCEEDED=$((SUCCEEDED + 1))
            TASK_RESULTS[$LABEL]="OK"
        else
            echo "[FAILED] $LABEL (exit code: $EXIT_CODE)"
            FAILED=$((FAILED + 1))
            TASK_RESULTS[$LABEL]="FAILED"
        fi
    done
}

gpu_has_libero_evaluator() {
    local target_gpu=$1
    local pid visible gpu
    local saw_unknown=false

    while read -r pid; do
        [[ -n "$pid" && -r "/proc/$pid/environ" ]] || continue
        visible=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -n 1)
        if [[ -z "$visible" ]]; then
            saw_unknown=true
            continue
        fi
        IFS=',' read -r -a visible_gpus <<< "$visible"
        for gpu in "${visible_gpus[@]}"; do
            if [[ "$gpu" == "$target_gpu" ]]; then
                return 0
            fi
        done
    done < <(pgrep -u "$(id -u)" -f '[r]un_libero_eval.py' || true)

    # An evaluator whose physical GPU cannot be identified is treated
    # conservatively as occupying every candidate GPU.
    [[ "$saw_unknown" == "true" ]]
}

global_resources_allow_launch() {
    local ram_available_kib disk_free_kib

    if (( MIN_RAM_AVAILABLE_KIB > 0 )); then
        ram_available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        [[ "$ram_available_kib" =~ ^[0-9]+$ ]] || return 1
        (( ram_available_kib >= MIN_RAM_AVAILABLE_KIB )) || return 1
    fi

    if (( MIN_DISK_FREE_KIB > 0 )); then
        disk_free_kib=$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')
        [[ "$disk_free_kib" =~ ^[0-9]+$ ]] || return 1
        (( disk_free_kib >= MIN_DISK_FREE_KIB )) || return 1
    fi

    return 0
}

gpu_resources_allow_launch() {
    local gpu_id=$1
    local free_mib

    if (( MIN_GPU_FREE_MIB > 0 )); then
        free_mib=$(nvidia-smi --id="$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
        [[ "$free_mib" =~ ^[0-9]+$ ]] || return 1
        (( free_mib >= MIN_GPU_FREE_MIB )) || return 1
    fi

    return 0
}

select_available_gpu() {
    local now gpu_id
    while true; do
        reap_finished
        if global_resources_allow_launch; then
            for ((i=0; i<NUM_GPUS; i++)); do
                (( GPU_SLOTS[i] < TASKS_PER_GPU )) || continue
                gpu_id=${GPU_ARRAY[$i]}
                if [[ "$RESPECT_EXISTING_EVALUATORS" == "true" ]] && gpu_has_libero_evaluator "$gpu_id"; then
                    continue
                fi
                gpu_resources_allow_launch "$gpu_id" || continue
                SELECTED_GPU_IDX=$i
                return
            done
        fi
        now=$SECONDS
        if (( now - LAST_WAIT_LOG_SECONDS >= WAIT_LOG_SECONDS )); then
            echo "[$(date '+%H:%M:%S')] Waiting for an evaluator-free GPU and safe GPU/RAM/storage headroom..."
            LAST_WAIT_LOG_SECONDS=$now
        fi
        sleep "$RESOURCE_POLL_SECONDS"
    done
}

echo "============================================================"
echo "Frame Delay Evaluation: $TASK_SUITE_NAME"
echo "Checkpoint: $(basename "$CHECKPOINT")"
echo "GPUs: $GPU_IDS | tasks/GPU: $TASKS_PER_GPU"
echo "Delays (environment steps): $DELAYS_CSV"
echo "Trials per task: $NUM_TRIALS | seed: $SEED"
echo "Save rollout videos: $SAVE_ROLLOUTS"
echo "Respect existing evaluators: $RESPECT_EXISTING_EVALUATORS"
if [[ "$RESPECT_EXISTING_EVALUATORS" == "true" ]]; then
    echo "Safety thresholds: GPU ${MIN_GPU_FREE_MIB} MiB free, RAM ${MIN_RAM_AVAILABLE_KIB} KiB available, disk ${MIN_DISK_FREE_KIB} KiB free"
    echo "Resource polling: every ${RESOURCE_POLL_SECONDS}s | wait status: every ${WAIT_LOG_SECONDS}s"
fi
echo "Logs: $LOG_DIR"
echo "============================================================"

TOTAL_TASKS=${#DELAY_ARRAY[@]}
for TASK_IDX in "${!DELAY_ARRAY[@]}"; do
    DELAY=${DELAY_ARRAY[$TASK_IDX]}
    if [[ "$DELAY" -eq 0 ]]; then
        LABEL="baseline_d0_seed${SEED}"
        DELAY_ARGS=(--use_frame_delay_eval false)
    else
        LABEL="frame_delay_d${DELAY}_seed${SEED}"
        DELAY_ARGS=(--use_frame_delay_eval true --max_delay_steps_eval "$DELAY")
    fi

    select_available_gpu
    GPU_IDX=$SELECTED_GPU_IDX
    GPU_ID=${GPU_ARRAY[$GPU_IDX]}
    TASK_LOG="$LOG_DIR/${LABEL}.log"
    TASK_ERR="$LOG_DIR/${LABEL}.err"

    if [[ -e "$TASK_LOG" || -e "$TASK_ERR" ]]; then
        echo "ERROR: Refusing to overwrite existing logs for $LABEL in $LOG_DIR"
        exit 1
    fi

    echo "[$(date '+%H:%M:%S')] Task $((TASK_IDX + 1))/$TOTAL_TASKS: delay=$DELAY -> GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python "$EVAL_SCRIPT" \
        --pretrained_checkpoint "$CHECKPOINT" \
        "${COMMON_ARGS[@]}" \
        "${DELAY_ARGS[@]}" \
        --run_id_note "$LABEL" \
        > "$TASK_LOG" 2> "$TASK_ERR" &

    PID=$!
    PID_GPU_IDX[$PID]=$GPU_IDX
    PID_LABEL[$PID]=$LABEL
    GPU_SLOTS[$GPU_IDX]=$((GPU_SLOTS[$GPU_IDX] + 1))
    RUNNING=$((RUNNING + 1))

    if (( TASK_IDX + 1 < TOTAL_TASKS && STAGGER_DELAY > 0 )); then
        sleep "$STAGGER_DELAY"
    fi
done

while (( RUNNING > 0 )); do
    reap_finished
    if (( RUNNING > 0 )); then
        sleep 2
    fi
done

echo "============================================================"
echo "Evaluation Complete"
echo "Succeeded: $SUCCEEDED/$TOTAL_TASKS"
echo "Failed:    $FAILED/$TOTAL_TASKS"
echo "Duration:  $((SECONDS - START_SECONDS)) seconds"
echo "Logs:      $LOG_DIR"
echo "============================================================"

for DELAY in "${DELAY_ARRAY[@]}"; do
    if [[ "$DELAY" -eq 0 ]]; then
        LABEL="baseline_d0_seed${SEED}"
    else
        LABEL="frame_delay_d${DELAY}_seed${SEED}"
    fi
    printf '%5s | %s\n' "$DELAY" "${TASK_RESULTS[$LABEL]:-N/A}"
done

exit "$FAILED"
