#!/usr/bin/env bash
set -euo pipefail

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MEGATRON_LOGGING_LEVEL=${MEGATRON_LOGGING_LEVEL:-20}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
# export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
# export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
# export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
# export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}

# =========================
# User-configurable paths
# =========================
TOKENIZER_ARG=${TOKENIZER_ARG:-/mnt/bn/tt-search-ads-nas/shenjiarun/qwen3-c4-megatron/Qwen3-tokenizer}
TRAIN_DATA_ARG=${TRAIN_DATA_ARG:-/mnt/bn/tt-search-ads-nas/shenjiarun/c4_mcore_qwen3_train/c4_full}
VALIDATION_DATA_ARG=${VALIDATION_DATA_ARG:-/mnt/bn/tt-search-ads-nas/shenjiarun/c4_mcore_qwen3_valid/c4_full}
PRETRAIN_SCRIPT_PATH=${PRETRAIN_SCRIPT_PATH:-pretrain_gpt.py}

# =========================
# Distributed training setup
# =========================
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6015}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))

# =========================
# Qwen3-like Dense setup
# Target: ~0.40B dense params
#
# Param estimate:
#   untied embedding + lm head:
#     2 * 151936 * 768 ~= 233M
#   transformer blocks:
#     24 layers * ~6.8M/layer ~= 165M
#   total:
#     ~= 0.40B
# =========================
TP_SIZE=${TP_SIZE:-4}
CP_SIZE=${CP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-32}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}

NUM_LAYERS=${NUM_LAYERS:-24}
HIDDEN_SIZE=${HIDDEN_SIZE:-768}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-2304}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-12}
NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-4}

DTYPE=${DTYPE:-bf16}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-32768}

OPTIMIZER_TYPE=${OPTIMIZER_TYPE:-muon}
MUON_MODE=${MUON_MODE:-blockwise}              # blockwise / duplicated / distributed
MUON_USE_NESTEROV=${MUON_USE_NESTEROV:-true}
USE_MEGATRON_FSDP=${USE_MEGATRON_FSDP:-false}

# =========================
# Search controls
# =========================
# If you want strict 1x Chinchilla for dense 0.4B:
#   tokens ~= 0.4B * 20 = 8B
#   steps ~= 8B / (256 * 4096) ~= 7629
#
# If you want to keep the same 4000-step setting:
#   tokens = 256 * 4096 * 4000 ~= 4.19B
#   this is ~0.5x Chinchilla for dense 0.4B.
GRID_STEPS=${GRID_STEPS:-4000}
GRID_WARMUP=${GRID_WARMUP:-400}
EVAL_INTERVAL=${EVAL_INTERVAL:-200}
EVAL_ITERS=${EVAL_ITERS:-32}

# Fixed Muon defaults
MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
MUON_NUM_NS_STEPS=${MUON_NUM_NS_STEPS:-5}
MUON_SCALE_MODE=${MUON_SCALE_MODE:-unit_rms_norm}
MUON_EXTRA_SCALE=${MUON_EXTRA_SCALE:-1.0}
MUON_SNECV_Z_HIGH=${MUON_SNECV_Z_HIGH:-0.3}

# =========================
# Sweep grids
# =========================
# Override examples:
#   LR_GRID="1.0e-2 1.5e-2 2.0e-2"
#   WD_GRID="0.01 0.05 0.1"
#   MUON_CONFIG_MODE_GRID="blockwise muonbp snecv muon"
#   MONITOR_SIGNAL_GRID="spectral_norm_cv energy_cv directional_gram_cv"
LR_GRID=(${LR_GRID:-1.0e-2 1.5e-2 2.0e-2})
WD_GRID=(${WD_GRID:-0.01 0.05 0.1})
MUON_CONFIG_MODE_GRID=(${MUON_CONFIG_MODE_GRID:-blockwise muonbp snecv muon})
MONITOR_SIGNAL_GRID=(${MONITOR_SIGNAL_GRID:-spectral_norm_cv})

MUON_SNECV_MONITOR_SKETCH_Q=${MUON_SNECV_MONITOR_SKETCH_Q:-4}
MUON_SNECV_MONITOR_POWER_ITERS=${MUON_SNECV_MONITOR_POWER_ITERS:-2}

# Optional behavior
SKIP_EXISTING=${SKIP_EXISTING:-true}
AUTO_RESUME=${AUTO_RESUME:-true}

# Output locations
BASE_CHECKPOINT_DIR=${BASE_CHECKPOINT_DIR:-/mnt/bn/tt-search-ads-nas/shenjiarun/checkpoints/Qwen3_Dense_0p4B_muon_grid}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-${PWD}/cache_Qwen3_Dense_0p4B_bf16}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$DATA_CACHE_PATH" "$LOG_DIR" "$BASE_CHECKPOINT_DIR"

if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please run this script from the root of the Megatron-LM repository."
    exit 1
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTENTION_HEADS"
    --num-query-groups "$NUM_QUERY_GROUPS"

    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --position-embedding-type rope
    --rotary-base 1000000

    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-06
    --qk-layernorm
    --init-method-std 0.02

    --attention-backend fused
    --apply-layernorm-1p
    --untie-embeddings-and-output-weights
    --disable-bias-linear

    --optimizer "${OPTIMIZER_TYPE}"
    --logging-level 20
)

if [[ "$USE_MEGATRON_FSDP" == "true" ]]; then
    MODEL_ARGS+=(
        --use-megatron-fsdp
        --ckpt-format fsdp_dtensor
    )
fi

DTYPE_ARGS=()
if [[ "$DTYPE" == "bf16" ]]; then
    DTYPE_ARGS+=(
        --bf16
    )
elif [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        --bf16
        --fp8-format hybrid
        --fp8-amax-history-len 1024
        --fp8-amax-compute-algo max
        --fp8-param-gather
    )
else
    echo "Error: unsupported DTYPE=${DTYPE}. Supported values: bf16, fp8"
    exit 1
fi

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --context-parallel-size "$CP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
    --sequence-parallel
)

DATA_ARGS_LIST=(
    --train-data-path "$TRAIN_DATA_ARG"
    --valid-data-path "$VALIDATION_DATA_ARG"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_ARG"
    --data-cache-path "${DATA_CACHE_PATH}"
    --no-create-attention-mask-in-dataloader
    --no-mmap-bin-files
    --num-workers 8
    --vocab-size 151936
)

run_one() {
    local muon_config_mode="$1"
    local monitor_signal="$2"
    local current_lr="$3"
    local current_wd="$4"
    local current_extra_scale="$MUON_EXTRA_SCALE"

    local current_min_lr
    current_min_lr=$(awk "BEGIN {printf \"%e\", $current_lr * 1e-2}")

    local log_name
    log_name="Qwen3Dense0p4B_${muon_config_mode}_monitor_${monitor_signal}_lr${current_lr}_wd${current_wd}_TP_SIZE_${TP_SIZE}_${MUON_SNECV_Z_HIGH}_z_score_high"

    local ckpt_dir="${BASE_CHECKPOINT_DIR}/${log_name}"
    local tb_dir="${DATA_CACHE_PATH}/${log_name}"
    local log_file="${LOG_DIR}/${log_name}.log"
    local tracker_file="${ckpt_dir}/latest_checkpointed_iteration.txt"
    local -a load_args=()
    local -a tee_args=("$log_file")

    if [[ "$SKIP_EXISTING" == "true" && -f "$log_file" ]]; then
        echo "${log_name} already has a log file: ${log_file}... Reruning"
    fi

    mkdir -p "$ckpt_dir" "$tb_dir"

    if [[ "$AUTO_RESUME" == "true" && -f "$tracker_file" ]]; then
        local resume_target
        resume_target=$(<"$tracker_file")
        echo "[Info] Found checkpoint tracker at ${tracker_file}, resuming from ${resume_target}"
        load_args=(
            --load "$ckpt_dir"
            --auto-detect-ckpt-format
        )
        tee_args=(-a "$log_file")
    fi

    echo "============================================================"
    echo "Starting dense sweep run:"
    echo "  MODEL                = Qwen3Dense0p4B"
    echo "  NUM_LAYERS           = ${NUM_LAYERS}"
    echo "  HIDDEN_SIZE          = ${HIDDEN_SIZE}"
    echo "  FFN_HIDDEN_SIZE      = ${FFN_HIDDEN_SIZE}"
    echo "  NUM_ATTENTION_HEADS  = ${NUM_ATTENTION_HEADS}"
    echo "  NUM_QUERY_GROUPS     = ${NUM_QUERY_GROUPS}"
    echo "  MUON_CONFIG_MODE     = ${muon_config_mode}"
    echo "  MONITOR_SIGNAL       = ${monitor_signal}"
    echo "  LR                   = ${current_lr}"
    echo "  MIN_LR               = ${current_min_lr}"
    echo "  WEIGHT_DECAY         = ${current_wd}"
    echo "  MUON_SNECV_Z_HIGH    = ${MUON_SNECV_Z_HIGH}"
    echo "  MUON_EXTRA_SCALE     = ${current_extra_scale}"
    echo "  GRID_STEPS           = ${GRID_STEPS}"
    echo "  GRID_WARMUP          = ${GRID_WARMUP}"
    if [[ ${#load_args[@]} -gt 0 ]]; then
        echo "  RESUME_FROM          = ${ckpt_dir}"
    else
        echo "  RESUME_FROM          = <none>"
    fi
    echo "  LOG_NAME             = ${log_name}"
    echo "============================================================"

    torchrun \
        "${DISTRIBUTED_ARGS[@]}" \
        "$PRETRAIN_SCRIPT_PATH" \
        "${MODEL_ARGS[@]}" \
        --seed 42 \
        --use-flash-attn \
        --micro-batch-size "$MICRO_BATCH_SIZE" \
        --global-batch-size "$GLOBAL_BATCH_SIZE" \
        --train-iters "$GRID_STEPS" \
        --lr "$current_lr" \
        --min-lr "$current_min_lr" \
        --lr-decay-style cosine \
        --lr-decay-iters "$GRID_STEPS" \
        --lr-warmup-iters "$GRID_WARMUP" \
        --clip-grad 1.0 \
        --weight-decay "$current_wd" \
        --adam-beta1 0.9 \
        --adam-beta2 0.98 \
        "${DTYPE_ARGS[@]}" \
        --manual-gc \
        --empty-unused-memory-level 1 \
        --cross-entropy-loss-fusion \
        --no-gradient-accumulation-fusion \
        --use-checkpoint-opt_param-scheduler \
        --muon-scale-mode "$MUON_SCALE_MODE" \
        --muon-tp-mode "$MUON_MODE" \
        --muon-config-mode "$muon_config_mode" \
        --muon-momentum "$MUON_MOMENTUM" \
        --muon-num-ns-steps "$MUON_NUM_NS_STEPS" \
        --muon-extra-scale-factor "$current_extra_scale" \
        --muon-snecv-z-high "$MUON_SNECV_Z_HIGH" \
        --muon-snecv-monitor-signal "$monitor_signal" \
        --muon-snecv-monitor-sketch-q "$MUON_SNECV_MONITOR_SKETCH_Q" \
        --muon-snecv-monitor-power-iters "$MUON_SNECV_MONITOR_POWER_ITERS" \
        $( [[ "$MUON_USE_NESTEROV" == "true" ]] && echo "--muon-use-nesterov" ) \
        "${load_args[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${DATA_ARGS_LIST[@]}" \
        --log-interval 1 \
        --eval-iters "$EVAL_ITERS" \
        --eval-interval "$EVAL_INTERVAL" \
        --save-interval 1000 \
        --save "$ckpt_dir" \
        --log-throughput \
        --distributed-timeout-minutes 3600 \
        --tensorboard-dir "$tb_dir" \
        | tee "${tee_args[@]}"
}

echo "[Info] Running dense LR / WD / muon-config-mode sweep"
echo "[Info] MODEL=Qwen3Dense0p4B"
echo "[Info] LR_GRID=${LR_GRID[*]}"
echo "[Info] WD_GRID=${WD_GRID[*]}"
echo "[Info] MUON_CONFIG_MODE_GRID=${MUON_CONFIG_MODE_GRID[*]}"
echo "[Info] MONITOR_SIGNAL_GRID=${MONITOR_SIGNAL_GRID[*]}"
echo "[Info] MUON_SNECV_Z_HIGH=${MUON_SNECV_Z_HIGH}"
echo "[Info] GRID_STEPS=${GRID_STEPS}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}, SEQ_LENGTH=${SEQ_LENGTH}"

for CURRENT_MUON_CONFIG_MODE in "${MUON_CONFIG_MODE_GRID[@]}"; do
    for CURRENT_MONITOR_SIGNAL in "${MONITOR_SIGNAL_GRID[@]}"; do
        for CURRENT_LR in "${LR_GRID[@]}"; do
            for CURRENT_WD in "${WD_GRID[@]}"; do
                run_one "$CURRENT_MUON_CONFIG_MODE" "$CURRENT_MONITOR_SIGNAL" "$CURRENT_LR" "$CURRENT_WD"
            done
        done
    done
done

set +x