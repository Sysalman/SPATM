#!/usr/bin/env bash
# =========================================================
# train_gender.sh
# SPATM GENDER Bias Training (JarvisLabs)
#
# Usage:
#   chmod +x train_gender.sh
#   ./train_gender.sh                      # uses defaults below
#   ./train_gender.sh /path/to/gender_dataset.txt   # override dataset CSV
#
# Env overrides:
#   CHECKPOINT_ROOT=/home/jl_fs/checkpoints   # where checkpoints are saved
#   DATASET_ROOT=/home/jl_fs/dataset          # where datasets live
#
# NOTE: in-training validation is intentionally disabled (it regenerated all
# professions every save and stalled the run). Evaluate the finished
# checkpoint with run_gender.sh instead.
# =========================================================

set -euo pipefail

SEED=666
NOTE="gender-spatm-v3"
MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"

# Persist checkpoints + model cache on the JarvisLabs filesystem.
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/jl_fs/checkpoints}"
export HF_HOME="${HF_HOME:-/home/jl_fs/hf_cache}"

# Dataset CSV produced by generate_data.sh (BUILD_CSV=1) / build_dataset_csv.py.
DATASET_ROOT="${DATASET_ROOT:-/home/jl_fs/dataset}"
DATA_DIR="${1:-${DATASET_ROOT}/gender_dataset/gender_dataset.txt}"

if [ ! -f "$DATA_DIR" ]; then
  echo "ERROR: dataset CSV not found: $DATA_DIR"
  echo "  Build it first, e.g.:"
  echo "    DATASET_ROOT=$DATASET_ROOT BUILD_CSV=1 ./generate_data.sh gender train"
  exit 1
fi

echo "Training SPATM for GENDER bias"
echo "Model      : $MODEL_NAME"
echo "Dataset    : $DATA_DIR"
echo "Ckpt root  : $CHECKPOINT_ROOT  (folder: <timestamp>-${NOTE})"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# If accelerate has never been configured on this box, run once:  accelerate config default
accelerate launch train_spatm.py \
    --pretrained_model_name_or_path "$MODEL_NAME" \
    --train_data_dir "$DATA_DIR" \
    --bias_attribute "gender" \
    --resolution 512 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 3000 \
    --learning_rate 1e-5 \
    --lr_scheduler "constant_with_warmup" \
    --lr_warmup_steps 100 \
    --output_dir "$NOTE" \
    --seed "$SEED" \
    --save_steps 1000 \
    --checkpointing_steps 1000 \
    --checkpoints_total_limit 2 \
    --anchor_loss 0.1 \
    --train_adaptive_token_mapping \
    --is_run

# Resolve the actual (timestamped) checkpoint dir and print the eval command.
CKPT_DIR="$(ls -dt "${CHECKPOINT_ROOT}"/*-"${NOTE}" 2>/dev/null | head -1 || true)"
echo ""
echo "========================================="
echo "GENDER training complete."
if [ -n "$CKPT_DIR" ]; then
  echo "Checkpoint: $CKPT_DIR"
  echo "Evaluate with:"
  echo "  ./run_gender.sh \"$CKPT_DIR\""
else
  echo "Checkpoint under: $CHECKPOINT_ROOT (look for *-${NOTE})"
fi
echo "========================================="