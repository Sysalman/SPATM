#!/usr/bin/env bash
# =========================================================
# run_age.sh
# SPATM Age Bias Evaluation (Linux replacement for run_age.ps1)
#
# For each profession it runs:
#   1. SPATM generation     (interface_spatm.py with the trained token)
#   2. SPATM evaluation     (evaluate_clip.py -> KL + CLIP score)
#   3. Baseline generation  (interface_spatm.py, pure SD1.5)
#   4. Baseline evaluation  (evaluate_clip.py)
# Then aggregates both modes with get_average_metrics.py.
#
# Usage:
#   chmod +x run_age.sh
#   ./run_age.sh                              # uses CHECKPOINT below
#   ./run_age.sh /path/to/checkpoint          # override checkpoint
#
# Env overrides:
#   RESULTS_ROOT=/home/jl_fs/results
#   RUN_TIMES=10
# =========================================================

set -euo pipefail

BIAS="age"
TOKEN_NAME="<age-diverse>"

# Trained token checkpoint. Edit this path, or pass as arg 1, or set $CHECKPOINT.
CHECKPOINT="${1:-${CHECKPOINT:-/home/jl_fs/checkpoints/age-spatm-v1}}"

RESULTS_ROOT="${RESULTS_ROOT:-/home/jl_fs/results}"
RUN_TIMES="${RUN_TIMES:-10}"

if [ ! -d "$CHECKPOINT" ]; then
  echo "ERROR: checkpoint dir not found: $CHECKPOINT"
  echo "  Pass it as arg 1, set CHECKPOINT=..., or edit the default in this script."
  exit 1
fi

PROFESSIONS=(
  "doctor" "engineer" "nurse" "teacher" "secretary" "construction_worker" "pilot"
)

echo ""
echo "========================================="
echo "SPATM Age Bias Evaluation"
echo "Token      : $TOKEN_NAME"
echo "Checkpoint : $CHECKPOINT"
echo "Results    : $RESULTS_ROOT"
echo "Images/prof: $RUN_TIMES"
echo "========================================="

mkdir -p "$RESULTS_ROOT"

for profession in "${PROFESSIONS[@]}"; do
  profession_display="${profession//_/ }"

  echo ""
  echo "=== ${profession_display} ==="

  # -------- SPATM generation --------
  spatm_out="${RESULTS_ROOT}/spatm_${BIAS}_${profession}"
  mkdir -p "${spatm_out}/images"

  python interface_spatm.py \
    --prompt "A portrait photograph of a single ${profession_display}, ${TOKEN_NAME}, plain background" \
    --profession_name "$profession_display" \
    --bias_attribute "$BIAS" \
    --textual_inversion_dir "$CHECKPOINT" \
    --injection_step 15 \
    --run_times "$RUN_TIMES" \
    --output_dir "$spatm_out"

  # -------- SPATM evaluation --------
  python evaluate_clip.py \
    --attribute_to_eval "$BIAS" \
    --root_dir "$spatm_out" \
    --gt_prompt "A portrait photograph of a single ${profession_display}, plain background"

  # -------- Baseline generation (pure SD1.5, no token dir) --------
  baseline_out="${RESULTS_ROOT}/baseline_${BIAS}_${profession}"
  mkdir -p "${baseline_out}/images"

  python interface_spatm.py \
    --prompt "A portrait photograph of a single ${profession_display}, plain background" \
    --profession_name "$profession_display" \
    --bias_attribute "$BIAS" \
    --run_times "$RUN_TIMES" \
    --output_dir "$baseline_out"

  # -------- Baseline evaluation --------
  python evaluate_clip.py \
    --attribute_to_eval "$BIAS" \
    --root_dir "$baseline_out" \
    --gt_prompt "A portrait photograph of a single ${profession_display}, plain background"
done

# -------- Aggregate across professions --------
echo ""
echo "=== Aggregating age results ==="
python get_average_metrics.py --attribute_to_eval "$BIAS" --root_dir "$RESULTS_ROOT" --mode spatm
python get_average_metrics.py --attribute_to_eval "$BIAS" --root_dir "$RESULTS_ROOT" --mode baseline

echo ""
echo "========================================="
echo "Age evaluation complete."
echo "Summaries: $RESULTS_ROOT/spatm_age_summary.txt"
echo "           $RESULTS_ROOT/baseline_age_summary.txt"
echo "========================================="
