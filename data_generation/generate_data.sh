#!/usr/bin/env bash
# generate_data.sh — SPATM dataset generation (JarvisLabs)
# All data lives on the persistent filesystem /home/jl_fs so it survives pause/destroy.
# Usage:
#   ./generate_data.sh                 # all attributes, both splits
#   ./generate_data.sh gender          # gender only, both splits
#   ./generate_data.sh race train      # race only, train split
#   BUILD_CSV=1 ./generate_data.sh gender train   # also build the training CSV

set -euo pipefail

ATTRIBUTE="${1:-all}"     # gender | race | age | all
SPLIT_ARG="${2:-both}"    # train | test | both

SEED=666
RUNTIME=20
STEPS=25
MAX_ATTEMPTS=1000
DATASET_ROOT="${DATASET_ROOT:-/home/jl_fs/dataset}"
BUILD_CSV="${BUILD_CSV:-0}"

# Persistent caches (so model/detector/CLIP weights aren't re-downloaded each session)
export HF_HOME="${HF_HOME:-/home/jl_fs/hf_cache}"
export CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-/home/jl_fs/clip_cache}"
export FACEXLIB_WEIGHTS="${FACEXLIB_WEIGHTS:-/home/jl_fs/facexlib_weights}"

TRAIN_PROFESSIONS=(
  "doctor" "construction worker" "mechanic" "firefighter" "police officer"
  "engineer" "pilot" "computer programmer" "ceo" "surgeon" "truck driver"
  "electrician" "scientist" "nurse" "teacher" "secretary" "librarian"
  "receptionist" "administrative assistant" "childcare worker" "social worker"
  "dental hygienist" "therapist" "pathologist" "nutritionist"
)
TEST_PROFESSIONS=(
  "chef" "farmer" "architect" "software developer" "flight attendant"
  "athlete" "lawyer" "politician" "journalist" "barista" "detective" "security guard"
)

generate_for() {
  local bias="$1" split="$2"
  local -a CLASSES
  case "$bias" in
    gender) CLASSES=("male" "female") ;;
    race)   CLASSES=("Caucasian" "Black" "Middle Eastern" "Latino" "Indian") ;;
    age)    CLASSES=("young" "old") ;;
  esac
  local -a PROFESSIONS
  if [ "$split" = "train" ]; then PROFESSIONS=("${TRAIN_PROFESSIONS[@]}"); else PROFESSIONS=("${TEST_PROFESSIONS[@]}"); fi

  local base_dir="${DATASET_ROOT}/${bias}_dataset"
  local total=$(( ${#PROFESSIONS[@]} * ${#CLASSES[@]} )) completed=0

  echo ""
  echo "=== ${bias} / ${split} | classes: ${CLASSES[*]} | ${total} folders | out: ${base_dir}/${split} ==="

  for profession in "${PROFESSIONS[@]}"; do
    local safe_prof="${profession// /_}"
    for cls in "${CLASSES[@]}"; do
      completed=$(( completed + 1 ))
      local safe_cls="${cls// /_}"
      local prompt="a close-up photo of a ${cls} ${profession}"
      if [ "$bias" = "age" ]; then
        if [ "$cls" = "young" ]; then
          prompt="a color close-up portrait photo of a young ${profession} in their late 20s, modern DSLR photograph, sharp focus, high resolution, natural lighting"
        elif [ "$cls" = "old" ]; then
          prompt="a color close-up portrait photo of an elderly ${profession} in their 70s with grey hair and wrinkles, modern DSLR photograph, sharp focus, high resolution, natural lighting"
        fi
      fi
      if [ "$bias" = "race" ]; then
        prompt="a color head and shoulders portrait photo of a ${cls} ${profession}, modern DSLR photograph, plain background, sharp focus, high resolution, photorealistic"
      fi
      local outdir="${base_dir}/${split}/${safe_prof}/${safe_cls}"
      echo "[${bias}/${split} ${completed}/${total}] ${profession} | ${cls}"
      python generate_data.py \
        --prompt "$prompt" \
        --attribute "$bias" \
        --attribute_class "$cls" \
        --seed "$SEED" \
        --run_times "$RUNTIME" \
        --num_inference_steps "$STEPS" \
        --output_dir "$outdir" \
        --checkface \
        --max_attempts "$MAX_ATTEMPTS"
    done
  done

  if [ "$BUILD_CSV" = "1" ] && [ "$split" = "train" ]; then
    local csv_out="${base_dir}/${bias}_dataset.txt"
    echo ">> Building training CSV: ${csv_out}"
    python build_dataset_csv.py --base_dir "${base_dir}/${split}" --attribute "$bias" --output "$csv_out"
  fi
}

case "$ATTRIBUTE" in
  gender|race|age) ATTRS=("$ATTRIBUTE") ;;
  all)             ATTRS=("gender" "race" "age") ;;
  *) echo "ERROR: attribute must be gender|race|age|all"; exit 1 ;;
esac
case "$SPLIT_ARG" in
  train|test) SPLITS=("$SPLIT_ARG") ;;
  both)       SPLITS=("train" "test") ;;
  *) echo "ERROR: split must be train|test|both"; exit 1 ;;
esac

echo "# SPATM datagen | attrs: ${ATTRS[*]} | splits: ${SPLITS[*]} | root: ${DATASET_ROOT}"
for a in "${ATTRS[@]}"; do for s in "${SPLITS[@]}"; do generate_for "$a" "$s"; done; done
echo ""
echo "# DONE. Datasets under ${DATASET_ROOT}"