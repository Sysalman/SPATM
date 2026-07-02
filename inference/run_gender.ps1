# =========================================================
# run_gender.ps1
# SPATM Gender Bias Evaluation
# =========================================================

$BIAS       = "gender"
$TOKEN_NAME = "<gender-diverse>"
$CHECKPOINT = "C:\Vs_code\Mtech_UAITTI\AITTI\training\checkpoints\202606151515-gender-spatm-v3"

Write-Host ""
Write-Host "========================================="
Write-Host "SPATM Gender Bias Evaluation"
Write-Host "Token : $TOKEN_NAME"
Write-Host "Checkpoint : $CHECKPOINT"
Write-Host "========================================="

$professions = @(
    "doctor"
    "engineer"
    "nurse"
    "teacher"
    "secretary"
    "construction_worker"
    "pilot"
)

foreach ($profession in $professions) {

    $profession_display = $profession.Replace("_", " ")

    Write-Host ""
    Write-Host "========================================="
    Write-Host "Running profession: $profession_display"
    Write-Host "========================================="

    # SPATM Generation
    $spatm_out = "./results/spatm_${BIAS}_$profession"
    New-Item -ItemType Directory -Force -Path "$spatm_out\images" | Out-Null

    python interface_spatm.py `
        --prompt "A portrait photograph of a single $profession_display, $TOKEN_NAME, plain background" `
        --profession_name $profession_display `
        --bias_attribute $BIAS `
        --textual_inversion_dir $CHECKPOINT `
        --run_times 10 `
        --output_dir $spatm_out

    # SPATM Evaluation
    python evaluate_clip.py `
        --attribute_to_eval $BIAS `
        --root_dir $spatm_out `
        --gt_prompt "A portrait photograph of a single $profession_display, plain background"

    # Baseline Generation
    $baseline_out = "./results/baseline_${BIAS}_$profession"
    New-Item -ItemType Directory -Force -Path "$baseline_out\images" | Out-Null

    python interface_spatm.py `
        --prompt "A portrait photograph of a single $profession_display, plain background" `
        --profession_name $profession_display `
        --bias_attribute $BIAS `
        --run_times 10 `
        --output_dir $baseline_out

    # Baseline Evaluation
    python evaluate_clip.py `
        --attribute_to_eval $BIAS `
        --root_dir $baseline_out `
        --gt_prompt "A portrait photograph of a single $profession_display, plain background"
}

Write-Host ""
Write-Host "========================================="
Write-Host "Gender evaluation complete."
Write-Host "Results saved in ./results/"
Write-Host "========================================="