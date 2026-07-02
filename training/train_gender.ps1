# =========================================================
# train_gender.ps1
# SPATM Gender Bias Training
# =========================================================

$SEED       = 666
$NOTE       = "gender-spatm-v3"
$MODEL_NAME = "runwayml/stable-diffusion-v1-5"
$DATA_DIR   = "C:\Vs_code\Mtech_UAITTI\AITTI\dataset\gender_dataset.txt"
$CHECKPOINT_DIR = "C:\Vs_code\Mtech_UAITTI\AITTI\training\checkpoints"
$OUTPUT_DIR = "$CHECKPOINT_DIR\$(Get-Date -Format 'yyyyMMddHHmm')-$NOTE"

Write-Host "Training SPATM for GENDER bias"
Write-Host "Output: $OUTPUT_DIR"

$env:CUDA_VISIBLE_DEVICES = "0"

accelerate launch train_spatm.py `
    --pretrained_model_name_or_path $MODEL_NAME `
    --train_data_dir $DATA_DIR `
    --bias_attribute "gender" `
    --resolution 512 `
    --train_batch_size 1 `
    --gradient_accumulation_steps 4 `
    --max_train_steps 3000 `
    --learning_rate 1e-5 `
    --lr_scheduler "constant_with_warmup" `
    --lr_warmup_steps 100 `
    --output_dir $NOTE `
    --seed $SEED `
    --save_steps 1000 `
    --checkpointing_steps 1000 `
    --checkpoints_total_limit 2 `
    --validation_prompt "A portrait photograph of a single doctor, <gender-diverse>, plain background" `
    --validation_steps 1000 `
    --anchor_loss 0.1 `
    --train_adaptive_token_mapping `
    --is_run

Write-Host "Gender training complete: $OUTPUT_DIR"