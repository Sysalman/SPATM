# =========================================================
# generate_gender_data_spatm.ps1
# =========================================================

$biased_professions = @(
    "Doctor",
    "construction worker",
    "secretary",
    "nurse"
    # "mechanic",
    # "firefighter",
    # "police officer",
    # "engineer",
    # "pilot",
    # "computer programmer",
    # "ceo",
    # "surgeon",
    # "truck driver",
    # "electrician",
    # "scientist",
    # "nurse",
    # "teacher",
    # "secretary",
    # "librarian",
    # "receptionist",
    # "administrative assistant",
    # "childcare worker",
    # "social worker",
    # "dental hygienist",
    # "therapist",
    # "pathologist",
    # "nutritionist"
)

$genders = @(
    "male",
    "female"
)

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

$SEED = 666

$RUNTIME = 500

$NUMCOL = 10

# ---------------------------------------------------------
# Main Loop
# ---------------------------------------------------------

foreach ($profession in $biased_professions) {

    foreach ($gender in $genders) {

        $PROMPT = "a high-quality realistic photo of a $gender $profession"

        $safe_prompt = $PROMPT -replace " ", "_"

        $OUTDIR = "C:\Vs_code\Mtech_UAITTI\AITTI\dataset\generated_gender\$safe_prompt"

        Write-Host ""
        Write-Host "====================================="
        Write-Host "PROMPT: $PROMPT"
        Write-Host "OUTDIR: $OUTDIR"
        Write-Host "====================================="
        Write-Host ""

        python generate_data.py `
            --seed $SEED `
            --run_times $RUNTIME `
            --num_col $NUMCOL `
            --prompts "$PROMPT" `
            --output_dir "$OUTDIR" `
            --checkface
    }
}