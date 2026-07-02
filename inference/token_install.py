from huggingface_hub import hf_hub_download

repo_id = "itsmag11/gender-inclusive"

print("Downloading adaptive_mapping...")
hf_hub_download(
    repo_id=repo_id,
    filename="adaptive_mapping.safetensors",
    local_dir="./checkpoints/gender-inclusive",
    force_download=True
)

print("Downloading learned_embeds...")
hf_hub_download(
    repo_id=repo_id,
    filename="learned_embeds.safetensors",
    local_dir="./checkpoints/gender-inclusive",
    force_download=True
)

print("✅ DONE")