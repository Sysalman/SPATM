import os
import sys
import torch
import argparse
import torch.nn.functional as F

from pytorch_lightning import seed_everything
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image

# Make spatm_pipeline.py importable whether it sits next to this script,
# in a ./pipelines subfolder, or in a sibling ../pipelines folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "pipelines"), os.path.join(_HERE, "..", "pipelines")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from spatm_pipeline_delayed import (
    StableDiffusionAdaptiveTokenPipeline,
    AdaptiveTokenMapping_v2
)

from safetensors.torch import load_file


# =========================================================
# Parse Args
# =========================================================
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        default="A professional medical doctor wearing a white coat and stethoscope, hospital background, realistic portrait photography",
        type=str
    )

    parser.add_argument(
        "--profession_name",
        default="doctor",
        type=str
    )

    parser.add_argument(
        "--textual_inversion_dir",
        type=str,
        default=None
    )

    parser.add_argument(
        "--sd_model",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5"
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=666
    )

    parser.add_argument(
        "--run_times",
        type=int,
        default=1
    )

    parser.add_argument(
        "--num_col",
        type=int,
        default=10
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='./results/spatm'
    )

    parser.add_argument(
        "--disable_spatm",
        action="store_true"
    )

    parser.add_argument(
        "--bias_attribute",
        type=str,
        default="gender",
        choices=["gender", "race", "age"],
        help="Bias attribute to evaluate: gender, race, or age"
    )

    parser.add_argument(
        "--injection_step",
        type=int,
        default=0,
        help="Delayed injection: steps < this use base prompt (no token); >= use inclusive token. 0 = off."
    )

    return parser.parse_args()


# =========================================================
# Main
# =========================================================
if __name__ == '__main__':

    args = parse_args()

    seed_everything(args.seed)

    # =====================================================
    # Derive token name from bias_attribute argument
    # Supports: gender → <gender-diverse>
    #           race   → <race-diverse>
    #           age    → <age-diverse>
    # =====================================================
    ATTRIBUTE_TOKEN = {
        "gender": "<gender-diverse>",
        "race":   "<race-diverse>",
        "age":    "<age-diverse>",
    }
    TOKEN_NAME = ATTRIBUTE_TOKEN[args.bias_attribute]

    print(f"\n===== BIAS ATTRIBUTE =====")
    print(f"Attribute : {args.bias_attribute}")
    print(f"Token     : {TOKEN_NAME}")
    print("==========================\n")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f'device: {device}')

    PROMPT = args.prompt
    print("\n===== FINAL PROMPT =====")
    print(PROMPT)

    OUT_DIR = args.output_dir

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)

    # =====================================================
    # SPATM Mapping
    # =====================================================
    adaptive_mapping = None

    if args.disable_spatm:

        print("\n===== SPATM DISABLED =====")
        adaptive_mapping = None

    else:
        embed_dim = (
            1024
            if args.sd_model == "stabilityai/stable-diffusion-2-1"
            else 768
        )

        adaptive_mapping = AdaptiveTokenMapping_v2(
            input_dim=embed_dim,
            hidden_dim=1024
        )

        adaptive_mapping = adaptive_mapping.to(
            device=device,
            dtype=torch.float16
        )

        adaptive_mapping.eval()

        # =====================================================
        # Load SPATM Weights
        # =====================================================
        if args.textual_inversion_dir is not None:

            mapping_path = os.path.join(
                args.textual_inversion_dir,
                'adaptive_mapping.safetensors'
            )

            try:
                print("\n===== MAPPING FILE =====")
                print(mapping_path)
                print(os.path.exists(mapping_path))
                print("========================")

                from safetensors import safe_open

                with safe_open(
                    mapping_path,
                    framework="pt",
                    device="cpu"
                ) as f:

                    print(
                        "Keys:",
                        list(f.keys())
                    )

                state_dict = load_file(mapping_path)
                print("\n===== SAMPLE WEIGHT VALUES =====")

                for k, v in state_dict.items():

                    print(k, v.flatten()[0:5])

                    break

                missing, unexpected = adaptive_mapping.load_state_dict(
                    state_dict,
                    strict=True
                )

                print("\n===== WEIGHT LOADING DEBUG =====")

                print("\nMissing Keys:")
                print(missing)

                print("\nUnexpected Keys:")
                print(unexpected)

                print("\n===== WEIGHTS LOADED =====")

            except Exception as e:

                print("\nWARNING:")
                print("Failed to load SPATM mapping weights.")
                print(e)

                print("\nUsing randomly initialized mapping.\n")

    # =====================================================
    # Load Pipeline
    # =====================================================
    pipe = StableDiffusionAdaptiveTokenPipeline.from_pretrained(
        args.sd_model,
        adaptive_mapping=adaptive_mapping,
        torch_dtype=torch.float16
    ).to(device)

        # =====================================================
        # REGISTER SPECIAL TOKEN ONLY FOR SPATM
        # =====================================================

    pipe.safety_checker = None

    USE_TI = args.textual_inversion_dir is not None
    USE_SPATM = (
        USE_TI
        and not args.disable_spatm
    )

    if USE_TI:
        learned_embed_path = os.path.join(
                args.textual_inversion_dir,
                "learned_embeds.safetensors"
            )
        print("\n===== LEARNED EMBEDS FILE =====")
        print(learned_embed_path)
        print(os.path.exists(learned_embed_path))
        print("===============================\n")

        if os.path.exists(learned_embed_path):

                print(
                    f"Loading learned embedding: {learned_embed_path}"
                )

                pipe.load_textual_inversion(
                    learned_embed_path,
                    token=TOKEN_NAME
                )
                token_id = pipe.tokenizer.convert_tokens_to_ids(TOKEN_NAME)
                person_id = pipe.tokenizer.convert_tokens_to_ids("person")

                emb = pipe.text_encoder.get_input_embeddings().weight

                placeholder_emb = emb[token_id]
                person_emb = emb[person_id]

                cos = F.cosine_similarity(
                    placeholder_emb.unsqueeze(0),
                    person_emb.unsqueeze(0)
                )

                print("\n===== TOKEN CHECK =====")
                print("placeholder norm:", placeholder_emb.norm().item())
                print("person norm:", person_emb.norm().item())
                print("cosine:", cos.item())

                doctor_id = pipe.tokenizer.convert_tokens_to_ids("doctor")
                doctor_embedding = pipe.text_encoder.get_input_embeddings().weight[
                    doctor_id
                ]

                print(
                    "doctor cosine:",
                    F.cosine_similarity(
                        placeholder_emb.unsqueeze(0),
                        doctor_embedding.unsqueeze(0)
                    ).item()
                )

                human_id = pipe.tokenizer.convert_tokens_to_ids("human")
                human_embedding = pipe.text_encoder.get_input_embeddings().weight[
                    human_id
                ]

                print(
                    "human cosine:",
                    F.cosine_similarity(
                        placeholder_emb.unsqueeze(0),
                        human_embedding.unsqueeze(0)
                    ).item()
                )
                                    
                print("=======================\n")

                placeholder_token_id = (
                    pipe.tokenizer.convert_tokens_to_ids(
                        TOKEN_NAME
                    )
                )

                print(
                    f"Registered token id: {placeholder_token_id}"
                )

                print(
                    "Tokenizer test:",
                    pipe.tokenizer.tokenize(
                        f"a {TOKEN_NAME} doctor"
                    )
                )

        else:

                print(
                    "WARNING: learned_embeds.safetensors NOT FOUND."
                )

    if USE_TI and USE_SPATM:
        print("=== TI + SPATM MODE ===")
    elif USE_TI:
        print("=== TI ONLY MODE ===")
    else:
        print("=== PURE SD1.5 BASELINE ===")

    # =====================================================
    # Generate Images
    # =====================================================
    imgs = []

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    for i in range(args.run_times):

        # Reset seed per image for diversity
        seed_everything(args.seed + i)

        print(
            f'Generating image {i+1}/{args.run_times}'
        )

        with torch.no_grad():

            _base_prompt = None
            if args.injection_step > 0:
                _base_prompt = (
                    args.prompt
                    .replace(", " + TOKEN_NAME, "")
                    .replace(TOKEN_NAME + ", ", "")
                    .replace(TOKEN_NAME, "")
                )
            out = pipe(
                args.prompt,
                negative_prompt="blurry, distorted, low quality, two people, couple, group photo, crowd, multiple people, team, food, building, vehicle, artwork, text, watermark, game, cartoon, anime, military, police, full body, far away, small face, body only, headless",
                num_inference_steps=args.num_inference_steps,
                guidance_scale=7.5,
                profession_name=args.profession_name.replace('_', ' '),
                token_name=TOKEN_NAME if USE_SPATM else None,
                injection_step=args.injection_step,
                base_prompt=_base_prompt
            )

        image = out.images[0]

        # -------------------------------------------------
        # Save Image
        # -------------------------------------------------
        image_path = os.path.join(
            OUT_DIR,
            'images',
            f'{str(i).zfill(3)}.jpg'
        )

        image.save(image_path)

        imgs.append(transform(image))

    # =====================================================
    # Grid Visualization
    # =====================================================
    grid = make_grid(
        imgs,
        nrow=args.num_col
    )

    grid = transforms.ToPILImage()(grid)

    grid.save(
        os.path.join(
            OUT_DIR,
            'grid.jpg'
        )
    )

    print("\nSPATM generation completed.\n")