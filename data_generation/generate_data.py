import os
import torch
import argparse
import random
from pytorch_lightning import seed_everything

from diffusers import StableDiffusionPipeline

from facexlib.detection import init_detection_model

import cv2
import clip
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# =====================================================
# OPTIONAL PERSISTENT CACHE LOCATIONS
# Set these env vars to your Lambda filesystem so weights
# survive instance termination, e.g.:
#   export CLIP_CACHE_DIR=/lambda/nfs/<fs>/clip_cache
#   export FACEXLIB_WEIGHTS=/lambda/nfs/<fs>/facexlib_weights
# If unset, the libraries fall back to ~/.cache (ephemeral).
# =====================================================
CLIP_CACHE_DIR = os.environ.get("CLIP_CACHE_DIR")        # None -> library default
FACEXLIB_WEIGHTS = os.environ.get("FACEXLIB_WEIGHTS")    # None -> library default


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def crop_face(img, left, top, right, bottom, expansion_factor=0.5):
    width = right - left
    height = bottom - top
    expanded_left   = max(0, left   - expansion_factor * width)
    expanded_top    = max(0, top    - expansion_factor * height)
    expanded_right  = min(img.width,  right  + expansion_factor * width)
    expanded_bottom = min(img.height, bottom + expansion_factor * height)
    return img.crop((expanded_left, expanded_top, expanded_right, expanded_bottom))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training dataset images using SD1.5 with face and attribute filtering"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Generation prompt. Must contain the target attribute keyword "
             "(e.g. 'male', 'female', 'young', 'old', 'White', 'Black', etc.)"
    )
    parser.add_argument(
        "--sd_model",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="Hugging Face model id for the base SD1.5 checkpoint. "
             "The original 'runwayml/stable-diffusion-v1-5' repo was removed; "
             "this community mirror is the current canonical location."
    )
    parser.add_argument(
        "--attribute",
        type=str,
        default="gender",
        choices=["gender", "race", "age"],
        help="Bias attribute being collected. Determines the CLIP classifier used."
    )
    parser.add_argument(
        "--attribute_class",
        type=str,
        default=None,
        help="Target class label within the attribute "
             "(e.g. 'male'/'female' for gender, 'White'/'Black'/... for race, 'young'/'old' for age). "
             "If not set, no attribute filtering is applied — only face detection."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Seed for reproducibility. -1 = random."
    )
    parser.add_argument(
        "--run_times",
        type=int,
        default=200,
        help="Number of valid images to collect."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--checkface",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter out images without exactly one detected face. "
             "Use --no-checkface to disable."
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=2000,
        help="Maximum generation attempts before giving up."
    )

    return parser.parse_args()


# =====================================================
# CLIP CLASS PROMPTS PER ATTRIBUTE
# =====================================================
ATTRIBUTE_CLASSES = {
    "gender": [
        "a photo of a male",
        "a photo of a female",
    ],
    "race": [
        "a photo of a Caucasian person",
        "a photo of a Black person",
        "a photo of a Middle Eastern person",
        "a photo of a Latino person",
        "a photo of an Indian person",
    ],
    "age": [
        "a photo of a young person",
        "a photo of an old person",
    ],
}


def get_target_class_idx(attribute, attribute_class):
    """Return index of target class in CLIP classifier list."""
    classes = ATTRIBUTE_CLASSES[attribute]
    for i, cls_prompt in enumerate(classes):
        # Match last word of class prompt to attribute_class arg
        # e.g. "a photo of a male" → "male"
        if attribute_class.lower() in cls_prompt.lower():
            return i
    raise ValueError(
        f"Could not find '{attribute_class}' in class list for attribute '{attribute}'.\n"
        f"Available: {classes}"
    )


if __name__ == '__main__':

    args = parse_args()

    # =====================================================
    # SEED
    # =====================================================
    SEED = args.seed if args.seed > 0 else random.randint(0, 10000)
    seed_everything(SEED)
    print(f"Seed: {SEED}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"Device: {device}")

    # =====================================================
    # TARGET CLASS INDEX FOR CLIP FILTERING
    # =====================================================
    CLASSES_prompts = ATTRIBUTE_CLASSES[args.attribute]
    target_class_idx = None

    if args.attribute_class is not None:
        target_class_idx = get_target_class_idx(args.attribute, args.attribute_class)
        print(f"Attribute      : {args.attribute}")
        print(f"Target class   : {args.attribute_class} (index {target_class_idx})")
        print(f"Classifier     : {CLASSES_prompts}")

    # =====================================================
    # OUTPUT DIR
    # =====================================================
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Resume from existing images
    existing_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    existing_count = len(existing_images)

    if existing_count >= args.run_times:
        print(f"Already have {existing_count} images (>= {args.run_times}). Skipping.")
        exit(0)
    elif existing_count > 0:
        print(f"Resuming from {existing_count} existing images.")

    # =====================================================
    # LOAD CLIP
    # =====================================================
    clip_model, _ = clip.load("ViT-B/32", device=device, download_root=CLIP_CACHE_DIR)
    clip_model.eval()
    CLASSES_text = clip.tokenize(CLASSES_prompts).to(device)

    clip_transforms = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275,  0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    # =====================================================
    # LOAD SD1.5 PIPELINE
    # NOTE: 'runwayml/stable-diffusion-v1-5' was removed from the Hub.
    # The default --sd_model points at the current community mirror and
    # downloads on first run (no local_files_only). Point HF_HOME at your
    # persistent filesystem to avoid re-downloading each session.
    # =====================================================
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model,
        torch_dtype=dtype,
    ).to(device)
    pipe.safety_checker = None

    # =====================================================
    # LOAD FACE DETECTOR
    # =====================================================
    det_net = None
    if args.checkface:
        det_net = init_detection_model(
            'retinaface_resnet50', half=True, device=device,
            model_rootpath=FACEXLIB_WEIGHTS
        )

    # =====================================================
    # GENERATION LOOP
    # =====================================================
    valid_generation = existing_count
    attempt = 0

    print(f"\nGenerating {args.run_times} images...")
    print(f"Prompt: {args.prompt}\n")

    while valid_generation < args.run_times:

        if attempt >= args.max_attempts:
            print(f"Reached max attempts ({args.max_attempts}). "
                  f"Collected {valid_generation}/{args.run_times} images.")
            break

        attempt += 1
        print(f"  Attempt {attempt} | Valid: {valid_generation}/{args.run_times}")

        # Generate image
        image = pipe(
            prompt=args.prompt,
            negative_prompt=(
                "blurry, low quality, distorted, multiple people, "
                "crowd, cartoon, painting, text, watermark, "
                "black and white, monochrome, grayscale, sepia, "
                "vintage, antique, old photograph, film grain, faded, retro"
            ),
            num_inference_steps=args.num_inference_steps
        ).images[0]

        # ── Face detection filter ──────────────────────
        if args.checkface and det_net is not None:

            # Convert PIL to cv2 for face detector
            img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            with torch.no_grad():
                face_locations = det_net.detect_faces(img_cv2, 0.97)

            if len(face_locations) != 1:
                print(f"    Skipped: {len(face_locations)} face(s) detected")
                continue

            left, top, right, bottom, conf = face_locations[0][:5]
            cropped_img = crop_face(image, left, top, right, bottom, 0.5)

        else:
            cropped_img = image

        # ── Attribute class filter (CLIP) ──────────────
        if target_class_idx is not None:

            cropped_transformed = clip_transforms(cropped_img).to(device)

            with torch.no_grad():
                logits, _ = clip_model(
                    cropped_transformed.unsqueeze(0), CLASSES_text
                )
                probs = logits.softmax(dim=-1).cpu().numpy()
                predicted_class = int(np.argmax(probs, axis=1)[0])

            if predicted_class != target_class_idx:
                print(f"    Skipped: predicted class {predicted_class} "
                      f"!= target {target_class_idx}")
                continue

        # ── Save valid image ───────────────────────────
        save_path = os.path.join(
            images_dir, f'{str(valid_generation).zfill(3)}.jpg'
        )
        image.save(save_path)
        valid_generation += 1
        print(f"    Saved: {save_path}")

    print(f"\nDone. Collected {valid_generation}/{args.run_times} images.")
    print(f"Output: {args.output_dir}")

    # Verify count
    final_count = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    if final_count < args.run_times:
        print(f"WARNING: Only {final_count} images collected (target: {args.run_times})")
    else:
        print(f"SUCCESS: {final_count} images collected.")
