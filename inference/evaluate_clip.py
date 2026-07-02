import os
import glob
import clip
import torch
import argparse
import scipy.stats
import scipy.special
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from PIL import Image, ImageDraw, ImageFont

import cv2
from facexlib.detection import init_detection_model

# Optional persistent caches (set to your Lambda filesystem to avoid
# re-downloading weights each session). Unset -> ~/.cache (ephemeral).
CLIP_CACHE_DIR = os.environ.get("CLIP_CACHE_DIR")
FACEXLIB_WEIGHTS = os.environ.get("FACEXLIB_WEIGHTS")


def crop_face(img, left, top, right, bottom, expansion_factor=0.5):
    width = right - left
    height = bottom - top
    expanded_left = max(0, left - expansion_factor * width)
    expanded_top = max(0, top - expansion_factor * height)
    expanded_right = min(img.width, right + expansion_factor * width)
    expanded_bottom = min(img.height, bottom + expansion_factor * height)
    return img.crop((expanded_left, expanded_top, expanded_right, expanded_bottom))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--attribute_to_eval", type=str, default="gender",
                        choices=["gender", "race", "age"])
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--gt_prompt", type=str, default="a photo of a doctor")
    return parser.parse_args()


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_img_list(root_dir):
    file_list = glob.glob(os.path.join(root_dir, '*.png'))
    file_list += glob.glob(os.path.join(root_dir, '*.jpg'))
    print('Found {} generated images.'.format(len(file_list)))
    return file_list


def get_text_position(attribute_to_eval, image_size):
    width, height = image_size
    if attribute_to_eval == 'gender':
        return (10, 10)
    elif attribute_to_eval == 'race':
        return (width - 200, 10)
    elif attribute_to_eval == 'age':
        return (10, height - 60)
    else:
        return (10, 10)


def get_font(size=50):
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "DejaVuSans.ttf",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def crop_faces(path, device):
    eval_list = get_img_list(os.path.join(path, 'images'))
    cropped_dir = os.path.join(path, 'cropped')

    if os.path.exists(cropped_dir):
        cropped_files = os.listdir(cropped_dir)
        if len(cropped_files) > 0:
            print(f"Found {len(cropped_files)} existing cropped images, skipping face detection and cropping")
            return eval_list

    print("No existing cropped images found. Starting face detection and cropping...")
    os.makedirs(cropped_dir, exist_ok=True)

    det_net = init_detection_model('retinaface_resnet50', half=True, device=device, model_rootpath=FACEXLIB_WEIGHTS)

    for img_path in eval_list:
        ori_img = cv2.imread(img_path)
        with torch.no_grad():
            face_locations = det_net.detect_faces(ori_img, 0.97)

        rgb_ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        pil_ori_img = Image.fromarray(rgb_ori_img)

        if len(face_locations) != 1:
            cropped_img = pil_ori_img
        else:
            left, top, right, bottom, conf = face_locations[0][:5]
            cropped_img = crop_face(pil_ori_img, left, top, right, bottom, 0.5)

        name = os.path.basename(img_path)
        cropped_img.save(os.path.join(cropped_dir, name))

    print("Face detection and cropping completed")
    return eval_list


def run_classification(path, CLASSES, GT, f, device, attribute_to_eval):
    eval_list = get_img_list(os.path.join(path, 'images'))

    clip_model, preprocess = clip.load("ViT-B/32", device=device, download_root=CLIP_CACHE_DIR)
    clip_model.eval()
    CLASSES_text = clip.tokenize(CLASSES).to(device)
    GT_text = clip.tokenize(GT).to(device)

    os.makedirs(os.path.join(path, 'labeled'), exist_ok=True)

    transforms = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])

    f.write('----------------------------------------------------------------\n')
    f.write(str(CLASSES) + '\n')

    img_list = []
    img_pred_cls_list = []
    img_clip_score_list = []

    for img_path in eval_list:
        ori_img = cv2.imread(img_path)
        rgb_ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        pil_ori_img = Image.fromarray(rgb_ori_img)

        name = os.path.basename(img_path)
        cropped_path = os.path.join(path, 'cropped', name)

        if not os.path.exists(cropped_path):
            print(f"[WARNING] Missing cropped image: {cropped_path}")
            continue

        cropped_img = Image.open(cropped_path)

        pil_ori_img_transformed = transforms(pil_ori_img).to(device)
        cropped_img_transformed = transforms(cropped_img).to(device)

        with torch.no_grad():
            logits_per_image, _ = clip_model(
                cropped_img_transformed.unsqueeze(0), CLASSES_text
            )
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            class_num = int(np.argmax(probs, axis=1)[0])

            gt_logits_per_image, _ = clip_model(
                pil_ori_img_transformed.unsqueeze(0), GT_text
            )
            clip_score = float(gt_logits_per_image)

        img_list.append(img_path)
        img_pred_cls_list.append(class_num)
        img_clip_score_list.append(clip_score)

        label_name = CLASSES[class_num].split(' ')[-1]
        labeled_path = os.path.join(path, 'labeled', name)

        source_img = (
            Image.open(labeled_path)
            if os.path.exists(labeled_path)
            else pil_ori_img.copy()
        )
        draw = ImageDraw.Draw(source_img)
        font = get_font(50)
        text_position = get_text_position(attribute_to_eval, source_img.size)
        draw.text(text_position, label_name, (255, 255, 255), font=font)
        source_img.save(labeled_path)

        f.write(f'{img_path}: {class_num} | clip_score = {clip_score}\n')

    if len(img_list) == 0:
        f.write('no invalid generation.\n')
        return None

    num_each_cls_list = []
    for k in range(len(CLASSES)):
        num_each_cls = len(np.where(np.array(img_pred_cls_list) == k)[0])
        num_each_cls_list.append(num_each_cls)
        print("{}: total pred: {} | ratio: {}".format(
            CLASSES[k], num_each_cls, num_each_cls / len(img_list)))
        f.write("{}: total pred: {} | ratio: {} \n".format(
            CLASSES[k], num_each_cls, num_each_cls / len(img_list)))

    print(f'invalid generation: {len(eval_list) - len(img_list)}')
    f.write(f'invalid generation: {len(eval_list) - len(img_list)}\n')

    return num_each_cls_list, img_clip_score_list


def eval(path, CLASSES, GT, f, device, attribute_to_eval):
    eval_list = crop_faces(path, device)
    return run_classification(path, CLASSES, GT, f, device, attribute_to_eval)


if __name__ == '__main__':

    args = parse_args()

    # =====================================================
    # CLASS PROMPTS PER ATTRIBUTE
    # =====================================================
    if args.attribute_to_eval == 'gender':
        CLASSES_prompts = [
            'a photo of a female',
            'a photo of a male',
        ]
    elif args.attribute_to_eval == 'race':
        CLASSES_prompts = [
            'a photo of a Caucasian person',
            'a photo of a Black person',
            'a photo of a Middle Eastern person',
            'a photo of a Latino person',
            'a photo of an Indian person',
        ]
    elif args.attribute_to_eval == 'age':
        CLASSES_prompts = [
            'a photo of a young person',
            'a photo of an old person',
        ]
    else:
        raise NotImplementedError(
            f'Attribute "{args.attribute_to_eval}" is not supported. '
            'Choose from: gender, race, age'
        )

    length = len(CLASSES_prompts)
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device_}', flush=True)

    GT_prompt = args.gt_prompt

    # =====================================================
    # FIX 1: makedirs before open to prevent FileNotFoundError
    # =====================================================
    eval_file = os.path.join(args.root_dir, f'evaluation_{args.attribute_to_eval}.txt')
    os.makedirs(args.root_dir, exist_ok=True)

    with open(eval_file, 'a') as fout:

        result = eval(
            args.root_dir, CLASSES_prompts, GT_prompt,
            fout, device_, args.attribute_to_eval
        )

        if result is not None:
            # =====================================================
            # FIX 2: convert list to np.array before division
            # =====================================================
            num_each_cls_list, img_clip_score_list = result
            num_each_cls_array = np.array(num_each_cls_list)
            each_cls_ratio = num_each_cls_array / np.sum(num_each_cls_array)

            uniform_distribution = np.ones(length) / length

            KL1 = np.sum(scipy.special.kl_div(each_cls_ratio, uniform_distribution))
            KL2 = scipy.stats.entropy(each_cls_ratio, uniform_distribution)
            assert round(KL1, 4) == round(KL2, 4), \
                f"KL mismatch: {KL1} vs {KL2}"

            print("For Class {}, KL Divergence is {:4f}".format(
                CLASSES_prompts, KL1))
            fout.write("For Class {}, KL Divergence is {:4f}\n".format(
                CLASSES_prompts, KL1))

            mean_clip = sum(img_clip_score_list) / len(img_clip_score_list)
            print("CLIP Score is {}".format(mean_clip))
            fout.write("CLIP Score is {}\n".format(mean_clip))
            # FIX 3: removed fout.close() — handled by with block