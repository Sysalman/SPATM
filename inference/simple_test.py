import torch
import os
from diffusers import StableDiffusionPipeline

# This forces the path to your desktop so you can find it instantly
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
save_path = os.path.join(desktop_path, "baseline_doctor.png")

model_id = "runwayml/stable-diffusion-v1-5"
# Force float32 for CPU/GPU compatibility
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "A professional medical doctor wearing a white lab coat, realistic portrait"
print(f"Generating image... please wait (running on {'GPU' if torch.cuda.is_available() else 'CPU'})")

image = pipe(prompt, num_inference_steps=25).images[0]
image.save(save_path)

print(f"--- SUCCESS ---")
print(f"The image is now on your WINDOWS DESKTOP as: baseline_doctor.png")