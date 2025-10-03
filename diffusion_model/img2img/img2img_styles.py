from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os

model_id = "runwayml/stable-diffusion-v1-5"

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available(): #Apple Silicon
    device = "mps"
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32
    
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtpe = dtype
)
pipe = pipe.to(device)
pipe.enable_attention_slicing() # memory saver

# Load input image (must be RGB)
input_image = Image.open("input.png").convert("RGB")
input_image = input_image.resize((512, 512)) # optional: resize to a model-friendly size

prompts = [
    "A cinematic film still, moody lighting, dramatic color grading, ultra detailed",
    "A watercolor painting of the scene, soft textured paper, painterly brush strokes",
    "Cyberpunk neon style, high contrast, futuristic city reflections, highly detailed"
]
# A seed fixed the initial noise -> same input + same seed = reproducible output.
seeds = [42, 1337, 2025]
strength = 0.6
num_steps = 30
guidance = 7.5

for i, (prompt, seed) in enumerate(zip(prompts, seeds), start=1):
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt = prompt,
        image = input_image,
        strength = strength,
        num_inference_steps = num_steps,
        guidance_scale=guidance,
        generator=generator
    )
    image = result.images[0]
    out_path = f"out_style_{i}.png"
    image.save(out_path)
    print("Saved:", out_path)
    
print("Done -3 stylistic variations generated.")
