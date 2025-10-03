from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

model_id ="runwayml/stable-diffusion-inpainting"
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available(): #Apple Silicon
    device = "mps"
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32
    
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

image = Image.open("input.jpg").convert("RGB").resize((512,512))
mask = Image.open("mask.png").convert("RGB").resize((512,512))
prompt = "Replace masked area with a retro wooden door with brass handle, photorealistic"
result = pipe(prompt=prompt, image=image, mask_image=mask, guidance_scale=7.5, num_inference_steps=50)
out = result.images[0]
out.save("inpainted_result.png")
print("Saved inpained_result.png")