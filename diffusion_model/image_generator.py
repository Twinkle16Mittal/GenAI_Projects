import torch # needed for running neural networks
from diffusers import StableDiffusionPipeline # handling prompt-> denoising-> image generation

model_id = "runwayml/stable-diffusion-v1-5" # Hugging face model

# Detect best device
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available(): #Apple Silicon
    device = "mps"
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32
    
print(f"Using device: {device}, dtype: {dtype}")

#Load Pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtpe=dtype, local_files_only=False)
pipe = pipe.to(device)

# Reduce memory Usage
# for mac for 4 - 8 gb (reduce memory usage by processing attention layers in smaller chunks)
pipe.enable_attention_slicing()

# Your Prompts
prompts = [
    "A cyberpunk cat with neon lights",
    "A fantasy castle on floating islands",
    "A realistic photo of a robot chef cooking",
    "A magical forest with glowing mushrooms",
    "A spaceship landing on Mars"
]

# Generate and save images
# Summary of What Happens Internally

# Your code loads Stable Diffusion v1.5 model.

# Chooses CPU/GPU automatically.

# For each prompt:

# Starts from pure noise.

# Applies reverse diffusion guided by text embeddings.

# Gradually denoises noise into a coherent image.

# Saves the result.

# That’s diffusion → controlled noise removal until the noise becomes an image.
for i, prompt in enumerate(prompts,1):
    print(f"Generating image {i} for prompt: {prompt}")
    image = pipe(prompt).images[0] # runs. the diffusion denoising process
    # image = pipe(prompt, num_inference_steps=25).images[0] 
    filename = f"generated_{i}.png"
    image.save(filename)
    print(f"saved {filename}")

print("All images generated successu=fully!")
