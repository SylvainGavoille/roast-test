from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid
from dotenv import load_dotenv
import os
#import torch._dynamo


load_dotenv()

# Set up your Hugging Face token
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# You can set the token globally for your session
from huggingface_hub import login

#login(HUGGING_FACE_TOKEN)

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
)

#offload the model to the GPU while the other pipeline components wait on the CPU
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()
# boost your inference speed even more by wrapping your UNet with it
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

# Load local image
image_path = "./original_image.webp"
init_image = Image.open(image_path).convert("RGB")

prompt = "With blonde hair, and smiling."
negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"

# pass prompt and image to pipeline
image = pipeline(prompt, negative_prompt=negative_prompt, image=init_image, strength=0.05, guidance_scale=1.0).images[0]
# Save the upscaled and resized image as a .webp file
image.save("featured_image.webp", format="WEBP")
#make_image_grid([init_image, image], rows=1, cols=2)