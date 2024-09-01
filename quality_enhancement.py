from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch

# Clear CUDA memory
torch.cuda.empty_cache()

# Load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16
)

# Move pipeline to CPU to reduce GPU memory usage (optional)
pipeline = pipeline.to("cuda")

# Load local image
image_path = "./generated_image.webp"
low_res_img = Image.open(image_path).convert("RGB")

# Get the original size of the image
original_size = low_res_img.size

# Reduce the image size before processing to save memory, otherwise the model may run out of memory on my machine
low_res_img_resized = low_res_img.resize(
    (low_res_img.width // 4, low_res_img.height // 4), Image.Resampling.LANCZOS
)

# Upscale the image (using the resized low-res image), the more inference steps, the better the quality
upscaled_image = pipeline(prompt="a man", image=low_res_img_resized, num_inference_steps=200).images[0]

# Resize the upscaled image back to the original size
upscaled_image_resized = upscaled_image.resize(original_size, Image.Resampling.LANCZOS)

# Save the upscaled and resized image as a .webp file
upscaled_image_resized.save("upsampled_image.webp", format="WEBP")
