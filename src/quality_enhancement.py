from PIL import Image
from diffusers import LDMSuperResolutionPipeline
import torch

# Clear CUDA memory
torch.cuda.empty_cache()

# Load model and scheduler
model_id = "CompVis/ldm-super-resolution-4x-openimages"

pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
# Move pipeline to CPU to reduce GPU memory usage (optional)
pipeline = pipeline.to("cuda")

# Load local image
image_path = "./generated_image.webp"
low_res_img = Image.open(image_path).convert("RGB")

# Get the original size of the image
original_size = low_res_img.size

# Reduce the image size before processing to save memory, otherwise the model may run out of memory on my machine
low_res_img_resized = low_res_img.resize(
    (low_res_img.width//2, low_res_img.height//2), Image.Resampling.LANCZOS
)
print("Original image size:", original_size)
# Upscale the image (using the resized low-res image), the more inference steps, the better the quality
upscaled_image = pipeline(image=low_res_img_resized, num_inference_steps=300, eta=1).images[0]
print("Upscale image size:",upscaled_image.size)

# Resize the generated image to the enhanced image size so it can be compared
low_res_img_resized = low_res_img.resize(
    (upscaled_image.width, upscaled_image.height), Image.Resampling.LANCZOS
)
# Save the upscaled and resized image as a .webp file
upscaled_image.save("upsampled_image.webp", format="WEBP")
low_res_img_resized.save("generated_image_resized.webp", format="WEBP")

