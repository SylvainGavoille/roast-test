from huggingface_hub import login
import torch
from diffusers import AutoPipelineForImage2Image
from dotenv import load_dotenv
import os
import logging
from tools import (
    load_configuration,
    load_and_process_image,
    resize_and_save_image,
    timer,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def initialize_pipeline(model_name, fast=True):
    """
    Initialize the image-to-image pipeline with the specified model.

    Args:
        model_name (str): The name of the model to load.
        fast (bool): Whether to use fast settings for the pipeline. Defaults to True.

    Returns:
        AutoPipelineForImage2Image: The initialized pipeline.
    """
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        model_name, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    # Enable fast settings if specified
    if fast:
        # offload the model to the GPU while the other pipeline components wait on the CPU
        pipeline.enable_model_cpu_offload()
        # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
        pipeline.enable_xformers_memory_efficient_attention()
        # boost your inference speed even more by wrapping your UNet with it
        pipeline.unet = torch.compile(
            pipeline.unet, mode="reduce-overhead", fullgraph=True
        )
    return pipeline


@timer
def enhance_image(
    pipeline,
    image,
    prompt,
    negative_prompt,
    strength,
    guidance_scale,
    num_inference_steps,
):
    """
    Image-to-image is similar to text-to-image, but in addition to a prompt, you can also pass an initial image as a starting point for the diffusion process. 
    The initial image is encoded to latent space and noise is added to it. Then the latent diffusion model takes a prompt and the noisy latent image, predicts
    the added noise, and removes the predicted noise from the initial latent image to get the new latent image. Lastly, a decoder decodes the new latent image
    back into an image.

    Args:
        pipeline (AutoPipelineForImage2Image): The initialized pipeline.
        image (PIL.Image.Image): The initial image to enhance.
        prompt (str): The enhancement prompt.
        negative_prompt (str): The negative prompt conditions the model to not include things in an image, and it can be used to improve image quality or modify an image
        strength (float): The strength value gives the model more “creativity” to generate an image that’s different from the initial image; a strength value of 1.0 means the initial image is more or less ignored
        guidance_scale (float): The guidance_scale parameter is used to control how closely aligned the generated image and text prompt are
        num_inference_steps (int): The number of inference steps.

    Returns:
        PIL.Image.Image: The enhanced image.
    """
    return pipeline(
        prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        original_size=image.size,
        target_size=image.size,
    ).images[0]


def main():
    """
    Main function to load configuration, initialize the pipeline, enhance the image, and save the result.
    """
    # Load environment variables and configuration
    load_dotenv()
    config = load_configuration()
    hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")
    login(hugging_face_token)

    # Initialize the pipeline
    pipeline = initialize_pipeline(
        config["enhance_image"]["model_id"], config["enhance_image"]["fast"]
    )

    # Load the local image
    init_image = load_and_process_image(config["paths"]["original_image"])

    # Enhance the image
    enhanced_image = enhance_image(
        pipeline,
        init_image,
        config["enhance_image"]["prompt"],
        config["enhance_image"]["negative_prompt"],
        config["enhance_image"]["strength"],
        config["enhance_image"]["guidance_scale"],
        config["enhance_image"]["num_inference_steps"],
    )

    # Resize and save the enhanced image
    resize_and_save_image(
        enhanced_image,
        (init_image.width, init_image.height),
        config["paths"]["featured_image"],
    )


if __name__ == "__main__":
    main()
