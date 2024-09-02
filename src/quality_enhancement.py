from diffusers import LDMSuperResolutionPipeline
import torch
import logging
from tools import load_configuration, load_and_process_image, resize_image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clear_cuda_memory():
    """
    Clears CUDA memory to free up GPU resources.
    This function is particularly useful when dealing with large models or images
    that require significant GPU memory.
    """
    torch.cuda.empty_cache()

def load_pipeline(model_id):
    """
    Loads the LDMSuperResolutionPipeline model and moves it to the GPU.

    Args:
        model_id (str): The model identifier used to load the pipeline.

    Returns:
        LDMSuperResolutionPipeline: The loaded pipeline, ready for inference.
    """
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to("cuda")
    return pipeline

def upscale_image(pipeline, image, num_inference_steps, eta):
    """
    Upscales an image using the specified pipeline.

    Args:
        pipeline (LDMSuperResolutionPipeline): The pipeline used for upscaling.
        image (PIL.Image.Image): The image to be upscaled.
        num_inference_steps (int): Number of inference steps for upscaling.
        eta (float): Parameter controlling the randomness of the generation process.

    Returns:
        PIL.Image.Image: The upscaled image.
    """
    return pipeline(image=image, num_inference_steps=num_inference_steps, eta=eta).images[0]

def save_images(upscaled_image, resized_image, upsampled_image_path, resized_image_path):
    """
    Saves the upscaled and resized images to the specified file paths.

    Args:
        upscaled_image (PIL.Image.Image): The upscaled image.
        resized_image (PIL.Image.Image): The resized image for comparison.
        upsampled_image_path (str): File path to save the upscaled image.
        resized_image_path (str): File path to save the resized image.
    """
    upscaled_image.save(upsampled_image_path, format="WEBP")
    resized_image.save(resized_image_path, format="WEBP")

def main():
    """
    Main function that orchestrates the process of loading, upscaling, resizing, and saving images.
    It also logs the sizes of the original and upscaled images.
    """
    # Load configuration
    config = load_configuration()

    # Clear CUDA memory
    clear_cuda_memory()

    # Load model and scheduler
    pipeline = load_pipeline(config['quality_enhancement']['model_id'])

    # Load local image
    low_res_img = load_and_process_image(config['paths']['generated_image'])

    # Get the original size of the image
    original_size = low_res_img.size
    logging.info(f"Original image size: {original_size}")

    # Reduce the image size before processing to save memory
    low_res_img_resized = resize_image(low_res_img, config['quality_enhancement']['resize_factor'])

    # Upscale the image
    upscaled_image = upscale_image(
        pipeline,
        low_res_img_resized,
        config['quality_enhancement']['num_inference_steps'],
        config['quality_enhancement']['eta']
    )
    logging.info(f"Upscaled image size: {upscaled_image.size}")

    # Resize the generated image to the enhanced image size for comparison
    low_res_img_resized = resize_image(low_res_img, 1)  # resizing to original dimensions

    # Save the upscaled and resized images as .webp files
    save_images(upscaled_image, low_res_img_resized, config['paths']['upsampled_image'], config['paths']['generated_image_resized'])

if __name__ == '__main__':
    main()
