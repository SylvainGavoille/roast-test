from pspnet import PSPNet
import cv2
import numpy as np
import torch
import torchvision.transforms as std_trnsf
from diffusers import StableDiffusionInpaintPipeline
import logging
from tools import load_configuration, load_and_process_image, timer

# Load configuration from YAML
config = load_configuration("config.yml")


# PSPNet model loading function
def load_pspnet_model(model_path, num_classes=1, base_network="resnet101"):
    """
    Load the pre-trained PSPNet model.

    Args:
        model_path (str): Path to the model weights.
        num_classes (int): Number of classes for segmentation.
        base_network (str): Backbone architecture for PSPNet.

    Returns:
        model: Loaded PSPNet model with weights.
    """
    net = PSPNet(num_class=num_classes, base_network=base_network).to("cuda")
    state = torch.load(model_path)
    net.load_state_dict(state["weight"])
    return net


# Image preprocessing function
def preprocess_image(image_path):
    """
    Load and preprocess the image for PSPNet.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    img = load_and_process_image(image_path)
    test_image_transforms = std_trnsf.Compose(
        [
            std_trnsf.ToTensor(),
            std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    data = test_image_transforms(img)
    data = torch.unsqueeze(data, dim=0).to("cuda")
    return data


# PSPNet segmentation and mask generation
@timer
def generate_segmentation_mask(model, image_data, threshold=0.5):
    """
    Generate a segmentation mask from the image using the PSPNet model.

    Args:
        model (PSPNet): Pre-trained PSPNet model.
        image_data (torch.Tensor): Preprocessed image data.
        threshold (float): Threshold for segmentation.

    Returns:
        np.array: Generated binary mask.
    """
    model.eval()
    logit = model(image_data)
    pred = torch.sigmoid(logit.cpu())[0][0].data.numpy()
    mask = pred >= threshold
    mh, mw = image_data.size(2), image_data.size(3)
    mask_n = np.zeros((mh, mw, 3))
    mask_n[:, :, 0] = 255  # Red channel
    mask_n[:, :, 1] = 255  # Green channel
    mask_n[:, :, 2] = 255  # Blue channel
    mask_n *= mask[..., None]
    return mask_n


# Function to save mask image
def save_mask(mask, save_path):
    """
    Save the generated mask as a PNG file.

    Args:
        mask (np.array): Segmentation mask.
        save_path (str): Path to save the mask file.
    """
    cv2.imwrite(save_path, mask)
    logging.info(f"Mask saved at: {save_path}")


# Inpainting function using Stable Diffusion
@timer
def inpaint_image(
    init_image_path,
    mask_image_path,
    prompt,
    strength,
    guidance_scale,
    num_inference_steps,
    fast,
    blur_factor=0.0,
):
    """
    Perform image inpainting using the Stable Diffusion model.

    Args:
        init_image_path (str): Path to the initial image.
        mask_image_path (str): Path to the mask image.
        prompt (str): The enhancement prompt.
        strength (float): The strength value gives the model more “creativity” to generate an image that’s different from the initial image; a strength value of 1.0 means the initial image is more or less ignored
        guidance_scale (float): The guidance_scale parameter is used to control how closely aligned the generated image and text prompt are
        num_inference_steps (int): The number of inference steps.
        fast: (bool): Enable fast mode for inference.
        blur_factor (float): The factor by which to blur the mask image.

    Returns:
        PIL.Image.Image: Inpainted image.
    """
    # Load the Stable Diffusion Inpainting Pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        config["enhance_image"]["model_id"], torch_dtype=torch.bfloat16
    ).to("cuda")
    if fast:
        # offload the model to the GPU while the other pipeline components wait on the CPU
        pipe.enable_model_cpu_offload()
        # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
        pipe.enable_xformers_memory_efficient_attention()
        # boost your inference speed even more by wrapping your UNet with it
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    # Load initial and mask images
    init_image = load_and_process_image(init_image_path)
    mask_image = load_and_process_image(mask_image_path)
    mask = pipe.mask_processor.blur(mask_image, blur_factor=blur_factor)
    # Inpaint image using the prompt
    image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]
    return image


# Main function to process the images
def main():
    
    image_data = preprocess_image(config["paths"]["generated_image"])
    if config["enhance_image"]["generate_hair_mask"]:
        mask = preprocess_image(config["paths"]["mask_image"])
        # Load PSPNet model
        pspnet_model = load_pspnet_model(config["paths"]["model_path"])
        # Preprocess image and generate mask
        mask = generate_segmentation_mask(pspnet_model, image_data)
        save_mask(mask, config["paths"]["mask_image"])
    else:
        mask = preprocess_image(config["paths"]["mask_image"])
    
    # Perform inpainting using the generated mask
    inpainted_image = inpaint_image(
        config["paths"]["generated_image"],
        config["paths"]["mask_image"],
        config["enhance_image"]["prompt"],
        config["enhance_image"]["strength"],
        config["enhance_image"]["guidance_scale"],
        config["enhance_image"]["num_inference_steps"],
        config["enhance_image"]["fast"],
        config["enhance_image"]["blur_factor"],
    )

    # Save the inpainted image
    inpainted_image.save(config["paths"]["inpainted_image"], format="WEBP")
    logging.info(f"Inpainted image saved at: {config['paths']['inpainted_image']}")


if __name__ == "__main__":
    main()
