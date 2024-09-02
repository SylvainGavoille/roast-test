from PIL import Image
import yaml
import logging
import time

def load_configuration(config_path="config.yml"):
    """
    Load configuration parameters from a YAML file.

    Args:
        config_path (str): Path to the configuration file. Defaults to "config.yml".

    Returns:
        dict: A dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_and_process_image(image_path):
    """
    Load and convert an image to RGB format from the specified path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        PIL.Image.Image: The loaded and converted image in RGB format.
    """
    return Image.open(image_path).convert("RGB")

def resize_image(image, resize_factor):
    """
    Resize the image by a specified factor using the LANCZOS resampling filter.

    Args:
        image (PIL.Image.Image): The image to resize.
        resize_factor (int): The factor by which to resize the image (e.g., 2 to reduce by half).

    Returns:
        PIL.Image.Image: The resized image.
    """
    return image.resize(
        (image.width // resize_factor, image.height // resize_factor),
        Image.Resampling.LANCZOS
    )

def resize_and_save_image(image, target_size, save_path):
    """
    Resize the image to the target size and save it as a WEBP file.

    Args:
        image (PIL.Image.Image): The image to resize.
        target_size (tuple): The target size as a tuple of (width, height).
        save_path (str): The file path where the resized image will be saved.
    """
    image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
    image_resized.save(save_path, format="WEBP")
    logging.info(f"Image saved at: {save_path}")

def timer(func):
    """
    Decorator to measure the execution time of a function.

    Args:
        func (function): The function to be timed.

    Returns:
        function: The wrapped function with timing logic.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time

        logging.info(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper