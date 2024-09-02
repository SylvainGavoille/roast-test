import cv2
import torch
import piq
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import logging
from tools import load_configuration

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_images(config):
    """
    Load generated and upsampled images based on the paths specified in the configuration.

    Args:
        config (dict): Configuration dictionary containing image paths.

    Returns:
        tuple: Loaded generated and upsampled images in BGR format.
    """
    generated_image = cv2.imread(config['paths']['generated_image_resized'], cv2.IMREAD_COLOR)
    upsampled_image = cv2.imread(config['paths']['upsampled_image'], cv2.IMREAD_COLOR)
    return generated_image, upsampled_image

def compute_metrics(generated_image, upsampled_image):
    """
    Compute PSNR, SSIM, FSIM, and GMSD metrics between the generated and upsampled images.

    Args:
        generated_image (numpy.ndarray): The generated image.
        upsampled_image (numpy.ndarray): The upsampled image.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    # Convert images to grayscale for SSIM (optional)
    generated_image_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
    upsampled_image_gray = cv2.cvtColor(upsampled_image, cv2.COLOR_BGR2GRAY)

    # Compute PSNR
    psnr_value = psnr(generated_image, upsampled_image)
    logging.info(f"PSNR value: {psnr_value:.2f} dB")

    # Compute SSIM
    ssim_value, _ = ssim(generated_image_gray, upsampled_image_gray, full=True)
    logging.info(f"SSIM value: {ssim_value:.4f}")

    # Convert images to PyTorch tensors for piq metrics
    generated_image_tensor = torch.tensor(generated_image.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    upsampled_image_tensor = torch.tensor(upsampled_image.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

    # Compute FSIM
    fsim_value = piq.fsim(generated_image_tensor, upsampled_image_tensor).item()
    logging.info(f"FSIM value: {fsim_value:.4f}")

    # Compute GMSD
    gmsd_value = piq.gmsd(generated_image_tensor, upsampled_image_tensor).item()
    logging.info(f"GMSD value: {gmsd_value:.4f}")

    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'FSIM': fsim_value,
        'GMSD': gmsd_value
    }

def plot_metrics(metrics, save_path='metrics_plot.webp'):
    """
    Plot the computed metrics as a bar chart and save the plot as a .webp image.

    Args:
        metrics (dict): A dictionary containing metric names and their corresponding values.
        save_path (str): The file path where the plot will be saved. Defaults to 'metrics_plot.webp'.
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red'])

    # Set limits for y-axis based on metric values
    plt.ylim(0, 1 if max(metrics.values()) <= 1 else 100)
    plt.ylabel('Metric Value')
    plt.title('Image Consistency Metrics')

    # Annotate the plot with the exact metric values
    for bar, value in zip(bars, metrics.values()):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                 f'{value:.4f}' if value <= 1 else f'{value:.2f}', ha='center', va='bottom', fontsize=12)

    # Save the plot as a .webp image
    plt.savefig(save_path, format='webp')
    plt.close()

    logging.info(f"Metrics plot saved at: {save_path}")

def interpret_metrics(metrics, config):
    """
    Interpret the metrics to determine the consistency between images.

    Args:
        metrics (dict): A dictionary containing the computed metrics.
        config (dict): Configuration dictionary containing interpretation thresholds.

    Returns:
        str: A conclusion string based on the metrics.
    """
    if (metrics['PSNR'] > config['interpretation']['high_consistency']['psnr'] and
        metrics['SSIM'] > config['interpretation']['high_consistency']['ssim'] and
        metrics['FSIM'] > config['interpretation']['high_consistency']['fsim'] and
        metrics['GMSD'] < config['interpretation']['high_consistency']['gmsd']):
        return "The images are highly consistent."
    elif (metrics['PSNR'] > config['interpretation']['moderate_consistency']['psnr'] and
          metrics['SSIM'] > config['interpretation']['moderate_consistency']['ssim'] and
          metrics['FSIM'] > config['interpretation']['moderate_consistency']['fsim'] and
          metrics['GMSD'] < config['interpretation']['moderate_consistency']['gmsd']):
        return "The images are moderately consistent."
    else:
        return "The images show significant differences."

def main():
    """
    Main function to load configuration, compute image metrics, plot results, and interpret the consistency between images.
    """
    # Load configuration
    config = load_configuration()

    # Load images
    generated_image, upsampled_image = load_images(config)

    # Compute metrics
    metrics = compute_metrics(generated_image, upsampled_image)

    # Plot metrics
    plot_metrics(metrics, config['interpretation']['paths']['metrics_plot'])

    # Interpret and log conclusion
    conclusion = interpret_metrics(metrics, config)
    logging.info(f"Conclusion: {conclusion}")

if __name__ == '__main__':
    main()
