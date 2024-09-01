import cv2
import torch
import piq
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np

# Load the images
generated_image = cv2.imread('generated_image_resized.webp', cv2.IMREAD_COLOR)
upsampled_image = cv2.imread('upsampled_image.webp', cv2.IMREAD_COLOR)

# Convert images to grayscale for SSIM (optional)
generated_image_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
upsampled_image_gray = cv2.cvtColor(upsampled_image, cv2.COLOR_BGR2GRAY)

# Compute PSNR
psnr_value = psnr(generated_image, upsampled_image)
print(f"PSNR value: {psnr_value:.2f} dB")

# Compute SSIM
ssim_value, ssim_map = ssim(generated_image_gray, upsampled_image_gray, full=True)
print(f"SSIM value: {ssim_value:.4f}")

# Convert images to PyTorch tensors for piq metrics
generated_image_tensor = torch.tensor(generated_image.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
upsampled_image_tensor = torch.tensor(upsampled_image.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

# Compute FSIM
fsim_value = piq.fsim(generated_image_tensor, upsampled_image_tensor).item()
print(f"FSIM value: {fsim_value:.4f}")

# Compute GMSD
gmsd_value = piq.gmsd(generated_image_tensor, upsampled_image_tensor).item()
print(f"GMSD value: {gmsd_value:.4f}")

# Create a plot to visualize the results
metrics = ['PSNR', 'SSIM', 'FSIM', 'GMSD']
values = [psnr_value, ssim_value, fsim_value, gmsd_value]

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])

# Set limits for y-axis for better visualization (based on expected value ranges)
plt.ylim(0, 1 if max(values) <= 1 else 100)
plt.ylabel('Metric Value')
plt.title('Image Consistency Metrics')

# Annotate the plot with the exact metric values
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{value:.4f}' if value <= 1 else f'{value:.2f}', ha='center', va='bottom', fontsize=12)

plt.show()

# Interpretation based on the results
if psnr_value > 30 and ssim_value > 0.9 and fsim_value > 0.9 and gmsd_value < 0.1:
    conclusion = "The images are highly consistent."
elif psnr_value > 20 and ssim_value > 0.7 and fsim_value > 0.7 and gmsd_value < 0.2:
    conclusion = "The images are moderately consistent."
else:
    conclusion = "The images show significant differences."

print("Conclusion:", conclusion)
