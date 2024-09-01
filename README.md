# Installation:

```
conda env create -f environment.yaml
conda activate myenv
```

# Interpretation of Results:

- High PSNR (>30 dB), High SSIM (>0.9), High FSIM (>0.9), and Low GMSD (<0.1): The images are highly consistent across all metrics, indicating that the upsampled image closely resembles the generated image both in pixel accuracy and structural features.

- Moderate PSNR (20-30 dB), Moderate SSIM (0.7-0.9), Moderate FSIM (0.7-0.9), and Moderate GMSD (<0.2): The images are moderately consistent, with some differences that may be noticeable, particularly in edges and textures.

- Low PSNR (<20 dB), Low SSIM (<0.7), Low FSIM (<0.7), and High GMSD (>0.2): The images show significant differences, suggesting that the upsampling process introduced noticeable artifacts or that the images are not well-aligned or consistent.
