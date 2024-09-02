# Image Enhancement and Consistency Check Project

This project provides a set of scripts and tools for enhancing image quality and checking the consistency of generated images. The project is organized to use a `Makefile` for easy execution of tasks and an `environment.yml` file for setting up the necessary environment.

## Project Structure

- **src/**: Contains the Python scripts.
  - `enhance_image.py`: Enhances an image based on a given prompt using a diffusion model.
  - `quality_enhancement.py`: Upscales an image using a super-resolution pipeline.
  - `check_consistency.py`: Compares two images using various metrics to check for consistency.
- **data/**: Stores the input and output images.
  - `original_image.webp`: The original input image.
  - `generated_image.webp`: The initial generated image.
  - `upsampled_image.webp`: The upscaled image.
  - `generated_image_resized.webp`: The resized generated image for comparison.
- **config.yml**: Configuration file containing paths and parameters for the scripts.
- **environment.yml**: Defines the environment setup, including all dependencies needed for the project.
- **Makefile**: Provides commands to run the different scripts easily.

## Prerequisites

Ensure you have Conda installed to create the environment from the provided `environment.yml` file.

### Setting Up the Environment

To set up the environment for this project:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create the environment using Conda:

```bash
   conda env create -f environment.yml
```