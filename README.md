# Image Enhancement and Consistency Check Project

This project provides a set of scripts and tools for enhancing image quality and checking the consistency of generated images. The project is organized to use a `Makefile` for easy execution of tasks and an `environment.yml` file for setting up the necessary environment.

## Project Structure

- **src/**: Contains the Python scripts.
  - `inpainting.py`: Enhances an image based on a given prompt using a diffusion model.
  - `quality_enhancement.py`: Upscales an image using a super-resolution pipeline.
  - `check_consistency.py`: Compares two images using various metrics to check for consistency.
  - `mask.py`: Creates a background mask.
  - `pspnet.py`: Mask RCNN network.
  - `tools.py`: A set of functions used in the different files
- **data/**: Stores the input and output images.
  - `original_image.webp`: The original input image.
  - `generated_image.webp`: The initial generated image.
  - `upsampled_image.webp`: The upscaled image.
  - `generated_image_resized.webp`: The resized generated image for comparison.
  - `inpainted_image.webp`: The inpainting obtained with the hair mask.
  - `inpainted_image2.webp`: The inpainting obtained with the background mask.
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

### Loading the Model

You must create a `.env` file with your HuggingFace token in the environment variable `HUGGING_FACE_TOKEN`. 

Then, download the Mask R-CNN model for hair segmentation [here](https://drive.google.com/file/d/1ZbWTqWLi7w-lVvf7TQ59Gqil_SJnofbE/view), and fill in the path where the model is located in `config.yml`.

## Available Commands

### 1. Run Quality Enhancement

This command is used to enhance the overall quality of an image, likely improving sharpness, resolution, or visual details.

**Command:**

```bash
make quality
```

### 2. Run Consistency Check

This command checks the consistency of the images, ensuring that there are no anomalies or issues with image data integrity.

**Command:**

```bash
make inpainting
```

### 3. Run Image Enhancement

This command runs the image enhancement process using an inpainting technique. The enhancement is created based on a prompt defined in config.yml.

**Command:**

```bash
make inpainting
```


