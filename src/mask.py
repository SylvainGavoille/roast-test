# Import the pipeline function from the transformers library
from transformers import pipeline

# Define the path to the image from which we want to remove the background
image_path = "./data/generated_image.webp"

# Initialize the image segmentation pipeline using the "briaai/RMBG-1.4" model,
# which is specifically designed for background removal. The parameter 'trust_remote_code=True'
# allows running custom code from the model repository.
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

# Generate a mask for the image that separates the foreground from the background.
# The 'return_mask=True' argument ensures that only the mask (without applying it to the image)
# is returned as a Pillow object.
pillow_mask = pipe(image_path, return_mask=True)  # outputs a pillow mask

# Convert the mask to grayscale mode (if needed)
pillow_mask = pillow_mask.convert("L")

# Ensure that the mask is binary (black and white), we can threshold it.
# In this case, we assume that the mask is semi-transparent or grayscale, so we apply a threshold
# operation to make it strictly black and white.
threshold_value = 128  # Adjust as needed for your image
binary_mask = pillow_mask.point(lambda p: 255 if p < threshold_value else 0)

# Save the binary mask as a PNG
binary_mask.save("./data/mask.png")

# Apply the generated mask to the input image. This returns the image with the background removed.
pillow_image = pipe(image_path)  # applies mask on input and returns a pillow image
pillow_image.save("./data/without_background.png")
