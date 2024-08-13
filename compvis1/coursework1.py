import numpy as np
import cv2
from matplotlib import pyplot as plt

# Function to perform convolution
def convolve2d(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    image_padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float32)  # Use float32 for precision during convolution
    
    # Perform convolution
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            output[y, x] = np.sum(image_padded[y:y+kernel_height, x:x+kernel_width] * kernel)
    
    return output



# Load the image
image_path = 'kitty.bmp' # Make sure to update this path to where your image is located
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded properly
if image is None:
    print("Error: Image not found. Check the path.")
else:
    # Define a 3x3 average smoothing kernel
    average_kernel = np.ones((3, 3)) / 9

    # Apply the average smoothing kernel
    convolved_image_average = convolve2d(image, average_kernel)

    # Display the original and convolved images
"""    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Convolved with Average Kernel')
    plt.imshow(convolved_image_average, cmap='gray')
    plt.show()
"""
#2 
# Define Sobel kernels
sobel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

# Apply Sobel kernels to the image to compute gradients
gradient_horizontal = convolve2d(image, sobel_horizontal)
gradient_vertical = convolve2d(image, sobel_vertical)

# Calculate the magnitude of gradients (Edge Strength)
edge_strength = np.sqrt(np.square(gradient_horizontal) + np.square(gradient_vertical))

# Assuming the edge strengths are already within the range [0, 255]
edge_strength_normalized = edge_strength.astype(np.uint8)

# Display the edge strength image
plt.title('Edge Strength')
plt.imshow(edge_strength_normalized, cmap='gray')
plt.show()
