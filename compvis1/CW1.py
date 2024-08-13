import cv2
import numpy as np

def convolve_3x3(image, kernel):
    # Convert image to float for precision
    image = image.astype(np.float32)
    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Pad the image with zeros on all sides
    padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)
    
    # Create an empty image to store the convolution output
    output = np.zeros((height, width, image.shape[2]))
    
    # Perform convolution
    for y in range(height):
        for x in range(width):
            for c in range(image.shape[2]):  # Handle each color channel
                # Extract the current region of interest
                region = padded_image[y:y+3, x:x+3, c]
                # Apply the kernel to the region (element-wise multiplication followed by sum)
                output[y, x, c] = np.sum(region * kernel)
                
    # Normalize the output to ensure the pixel values are valid [0-255]
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output

average_kernel = np.ones((3, 3), np.float32) / 9
weighted_average_kernel = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], np.float32) / 16

# Load the image
image = cv2.imread('kitty.bmp')

# Apply the average kernel
average_result = convolve_3x3(image, average_kernel)

# Apply the weighted average kernel
weighted_average_result = convolve_3x3(image, weighted_average_kernel)

# Display the results
"""cv2.imshow('Original', image)
cv2.imshow('Average Kernel', average_result)
cv2.imshow('Weighted Average Kernel', weighted_average_result)
cv2.waitKey(0)
cv2.destroyAllWindows()"""


# Sobel kernels for horizontal and vertical gradients
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

# Convolve with Sobel kernels
gradient_x = convolve_3x3(image, sobel_x)
gradient_y = convolve_3x3(image, sobel_y)

# Compute the gradient magnitude (edge strength)
gradient_magnitude = np.sqrt(np.square(gradient_x.astype(np.float32)) + np.square(gradient_y.astype(np.float32)))
gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

# Display the gradient images and the gradient magnitude image
#cv2.imshow('Gradient X', gradient_x)
#cv2.imshow('Gradient Y', gradient_y)
cv2.imshow('Edge Strength', gradient_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()
