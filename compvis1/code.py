
import numpy as np

import cv2

import matplotlib.pyplot as plt

def convolve_3x3(image, kernel):
    # Ensure kernel is 3x3
    if kernel.shape != (3, 3):
        raise ValueError("Kernel must be 3x3.")
    
    # Check to ensure the image is grayscale
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale.")
    
    # Convert image to float for precision
    image = image.astype(np.float32)
    
    # Get image dimensions
    height, width = image.shape
    
    # Manual padding with zeros around the original image
    padded_image = np.zeros((height + 2, width + 2), dtype=np.float32)
    padded_image[1:-1, 1:-1] = image
    
    # Initialize the output image
    output = np.zeros((height, width), dtype=np.float32)
    
    # Perform convolution using the kernel
    for y in range(height):
        for x in range(width):
            # Extract the current region of interest
            region = padded_image[y:y+3, x:x+3]
            # Apply the kernel to the region (element-wise multiplication followed by sum)
            output[y, x] = np.sum(region * kernel)
    
    # Normalize the output to ensure the pixel values are valid [0-255]
    #output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output

average_kernel = np.ones((3, 3), np.float32) / 9
weighted_average_kernel = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], np.float32) / 16

# Load the image
# Load the image as grayscale
image = cv2.imread('kitty.bmp', cv2.IMREAD_GRAYSCALE)

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

# Calculate the gradient magnitude from the gradients
gradient_magnitude = np.sqrt(np.square(gradient_x.astype(np.float32)) + np.square(gradient_y.astype(np.float32)))

# Normalize the gradient magnitude image to the range 0-255
# First, find the maximum value in the gradient magnitude image
max_val = gradient_magnitude.max()
# Then, scale all values to be within the range 0-255
gradient_magnitude = (gradient_magnitude / max_val) * 255 if max_val > 0 else gradient_magnitude
# Finally, convert the scaled values to unsigned 8-bit integers
gradient_magnitude = gradient_magnitude.astype(np.uint8)

# Ensure all images are correctly scaled to uint8 for display purposes
original_uint8 = image.astype(np.uint8)
average_result_uint8 = average_result.astype(np.uint8)
weighted_average_result_uint8 = weighted_average_result.astype(np.uint8)

# Use matplotlib to display the images side by side
plt.figure(figsize=(15, 5))  # Adjust the figure size as needed

# Display original image
plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st subplot
plt.imshow(original_uint8, cmap='gray')
plt.title('Original Image')
plt.axis('off')  # Hide axes ticks

# Display image after applying average kernel
plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd subplot
plt.imshow(average_result_uint8, cmap='gray')
plt.title('Average Kernel Applied')
plt.axis('off')

# Display image after applying weighted average kernel
plt.subplot(1, 3, 3)  # 1 row, 3 columns, 3rd subplot
plt.imshow(weighted_average_result_uint8, cmap='gray')
plt.title('Weighted Average Kernel Applied')
plt.axis('off')

plt.show()


# Display the gradient images and the gradient magnitude image
#cv2.imshow('Gradient X', gradient_x)
#cv2.imshow('Gradient Y', gradient_y)
cv2.imshow('Edge Strength', gradient_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate the histogram with more bins for more detail
# 256 bins cover the full range, but you can increase the number of bins if needed for more granularity
hist = cv2.calcHist([gradient_magnitude], [0], None, [256], [0, 256])

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.plot(hist, color='black')
plt.title('Edge Strength Histogram')
plt.xlabel('Edge Strength Value')
plt.ylabel('Pixel Count')

# Set the x-axis limits to zoom in on a particular range if needed
# For example, if most values are below 50, you could use plt.xlim([0, 50])
plt.xlim([0, 256])

# To zoom in on the y-axis, you can set the ylim to the range of interest
# For example, if the pixel count range of interest is 0 to 1000, you can use plt.ylim([0, 1000])
# Uncomment and adjust the following line based on your specific data
# plt.ylim([0, 1000])

plt.grid(True)
plt.show()




def manual_threshold(image, threshold_value):
    # Initialize a new image of zeros (black) with the same dimensions as the input image
    thresholded_image = np.zeros_like(image)
    # Set pixels to white (255) where the pixel value is greater than the threshold
    thresholded_image[image > threshold_value] = 255
    return thresholded_image

def update_threshold(x):
    # Retrieve the current position/value of the slider
    threshold_value = cv2.getTrackbarPos('Threshold', 'Edges')
    # Apply manual thresholding to the gradient magnitude image
    thresholded_image = manual_threshold(gradient_magnitude, threshold_value)
    # Show the thresholded image in the "Edges" window
    cv2.imshow('Edges', thresholded_image)

# Assuming 'gradient_magnitude' is your edge strength image
# Convert it to 8-bit if it's not already
#gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

# Create a window to display results
cv2.namedWindow('Edges')

# Create a trackbar (slider) in the window for the threshold value
cv2.createTrackbar('Threshold', 'Edges', 0, 255, update_threshold)

# Initialize the display with the threshold value 0
update_threshold(0)

# Display the window until any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()


# WEIGHTED-MEAN SMOOTHING

# Apply the weighted average kernel to smooth the image
smoothed_image = convolve_3x3(image, weighted_average_kernel)

# Sobel kernels for horizontal and vertical gradients
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

# Convolve the smoothed image with Sobel kernels to get gradients
smoothed_gradient_x = convolve_3x3(smoothed_image, sobel_x)
smoothed_gradient_y = convolve_3x3(smoothed_image, sobel_y)

# Compute the gradient magnitude (edge strength) from the smoothed gradients
smoothed_gradient_magnitude = np.sqrt(np.square(smoothed_gradient_x.astype(np.float32)) + np.square(smoothed_gradient_y.astype(np.float32)))

# Normalize the gradient magnitude image to the range 0-255
# First, find the maximum value in the smoothed gradient magnitude image
max_val_smoothed = smoothed_gradient_magnitude.max()

# Then, scale all values to be within the range 0-255, but only if max_val_smoothed is greater than 0 to avoid division by zero
smoothed_gradient_magnitude = (smoothed_gradient_magnitude / max_val_smoothed * 255 if max_val_smoothed > 0 else smoothed_gradient_magnitude).astype(np.uint8)


smoothed_gradient_magnitude = smoothed_gradient_magnitude.astype(np.uint8)
