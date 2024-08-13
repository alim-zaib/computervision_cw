import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolve(image, kernel):
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel dimensions should be odd.")

    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale.")
    
    image = image.astype(np.float32)
    
    height, width = image.shape
    k_height, k_width = kernel.shape
    
    pad_height = k_height // 2
    pad_width = k_width // 2
    
    padded_image = np.zeros((height + 2*pad_height, width + 2*pad_width), dtype=np.float32)
    padded_image[pad_height:-pad_height, pad_width:-pad_width] = image
    
    output = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            region = padded_image[y:y+k_height, x:x+k_width]
            output[y, x] = np.sum(region * kernel)

    return output


def getGaussianKernel(ksize, sigma):
    offset = ksize // 2
    y, x = np.ogrid[-offset:offset+1, -offset:offset+1]
    gaussian_kernel = np.exp(-(x**2 + y**2) / (2*sigma**2))
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize the kernel
    return gaussian_kernel


image = cv2.imread('kitty.bmp', cv2.IMREAD_GRAYSCALE)


# Define kernels
average_kernel = np.ones((3, 3), np.float32) / 9
weighted_average_kernel = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], np.float32) / 16


# First Gaussian Kernel: 3x3, Sigma=1
gaussian_kernel_1 = getGaussianKernel(3, 2)
gaussian_kernel_params_1 = "5x5, $\\sigma=1$"

# Second Gaussian Kernel: 3x3, Sigma=2
gaussian_kernel_2 = getGaussianKernel(5, 2)
gaussian_kernel_params_2 = "5x5, $\\sigma=2$"

# Second Gaussian Kernel: 3x3, Sigma=1
gaussian_kernel_3 = getGaussianKernel(5, 8)
gaussian_kernel_params_3 = "5x5, $\\sigma=8$"

# Load the image as grayscale
image = cv2.imread('kitty.bmp', cv2.IMREAD_GRAYSCALE)

# Apply the kernels
average_result = convolve(image, average_kernel)
weighted_average_result = convolve(image, weighted_average_kernel)
gaussian_result_1 = convolve(image, gaussian_kernel_1)
# Apply the second Gaussian kernel
gaussian_result_2 = convolve(image, gaussian_kernel_2)

gaussian_result_3 = convolve(image, gaussian_kernel_3)


# Display the results side by side using matplotlib
"""plt.figure(figsize=(20, 6))  # Adjusted figure size for better visualization
plt.subplot(1, 3, 1)  # Changed the subplot parameters to accommodate 3 images
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)  # Changed the subplot parameters
plt.imshow(average_result, cmap='gray')
plt.title('Average Kernel Applied')
plt.axis('off')

plt.subplot(1, 3, 3)  # Changed the subplot parameters
plt.imshow(weighted_average_result, cmap='gray')
plt.title('Weighted Average Kernel Applied')
plt.axis('off')

plt.show()"""

plt.figure(figsize=(20, 5))  # Adjusted figure size

# Original Image
plt.subplot(1, 4, 1)  
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Gaussian Applications
plt.subplot(1, 4, 2)  
plt.imshow(gaussian_result_1, cmap='gray')
plt.title(f'Gaussian Kernel Applied\n{gaussian_kernel_params_1}')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(gaussian_result_2, cmap='gray')
plt.title(f'Gaussian Kernel Applied\n{gaussian_kernel_params_2}')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(gaussian_result_3, cmap='gray')
plt.title(f'Gaussian Kernel Applied\n{gaussian_kernel_params_3}')
plt.axis('off')

plt.show()


##########################################################################################

def calculate_gradients(image):
    # Sobel kernels for horizontal and vertical gradients
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2], 
                        [-1, 0, 1]], np.float32)
    sobel_y = np.array([[1, 2, 1], 
                        [0, 0, 0], 
                        [-1, -2, -1]], np.float32)

    gradient_x = convolve(image, sobel_x)
    gradient_y = convolve(image, sobel_y)

    gradient_magnitude = np.sqrt(np.square(gradient_x.astype(np.float32)) + np.square(gradient_y.astype(np.float32)))

    
    # Normalise the gradient magnitude
    min_val = gradient_magnitude.min()
    max_val = gradient_magnitude.max()
    gradient_magnitude_normalised = (gradient_magnitude - min_val) / (max_val - min_val) * 255

    return gradient_x, gradient_y, gradient_magnitude_normalised.astype(np.uint8)

# Calculate gradients for the weighted average image

original_gradient_x, original_gradient_y, original_gradient_magnitude = calculate_gradients(image)

average_gradient_x, average_gradient_y, average_gradient_magnitude = calculate_gradients(average_result)

weighted_average_gradient_x, weighted_average_gradient_y, weighted_average_gradient_magnitude = calculate_gradients(weighted_average_result)

gaussian_gradient1_x, gaussian_gradient1_y, gaussian_gradient1_magnitude = calculate_gradients(gaussian_result_1)

gaussian_gradient2_x, gaussian_gradient2_y, gaussian_gradient2_magnitude = calculate_gradients(gaussian_result_2)

gaussian_gradient3_x, gaussian_gradient3_y, gaussian_gradient3_magnitude = calculate_gradients(gaussian_result_3)

# Display the Ix and Iy images along with the edge strength image
"""plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_gradient_magnitude, cmap='gray')
plt.title('Edge Strength: Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(average_gradient_magnitude, cmap='gray')
plt.title('Edge Strength: Average')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(weighted_average_gradient_magnitude, cmap='gray')
plt.title('Edge Strength: Weighted Average')
plt.axis('off')

plt.show()"""

#######################################################################################

def plot_uniform_histograms(images, titles):
    plt.figure(figsize=(20, 10))
    
    max_hist_values = [np.max(cv2.calcHist([img], [0], None, [256], [0, 256])) for img in images]

    for i, (image, title, max_val) in enumerate(zip(images, titles, max_hist_values), start=1):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.subplot(1, len(images), i)
        plt.plot(hist, color='black')
        plt.title(f'Edge Strength Histogram - {title}', fontsize=14)
        plt.xlabel('Edge Strength Value', fontsize=12)
        plt.ylabel('Pixel Count', fontsize=12)
        plt.xlim([0, 256])
        plt.ylim([0, max_val + 0.05*max_val]) 
        plt.grid(True)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

    plt.subplots_adjust(wspace=0.3)
    plt.show()

# Load your images into the 'images' list.
#Example usage:

images = [original_gradient_magnitude, gaussian_gradient1_magnitude, gaussian_gradient2_magnitude, gaussian_gradient3_magnitude]
titles = ['Original', 'Gaussian 5x5 (σ = 1)', 'Gaussian 5x5 (σ = 2)', 'Gaussian 5x5 (σ = 5)']

plot_uniform_histograms(images, titles)

#images = [original_gradient_magnitude, average_gradient_magnitude, weighted_average_gradient_magnitude, gaussian_gradient1_magnitude]


###############################################################################################

def manual_threshold(image, threshold_value):
    # Initialise a new image of zeros (black) with the same dimensions as the input image
    thresholded_image = np.zeros_like(image)
    thresholded_image[image > threshold_value] = 255
    return thresholded_image

def update_threshold(threshold_value):
    # Apply manual thresholding to each gradient magnitude image
    original_thresholded = manual_threshold(original_gradient_magnitude, threshold_value)
    #average_thresholded = manual_threshold(average_gradient_magnitude, threshold_value)
    #weighted_average_thresholded = manual_threshold(weighted_average_gradient_magnitude, threshold_value)
    gaussian1_thresholded = manual_threshold(gaussian_gradient1_magnitude, threshold_value)
    gaussian2_thresholded = manual_threshold(gaussian_gradient2_magnitude, threshold_value)
    gaussian3_thresholded = manual_threshold(gaussian_gradient3_magnitude, threshold_value)


    cv2.imshow('Original Edges', original_thresholded)
    #cv2.imshow('Average Kernel Edges', average_thresholded)
    #cv2.imshow('Weighted Average Kernel Edges', weighted_average_thresholded)
    cv2.imshow('Gaussian Kernel Edges 1', gaussian1_thresholded)
    cv2.imshow('Gaussian Kernel Edges 2', gaussian2_thresholded)
    cv2.imshow('Gaussian Kernel Edges 3', gaussian3_thresholded)


def on_trackbar(val):
    # This function is called whenever the trackbar slider is moved
    update_threshold(val)

#Create windows for displaying the images
cv2.namedWindow('Original Edges')
#cv2.namedWindow('Average Kernel Edges')
#cv2.namedWindow('Weighted Average Kernel Edges')
cv2.namedWindow('Gaussian Kernel Edges 1')
cv2.namedWindow('Gaussian Kernel Edges 2')
cv2.namedWindow('Gaussian Kernel Edges 3')

cv2.createTrackbar('Threshold', 'Original Edges', 0, 255, on_trackbar)

# Initialise the display with the threshold value 0
update_threshold(0)

cv2.waitKey(0)
cv2.destroyAllWindows()

def apply_otsu_threshold(image):
    # Convert image to 8-bit if not already
    image = np.uint8(image)
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded_image

# Apply Otsu's thresholding to each gradient magnitude image
otsu_original = apply_otsu_threshold(original_gradient_magnitude)
otsu_average = apply_otsu_threshold(average_gradient_magnitude)
otsu_weighted_average = apply_otsu_threshold(weighted_average_gradient_magnitude)
otsu_gaussian = apply_otsu_threshold(gaussian_gradient1_magnitude)

"""plt.figure(figsize=(15, 5))  # Adjusted figure size

plt.subplot(1, 3, 1)
plt.imshow(otsu_original, cmap='gray')
plt.title('Original (Otsu)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(otsu_average, cmap='gray')
plt.title('Average Kernel (Otsu)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(otsu_weighted_average, cmap='gray')
plt.title('Weighted Average Kernel (Otsu)')
plt.axis('off')

plt.show()
"""