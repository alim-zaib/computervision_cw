import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

# Global variables
numDisparities = 16  # Starting value, must be a multiple of 16
blockSize = 5        # Must be an odd number, minimum 5
k = 1.0              # Depth calculation factor, adjustable via trackbar

# ================================================
#
def getDisparityMap(imL, imR):
    global numDisparities, blockSize
    stereo = cv2.StereoBM_create(numDisparities, blockSize)
    disparity = stereo.compute(imL, imR).astype(np.float32) / 16
    return disparity

# ================================================
#
def updateImages():
    global imgL, imgR, imgL_colour, disparity
    disparity = getDisparityMap(imgL, imgR)
    selective_focus_img = applySelectiveFocus(disparity, imgL_colour)
    cv2.imshow('Selective Focus Image', selective_focus_img)
    disparity_visual = (disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255
    cv2.imshow('Disparity', disparity_visual.astype(np.uint8))

# ================================================
#
def on_numDisparities_trackbar(val):
    global numDisparities
    numDisparities = max(16, val * 16)
    updateImages()

# ================================================
#
def on_blockSize_trackbar(val):
    global blockSize
    if val % 2 == 0:  # Convert even values to the nearest odd number
        val += 1
    blockSize = max(5, val)
    updateImages()

# ================================================
#
def on_k_trackbar(val):
    global k
    k = val / 10.0  # Scale k to make the effect perceptible
    updateImages()

# ================================================
#
def applySelectiveFocus(disparity, img):
    global blockSize, k
    # Compute depth and normalize
    k_adjusted = max(k, 0.1)
    disparity_adjusted = disparity + 0.001
    depth = 1.0 / (disparity_adjusted + k_adjusted)
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert the entire image to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Always blur the background heavily, independent of blockSize
    heavy_blur_size = 31  # Use a large fixed kernel size for heavy blurring
    blurred_img = cv2.GaussianBlur(grayscale_img, (heavy_blur_size, heavy_blur_size), 0)

    # Use blockSize to adjust the sensitivity of the depth mask threshold
    threshold_value = int(blockSize / 50 * 255)  # Scale threshold with blockSize
    _, mask = cv2.threshold(depth_normalized, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Blend sharp foreground with blurred background using the mask
    foreground = cv2.bitwise_and(grayscale_img, grayscale_img, mask=mask)
    background = cv2.bitwise_and(blurred_img, blurred_img, mask=cv2.bitwise_not(mask))

    # Combine foreground and background
    combined = cv2.add(foreground, background)

    return combined

# ================================================
#
if __name__ == '__main__':
    imgL = cv2.imread('girlL.png', cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('girlR.png', cv2.IMREAD_GRAYSCALE)
    imgL_colour = cv2.imread('girlL.png', cv2.IMREAD_COLOR)  # Load color image for processing
    if imgL is None or imgR is None:
        print('\nError: failed to load images.\n')
        sys.exit()

    cv2.imwrite('girlL_gray.png', imgL)  # Save grayscale image
    cv2.imwrite('girlR_gray.png', imgR)  # Save grayscale image

    cv2.namedWindow('Selective Focus Parameters')
    cv2.createTrackbar('NumDisparities', 'Selective Focus Parameters', 1, 16, on_numDisparities_trackbar)
    cv2.createTrackbar('BlockSize', 'Selective Focus Parameters', 2, 50, on_blockSize_trackbar)
    cv2.createTrackbar('K-value', 'Selective Focus Parameters', 10, 100, on_k_trackbar)

    updateImages() 
    cv2.waitKey(0)
    cv2.destroyAllWindows()