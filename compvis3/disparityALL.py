import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

# Global variables
numDisparities = 16  # Starting value, must be a multiple of 16
blockSize = 5        # Must be an odd number, minimum 5
k = 1.0           

SENSOR_WIDTH_MM = 22.2
IMAGE_WIDTH_PIXELS = 3088
F_PIXELS = 5806.559
BASELINE = 174.019  # in millimeters
DOFFS = 114.291  # in pixels

# Focal LENGTH
F_MM = F_PIXELS * (SENSOR_WIDTH_MM / IMAGE_WIDTH_PIXELS)

# ================================================
#
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits
    return disparity

# ================================================
#
def updateImages():
    global imgL, imgR, imgL_colour, disparity
    disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize)
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
    k_adjusted = max(k, 0.1)
    disparity_adjusted = disparity + 0.001
    depth = 1.0 / (disparity_adjusted + k_adjusted)
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    heavy_blur_size = 31  # Use a large fixed kernel size for heavy blurring
    blurred_img = cv2.GaussianBlur(grayscale_img, (heavy_blur_size, heavy_blur_size), 0)

    threshold_value = int(blockSize / 50 * 255) 
    _, mask = cv2.threshold(depth_normalized, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Blend sharp foreground with blurred background using the mask
    foreground = cv2.bitwise_and(grayscale_img, grayscale_img, mask=mask)
    background = cv2.bitwise_and(blurred_img, blurred_img, mask=cv2.bitwise_not(mask))

    combined = cv2.add(foreground, background)

    return combined

# ================================================
#
def update(_):
    minVal = cv2.getTrackbarPos('Min Threshold', 'Disparity')
    maxVal = cv2.getTrackbarPos('Max Threshold', 'Disparity')
    numDisparities = cv2.getTrackbarPos('numDisparities', 'Disparity') * 16  # Ensure multiple of 16
    blockSize = cv2.getTrackbarPos('blockSize', 'Disparity')
    if blockSize < 5:
        blockSize = 5  # Enforce minimum blockSize
    if blockSize % 2 == 0:
        blockSize += 1  # Make blockSize odd if it is even

    edgesL = cv2.Canny(imgL, minVal, maxVal)
    edgesR = cv2.Canny(imgR, minVal, maxVal)

    disparity_edges = getDisparityMap(edgesL, edgesR, numDisparities, blockSize)
    disparity_edges_img = np.interp(disparity_edges, (disparity_edges.min(), disparity_edges.max()), (0.0, 1.0))

    cv2.imshow('Disparity', disparity_edges_img)

# ================================================
#
def plot(disparity, h, w, f, B, doffs):
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    Z = (f * B) / (disparity + doffs + 1e-6)
    
    # Mask out invalid or zero disparity values to avoid infinite depths
    valid_mask = disparity > 0
    x = x[valid_mask]
    y = y[valid_mask]
    Z = Z[valid_mask]
    
    max_Z = np.max(Z)
    distance_threshold = 0.98 * max_Z
    close_enough_mask = Z < distance_threshold

    # Removing statistical outliers
    mean_Z = np.mean(Z[close_enough_mask])
    std_Z = np.std(Z[close_enough_mask])
    inlier_mask = (Z > (mean_Z - 2 * std_Z)) & (Z < (mean_Z + 2 * std_Z))

    final_mask = close_enough_mask & inlier_mask

    # Filter x, y, and Z based on final combined mask
    X = (x[final_mask] - w // 2) * Z[final_mask] / f
    Y = (y[final_mask] - h // 2) * Z[final_mask] / f
    Z = Z[final_mask]

    # Shift coordinates to ensure all are positive
    X += np.abs(np.min(X))
    Y += np.abs(np.min(Y))

    # Create figure for 3D subplots
    fig = plt.figure(figsize=(18, 6))

    # 3D View
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X, Z, Y, s=0.5)   # Swapped Y and Z
    ax1.set_title('3D View')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')  # Swapped Y and Z for all
    ax1.set_zlabel('Y')

    # Top View - Looking from above, X horizontal, Z vertical
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(X, Z, Y, s=0.5)
    ax2.view_init(elev=90, azim=-90)  
    ax2.set_title('Top View')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z') 
    ax2.set_zlabel('Y')  

    # Side View - Looking from the side, Z horizontal, Y vertical
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(X, Z, Y, s=0.5)
    ax3.view_init(elev=0, azim=-180)  
    ax3.set_title('Side View')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z') 
    ax3.set_zlabel('Y')  

    plt.show()

if __name__ == '__main__':
    imgL = cv2.imread('girlL.png', cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('girlR.png', cv2.IMREAD_GRAYSCALE)
    imgL_colour = cv2.imread('girlL.png', cv2.IMREAD_COLOR)  # Load color image for processing
    if imgL is None or imgR is None:
        print('\nError: failed to load images.\n')
        sys.exit()

    # Utilise functions
    print("Focal Length (mm): "+ str(F_MM))
