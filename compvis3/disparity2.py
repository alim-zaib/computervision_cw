import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

# Camera parameters
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

    return disparity # floating point image
# ================================================

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
    # Create meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Calculate depth (Z) from disparity map, adding a tiny constant to avoid division by zero
    Z = (f * B) / (disparity + doffs + 1e-6)
    
    # Mask out invalid or zero disparity values to avoid infinite depths
    valid_mask = disparity > 0
    x = x[valid_mask]
    y = y[valid_mask]
    Z = Z[valid_mask]
    
    # Apply a threshold to Z to remove points that are too far away
    max_Z = np.max(Z)
    distance_threshold = 0.98 * max_Z
    close_enough_mask = Z < distance_threshold

    # Further refine by removing statistical outliers
    mean_Z = np.mean(Z[close_enough_mask])
    std_Z = np.std(Z[close_enough_mask])
    inlier_mask = (Z > (mean_Z - 2 * std_Z)) & (Z < (mean_Z + 2 * std_Z))

    # Combine masks
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
    ax2.view_init(elev=90, azim=-90)  # Looking directly downwards
    ax2.set_title('Top View')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')  # Z is now vertical
    ax2.set_zlabel('Y')  # Y is depth, won't be seen from top view

    # Side View - Looking from the side, Z horizontal, Y vertical
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(X, Z, Y, s=0.5)
    ax3.view_init(elev=0, azim=-180)  # Looking directly downwards
    ax3.set_title('Side View')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')  # Z is now vertical
    ax3.set_zlabel('Y')  # Y is depth, won't be seen from top view

    # Show the plots
    plt.show()
# ================================================
#
if __name__ == '__main__':
    # Load images
    filenameL = 'umbrellaL.png'
    filenameR = 'umbrellaR.png'
    imgL = cv2.imread(filenameL, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(filenameR, cv2.IMREAD_GRAYSCALE)

    if imgL is None or imgR is None:
        print('Error: failed to open image files.')
        sys.exit()

    # Apply Canny edge detection to enhance edges
    edgesL = cv2.Canny(imgL, 60, 165)  # Thresholds might need tuning
    edgesR = cv2.Canny(imgR, 60, 165)

    # Parameters
    SENSOR_WIDTH_MM = 22.2
    IMAGE_WIDTH_PIXELS = 3088
    F_PIXELS = 5806.559
    BASELINE = 174.019  # in millimeters
    DOFFS = 114.291  # in pixels

    # Compute the focal length in millimeters
    F_MM = F_PIXELS * (SENSOR_WIDTH_MM / IMAGE_WIDTH_PIXELS)

    # Compute disparity map using the preprocessed images
    numDisparities = 64  # Placeholder values, adjust as needed
    blockSize = 5  # Placeholder values, adjust as needed
    disparity_map = getDisparityMap(edgesL, edgesR, numDisparities, blockSize)

    # Call the plot function with the disparity map and camera parameters
    h, w = disparity_map.shape
    plot(disparity_map, h, w, F_PIXELS, BASELINE, DOFFS)