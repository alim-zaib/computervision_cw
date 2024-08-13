import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 1. Feature detection

def HarrisPointsDetector(image, k=0.05, window_size=5, sobel_size=3, threshold=0.01):
    """
    Detects keypoints in an image using the Harris corner detection method.

    :param image: Grayscale image to detect keypoints on.
    :param k: Harris corner constant. Usually 0.04 to 0.06.
    :param window_size: Window size used for the Gaussian filter.
    :param sobel_size: Size of the Sobel kernel.
    :param threshold: Threshold for detecting strong corners.
    :return: List of detected keypoints.
    """
    
    # Compute x and y derivatives using Sobel operator
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_size)
    
    # Initialize variables
    offset = window_size // 2
    height, width = image.shape
    harris_response = np.zeros((height, width))

    # Define Gaussian weights and normalize
    gaussian_weights = cv2.getGaussianKernel(ksize=window_size, sigma=0.5)
    gaussian_weights = gaussian_weights * gaussian_weights.T
    gaussian_weights /= np.sum(gaussian_weights)

    # Apply padding to the derivative images
    padded_sobelx = cv2.copyMakeBorder(sobelx, offset, offset, offset, offset, cv2.BORDER_REFLECT)
    padded_sobely = cv2.copyMakeBorder(sobely, offset, offset, offset, offset, cv2.BORDER_REFLECT)

    # Compute the Harris matrix M for each pixel
    for y in range(offset, height + offset):
        for x in range(offset, width + offset):
            window_ix2 = padded_sobelx[y-offset:y+offset+1, x-offset:x+offset+1] ** 2
            window_iy2 = padded_sobely[y-offset:y+offset+1, x-offset:x+offset+1] ** 2
            window_ixy = padded_sobelx[y-offset:y+offset+1, x-offset:x+offset+1] * padded_sobely[y-offset:y+offset+1, x-offset:x+offset+1]

            m_ix2 = np.sum(window_ix2 * gaussian_weights)
            m_iy2 = np.sum(window_iy2 * gaussian_weights)
            m_ixy = np.sum(window_ixy * gaussian_weights)

            det_m = m_ix2 * m_iy2 - m_ixy ** 2
            trace_m = m_ix2 + m_iy2

            # Compute the corner strength function R
            r = det_m - k * (trace_m ** 2)
            harris_response[y-offset, x-offset] = r


    # Apply non-maximum suppression in a 7x7 neighborhood
    keypoints = []
    border = 3  # To avoid out-of-bounds access
    for y in range(offset + border, height - offset - border):
        for x in range(offset + border, width - offset - border):
            if harris_response[y, x] > threshold:
                local_max = np.amax(harris_response[y-3:y+4, x-3:x+4])
                if harris_response[y, x] == local_max:
                    # Compute the orientation (Optional: if you decide to include orientation calculation)
                    orientation = np.arctan2(sobely[y, x], sobelx[y, x]) * (180.0 / np.pi)
                    orientation = np.mod(orientation, 360)  # Normalize angle to [0, 360)
                    keypoints.append(cv2.KeyPoint(float(x), float(y), size=float(window_size), angle=float(orientation)))
    print(f"Number of keypoints detected: {len(keypoints)}")
    return keypoints

def visualize_harris_keypoints(image_path, k=0.05, window_size=5, sobel_size=3, threshold=0.9):
   # Load image
   image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
   # Ensure image was loaded
   if image is None:
       print(f"Failed to load image: {image_path}")
       return

   # Detect keypoints using custom Harris detector
   keypoints = HarrisPointsDetector(image, k, window_size, sobel_size, threshold)

   # Convert grayscale image to BGR for visualization
   image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

   # Draw keypoints on the image
   image_with_keypoints = cv2.drawKeypoints(image_color, keypoints, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

   # Display the image
   cv2.imshow("Harris Keypoints", image_with_keypoints)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

   # Optionally, save the image
   cv2.imwrite('harris_keypoints.jpg', image_with_keypoints)

# 2. Feature description


def featureMatchingAndComparison(reference_image_path, target_image_path, detector='custom_harris'):
   reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
   target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
   orb = cv2.ORB_create()
   
   # Choose the detector for keypoints
   if detector == 'custom_harris':
       # Custom Harris detector
       ref_keypoints = HarrisPointsDetector(reference_image)
       target_keypoints = HarrisPointsDetector(target_image)
   elif detector == 'built_in_harris':
       # Built-in Harris detector
       orb_harris = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE)
       ref_keypoints, _ = orb_harris.detectAndCompute(reference_image, None)
       target_keypoints, _ = orb_harris.detectAndCompute(target_image, None)
   elif detector == 'built_in_fast':
       # Built-in FAST detector
       orb_fast = cv2.ORB_create()
       ref_keypoints, _ = orb_fast.detectAndCompute(reference_image, None)
       target_keypoints, _ = orb_fast.detectAndCompute(target_image, None)
   else:
       raise ValueError("Detector type not recognized. Choose 'custom_harris', 'built_in_harris', or 'built_in_fast'.")

   # Compute descriptors for both sets of keypoints
   _, ref_descriptors = orb.compute(reference_image, ref_keypoints)
   _, target_descriptors = orb.compute(target_image, target_keypoints)

   # Match descriptors
   matches = find_best_matches_with_ratio_test(ref_descriptors, target_descriptors, ratio_threshold=0.75, max_matches=100, distance_threshold=30)
   
   # Visualize the matches
   matched_image = cv2.drawMatches(reference_image, ref_keypoints, target_image, target_keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


   # Resize image for display if it's too large
   screen_res = 1280, 720  # Example screen resolution; adjust as needed
   scale_width = screen_res[0] / matched_image.shape[1]
   scale_height = screen_res[1] / matched_image.shape[0]
   scale = min(scale_width, scale_height)
   window_width = int(matched_image.shape[1] * scale)
   window_height = int(matched_image.shape[0] * scale)

   matched_image_resized = cv2.resize(matched_image, (window_width, window_height))

   output_file_name = f'matched_image_{detector}.jpg'
   cv2.imwrite(output_file_name, matched_image_resized)
   cv2.imshow('Matched Image', matched_image_resized)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

# 3. Feature matching

def compute_ssd(descriptor1, descriptor2):
   # Ensure the descriptors are in a numpy array format
   d1 = np.array(descriptor1, dtype=np.float32)
   d2 = np.array(descriptor2, dtype=np.float32)
   
   # Compute the sum of squared differences
   ssd = np.sum((d1 - d2) ** 2)
   return ssd


# Adjust this function to get a subset of the best matches
def find_best_matches_with_ratio_test(descriptors1, descriptors2, ratio_threshold=0.75, max_matches=500, distance_threshold=None):
   matches = []
   for i, d1 in enumerate(descriptors1):
       ssds = np.array([compute_ssd(d1, d2) for d2 in descriptors2])
       sorted_indices = np.argsort(ssds)
       closest, second_closest = sorted_indices[:2]

       # Apply ratio test
       if ssds[closest] < ratio_threshold * ssds[second_closest]:
           match = cv2.DMatch(_queryIdx=i, _trainIdx=closest, _distance=ssds[closest])
           
           # Apply distance threshold if specified
           if distance_threshold is None or match.distance <= distance_threshold:
               matches.append(match)
   
   # Sort the matches based on distance so that the best matches (lowest distance) come first
   matches = sorted(matches, key=lambda x: x.distance)
   return matches[:max_matches]  # Return only the top 'max_matches' matches



###################################################################

if __name__ == '__main__':
   reference_image_path = 'bernieSanders.jpg'
   target_image_path = 'bernieShoolLunch.jpeg'
   if not os.path.exists(reference_image_path) or not os.path.exists(target_image_path):
       print("One or both images could not be opened.")
       sys.exit()
   
   visualize_harris_keypoints(reference_image_path, k=0.05, window_size=5, sobel_size=3, threshold=0.01)
   #featureMatchingAndComparison(reference_image_path, target_image_path, detector='custom_harris')