import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter, sobel, maximum_filter
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def HarrisPointsDetector(image_path, threshold=0.05):
    image_color = cv2.imread(image_path)
    
    if image_color is None:
        raise ValueError(f"The image at {image_path} could not be loaded.")
    
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Compute x and y derivatives of the image using Sobel filter
    Ix = sobel(image_gray, axis=0, mode='reflect')
    Iy = sobel(image_gray, axis=1, mode='reflect')

    # Apply a 5x5 Gaussian mask to the squared gradients
    Ix2 = gaussian_filter(Ix**2, sigma=0.5, mode='reflect')
    Iy2 = gaussian_filter(Iy**2, sigma=0.5, mode='reflect')
    Ixy = gaussian_filter(Ix*Iy, sigma=0.5, mode='reflect')

    # Compute the Harris response R for each pixel
    detM = Ix2 * Iy2 - Ixy**2
    traceM = Ix2 + Iy2
    R = detM - 0.05 * (traceM ** 2)

    # Identify local maxima
    R_max = maximum_filter(R, size=7)
    local_maxima = (R == R_max)  
    thresholded_maxima = (R > threshold * R.max()) & local_maxima  

    # Compute the orientation for keypoints
    orientation = np.arctan2(Iy, Ix)
    orientation_degrees = np.rad2deg(orientation)
    orientation_adjusted = (orientation_degrees + 360) % 360  #

    # Generate keypoints for OpenCV
    keypoints_indices = np.argwhere(thresholded_maxima)
    keypoints = [cv2.KeyPoint(x=float(pt[1]), y=float(pt[0]), size=1, angle=orientation_adjusted[pt[0], pt[1]])
                 for pt in keypoints_indices]

    return keypoints


def featureDescriptor(image, keypoints):
    orb = cv2.ORB_create()
    _, descriptors = orb.compute(image, keypoints)
    return descriptors

def SSDFeatureMatcher(descriptors1, descriptors2):
    ssd_matrix = cdist(descriptors1, descriptors2, 'sqeuclidean')
    
    matches = []
    for i, distances in enumerate(ssd_matrix):
        train_idx = np.argmin(distances)
        matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=train_idx, _distance=float(distances[train_idx])))
    return matches

def RatioFeatureMatcher(descriptors1, descriptors2, ratio_threshold=None):
    ssd_matrix = cdist(descriptors1, descriptors2, 'sqeuclidean')
    
    initial_matches_count = ssd_matrix.size  # Total possible matches before applying the ratio test
    print(f"Ratio threshold: {ratio_threshold}")
    print(f"Initial number of potential matches: {initial_matches_count}")

    matches = []
    for i, distances in enumerate(ssd_matrix):
        sorted_indices = np.argsort(distances)
        closest, second_closest = sorted_indices[:2]
        
        if distances[closest] < ratio_threshold * distances[second_closest]:
            matches.append(cv2.DMatch(i, closest, distances[closest]))

    final_matches_count = len(matches) 
    print(f"Number of matches after applying ratio test: {final_matches_count}")

    return matches

def featureMatchingAndComparison(reference_image_path, target_image_path, detector='built_in_harris', match_method='ssd', ratio_threshold=None):
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    
    orb = cv2.ORB_create()

    if detector == 'custom_harris':
        ref_keypoints = HarrisPointsDetector(reference_image_path)
        target_keypoints = HarrisPointsDetector(target_image_path)
    else:
        if detector == 'built_in_harris':
            orb.setScoreType(cv2.ORB_HARRIS_SCORE)
        ref_keypoints = orb.detect(reference_image, None)
        target_keypoints = orb.detect(target_image, None)

    _, ref_descriptors = orb.compute(reference_image, ref_keypoints)
    _, target_descriptors = orb.compute(target_image, target_keypoints)


    if match_method == 'ssd':
        matches = SSDFeatureMatcher(ref_descriptors, target_descriptors)
        output_file_suffix = 'ssd'
    elif match_method == 'ratio_test':
        matches = RatioFeatureMatcher(ref_descriptors, target_descriptors, ratio_threshold)
        output_file_suffix = f'ratio{ratio_threshold}'

    matched_image = cv2.drawMatches(cv2.imread(reference_image_path), ref_keypoints, 
                                    cv2.imread(target_image_path), target_keypoints, 
                                    matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    matched_image_resized = resizeMatchedImage(matched_image)

    annotateMatchedImage(matched_image_resized, detector)

    target_image_basename = os.path.splitext(os.path.basename(target_image_path))[0]
    output_file_name = f'matched_image_{detector}_{output_file_suffix}_{target_image_basename}.jpg'
    cv2.imwrite(output_file_name, matched_image_resized)

    displayMatchedImage(matched_image_resized, detector)

def orbFeatureMatchingAndComparison(reference_image_path, target_image_path, detector='built_in_harris'):
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    target_image_gray = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    target_image_color = cv2.imread(target_image_path)  
    
    orb = cv2.ORB_create()

    if detector == 'built_in_harris':
        orb.setScoreType(cv2.ORB_HARRIS_SCORE)
    elif detector == 'built_in_fast':
        orb.setScoreType(cv2.ORB_FAST_SCORE)
    else:
        raise ValueError("Invalid detector type. Choose 'built_in_harris' or 'built_in_fast'.")

    ref_keypoints, ref_descriptors = orb.detectAndCompute(reference_image, None)
    target_keypoints, target_descriptors = orb.detectAndCompute(target_image_gray, None)

    target_image_with_keypoints = cv2.drawKeypoints(target_image_color, target_keypoints, None, color=(0,255,0), flags=0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(ref_descriptors, target_descriptors)
    matches = sorted(matches, key=lambda x:x.distance)

    print(f"Number of matches: {len(matches)}")

    matched_image = cv2.drawMatches(reference_image, ref_keypoints, 
                                    target_image_with_keypoints, target_keypoints, 
                                    matches, None, flags=2)

    matched_image_resized = resizeMatchedImage(matched_image)
    cv2.putText(matched_image_resized, f'Detector: {detector}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    target_image_basename = os.path.splitext(os.path.basename(target_image_path))[0]
    output_file_name = f'matched_image_{detector}_{target_image_basename}.jpg'
    cv2.imwrite(output_file_name, matched_image_resized)

    cv2.imshow("Matched Image", matched_image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def orbKeypointComparison(reference_image_path, target_image_path, detector='built_in_harris'):
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    
    orb = cv2.ORB_create()

    if detector == 'built_in_harris':
        orb.setScoreType(cv2.ORB_HARRIS_SCORE)
    elif detector == 'built_in_fast':
        orb.setScoreType(cv2.ORB_FAST_SCORE)
    else:
        raise ValueError("Invalid detector type. Choose 'built_in_harris' or 'built_in_fast'.")
    
    ref_keypoints = orb.detect(reference_image, None)
    target_keypoints = orb.detect(target_image, None)

    reference_keypoints_image = cv2.drawKeypoints(reference_image, ref_keypoints, None, color=(0, 255, 0), flags=0)
    target_keypoints_image = cv2.drawKeypoints(target_image, target_keypoints, None, color=(0, 255, 0), flags=0)

    reference_image_resized = resizeMatchedImage(reference_keypoints_image)
    target_image_resized = resizeMatchedImage(target_keypoints_image)
    annotateMatchedImage(reference_image_resized, detector)
    annotateMatchedImage(target_image_resized, detector)

    target_image_basename = os.path.splitext(os.path.basename(target_image_path))[0]
    output_reference_image_name = f'reference_keypoints_{detector}_{target_image_basename}.jpg'
    output_target_image_name = f'target_keypoints_{detector}_{target_image_basename}.jpg'
    cv2.imwrite(output_reference_image_name, reference_image_resized)
    cv2.imwrite(output_target_image_name, target_image_resized)

    displayMatchedImage(reference_image_resized, detector)
    displayMatchedImage(target_image_resized, detector)

def resizeMatchedImage(matched_image):
    screen_res = 1280, 720
    scale_width = screen_res[0] / matched_image.shape[1]
    scale_height = screen_res[1] / matched_image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(matched_image.shape[1] * scale)
    window_height = int(matched_image.shape[0] * scale)
    return cv2.resize(matched_image, (window_width, window_height))

def annotateMatchedImage(matched_image_resized, detector):
    cv2.putText(matched_image_resized, f"Detector: {detector}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)

def displayMatchedImage(matched_image_resized, detector):
    cv2.imshow(f'Keypoints - {detector}', matched_image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_keypoints_vs_threshold(image_path, save_path='keypoints_vs_threshold.png'):
    thresholds = np.linspace(0.01, 0.1, 10)  # from 0.01 to 0.1 in 10 steps
    num_keypoints = []

    for threshold in thresholds:
        keypoints = HarrisPointsDetector(image_path, threshold)
        num_keypoints.append(len(keypoints))

    # Plotting the number of keypoints vs. threshold values
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, num_keypoints, marker='o', linestyle='-', color='b')
    plt.title('Number of Keypoints Detected vs. Threshold Value')
    plt.xlabel('Threshold Value')
    plt.ylabel('Number of Keypoints Detected')
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")
    plt.show()

# Function to help viewing
def resizeImage(image):
    screen_res = 1280, 720
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)
    return cv2.resize(image, (window_width, window_height))



################################################ Usage


#compare_harris_detectors('bernieSanders.jpg', threshold=0.01)

# Looping through detectors for comparison
"""for detector in ['custom_harris', 'built_in_harris', 'built_in_fast']:
    print(f"Comparing using {detector} detector...")
    featureMatchingAndComparison('bernieSanders.jpg', 'brighterBernie.jpg', detector=detector)"""

#smoothed_image_path = smooth_image('bernieNoisy2.png', kernel_size=15, sigma_x=0)
#featureMatchingAndComparison('bernieSanders.jpg', smoothed_image_path, detector='custom_harris', match_method='ratio_test', ratio_threshold=0.7)


# Assuming HarrisPointsDetector returns a list of cv2.KeyPoint objects
#keypoints = HarrisPointsDetector('bernieSanders.jpg', threshold = 0.05)

# Visualize the detected keypoints on the image
#visualize_keypoints('bernieSanders.jpg', keypoints)

#plot_keypoints_vs_threshold('bernieSanders.jpg')


#featureMatchingAndComparison('bernieSanders.jpg', 'bernie180.jpg', detector='custom_harris', match_method='ratio_test', ratio_threshold=0.7)
#featureMatchingAndComparison('bernieSanders.jpg', 'bernie180.jpg', detector='in_built', match_method='ratio_test', ratio_threshold=0.7)


#featureMatchingAndComparison('bernieSanders.jpg', 'brighterBernie.jpg', detector='custom_harris', match_method='ssd')
#featureMatchingAndComparison('bernieSanders.jpg', 'brighterBernie.jpg', detector='custom_harris', match_method='ratio_test', ratio_threshold=0.7)
#featureMatchingAndComparison('bernieSanders.jpg', 'brighterBernieClip.jpg', detector='custom_harris', match_method='ssd')
#featureMatchingAndComparison('bernieSanders.jpg', 'brighterBernieClip.jpg', detector='custom_harris', match_method='ratio_test', ratio_threshold=0.7)

blurred_image = cv2.imread('bernieMoreblurred.jpg', cv2.IMREAD_GRAYSCALE)
sharpened_image = cv2.addWeighted(blurred_image, 1.7, cv2.GaussianBlur(blurred_image, (0, 0), 3), -0.6, 0)
sharpened_image_path = 'sharpened_bernieMoreblurred.jpg'
cv2.imwrite(sharpened_image_path, sharpened_image)

#featureMatchingAndComparison('bernieSanders.jpg', 'sharpened_bernieMoreblurred.jpg', detector='custom_harris', match_method='ratio_test', ratio_threshold=0.7)
featureMatchingAndComparison('bernieSanders.jpg', 'bernieNoisy2.png', detector='custom_harris', match_method='ssd')


#featureMatchingAndComparison('bernieSanders.jpg', 'bernie180.jpg', detector='custom_harris', match_method='ssd')
#featureMatchingAndComparison('bernieSanders.jpg', 'bernie180.jpg', detector='custom_harris', match_method='ratio_test', ratio_threshold=0.7)
#featureMatchingAndComparison('bernieSanders.jpg', 'bernie180.jpg', detector='custom_harris', match_method='ratio_test', ratio_threshold=0.01)
#featureMatchingAndComparison('bernieSanders.jpg', sharpened_image_path, detector='custom_harris', match_method='ssd')
#featureMatchingAndComparison('bernieSanders.jpg', deblurred_image_path, detector='custom_harris', match_method='ssd')
#featureMatchingAndComparison('bernieSanders.jpg', sharpened_image_path, detector='custom_harris', match_method='ratio_test', ratio_threshold=0.7)

#featureMatchingAndComparison('bernieSanders.jpg', 'bernieShoolLunch.jpeg', detector='custom_harris', match_method='ratio_test', ratio_threshold=0.80)
#orbKeypointComparison('bernieSanders.jpg', 'darkerBernie.jpg', detector='built_in_harris')
#orbKeypointComparison('bernieSanders.jpg', 'darkerBernie.jpg', detector='built_in_fast')
#orbKeypointComparison('bernieSanders.jpg', 'darkerBernieClip.jpg', detector='built_in_harris')
#orbKeypointComparison('bernieSanders.jpg', 'darkerBernieClip.jpg', detector='built_in_fast')


#compareSSDandRatioTest('bernieSanders.jpg', 'darkerBernieClip.jpg', ratio_threshold=0.95)
#orbFeatureMatchingAndComparison('bernieSanders.jpg', 'bernieFriends.png', detector='built_in_harris')
#orbFeatureMatchingAndComparison('bernieSanders.jpg', 'bernieFriends.png', detector='built_in_fast')
#orbFeatureMatchingAndComparison('bernieSanders.jpg', 'brighterBernieClip.jpg', detector='built_in_harris')


"""orbFeatureMatchingAndComparison('bernieSanders.jpg', 'bernieMoreblurred.jpg', detector='built_in_harris')
orbFeatureMatchingAndComparison('bernieSanders.jpg', 'bernieMoreblurred.jpg', detector='built_in_fast')
orbFeatureMatchingAndComparison('bernieSanders.jpg', 'bernieNoisy2.png', detector='built_in_harris')
orbFeatureMatchingAndComparison('bernieSanders.jpg', 'bernieNoisy2.png', detector='built_in_fast')
orbFeatureMatchingAndComparison('bernieSanders.jpg', 'berniePixelated2.png', detector='built_in_harris')
orbFeatureMatchingAndComparison('bernieSanders.jpg', 'berniePixelated2.png', detector='built_in_fast')"""

#orbFeatureMatchingAndComparison('bernieSanders.jpg', 'bernieFriends.png', detector='built_in_harris')
#orbFeatureMatchingAndComparison('bernieSanders.jpg', 'bernieBenefitBeautySalon.jpeg', detector='built_in_harris')
#orbFeatureMatchingAndComparison('bernieSanders.jpg', 'bernieShoolLunch.jpeg', detector='built_in_harris')
#compareCornerDetectors('bernieSanders.jpg')