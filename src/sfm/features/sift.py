import cv2
import numpy as np

from dataclasses import dataclass

from io_utils.image_loader import ImageData


""" FeatureSet contains image specific information about features detected by SIFT """
@dataclass
class FeatureSet:
    image_id: int
    keypoints: list[cv2.KeyPoint]
    points: np.ndarray
    descriptors: np.ndarray


"""
For a single image, use library SIFT implementation to detect features 
Input: 
    image: ImageData, information about the image
Output:
    features: FeatureSet: information about the keypoints detected and their descriptors
"""
def detect_sift_features(image: ImageData) -> FeatureSet:

    gray = image.gray

    # Apply the SIFT detector
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Load data into the struct and return
    pts = np.array([kp.pt for kp in keypoints], dtype=np.float64)
    temp = FeatureSet(image.image_id, keypoints, pts, descriptors)
    return temp

"""
For all images, detect features and build the list
Input:
    images: list[ImageData], list of image structs
Output:
    features: dict[int, FeatureSet], dictionary mapping image index to feature struct
"""
def detect_features_for_images(images: list[ImageData]) -> dict[int, FeatureSet]:

    features = {}
    for image in images:
        features[image.image_id] = detect_sift_features(image)
    
    return features

