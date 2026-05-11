import numpy as np
from itertools import combinations

from dataclasses import dataclass

from sift import FeatureSet

from config import FEATURE_MATCH_THRESHOLD

""" ImageMatchPair contains information about the keypoint matches between two images """
@dataclass
class ImageMatchPair:
    image_id1: int
    image_id2: int
    keypoint_indices1: np.ndarray
    keypoint_indices2: np.ndarray
    distances: np.ndarray

"""
Given detected features in two images, find matches via SSD, build struct, return
Input:
    features1: FeatureSet, SIFT features detected in the first image
    features2: FeatureSet, SIFT features detected in the second image
Output:
    matches: ImageMatchPair, indices of matches and the distances between descriptors
"""
def match_image_pair(features1: FeatureSet, features2: FeatureSet) -> ImageMatchPair:
    """
    Overview of the algorithm:
        - Compute the SSD between each descriptor to every descriptor in the second image
        - Compute the ratio of the best match to the second best match
        - If below a threshold, add the match
    """

    matches = ImageMatchPair(features1.image_id, features2.image_id, None, None, None)

    keypoint_indices1 = []
    keypoint_indices2 = []
    distances = []

    desc1 = features1.descriptors
    desc2 = features2.descriptors

    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) < 2:
        return ImageMatchPair(
            features1.image_id,
            features2.image_id,
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )
    
    for i, descriptor1 in enumerate(desc1):
        dist = []

        for j, descriptor2 in enumerate(desc2):
            
            # Compute the difference between the two descriptors and the SSD
            diff = descriptor1 - descriptor2
            ssd = np.sum(diff * diff)
            dist.append((ssd, j))
        
        # Sort according to the SSD term
        dist.sort(key=lambda x: x[0])

        best_distance, best_index = dist[0]
        second_distance, second_index = dist[1]

        if second_distance == 0:
            continue

        if best_distance / second_distance < FEATURE_MATCH_THRESHOLD:
            keypoint_indices1.append(i)
            keypoint_indices2.append(best_index)
            distances.append(best_distance)

    matches.keypoint_indices1 = np.array(keypoint_indices1, dtype=int)
    matches.keypoint_indices2 = np.array(keypoint_indices2, dtype=int)
    matches.distances = np.array(distances, dtype=float)

    return matches

"""
Determine feature matches between all pairs of images in the set
Input:
    features dict[int, FeatureSet], dictionary mapping indices to feature structs
Output:
    all_matches: dict[tuple[int, int], ImagePairMatch], dictionary mapping image pairs to structs describing their matches
"""
def match_all_image_pairs(features: dict[int, FeatureSet]) -> dict[tuple[int, int], ImageMatchPair]:

    all_matches = {}

    # Iterate over all combinations
    for (id1, f1), (id2, f2) in combinations(features.items(), 2):

        all_matches[(id1, id2)] = match_image_pair(f1, f2)
    
    return all_matches

