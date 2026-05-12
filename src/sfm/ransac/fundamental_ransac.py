import numpy as np

from geometry.fundamental import estimate_fundamental_matrix

from config import F_RANSAC_ITER, F_RANSAC_THRESHOLD

"""
Estimates the Fundamental matrix using RANSAC, identifies inliers and rejects outliers
Input:
    pts1: np.ndarray, all feature point coordinates in the first image
    pts2: np.ndarray, all feature point coordinates in the second image
Output:
    inlier_mask: np.ndarray, inlier point indices
    F_best: best estimate of the fundamental matrix
"""
def estimate_fundamental_matrix_ransac(pts1: np.ndarray, pts2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    best_F = None
    best_inlier_count = 0
    best_inlier_mask = None

    n = len(pts1)

    for i in range(F_RANSAC_ITER):
        # Randomly select 8 pairs of points for the 8 point algorithm
        rand_indices = np.random.choice(n, size=8, replace=False)

        pts1_sample = pts1[rand_indices]
        pts2_sample = pts2[rand_indices]

        est_F = estimate_fundamental_matrix(pts1_sample, pts2_sample)

        inliers = []

        # Check every point pair against the epipolar constraint (should be zero)
        for j in range(n):
            pt1_homo = np.array([pts1[j, 0], pts1[j, 1], 1.0])
            pt2_homo = np.array([pts2[j, 0], pts2[j, 1], 1.0])

            error = np.abs(pt1_homo.T @ est_F @ pt2_homo)

            if error < F_RANSAC_THRESHOLD:
                inliers.append(j)
        
        num_inliers = len(inliers)
        if num_inliers > best_inlier_count:
            best_inlier_count = num_inliers
            best_inlier_mask = inliers
            best_F = est_F
    
    # Recompute the best F using all of the inliers
    inliers1 = pts1[best_inlier_mask]
    inliers2 = pts2[best_inlier_mask]
    F_best = estimate_fundamental_matrix(inliers1, inliers2)

    if abs(F_best[2,2]) > 1e-8:
        F_best = F_best / F_best[2,2]
    
    return F_best, best_inlier_mask

