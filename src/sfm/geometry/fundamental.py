import numpy as np


""" 
Estimates the Fundamental Matrix between two sets of points using the 8-point algorithm
Input:
    pts1: np.ndarray, feature coordinates in image 1
    pts2: np.ndarray, feature coordinates in image 2
Output:
    F: np.ndarray, estimated fundamental matrix
"""
def estimate_fundamental_matrix(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:

    num_pts = len(pts1)

    # Step 1: Normalization
    #   Compute centroids of the points in each image
    #   Shift points to have zero mean by subtracting the centroid
    #   Compute scaling factors to make the mean distance of points from the origin equal to sqrt(2)
    #   Construct the normalization matrices for both images

    # Compute the centroids
    pts1_cent_x, pts1_cent_y = np.mean(pts1, axis=0)
    pts2_cent_x, pts2_cent_y = np.mean(pts2, axis=0)

    # Shift points by centroid
    shifted1 = pts1 - np.array([pts1_cent_x, pts1_cent_y])
    shifted2 = pts2 - np.array([pts2_cent_x, pts2_cent_y])

    # Compute the current mean distance from the origin of all points
    pts1_mean_dist = np.mean(np.linalg.norm(shifted1, axis=1))
    pts2_mean_dist = np.mean(np.linalg.norm(shifted2, axis=1))

    # Compute the scaling factor to make the mean distance equal to sqrt(2)
    scale_factor_pts1 = np.sqrt(2.0) / pts1_mean_dist
    scale_factor_pts2 = np.sqrt(2.0) / pts2_mean_dist

    # Build the normalization matrices that apply the shift and the scaling factors
    T1 = np.array([
        [scale_factor_pts1, 0, -scale_factor_pts1 * pts1_cent_x],
        [0, scale_factor_pts1, -scale_factor_pts1 * pts1_cent_y],
        [0,0,1]
    ], dtype=float)

    T2 = np.array([
        [scale_factor_pts2, 0, -scale_factor_pts2 * pts2_cent_x],
        [0, scale_factor_pts2, -scale_factor_pts2 * pts1_cent_y],
        [0,0,1]
    ], dtype=float)

    # Step 2: Construct the linear system for estimating F
    #   Set up the system Af = 0 where f is the vector form of the fundamental matrix
    #   Initialize the first row of A using the first pair of points
    #       Normalize the source point using the transformation matrix
    #       Normalize the target point using the transformation matrix
    #       Construct the first row of A using the epipolar constraint 
    #   Repeat for all other rows to build the full A matrix

    A = np.zeros((num_pts, 9))

    for i in range(num_pts):

        # Build the homogeneous point, apply transformation, normalize for the first image
        p1 = np.array([pts1[i, 0], pts1[i, 1], 1.0])
        p1_normalized = T1 @ p1
        _x1 = p1_normalized[0] / p1_normalized[2]
        _y1 = p1_normalized[1] / p1_normalized[2]

        # Same for the second image
        p2 = np.array([pts2[i, 0], pts2[i, 1], 1.0])
        p2_normalized = T1 @ p2
        _x2 = p2_normalized[0] / p2_normalized[2]
        _y2 = p2_normalized[1] / p2_normalized[2]

        # Add the row to the matrix
        A[i] = [_x2*_x1, _x2*_y1, _x2, _y2*_x1, _y2*_y1, _y2, _x1, _y1, 1]
    
    # Step 3: Solve the linear system using the SVD
    #   Take the SVD of A
    #   Use the right singular vector which corresponds to the smallest singular value
    #   Reshape into a matrix
    u, s, vh = np.linalg.svd(A)
    f_initial = vh[-1,:]
    F_initial = f_initial.reshape(3,3)

    # Step 4: Enforce the rank 2 constraint on F 
    #   Perform an SVD on F_initial
    #   Manually set the smallest singular value to 0 to enforce rank 2 constraint
    #   Reconstruct the matrix
    U, S, Vt = np.linalg.svd(F_initial)
    S[2] = 0
    F_tilde = U @ np.diag(S) @ Vt

    # Step 5: Scale, ensure the last element of F is one for consistency
    if abs(F_tilde[2,2] > 1e-8):
        F_tilde = F_tilde / F_tilde[2,2]
    
    # Denormalize
    F = T2.T @ F_tilde @ T1
    return

"""
Helper function to obtain the lists of matches feature points from a match struct and the feature structs
Input:
    match: ImageMatchPair, indices of matches and the distances between descriptors
    features1: FeatureSet, information about keypoints and their descriptors from image 1
    features2: FeatureSet, information about keypoints and their descriptors from image 2
Output:
    pts1: np.ndarray, feature locations in image1
    pts2: np.ndarray, feature locations in image2
"""
def get_matched_points(match, features1, features2):
    pts1 = features1.points(match.keypoint_indices)
    pts2 = features2.points(match.keypoint_indices)
    return pts1, pts2
