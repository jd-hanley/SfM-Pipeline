import numpy as np

"""
Compute the essential matrix from the fundamental matrix and camera intrinsic matrix
Input:
    F: np.ndarray, fundamental matrix relating corresponding points between two views in normalized image coordinates
    K: np.ndarray, intrinsic camera matrix describing the geometry of the camera
Output:
    E: np.ndarray, essential matrix with all constraints enforced
"""
def estimate_essential_matrix(F: np.ndarray, K: np.ndarray) -> np.ndarray:

    K_T = np.transpose(K)

    E_initial = K_T @ F @ K

    # Decompose to enforce essential matrix constraints
    U, S, Vt = np.linalg.svd(E_initial)

    # Constraint: Enforce two singular values to be 1 and the third to be 0
    S[0] = 1
    S[1] = 1
    S[2] = 0

    E_corrected = U @ np.diag(S) @ Vt

    return E_corrected

"""
Adjusts the rotation matrix R and the translation vector C if the determinant of R is negative
Ensures that R represents a valid rotation matrix with a determinant of +1
Input:
    R: np.ndarray, rotation matrix
    C: np.ndarray, translation vector
Output:
    R: np.ndarray, rotation matrix adjusted 
    C: np.ndarray, translation vector adjusted
"""
def check_det(R: np.ndarray, C: np.ndarray):

    if np.linalg.det(R) < 0:
        R = -R
        C = -C
    
    return R, C

"""
Extract four possible camera poses (rotation and translation pairs) from E using the SVD
Input:
    E: np.ndarray, essential matrix as previously estimated
Output:
    candidates: list[tuple[np.ndarray, np.ndarray]], rotation/translation candidate combinations
"""
def decompose_essential_matrix(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:

    # Perform the SVD of E
    U, D, Vt = np.linalg.svd(E)

    # Define the magical W matrix
    W = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Build the possible translation vectors and rotation matrices
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    t = U[:, 2]

    candidates = [
        (R1,  t),
        (R1, -t),
        (R2,  t),
        (R2, -t),
    ]

    fixed_candidates = []

    # Ensure the rotation matrices are valid
    for R, t in candidates:
        if np.linalg.det(R) < 0:
            R = -R
            t = -t

        fixed_candidates.append((R, t))

    return fixed_candidates


"""
Select the pose with most inliers based on the positive depth criterion (points in front of the camera)
Input:
    candidates: list[tuple[np.ndarray, np.ndarray]], rotation/translation candidate combinations
    xset: list[np.ndarray], list of sets of 3D points for each candidate camera pose
Output:
    t: np.ndarray, the correct camera translation vector
    R: np.ndarray, the correct rotation matrix
    x: np.ndarray, the set of 3D points corresponding to the correct camera pose
    index: int, the index of the correct pose in the input lists
"""
def disambiguate_camera_pose(candidates, xset):

    # List to store the points with positive depth for each candidate pose
    count_list = []

    max_count = 0
    max_index = -1

    for (R, t), curr_xset in zip(candidates, xset):

        # Extract the third row of the rotation matrix (z-axis direction)
        r3 = R[2,:]

        # Number of points with positive depth 
        count = 0

        # For each 3D point in the current candidate pose
        for x in curr_xset:
            # Transpose the 3D point to align
            X = x.reshape(3, 1)

            # Check if the point is in front of both cameras
            # Depth in camera 0/world frame
            depth0 = X[2, 0]

            # Transform point into camera 1 frame
            X_cam1 = R @ X + t
            depth1 = X_cam1[2, 0]

            if depth0 > 0 and depth1 > 0:
                count += 1

        count_list.append(count)
        if count > max_count:
            max_count = count
            max_index = len(count_list) - 1
    
    R, t = candidates[max_index]
    x = xset[max_index]

    return R, t, x, max_index
         




    