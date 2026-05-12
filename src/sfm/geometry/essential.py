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