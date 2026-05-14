import numpy as np

"""
Compute 3D points from two sets of 2D correspondences observed from two different camera poses
Inputs:
    K: np.ndarray, intrinsic camera matrix
    t0: np.ndarray, translation vector for the first camera pose
    R0: np.ndarray, rotation matrix for the first camera pose
    t1: np.ndarray, translation vector for the second camera pose
    R1: np.ndarray, rotation matrix for the second camera pose
    pts0: pts1: np.ndarray, feature coordinates in image 0
    pts1: pts1: np.ndarray, feature coordinates in image 1
Output:
    xset: np.ndarray, array of 3D points in Euclidean coordinates, shape (N, 3)
"""
def linear_triangulation(K, t0, R0, t1, R1, pts0, pts1):

    xset = []

    t0 = t0.reshape(3, 1)
    t1 = t1.reshape(3, 1)

    # Construct the projection matrices P1 and P2 with the form P = K[R|t]
    P0 = K @ np.hstack((R0, t0))
    P1 = K @ np.hstack((R1, t1))

    for pt0, pt1 in zip(pts0, pts1):

        u0 = pt0[0]
        v0 = pt0[1]

        u1 = pt1[0]
        v1 = pt1[1]

        # Construct matrix A for the linear triangulation system AX = 0.
        #
        # For a 3D homogeneous point X and camera projection matrix P,
        # the observed homogeneous image point x satisfies:
        #
        #     x ~ P X
        #
        # Since x and P X are proportional homogeneous vectors, their cross product is zero:
        #
        #     x × (P X) = 0
        #
        # Expanding this cross product gives two independent linear equations per image:
        #
        #     (u * P[2, :] - P[0, :]) X = 0
        #     (v * P[2, :] - P[1, :]) X = 0
        #
        # Stacking these equations from both cameras gives A X = 0.

        # Rows of P0
        p0_0 = P0[0,:]
        p0_1 = P0[1,:]
        p0_2 = P0[2,:]

        # Rows of P1
        p1_0 = P1[0,:]
        p1_1 = P1[1,:]
        p1_2 = P1[2,:]

        # Build the 4 x 4 A matrix
        A = np.zeros((4,4))
        A[0,:] = u0 * p0_2 - p0_0
        A[1,:] = v0 * p0_2 - p0_1
        A[2,:] = u1 * p1_2 - p1_0
        A[3,:] = v1 * p1_2 - p1_1

        # Solve via SVD
        U, S, Vt = np.linalg.svd(A)

        # Solution is the last row of Vt
        x_h = Vt[-1,:]

        if abs(x_h[-1]) > 1e-12:
            x_h = x_h / x_h[-1]

        xset.append([x_h[0], x_h[1], x_h[2]])

    return np.array(xset)



