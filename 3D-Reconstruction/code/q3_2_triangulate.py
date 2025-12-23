import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE
    N = pts1.shape[0]
    #print(pts1)
    P = np.zeros((N,4))
    err_sum =0
    for i in range(N):
        x1 = pts1[i,0]
        y1= pts1[i,1]
        x2 = pts2[i, 0]
        y2 = pts2[i, 1]
        # (1) For every input point, get A:
        # solve matrix [x y 1]^T = C[X Y Z 1]^T
        # A1: (C11 + C12 + C13 + C14)[X Y Z 1] - x(C31 +C31+C33+C34)[X Y Z 1]
        # A1: (C11 - xC31 , C12 - xC32 , C13 - xC33 , C14 - xC34)
        # same for A2
        C1_0= C1[0,:]
        C1_1 = C1[1, :]
        C1_2 = C1[2, :]
        C2_0 = C2[0, :]
        C2_1 = C2[1, :]
        C2_2 = C2[2, :]
        A = np.zeros((4,4))
        A[0,:] = C1_0 - x1*C1_2
        A[1, :] = C1_1 - y1 * C1_2
        A[2, :] = C2_0 - x2 * C2_2
        A[3, :] = C2_1 - y2 * C2_2
        # (2) Solve for the least square solution using np.linalg.svd
        u, s, vh = np.linalg.svd(A)
        X = vh[3, :]
        P[i, :] = X
        # (3) Calculate the reprojection error using the calculated 3D points and C1 & C2
        pts1_pred = C1 @ X
        pts2_pred = C2 @ X
        # (do not forget to convert from homogeneous coordinates to non-homogeneous ones)
        # make two compared pointers similar
        con1 = pts1_pred[2]
        con2 = pts2_pred[2]
        err = np.linalg.norm(pts1_pred/con1 - (x1, y1, 1), 1) ** 2
        err += np.linalg.norm(pts2_pred/con2 - (x2, y2, 1), 1) ** 2
        err_sum += err
    # normalize P
    for i in range(len(P)):
        z = P[i,3]
        P[i,:] = P[i,:]/z

    err = err_sum

    return P, err


'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
def findM2(F, pts1, pts2, intrinsics, filename = 'q3_3.npz'):
    '''
    Q2.2: Function to find camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)
    
    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track 
        of the projection error through best_error and retain the best one. 
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'. 

    '''
    # ----- TODO -----
    # YOUR CODE HERE
    # read with saved data
    E = np.load('q3_1.npz')['arr_0']
    # compute with F
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    E = essentialMatrix(F, K1, K2)
    # get M1 and M2
    M2s = camera2(E)
    M1 = np.eye(3,dtype=float)
    M1 = np.hstack((M1,np.array([[0],[0],[0]])))
    # from looking at camera2, reterive the M2 matrix from 'M2s'.
    _,_,n = M2s.shape
    best_error = float('inf')
    best_M2 =None
    best_C2 = None
    best_P = None
    threshold = 0.1
    for i in range(n):
        M2 = M2s[:,:,i]
        C1 = K1.dot(M1)
        C2 = K2.dot(M2)
        P, err = triangulate(C1, pts1, C2, pts2)
        # update and record best one
        #print(err)
        if err-threshold < best_error:
            best_error =err
            best_M2 = M2
            best_P = P
            best_C2 =C2
    M2 = best_M2
    P = best_P
    C2 = best_C2
    # normalize, or points will be on same plane
    for i in range(len(P)):
        z = P[i,3]
        P[i,:] = P[i,:]/z
    np.savez(filename, M2=M2, C2=C2,P=P)
    return M2, C2, P



if __name__ == "__main__":

    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    print(err)
    assert(err < 500)