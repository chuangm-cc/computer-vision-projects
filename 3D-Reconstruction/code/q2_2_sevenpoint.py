import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize

# Insert your package here
from sympy import symbols, solve, Matrix
import numpy as np

'''
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Solving this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
'''
def sevenpoint(pts1, pts2, M):

    Farray = []
    # ----- TODO -----
    # YOUR CODE HERE
    #raise NotImplementedError()
    # (1) Normalize the input pts1 and pts2 using the matrix T.(dividing each coordinate by M)
    T = np.array([[1 / M, 0], [0, 1 / M]])
    pts1_norm = pts1 @ T
    pts2_norm = pts2 @ T
    N = pts1.shape[0]
    # print(N)
    # print(pts1_norm.shape)
    # (2) Setup the eight point algorithm's equation.
    A = np.zeros((N, 9))
    for i in range(N):
        x1 = pts1_norm[i, 0]
        y1 = pts1_norm[i, 1]
        x2 = pts2_norm[i, 0]
        y2 = pts2_norm[i, 1]
        # x'->x1, x->x2
        A[i, :] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
    # (3) Solve for the least square solution using SVD.
    u, s, vh = np.linalg.svd(A)
    # (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    f1 = vh.T[:,8]
    f2 = vh.T[:,7]
    # from lecture, need to do transpose as .T
    f1 = f1.reshape((3,3))
    f2 = f2.reshape((3, 3))
    # (5) Use the singularity constraint to solve for the cubic polynomial equation
    a = symbols('a')
    # equation
    F = a*f1 + (1-a)*f2
    # to use det()
    F_mat = Matrix(F)
    d = F_mat.det()
    # get parameters
    #print(d)
    result = str(d)
    result = result.split()
    # get all parameters by splite
    a3 = float(result[0].split('*')[0])
    if(result[1]=='-'):
        a2 = -float(result[2].split('*')[0])
    else:
        a2 = float(result[2].split('*')[0])
    if (result[3] == '-'):
        a1 = -float(result[4].split('*')[0])
    else:
        a1 = float(result[4].split('*')[0])
    if (result[5] == '-'):
        a0 = -float(result[6].split('*')[0])
    else:
        a0 = float(result[6].split('*')[0])
    #print(a2)
    s=(a0,a1,a2,a3)
    #print(s)
    # get roots
    roots = np.polynomial.polynomial.polyroots(s)
    #print(roots)

    # (6) Unscale the fundamental matrixes and return as Farray
    T = np.array([[1 / M, 0, 0], [0, 1 / M, 0], [0, 0, 1]])
    for i in range(len(roots)):
        # equation
        if(roots[i].imag!=0):
            continue
        F=roots[i].real*f1 +(1-roots[i].real)*f2
        # Use the function `_singularize` (provided) to enforce the singularity condition.
        # F = _singularize(F)
        #(6) Unscale the fundamental matrixes and return as Farray
        F = T.T @ F @ T
        # for assert F[2,2]=1
        z = F[2,2]
        F =F/z
        Farray.append(F)

    return Farray



if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    #print(Farray)

    F = Farray[0]

    print(F)

    np.savez('q2_2.npz', F, M)

    # fundamental matrix must have rank 2!
    assert(np.linalg.matrix_rank(F) == 2)
    #print(F)
    #displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution. 
    np.random.seed(1) #Added for testing, can be commented out
    
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M=np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo,pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))
            
    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    #print(F)
    print("Error:", ress[min_idx])

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)
    #assert(2==1)
    #displayEpipolarF(im1, im2, F)