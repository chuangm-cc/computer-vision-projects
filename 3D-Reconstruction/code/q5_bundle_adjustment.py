import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2, triangulate

import scipy

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
'''
def ransacF(pts1, pts2, M, nIters=500, tol=2):
    # TODO: Replace pass by your implementation
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    best_inlier_num = 0
    best_inlier = None
    best_F = None
    best_errs=None
    np.random.seed(1)
    for i in range(nIters):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        # (2) Use the seven point alogrithm
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        # (3) Choose the resulting F that has the most number of inliers
        for F in Fs:
            #print(F)
            # use the calc_epi_error from q1 with threshold to calcualte inliers
            errs = calc_epi_error(pts1_homo, pts2_homo, F)
            #print(errs)
            #print(errs.shape)
            inliers = errs < tol
            num = inliers.sum()
            #print(inliers)
            #print(num)
            if num > best_inlier_num:
                best_inlier_num =num
                best_inlier = inliers
                best_F = F
                best_errs =errs
    F = best_F
    z = F[2,2]
    F = F/z
    #print(best_inlier_num)
    #print(best_errs)
    print(best_inlier_num/len(inliers))
    inliers = best_inlier
    return F,inliers


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # TODO: Replace pass by your implementation
    # make is as 3*1 form
    r = r.reshape((3,1))
    r_mag = np.linalg.norm(r)
    theta = r_mag
    #print(theta)
    if theta == 0:
        return np.eye(3,dtype=float)
    else:
        u = r/theta
        u1,u2,u3 = u[0][0],u[1][0],u[2][0]
        u_x = np.array([[0,-u3,u2],[u3,0,-u1],[-u2,u1,0]])
        R = np.eye(3,dtype=float) * np.cos(theta) + (1-np.cos(theta))* (u@u.T) + u_x * np.sin(theta)
        #print(R)
    return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # TODO: Replace pass by your implementation
    A = (R - R.T)/2
    p = np.array([A[2,1],A[0,2],A[1,0]]).reshape(3,1)
    s=np.linalg.norm(p)
    c = (R[0,0]+R[1,1]+R[2,2]-1)/2
    # case 1
    if s==0 and c==1:
        return [[0],[0],[0]].reshape(1,3)
    # case 2
    if s==0 and c==1:
        R_I = R + np.eye(3,dtype=float)
        for i in range(3):
            column = R_I[:,i]
            # a nonzero column
            if np.any(column):
                v=column
                u = v/np.linalg.norm(v)
                r = u*np.pi
                if (np.linalg.norm(r) == np.pi) and ((r[0,0]==r[0,1] and r[0,0]==0 and r[0,2]<0)
                     or(r[0,0]==0 and r[0,1]<0) or r[0,0]<0):
                    return -r.reshape(1,3)
                else:
                    return r.reshape(1,3)
    # case 3
    u = p/s
    theta = np.arctan2(s,c)
    return (u*theta).reshape(1,3)



'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # TODO: Replace pass by your implementation
    size = len(x)
    t2_size = 3
    r2_size = 3
    # get all values

    P = x[:size-t2_size-r2_size].reshape(-1,4)
    N = P.shape[0]
    #print(P.shape)
    #P = np.hstack((P,np.ones((N,1))))
    P=P.T
    #print(P.T)
    r2 = x[size-t2_size-r2_size:size-t2_size].reshape(3,1)
    t2 = x[size-t2_size:].reshape(3,1)
    # get p1_hat
    p1_hat = K1 @ M1 @ P
    #print(p1_hat)
    # get M2
    R2 = rodrigues(r2)
    M2 = np.hstack((R2,t2))
    p2_hat = K2 @ M2 @ P
    # normalize
    _,N = p1_hat.shape
    #print(N)
    p1_hat = p1_hat.T
    p2_hat = p2_hat.T
    for i in range(N):
        p1_hat[i,:]= p1_hat[i,:]/p1_hat[i,2]
    for i in range(N):
        p2_hat[i,:]= p2_hat[i,:]/p2_hat[i,2]
    #print(p1_hat)
    p1_hat = p1_hat[:,:2]
    p2_hat = p2_hat[:,:2]
    #print(p1_hat)
    residuals = np.concatenate([(p1 - p1_hat).reshape([-1]),(p2 - p2_hat).reshape([-1])])
    return residuals

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    
    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE
    # extract the rotation and translation from M2 init
    R2=M2_init[:,:3]
    t2 = M2_init[:,3]
    r2 = invRodrigues(R2)
    x = np.concatenate([P_init.reshape([-1]),r2.reshape([-1]),
                              t2.reshape([-1])])
    #  minimize the objective function, rodriguesResidual
    #rodriguesResidual(K1, M1, p1, K2, p2, x)
    fun = lambda x: np.linalg.norm(rodriguesResidual(K1, M1, p1, K2, p2, x))
    res = scipy.optimize.minimize(fun,x,method='SLSQP')
    # decompose it back to rotation using
    x = res.x
    size = len(x)
    t2_size = 3
    r2_size = 3
    # get all values
    P = x[:size - t2_size - r2_size].reshape(-1,4)
    r2 = x[size - t2_size - r2_size:size - t2_size].reshape(3, 1)
    t2 = x[size - t2_size:].reshape(3, 1)
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))
    #normalize
    for i in range(len(P)):
        z = P[i,3]
        P[i,:] = P[i,:]/z
    return M2, P, obj_start, obj_end



if __name__ == "__main__":
              
    np.random.seed(1) #Added for testing, can be commented out

    some_corresp_noisy = np.load('data/some_corresp_noisy.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    #using ransacF
    #F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    # using eightpoint
    # F = eightpoint(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    # displayEpipolarF(im1, im2, F)
    #
    # # Simple Tests to verify your implementation:
    # pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)
    #
    # assert(F.shape == (3, 3))
    # assert(F[2, 2] == 1)
    # assert(np.linalg.matrix_rank(F) == 2)
    
    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot
    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())
    assert(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    assert(np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)

    # Visualization:
    np.random.seed(1)
    correspondence = np.load('data/some_corresp_noisy.npz') # Loading noisy correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')
    M=np.max([*im1.shape, *im2.shape])

    # correspondence = np.load('data/some_corresp.npz')  # Loading correspondences
    # pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    # TODO: YOUR CODE HERE
    '''
    Call the ransacF function to find the fundamental matrix
    Call the findM2 function to find the extrinsics of the second camera
    Call the bundleAdjustment function to optimize the extrinsics and 3D points
    Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    '''
    # Call the ransacF function to find the fundamental matrix
    F, inliers = ransacF(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    print(len(inliers))
    pts1= pts1[inliers]
    pts2 = pts2[inliers]
    print(len(pts1))
    # Call the findM2 function to find the extrinsics of the second camera
    M2, C2, P = findM2(F, pts1, pts2, intrinsics)
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    print("before:"+str(err))
    #plot_3D_dual(P, P)
    #print(P)
    # Call the bundleAdjustment function to optimize the extrinsics and 3D points
    M1 = np.eye(3, dtype=float)
    M1 = np.hstack((M1, np.array([[0], [0], [0]])))
    M2, P_new, obj_start, obj_end = bundleAdjustment(K1, M1, pts1, K2, M2, pts2, np.copy(P))

    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    print("after:" + str(err))
    #print(P_new)
    # Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    print(len(P))
    plot_3D_dual(P,P_new)