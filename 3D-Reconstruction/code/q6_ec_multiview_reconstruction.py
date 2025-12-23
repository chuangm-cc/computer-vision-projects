import numpy as np
import matplotlib.pyplot as plt

import os

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors

# Insert your package here

'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''
def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres = 200):
    # TODO: Replace pass by your implementation
    #print(pts1)
    confi1 = pts1[:,2]
    confi2 = pts2[:, 2]
    confi3 = pts3[:, 2]

    N = pts1.shape[0]
    # print(pts1)
    P = np.zeros((N, 4))

    err_sum=0
    for i in range(N):
        # do the threshold
        if confi1[i] >Thres and confi2[i]>Thres and confi3[i]>Thres:
            X,err = triangulate_three(C1, pts1, C2, pts2, C3, pts3,i)
            P[i, :] = X
            err_sum+=err
        elif confi1[i] <Thres:
            X,err = triangulate_two(C3, pts3, C2, pts2, i)
            P[i, :] = X
            err_sum+=err
        elif confi2[i] <Thres:
            X,err = triangulate_two(C3, pts3, C1, pts1, i)
            P[i, :] = X
            err_sum+=err
        else:
            X,err = triangulate_two(C1, pts1, C2, pts2, i)
            P[i, :] = X
            err_sum+=err
    print(err_sum)
    plot_3d_keypoint(P)

    return P

def triangulate_two(C1, pts1, C2, pts2, i):
    x1 = pts1[i, 0]
    y1 = pts1[i, 1]
    x2 = pts2[i, 0]
    y2 = pts2[i, 1]
    # (1) For every input point, get A:
    # solve matrix [x y 1]^T = C[X Y Z 1]^T
    # A1: (C11 + C12 + C13 + C14)[X Y Z 1] - x(C31 +C31+C33+C34)[X Y Z 1]
    # A1: (C11 - xC31 , C12 - xC32 , C13 - xC33 , C14 - xC34)
    # same for A2
    C1_0 = C1[0, :]
    C1_1 = C1[1, :]
    C1_2 = C1[2, :]
    C2_0 = C2[0, :]
    C2_1 = C2[1, :]
    C2_2 = C2[2, :]
    A = np.zeros((4, 4))
    A[0, :] = C1_0 - x1 * C1_2
    A[1, :] = C1_1 - y1 * C1_2
    A[2, :] = C2_0 - x2 * C2_2
    A[3, :] = C2_1 - y2 * C2_2
    # (2) Solve for the least square solution using np.linalg.svd
    u, s, vh = np.linalg.svd(A)
    X = vh[3, :]
    pts1_pred = C1 @ X
    pts2_pred = C2 @ X
    # (do not forget to convert from homogeneous coordinates to non-homogeneous ones)
    # make two compared pointers similar
    con1 = pts1_pred[2]
    con2 = pts2_pred[2]
    err = np.linalg.norm(pts1_pred / con1 - (x1, y1, 1), 1) ** 2
    err += np.linalg.norm(pts2_pred / con2 - (x2, y2, 1), 1) ** 2
    X = X/X[3]
    return X,err

def triangulate_three(C1, pts1, C2, pts2, C3, pts3, i):
    x1 = pts1[i, 0]
    y1 = pts1[i, 1]
    x2 = pts2[i, 0]
    y2 = pts2[i, 1]
    x3 = pts3[i, 0]
    y3 = pts3[i, 1]
    # (1) For every input point, get A:
    # solve matrix [x y 1]^T = C[X Y Z 1]^T
    # A1: (C11 + C12 + C13 + C14)[X Y Z 1] - x(C31 +C31+C33+C34)[X Y Z 1]
    # A1: (C11 - xC31 , C12 - xC32 , C13 - xC33 , C14 - xC34)
    # same for A2
    C1_0 = C1[0, :]
    C1_1 = C1[1, :]
    C1_2 = C1[2, :]
    C2_0 = C2[0, :]
    C2_1 = C2[1, :]
    C2_2 = C2[2, :]
    C3_0 = C3[0, :]
    C3_1 = C3[1, :]
    C3_2 = C3[2, :]
    A = np.zeros((6, 4))
    A[0, :] = C1_0 - x1 * C1_2
    A[1, :] = C1_1 - y1 * C1_2
    A[2, :] = C2_0 - x2 * C2_2
    A[3, :] = C2_1 - y2 * C2_2
    A[4, :] = C3_0 - x3 * C3_2
    A[5, :] = C3_1 - y3 * C3_2
    # (2) Solve for the least square solution using np.linalg.svd
    u, s, vh = np.linalg.svd(A)
    X = vh[3, :]
    pts1_pred = C1 @ X
    pts2_pred = C2 @ X
    # (do not forget to convert from homogeneous coordinates to non-homogeneous ones)
    # make two compared pointers similar
    con1 = pts1_pred[2]
    con2 = pts2_pred[2]
    err = np.linalg.norm(pts1_pred / con1 - (x1, y1, 1), 1) ** 2
    err += np.linalg.norm(pts2_pred / con2 - (x2, y2, 1), 1) ** 2
    X = X / X[3]
    return X, err

'''
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
'''
def plot_3d_keypoint_video(pts_3d_video):
    # TODO: Replace pass by your implementation
    fig = plt.figure()
    N = len(pts_3d_video)
    #print(N)
    ax = fig.add_subplot(111, projection='3d')
    for i in range(N):
        pts_3d = pts_3d_video[i]
        # pts_3d = pts_3d_multi[i]
        for j in range(len(connections_3d)):
            index0, index1 = connections_3d[j]
            xline = [pts_3d[index0, 0], pts_3d[index1, 0]]
            yline = [pts_3d[index0, 1], pts_3d[index1, 1]]
            zline = [pts_3d[index0, 2], pts_3d[index1, 2]]
            ax.plot(xline, yline, zline, color=colors[j])
    np.set_printoptions(threshold=1e6, suppress=True)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


#Extra Credit
if __name__ == "__main__":

    pts_3d_video = []
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join('data/q6/','time'+str(loop)+'.npz')
        image1_path = os.path.join('data/q6/','cam1_time'+str(loop)+'.jpg')
        image2_path = os.path.join('data/q6/','cam2_time'+str(loop)+'.jpg')
        image3_path = os.path.join('data/q6/','cam3_time'+str(loop)+'.jpg')

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data['pts1']
        pts2 = data['pts2']
        pts3 = data['pts3']

        K1 = data['K1']
        K2 = data['K2']
        K3 = data['K3']

        M1 = data['M1']
        M2 = data['M2']
        M3 = data['M3']

        #Note - Press 'Escape' key to exit img preview and loop further 
        #img = visualize_keypoints(im2, pts2)

        # TODO: YOUR CODE HERE
        C1 = K1 @M1
        C2 = K2 @M2
        C3 = K3@M3
        P = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3)
        pts_3d_video.append(P)

    pts_3d_video = np.array(pts_3d_video)
    plot_3d_keypoint_video(pts_3d_video)