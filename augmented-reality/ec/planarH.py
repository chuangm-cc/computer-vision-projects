import numpy as np
import cv2


def computeH(x1, x2):
    #Q2.2.1
    #Compute the homography between two sets of points
    N = x1.shape[0]
    # A:2N * 9
    A = np.zeros((N*2, 9))
    for i in range(N):
        px1 = x1[i, 0]
        py1 = x1[i, 1]
        px2 = x2[i, 0]
        py2 = x2[i, 1]
        A[i*2, :] = np.array([px2, py2, 1, 0, 0, 0, -px1*px2, -px1*py2, -px1])
        A[i*2+1, :] = np.array([0, 0, 0, px2, py2, 1, -px2*py1, -py2*py1, -py1])
    u, s, vh = np.linalg.svd(A)
    # s in descending order, so choose the smallest eigenvalue of ATA(in column 9 of the matrix V)
    #print(vh.shape)
    H2to1 = vh[8, :]
    H2to1 = H2to1.reshape((3, 3))

    return H2to1


def computeH_norm(x1, x2):
    #Q2.2.2
    x1=x1.astype(float)
    x2 = x2.astype(float)
    #Compute the centroid of the points
    # for x1 moving transform
    p1_center_x = np.mean(x1[:, 0])
    p1_center_y = np.mean(x1[:, 1])
    # for x2
    p2_center_x = np.mean(x2[:, 0])
    p2_center_y = np.mean(x2[:, 1])
    #Shift the origin of the points to the centroid
    # for x1 moving transform
    for i in range(x1.shape[0]):
        x1[i, 0] = x1[i, 0] - p1_center_x
        x1[i, 1] = x1[i, 1] - p1_center_y
    # for x2
    for i in range(x2.shape[0]):
        x2[i, 0] = x2[i, 0] - p2_center_x
        x2[i, 1] = x2[i, 1] - p2_center_y
    #print(x2)
    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    sqrt2 = np.sqrt(2)
    p1_max_dis=0
    p2_max_dis = 0
    for i in range(x1.shape[0]):
        dis_p1 = np.linalg.norm(x1[i, :])
        p1_max_dis = np.maximum(p1_max_dis, dis_p1)
    for i in range(x2.shape[0]):
        dis_p2 = np.linalg.norm(x2[i, :])
        p2_max_dis = np.maximum(p2_max_dis, dis_p2)
    #Similarity transform 1
    # for have done moving, so use matrix for scale
    T1 = np.zeros((3,3))
    T1[0,0] = sqrt2/p1_max_dis
    T1[1, 1] = sqrt2 / p1_max_dis
    T1[2, 2] = 1
    new_col = np.ones((x1.shape[0], 1))
    new_x1 = np.hstack((x1, new_col))
    for i in range(x1.shape[0]):
        new_loc1 = (T1@new_x1[i, 0:3].T)
        x1[i, :] = new_loc1[0:2]
        #print(x1[i,:])
    #Similarity transform 2
    # matrix for scale
    T2 = np.zeros((3, 3))
    T2[0, 0] = sqrt2 / p2_max_dis
    T2[1, 1] = sqrt2 / p2_max_dis
    T2[2, 2] = 1
    new_col = np.ones((x2.shape[0], 1))
    new_x2 = np.hstack((x2, new_col))
    for i in range(x2.shape[0]):
        new_loc2 = (T2 @ new_x2[i, 0:3].T)
        x2[i, :] = new_loc2[0:2]
    #Compute homography
    H2to1 = computeH(x1,x2)
    #print(H2to1)
    #Denormalization
    # moving matrix @ scaling matrix for x1
    # moving matrix
    T1_cen = np.zeros((3, 3))
    T1_cen[0, 0] = 1
    T1_cen[1, 1] = 1
    T1_cen[2, 2] = 1
    T1_cen[0, 2] = -p1_center_x
    T1_cen[1, 2] = -p1_center_y
    T1 = T1 @ T1_cen
    # moving matrix @ scaling matrix for x2
    T2_cen = np.zeros((3, 3))
    T2_cen[0, 0] = 1
    T2_cen[1, 1] = 1
    T2_cen[2, 2] = 1
    T2_cen[0, 2] = -p2_center_x
    T2_cen[1, 2] = -p2_center_y
    T2 = T2 @ T2_cen
    H2to1 = np.linalg.inv(T1)@H2to1@T2

    return H2to1


def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    times = 0
    n_choice = 4
    N = locs2.shape[0]

    bestH2to1 = None
    best_inliers = None
    best_in_num = 0
    np.random.seed(2020)
    if N <= n_choice:
        max_iters = 1
    while times < max_iters:
        # Draw n points uniformly at random (from left image)
        # use length for a must be 1-D
        if N>n_choice:
            random_indexs=np.random.choice(N, n_choice, replace=False)
        else:
            random_indexs = np.arange(N)
        #print(random_indexs)
        random_x1 = locs1[random_indexs]
        #print(random_x1)
        # Fit (homography) warp to these n points and their correspondences
        random_x2 = locs2[random_indexs]
        H = computeH_norm(random_x1,random_x2)
        # Find inliers among the remaining left-image points(warped positions land close to right-image)
        new_col = np.ones((N, 1))
        new_loc2 = np.hstack((locs2, new_col))
        inlier_num = 0
        inliers = np.zeros((N))
        for i in range(N):
            warp_x1 = H @ new_loc2[i,0:3].T
            # divide by z ro get x,y
            z = warp_x1[-1]
            if z!=0:
                warp_x1_new = warp_x1[0:2]/z
            diff = warp_x1_new - locs1[i,:]
            dis = np.linalg.norm(diff)
            #print(dis)
            if dis < inlier_tol:
                inlier_num +=1
                inliers[i] = 1
            else:
                inliers[i]=0
        #print(H)
        # largest inlier set
        if inlier_num > best_in_num:
            bestH2to1 = H
            best_in_num = inlier_num
            best_inliers=inliers
        times += 1
    inliers=best_inliers
    return bestH2to1, inliers



def compositeH(H2to1, template, img):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    H2to1_inv = np.linalg.inv(H2to1)
    #H2to1_inv=H2to1


    #Create mask of same size as template
    mask = np.full(template.shape,True).astype(template.dtype)
    #print(img.shape)
    #Warp mask by appropriate homography
    h,w,c = img.shape
    mask_warp=cv2.warpPerspective(mask,H2to1_inv,(w,h))
    #Warp template by appropriate homography
    template_warp = cv2.warpPerspective(template, H2to1_inv,(w,h))
    #Use mask to combine the warped template and the image
    composite_img = img
    composite_img[mask_warp>0]=template_warp[mask_warp>0]
    #print(template_warp.shape)

    return composite_img


