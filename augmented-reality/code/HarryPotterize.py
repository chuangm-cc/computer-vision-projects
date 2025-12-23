import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

# Import necessary functions
from planarH import computeH_ransac
from matchPics import matchPics
from planarH import compositeH
# Q2.2.4

def warpImage(opts):
    cv_c = cv2.imread('../data/cv_cover.jpg')
    cv_d = cv2.imread('../data/cv_desk.png')
    hp_c = cv2.imread('../data/hp_cover.jpg')
    matches, locs1, locs2 = matchPics(cv_c, cv_d, opts)
    # get matched locs
    locs1 = locs1[matches[:, 0]]
    locs2 = locs2[matches[:, 1]]
    # solve: it is not filling up the same space as the book
    # for two size are different
    h_diff = hp_c.shape[0]/cv_c.shape[0]
    w_diff = hp_c.shape[1] / cv_c.shape[1]
    # x,y swap
    locs2[:,[0,1]] = locs2[:,[1,0]]
    locs1[:, [0, 1]] = locs1[:, [1, 0]]
    locs1[:,1] = locs1[:,1]*h_diff
    locs1[:, 0] = locs1[:, 0] * w_diff
    H,inliers = computeH_ransac(locs1,locs2,opts)
    #print(H)
    composite_img = compositeH(H,hp_c,cv_d)
    # show
    cv2.imshow("composite img", composite_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


