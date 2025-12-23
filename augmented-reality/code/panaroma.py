import numpy as np
import cv2

# Import necessary functions
from planarH import computeH_ransac
from matchPics import matchPics
from opts import get_opts
from planarH import computeH
# Q4
def get_move(img,H):
    h, w, c = img.shape
    x=np.arange(w)
    y=np.arange(h)
    xv, yv = np.meshgrid(x, y)
    new_x = xv*H[0][0]+yv*H[0][1]+H[0][2]
    new_y = xv * H[1][0] + yv * H[1][1] + H[1][2]
    return new_x.min(),new_y.min()


def compositeH_pano(H2to1, template, img):
    H2to1_inv = np.linalg.inv(H2to1)
    # Create mask of same size as template
    mask = np.full(template.shape, True).astype(template.dtype)
    # print(img.shape)
    # Warp mask by appropriate homography

    # move with matrix
    l,d=get_move(template,H2to1_inv)
    l=int(-l)-100
    d=int(-d)-100
    move = np.array([[1.0, 0, l], [0, 1, d], [0, 0, 1]])
    H2to1_inv=move@H2to1_inv
    img=cv2.copyMakeBorder(img, 0, d, 0, l, cv2.BORDER_CONSTANT, None, value = 0)
    h, w, c = img.shape
    img=cv2.warpPerspective(img, move, (w, h))

    mask_warp = cv2.warpPerspective(mask, H2to1_inv, (w, h))
    # Warp template by appropriate homography
    template_warp = cv2.warpPerspective(template, H2to1_inv, (w, h))
    # Use mask to combine the warped template and the image
    composite_img = np.zeros(img.shape).astype(img.dtype)
    # print(template_warp.shape)
    for i in range(c):
        for j in range(h):
            for k in range(w):
                if mask_warp[j, k, i] != 0:
                    composite_img[j, k, i] = template_warp[j, k, i]
                else:
                    composite_img[j, k, i] = img[j, k, i]
    return composite_img

left = cv2.imread('../data/d1_l.jpg')
right = cv2.imread('../data/d1_r.jpg')
ratio = left.shape[0]/ left.shape[1] # w/h
left= cv2.resize(left, (640,int(640*ratio)), interpolation = cv2.INTER_AREA)
right= cv2.resize(right, (640,int(640*ratio)), interpolation = cv2.INTER_AREA)
opts = get_opts()
matches, locs1, locs2 = matchPics(left, right, opts)
# get matched locs
locs1 = locs1[matches[:, 0]]
locs2 = locs2[matches[:, 1]]
# x,y swap
locs2[:,[0,1]] = locs2[:,[1,0]]
locs1[:, [0, 1]] = locs1[:, [1, 0]]
H,inliers = computeH_ransac(locs1,locs2,opts)
#print(H)
composite_img = compositeH_pano(H,left,right)

cv2.imwrite("../img/d1_result.jpg", composite_img)
cv2.imshow("composite img", composite_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

