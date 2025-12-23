import numpy as np
import cv2

#Import necessary functions
from planarH import computeH_ransac
from matchPics import matchPics
from planarH import compositeH
from helper import loadVid
from opts import get_opts

#Write script for Q3.1

def crop_center(source,template):
    t_h,t_w, t_c = template.shape
    s_h, s_w,s_c = source.shape
    # Crop black edges
    source = source[int(t_h//8*0.8):int(t_h//8*5.75), :, :]
    # keep center part
    ratio = t_h/t_w
    s_w_new = int(s_h//ratio)
    center_left = int((s_w-s_w_new)//2)
    crop_source = source[:,center_left:center_left+s_w_new,:]
    return crop_source


def process_one(opts,cv_c,cv_book,source):
    source = crop_center(source,cv_c)
    matches, locs1, locs2 = matchPics(cv_c, cv_book, opts)
    # get matched locs
    locs1 = locs1[matches[:, 0]]
    locs2 = locs2[matches[:, 1]]
    # solve: it is not filling up the same space as the book
    # for two size are different
    h_diff = source.shape[0] / cv_c.shape[0]
    w_diff = source.shape[1] / cv_c.shape[1]
    # x,y swap
    locs2[:, [0, 1]] = locs2[:, [1, 0]]
    locs1[:, [0, 1]] = locs1[:, [1, 0]]
    locs1[:, 1] = locs1[:, 1] * h_diff
    locs1[:, 0] = locs1[:, 0] * w_diff
    #print(locs1.shape, locs2.shape)
    H, inliers = computeH_ransac(locs1, locs2, opts)
    # print(H)
    composite_img = compositeH(H, source, cv_book)

    # show
    #cv2.imshow("composite img", composite_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return composite_img


if __name__ == "__main__":
    # load
    book_frames=loadVid("../data/book.mov")
    cv_c=cv_c = cv2.imread('../data/cv_cover.jpg')
    ar_frames=loadVid("../data/ar_source.mov")
    # get FPS
    cap = cv2.VideoCapture("../data/ar_source.mov")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    opts = get_opts()
    # make sure same size
    h, w = book_frames[0].shape[:2]
    out = cv2.VideoWriter('result_ar.avi', cv2.VideoWriter_fourcc(*'MJPG'),fps, (w, h))
    length = np.minimum(len(book_frames),len(ar_frames))
    for i in range(length):
        # get img
        composite_img=process_one(opts,cv_c,book_frames[i],ar_frames[i])
        out.write(composite_img)
        print("process: "+str(i+1)+"/"+str(length))
    out.release()
