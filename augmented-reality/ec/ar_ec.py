import time

import numpy as np
import cv2

# Import necessary functions

from loadVid import loadVid

# Q3.2
import numpy as np
import cv2

# Import necessary functions
from planarH import computeH_ransac
from planarH import computeH_norm
from matchPics import matchPics
#from matchPics import matchPics2
from planarH import compositeH
from opts import get_opts
import skimage.color
from helper import briefMatch
from helper import computeBrief_fast
from helper import corner_detection
import imagehash
from PIL import Image


# Write script for Q3.1

def crop_center(source, template):
    t_h, t_w, t_c = template.shape
    s_h, s_w, s_c = source.shape
    # Crop black edges
    source = source[int(t_h // 8 * 0.8):int(t_h // 8 * 5.75), :, :]
    # keep center part
    ratio = t_h / t_w
    s_w_new = int(s_h // ratio)
    center_left = int((s_w - s_w_new) // 2)
    crop_source = source[:, center_left:center_left + s_w_new, :]
    return crop_source


def process_one(opts, locs1, desc1, cv_c, cv_book, source, H, flag):

    source = crop_center(source, cv_c)
    time_match = time.time()

    if flag == True:
        matches, locs2 = matchPics(desc1, cv_book, opts)
        #print("matchPic: "+str(time.time() - time_match))
        # get matched locs
        locs1_new = locs1[matches[:, 0]]
        locs2 = locs2[matches[:, 1]]
        # solve: it is not filling up the same space as the book
        # for two size are different
        h_diff = source.shape[0] / cv_c.shape[0]
        w_diff = source.shape[1] / cv_c.shape[1]
        # x,y swap
        locs2[:, [0, 1]] = locs2[:, [1, 0]]
        locs1_new[:, [0, 1]] = locs1_new[:, [1, 0]]
        locs1_new[:, 1] = locs1_new[:, 1] * h_diff
        locs1_new[:, 0] = locs1_new[:, 0] * w_diff
        # print(locs1.shape, locs2.shape)
        time_rans = time.time()
        H, inliers = computeH_ransac(locs1_new, locs2, opts)
    #H= computeH_norm(locs1_new, locs2)
    #H, mask = cv2.findHomography(locs1, locs2, cv2.RANSAC, 2.0)
    #print("ransac: "+str(time.time()-time_rans))
    # print(H)

    time_comp = time.time()
    composite_img = compositeH(H, source, cv_book)
    #print("composite: "+str(time.time() - time_comp))

    return composite_img,H


def mse(I1, I2):
    err = np.sum((I1.astype("float") - I2.astype("float")) ** 2)
    err /= float(I1.shape[0] * I2.shape[1])
    return err


if __name__ == "__main__":
    # load
    cap1 = cv2.VideoCapture("../data/book.mov")
    cv_c = cv_c = cv2.imread('../data/cv_cover.jpg')
    cap2 = cv2.VideoCapture("../data/ar_source.mov")
    opts = get_opts()

    if (cap1.isOpened() == False) or (cap2.isOpened() == False):
        print("Error opening video stream or file")

    i=0
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0
    # Read until video is completed

    ratio = opts.ratio  # 'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  # 'threshold for corner detection using FAST feature detector'
    cv_c = cv2.resize(cv_c, (175, 220), interpolation=cv2.INTER_AREA)
    I1_gray = skimage.color.rgb2gray(cv_c)
    locs1 = corner_detection(I1_gray, sigma)
    desc1, locs1 = computeBrief_fast(I1_gray, locs1)

    fps_count =0
    H=None
    time_all_process = time.time()
    pre_frame = None
    while (cap1.isOpened()) and (cap2.isOpened()):
        i += 1
        # Capture frame-by-frame
        ret1, book_frame = cap1.read()
        ret2, ar_frame = cap2.read()


        if ret1 and ret2:

            cv_c = cv2.resize(cv_c, (175,220), interpolation=cv2.INTER_AREA)
            book_frame=cv2.resize(book_frame, (320,240), interpolation=cv2.INTER_AREA)
            ar_frame=cv2.resize(ar_frame, (320,180), interpolation=cv2.INTER_AREA)

            if(i>1):
                hash_now=imagehash.average_hash(Image.fromarray(book_frame, 'RGB'))
                hash_pre = imagehash.average_hash(Image.fromarray(pre_frame, 'RGB'))
                diff = np.abs(hash_now - hash_pre)
            #time_all = time.time()
            if(i%6==1 or diff>40):
                composite_img, H = process_one(opts, np.copy(locs1), np.copy(desc1),cv_c, book_frame, ar_frame, H, True)
            else:
                composite_img, H = process_one(opts, np.copy(locs1), np.copy(desc1), cv_c, book_frame, ar_frame, H,
                                               False)
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            fps_count+=fps
            prev_frame_time = new_frame_time
            print("fps: " + str(fps))
            composite_img=cv2.resize(composite_img, (640, 480), interpolation=cv2.INTER_AREA)
            cv2.imshow('Frame', composite_img)
            # Press Q on keyboard to  exit
            temp = cv2.waitKey(1)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
            pre_frame = book_frame

        else:
            break

    # When everything done, release the video capture object
    print("mean fps: "+str(i/(time.time()-time_all_process)))
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

