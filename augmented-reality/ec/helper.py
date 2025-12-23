import time

import numpy as np
import cv2
import scipy.io as sio
from matplotlib import pyplot as plt
import skimage.feature

PATCHWIDTH = 9

def briefMatch(desc1,desc2,ratio):

    matches = skimage.feature.match_descriptors(desc1,desc2,'hamming',cross_check=True,max_ratio=ratio)
    return matches



def plotMatches(im1,im2,matches,locs1,locs2):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.axis('off')
    skimage.feature.plot_matches(ax,im1,im2,locs1,locs2,matches,matches_color='r',only_matches=True)
    plt.show()
    return



def makeTestPattern_fast(patchWidth, nbits):

    half = patchWidth/2
    np.random.seed(0)
    compareX_row = (np.random.random((nbits,1))*patchWidth-half).astype(int)
    np.random.seed(1)
    compareX_col = (np.random.random((nbits,1))*patchWidth-half).astype(int)
    np.random.seed(2)
    compareY_row = (np.random.random((nbits, 1)) * patchWidth - half).astype(int)
    np.random.seed(3)
    compareY_col = (np.random.random((nbits, 1)) * patchWidth - half).astype(int)

    return (compareX_row, compareX_col, compareY_col, compareY_row)



def computePixel_fast(img, idx1, idx2,idex3,idex4, width, center):

    return 1 if img[int(center[0]+idx1)][int(center[1]+idx2)] < img[int(center[0]+idex3)][int(center[1]+idex4)] else 0


    return desc, locs

def computeBrief_fast(img, locs):

    patchWidth = 9
    nbits = 68
    compareX_row, compareX_col, compareY_col, compareY_row = makeTestPattern_fast(patchWidth,nbits)
    #print(compareX_row.shape)
    m, n = img.shape

    halfWidth = patchWidth//2
    time_loc = time.time()
    locs = np.array(list(filter(lambda x: halfWidth <= x[0] < m-halfWidth and halfWidth <= x[1] < n-halfWidth, locs)))

    time_desc = time.time()
    desc=[]
    for c in locs:
        desc_one=[]
        for i in range(len(compareX_row)):
            desc_one.append(computePixel_fast(img, compareX_row[i][0], compareX_col[i][0],
                                              compareY_col[i][0], compareY_row[i][0], patchWidth, c))
        desc.append(desc_one)
    desc=np.array(desc)
    return desc, locs


def corner_detection(img, sigma):

    # fast method
    result_img = skimage.feature.corner_fast(img, n=PATCHWIDTH, threshold=sigma)
    locs = skimage.feature.corner_peaks(result_img, min_distance=1)
    return locs


def loadVid(path):

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    cap = cv2.VideoCapture(path)

    # Append frames to list
    frames = []

    # Check if camera opened successfully
    if cap.isOpened()== False:
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            #Store the resulting frame
            frames.append(frame)
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    frames = np.stack(frames)

    return frames
