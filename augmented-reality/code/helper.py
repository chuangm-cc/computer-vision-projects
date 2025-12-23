import math

import numpy as np
import cv2
import scipy.io as sio
from matplotlib import pyplot as plt
import skimage.feature
from scipy.ndimage import gaussian_filter

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



def makeTestPattern(patchWidth, nbits):

    np.random.seed(0)
    compareX = patchWidth*patchWidth * np.random.random((nbits,1))
    compareX = np.floor(compareX).astype(int)
    np.random.seed(1)
    compareY = patchWidth*patchWidth * np.random.random((nbits,1))
    compareY = np.floor(compareY).astype(int)

    return (compareX, compareY)


def computePixel(img, idx1, idx2, width, center):

    halfWidth = width // 2
    col1 = idx1 % width - halfWidth
    row1 = idx1 // width - halfWidth
    col2 = idx2 % width - halfWidth
    row2 = idx2 // width - halfWidth
    return 1 if img[int(center[0]+row1)][int(center[1]+col1)] < img[int(center[0]+row2)][int(center[1]+col2)] else 0


def computeBrief(img, locs):

    patchWidth = 9
    nbits = 256
    compareX, compareY = makeTestPattern(patchWidth,nbits)
    m, n = img.shape

    halfWidth = patchWidth//2

    locs = np.array(list(filter(lambda x: halfWidth <= x[0] < m-halfWidth and halfWidth <= x[1] < n-halfWidth, locs)))
    desc = np.array([list(map(lambda x: computePixel(img, x[0], x[1], patchWidth, c), zip(compareX, compareY))) for c in locs])

    return desc, locs

def computePixel_scale(img, idx1, idx2, width, center,scale):
    halfWidth = width // 2
    col1 = idx1 % width - halfWidth
    row1 = idx1 // width - halfWidth
    col2 = idx2 % width - halfWidth
    row2 = idx2 // width - halfWidth
    p1x=int(center[0]+row1*scale)
    p1y=int(int(center[1]+col1*scale))
    p2x=int(center[0]+row2*scale)
    p2y=int(center[1]+col2*scale)

    h, w = img.shape
    p1x = min(p1x, h - 1)
    p2x = min(p2x, h - 1)
    p1y= min(p1y, w - 1)
    p2y = min(p2y, w - 1)
    return 1 if img[p1x][p1y] < img[p2x][p2y] else 0


def computeBrief_scale(img, locs):
    patchWidth = 9
    nbits = 256
    compareX, compareY = makeTestPattern(patchWidth, nbits)
    m, n = img.shape
    ori_locs = np.copy(locs)
    #print(locs.shape)

    halfWidth = patchWidth // 2

    locs = np.array(
        list(filter(lambda x: halfWidth <= x[0] < m - halfWidth and halfWidth <= x[1] < n - halfWidth, locs)))

    # set layer number
    img= gaussian_filter(img, sigma=1.6)
    layer_num = 4

    scale_num = 1/layer_num
    scale = scale_num
    desc=np.array([list(map(lambda x: computePixel_scale(img, x[0], x[1], patchWidth, c,scale), zip(compareX, compareY))) for c in locs])
    for i in range(layer_num):
        scale += scale_num
        desc_one =np.array([list(map(lambda x: computePixel_scale(img, x[0], x[1], patchWidth, c,scale), zip(compareX, compareY))) for c in ori_locs])
        #print(desc_one.shape)
        desc=np.vstack((desc,desc_one))
        locs = np.vstack((locs, ori_locs))
    #print(desc.shape)
    #print(locs.shape)

    return desc, locs


def computeBrief_rotation1(img, locs):

    patchWidth = 9
    nbits = 256
    compareX, compareY = makeTestPattern(patchWidth,nbits)
    m, n = img.shape

    halfWidth = patchWidth//2

    locs = np.array(list(filter(lambda x: halfWidth <= x[0] < m-halfWidth and halfWidth <= x[1] < n-halfWidth, locs)))
    desc = np.array([list(map(lambda x: computePixel_rotation1(img, x[0], x[1], patchWidth, c), zip(compareX, compareY))) for c in locs])

    #print(desc.shape)
    #desc = np.array(desc.sum(axis=1))
    #print(desc.shape[1])
    return desc, locs


def rotate(pre, center, radiant):
    cx, cy = center
    px, py = pre

    qx = cx + math.cos(radiant) * (px - cx) - math.sin(radiant) * (py - cy)
    qy = cy + math.sin(radiant) * (px - cx) + math.cos(radiant) * (py - cy)
    return int(qx), int(qy)

def computePixel_rotation1(img, idx1, idx2, width, center):

    h,w=img.shape
    half = width//2
    brightest =0
    b_i=0
    b_j=0
    # get brightest pointer
    for i in range(center[0] - half,center[0]+half):
        for j in range(center[1]-half,center[1]+half):
            value=img[i,j]
            if value>brightest:
                # record
                b_i=i
                b_j=j
                brightest=value
    # get radians
    radians = math.atan2(b_i - center[0], b_j - center[1])
    #print(radians)
    #print(brightest)

    halfWidth = width // 2
    col1 = idx1 % width - halfWidth
    row1 = idx1 // width - halfWidth
    col2 = idx2 % width - halfWidth
    row2 = idx2 // width - halfWidth

    pointer1_0=int(center[0]+row1)
    pointer1_1 = int(center[1]+col1)
    pointer2_0 = int(center[0]+row2)
    pointer2_1 = int(center[1]+col2)

    # rotate pointer around center
    pointer1_0,pointer1_1=rotate((pointer1_0,pointer1_1),(center[0],center[1]),radians)
    pointer2_0, pointer2_1 = rotate((pointer2_0, pointer2_1), (center[0], center[1]), radians)

    pointer2_0=min(pointer2_0,h-1)
    pointer1_0 = min(pointer1_0, h-1)
    pointer2_1 = min(pointer2_0, w-1)
    pointer1_1 = min(pointer1_0, w-1)

    return 1 if img[pointer1_0][pointer1_1] < img[pointer2_0][pointer2_1] else 0


def briefMatch_rotation(desc1,desc2,ratio):

    matches = skimage.feature.match_descriptors(desc1,desc2,'euclidean',cross_check=True,max_ratio=ratio)
    return matches

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
