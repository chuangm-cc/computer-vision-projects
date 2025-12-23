import time

import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief_fast
from helper import corner_detection
#from helper import computeBrief
# Q2.1.4

def matchPics(desc1, I2, opts):
        """
        Match features across images

        Input
        -----
        I1, I2: Source images
        opts: Command line args

        Returns
        -------
        matches: List of indices of matched features across I1, I2 [p x 2]
        locs1, locs2: Pixel coordinates of matches [N x 2]
        """
        
        ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
        sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
        # TODO: Convert Images to GrayScale
        I2_gray = skimage.color.rgb2gray(I2)

        time_cor = time.time()
        # TODO: Detect Features in Both Images
        locs2 = corner_detection(I2_gray, sigma)
        #print(locs2.shape)

        time_bri = time.time()
        desc2, locs2 = computeBrief_fast(I2_gray, locs2)
        #print("brief: " + str(time.time() - time_bri))

        time_match = time.time()
        # TODO: Match features using the descriptors
        matches = briefMatch(desc1, desc2, ratio)
        #print("match: " + str(time.time() - time_match))

        return matches, locs2
