import numpy as np
import cv2

from matchPics import matchPics
from opts import get_opts

from scipy import ndimage
import matplotlib.pyplot as plt
from helper import plotMatches
#Q2.1.6

def rotTest(opts):

    #Read the image and convert to grayscale, if necessary
    img = cv2.imread('../data/cv_cover.jpg')

    hist=[]
    rotations =[]
    for i in range(36):
        #Rotate Image
        rotation_img = ndimage.rotate(img, i*10)
        #Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(img, rotation_img, opts)
        #Update histogram
        hist.append(matches.shape[0])
        rotations.append(str(i*10))
        print(matches.shape)
        if i==9 or i==18 or i==27:
            plotMatches(rotation_img, img, matches, locs1, locs2)

    #Display histogram
    plt.xticks(rotation=90, ha='right')
    plt.bar(rotations,hist,align='center',width = 1)
    plt.xlabel('rotation')
    plt.ylabel('number of matches')
    plt.title('number of matches against rotation')
    plt.show()


if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
