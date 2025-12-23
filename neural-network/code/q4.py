import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################
    # estimate noise
    # denoise
    denoised = skimage.restoration.denoise_bilateral(image,channel_axis=-1)#, sigma_color=0.05, sigma_spatial=15)
    # -> greyscale
    grayscale = skimage.color.rgb2gray(denoised)
    # -> threshold
    thresh = skimage.filters.threshold_otsu(grayscale)
    binary = grayscale < thresh
    # -> morphology
    #tophat = skimage.morphology.black_tophat(binary)
    #binary = skimage.morphology.binary_opening(binary)
    dilation =skimage.morphology.binary_dilation(binary)
    dilation = skimage.morphology.binary_dilation(dilation)
    dilation = skimage.morphology.binary_dilation(dilation)
    dilation = skimage.morphology.binary_dilation(dilation)
    #dilation = skimage.morphology.binary_dilation(dilation)
    #dilation = skimage.morphology.binary_dilation(dilation)
    # -> label
    label_image = skimage.measure.label(dilation)
    regions = skimage.measure.regionprops(label_image)
    # -> skip small boxes
    # check
    #skimage.io.imshow(denoised)
    #skimage.io.imshow(binary)
    #skimage.io.imshow(tophat)
    # skimage.io.imshow(label_image)
    # skimage.io.imshow(dilation)
    # skimage.io.show()
    areas=[]
    for region in regions:
        areas.append(region.area)
    max_area = np.max(areas)
    for region in regions:
        #if region.area >= 500:
        if region.area >= max_area*0.3:
            bboxes.append(region.bbox)
    # with the characters in black and the background in white.
    # should be between 0.0 to 1.0
    dilation = ~dilation
    dilation=dilation.astype(float)
    bw = dilation
    return bboxes, bw