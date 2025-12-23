import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    #plt.show()
    #continue
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    # Group the letters based on which line of the text they are a part of, and sort each
    # group so that the letters are in the order they appear on the page.
    row_lists = []
    # box: minr, minc, maxr, maxc
    # sort with row
    bboxes.sort(key=lambda x: x[0])
    #h = bboxes[0][2] - bboxes[0][2]
    threshold_row = 100
    row_judge = bboxes[0][0]
    #print(row_num)
    #print(len(bboxes))
    row_list = []
    for bbox in bboxes:
        row = bbox[0]
        if abs(row - row_judge) < threshold_row:
            row_list.append(bbox)
        else:
            # sort each group with col and append
            row_list.sort(key=lambda x: x[1])
            row_lists.append(row_list)
            # start new row
            row_list = []
            row_list.append(bbox)
        #update new
        row_judge = row
    # last row
    row_list.sort(key=lambda x: x[1])
    row_lists.append(row_list)
    #print(row_lists)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    chars=[]
    counts = []
    for row_list in row_lists:
        count = 0
        for bbox in row_list:
            count+=1
            minr, minc, maxr, maxc = bbox
            crop_img = bw[minr:maxr,minc:maxc]
            # plt.imshow(crop_img)
            # plt.show()
            # square crop, and even using np.pad()
            h = maxr-minr
            w = maxc-minc
            diff = int(abs(h-w)//2)
            pad_h = int(h/7)
            pad_w = int(h / 7)
            if h > w:
                pad_img = np.pad(crop_img, ((pad_h, pad_h), (diff+pad_w, diff+pad_w)), 'maximum')
            else:
                pad_img = np.pad(crop_img, ((diff+pad_h, diff+pad_h),(pad_w, pad_w)), 'maximum')
            # plt.show()')
            # resize: input images are 32 × 32 images
            resized_img = skimage.transform.resize(pad_img, (32, 32))
            # before you flatten, transpose the image
            trans_img = np.transpose(resized_img)
            #plt.imshow(resized_img)
            #plt.show()
            flatten_img = trans_img.flatten()
            chars.append(flatten_img)
        counts.append(count)
    #print(counts)



    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################
    char_with = bboxes[0][3] - bboxes[0][1]
    #print(char_with)
    line_num = len(counts)
    chars = np.array(chars)
    h1 = forward(chars, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    res = []
    k=0
    count = counts[k]
    for prob in probs:
        if counts[k]-count>0 and \
                (row_lists[k][counts[k]-count][1] -
                 row_lists[k][counts[k]-count-1][3])>char_with*0.8:
            res.append(' ')
        index = np.argmax(prob)
        letter = letters[index]
        res.append(letter)
        count -= 1
        if count == 0:
            res.append('\n')
            k += 1
            if k < line_num:
                count = counts[k]
    print(''.join(res))


