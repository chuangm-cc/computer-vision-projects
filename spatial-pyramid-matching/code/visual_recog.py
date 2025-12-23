import os
import math
import multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    """
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    """

    K = opts.K
    # ----- TODO -----
    hist_shape = (K, )
    hist = np.zeros(hist_shape)
    for pixel_row in wordmap:
        for pixel in pixel_row:
            pixel = pixel.astype(int)
            hist[pixel] += 1
    hist /= (wordmap.shape[0]*wordmap.shape[1])
    return hist
    pass


def get_feature_from_wordmap_SPM(opts, wordmap):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape K*(4^(L+1) - 1) / 3
    """

    K = opts.K
    L = opts.L
    # ----- TODO -----
    # print(wordmap.shape)
    # divide into 2^l × 2^l cells
    level_num = pow(2, L)
    # list for histogram
    histogram_list = []
    # get pixel number per column and row
    pixel_row = wordmap.shape[1] // level_num
    pixel_col = wordmap.shape[0] // level_num
    # print(wordmap.shape)
    # print(pixel_row)
    # print(pixel_col)
    # for finest layer
    hist_all = []
    for i in range(level_num):
        for j in range(level_num):
            start_row = i * pixel_row
            start_col = j * pixel_col
            map_part = wordmap[start_col:start_col + pixel_col, start_row:start_row + pixel_row]
            # plt.imshow(map_part)
            # plt.show()
            hist, bins = np.histogram(map_part, K)
            # print(hist)
            # hist = get_feature_from_wordmap(opts, map_part)
            # print(np.sum(hist))
            histogram_list.append(hist)

            hist_new = hist#/(level_num*level_num)
            # print(np.sum(hist_new))
            if L > 1:
                hist_new =hist_new* pow(2, -1)
            else:
                hist_new =hist_new* pow(2, -L)
            # print(np.sum(hist_new))
            for hist_one in hist_new:
                hist_all.append(hist_one)
            # print(sum(histogram_final))
    # print(np.sum(hist_all))
    # histogram_final = np.mean(histogram_list, axis=0)
    # print(sum(histogram_final))
    # weight of layers 0 and 1 to 2^−L, and set the rest of to 2^l−L−1
    # if L > 1:
    #     histogram_final *= pow(2, -1)
    # else:
    #     histogram_final *= pow(2, -L)
    # print(sum(histogram_final))

    level = L - 1
    # histograms of coarser layers aggregated from finer ones
    for i in range(L):
        histogram_list_last_level = histogram_list
        histogram_list = []
        # print("----")
        for j in range(0, level_num, 2):
            for k in range(0, level_num, 2):
                hist=histogram_list_last_level[j*level_num+k]
                # print(np.sum(hist))
                hist+=histogram_list_last_level[j*level_num+k+1]
                # print(j * level_num + k+level_num)
                hist += histogram_list_last_level[j * level_num + k+level_num]
                hist += histogram_list_last_level[j * level_num + k + level_num+1]
                # hist /= 4
                # print(len(hist))
                # print(np.sum(hist))
                histogram_list.append(hist)

                hist_new = hist#/((level_num*level_num)/4)
                # print((level_num*level_num)/4)
                if level > 1:
                    hist_new =hist_new* pow(2, level - L - 1)
                else:
                    hist_new =hist_new* pow(2, -L)
                # print(np.sum(hist_new))
                # print(hist)
                for hist_one in hist_new:
                    hist_all.append(hist_one)
        # print(np.sum(hist_all))
        # histogram_tmp = np.mean(histogram_list, axis=0)
        # print(sum(histogram_tmp))
        # print(level)
        # if level > 1:
        #     histogram_tmp *= pow(2, level - L - 1)
        # else:
        #     histogram_tmp *= pow(2, -L)
        # # print(sum(histogram_tmp))
        # histogram_final += histogram_tmp
        level_num = pow(2, level)
        level -= 1

    hist_all = np.array(hist_all)
    return hist_all/np.sum(hist_all)
    # print(np.sum(hist_all))
    # print(hist_all.shape)
    pass


def get_image_feature(opts, img_path, dictionary):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    """

    # ----- TODO -----
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    word_hist = get_feature_from_wordmap_SPM(opts, wordmap)
    return word_hist
    pass


def build_recognition_system(opts, n_worker=1):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(out_dir, "dictionary.npy"))

    # ----- TODO -----
    # record process
    img_num = len(train_files)
    processed_num = 0

    features = []
    # get all feature
    for name in train_files:
        img_path = join(opts.data_dir, name)
        # get feature
        feature = get_image_feature(opts, img_path, dictionary)
        features.append(feature)
        processed_num += 1
        print("process: "+str(processed_num)+'/'+str(img_num))
    features = np.array(features)
    print(features.shape)
    print(train_labels.shape)
    print(dictionary.shape)

    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

    # pass


def distance_to_set(word_hist, histograms):
    """
    Compute distance between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    """

    # ----- TODO -----
    # print(word_hist.shape)
    # print(histograms.shape)
    # get minimum value of each corresponding bins
    result = np.minimum(word_hist,histograms)
    # print(result)
    # sum
    result=np.sum(result, axis=1)
    # print(result)
    return 1-result
    pass


def evaluate_recognition_system(opts, n_worker=1):
    """

    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, "trained_system.npz"))
    dictionary = trained_system["dictionary"]

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]

    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)

    # ----- TODO -----
    features = trained_system["features"]
    prediction_label = trained_system["labels"]
    # matrix of 8*8
    confusion_matrix = np.zeros((8, 8))

    img_num = len(test_files)
    processed_num = 0
    # get prediction
    for name in test_files:
        img_path = join(opts.data_dir, name)
        # get feature
        hist = get_image_feature(opts, img_path, dictionary)
        # get distance
        distances = distance_to_set(hist, features)
        # min index's label is prediction
        index = np.argmin(distances)
        prediction = prediction_label[index]
        real = test_labels[processed_num]
        # print(index, prediction, real)
        # add in the matrix
        confusion_matrix[real][prediction] += 1
        processed_num += 1
        print("process: " + str(processed_num) + '/' + str(img_num))
    trace = confusion_matrix.trace()
    accuracy = trace/img_num

    print(features.shape)
    print(test_opts.L)
    print(dictionary.shape)

    return confusion_matrix, accuracy

    pass
