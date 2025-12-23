import os
import multiprocessing
from os.path import join, isfile

import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
from sklearn.cluster import KMeans

from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_laplace
import visual_words


def extract_filter_responses(opts, img):
    """
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3) or (H,W,4) with range [0, 1]
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """

    filter_scales = opts.filter_scales
    # ----- TODO -----
    # result to store
    total_size = 3 * len(filter_scales) * 4
    result_shape = (img.shape[0], img.shape[1], total_size)
    result = np.zeros(result_shape)
    num = 0

    # check it's RGB img or grey img
    if len(img.shape) != 3:
        # simply duplicate grey into three channels
        img_shape = (img.shape[0], img.shape[1], 3)
        new_img = np.zeros(img_shape)
        for i in range(3):
            new_img[:, :, i] = img;
        img = new_img

    # Discard the last channel if more
    if img.shape[2] > 3:
        img_shape = (img.shape[0], img.shape[1], 3)
        new_img = np.zeros(img_shape)
        new_img[:, :, 0:3] = img[:, :, 0:3];
        img = new_img

    # convert image into the Lab color space
    img = skimage.color.rgb2lab(img)

    # first: scale
    for scale in filter_scales:
        # second: try all filters
        # third: only keep first 3(RGB) channel
        for i in range(3):
            # get each channel
            test_img = img[:, :, i]
            # use filter Gaussian
            gaussian_result = gaussian_filter(test_img, sigma=scale)
            # append individual result to the final
            result[:, :, num] = gaussian_result
            num += 1
        for i in range(3):
            # get each channel
            test_img = img[:, :, i]
            # use filter Laplacian of Gaussian
            laplace_result = gaussian_laplace(test_img, sigma=scale)
            # append individual result to the final
            result[:, :, num] = laplace_result
            num += 1
        for i in range(3):
            # get each channel
            test_img = img[:, :, i]
            # use filter derivative of Gaussian in the x direction
            gaussian_result_x = gaussian_filter(test_img, sigma=scale, order=(0, 1))
            # append individual result to the final
            result[:, :, num] = gaussian_result_x
            num += 1
        for i in range(3):
            # get each channel
            test_img = img[:, :, i]
            # use filter derivative of Gaussian in the y direction
            gaussian_result_y = gaussian_filter(test_img, sigma=scale, order=(1, 0))
            # append individual result to the final
            result[:, :, num] = gaussian_result_y
            num += 1

    # check
    # print(result.shape)
    return result
    pass


def compute_dictionary_one_image(img_path, total_size, opts):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    """

    # ----- TODO -----
    name = img_path.split(".")[0]
    name = name.split("/")[1]

    alpha = opts.alpha
    img_path = join(opts.data_dir, img_path)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    filter_responses = visual_words.extract_filter_responses(opts, img)

    # flatten first 2 dimension
    result_one = filter_responses.reshape(-1, filter_responses.shape[2])
    # get random pixel for each 3F
    random_pixels = np.zeros((alpha, total_size))
    for i in range(total_size):
        # False: same one can't be got twice
        random_pixels[:, i] = np.random.choice(result_one[:, i], alpha, False)
    feat_dir = opts.feat_dir
    np.save(join(feat_dir, name+'.npy'), random_pixels)
    pass





def compute_dictionary(opts, n_worker):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    # test for small data
    train_files = open(join(data_dir, "train_files_small.txt")).read().splitlines()
    # for whole test data
    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    # ----- TODO -----
    # record the processing
    img_num = len(train_files)
    processed_num = 0
    # check
    # print(train_files)
    # result for storing
    filter_scales = opts.filter_scales
    total_size = 3 * len(filter_scales) * 4
    result_shape = (0, total_size)
    result = np.zeros(result_shape)

    # get alpha
    alpha = opts.alpha

    # multithread
    num_per_process = len(train_files)//n_worker
    # get every thread
    for i in range(num_per_process):
        process_list = []
        for j in range(n_worker):
            p = multiprocessing.Process(target=compute_dictionary_one_image,
                                    args=(train_files[j+i*n_worker], total_size, opts))
            process_list.append(p)
        for process in process_list:
            process.start()
        for process in process_list:
            process.join()
            processed_num+=1
            print("multi-processing:"+str(processed_num)+"/"+str(img_num))
    while processed_num<img_num:
        compute_dictionary_one_image(train_files[processed_num], total_size, opts)
        processed_num+=1
        print("multi-processing:" + str(processed_num) + "/" + str(img_num))


    # iterate through the paths read the images
    # for img_path in train_files:
    #     img_path = join(opts.data_dir, img_path)
    #     # read and use filter
    #     img = Image.open(img_path)
    #     img = np.array(img).astype(np.float32) / 255
    #     filter_responses = visual_words.extract_filter_responses(opts, img)
    #
    #     # flatten first 2 dimension
    #     result_one = filter_responses.reshape(-1, filter_responses.shape[2])
    #     # get random pixel for each 3F
    #     random_pixels = np.zeros((alpha, total_size))
    #     for i in range(total_size):
    #         # False: same one can't be got twice
    #         random_pixels[:, i] = np.random.choice(result_one[:, i], alpha, False)
    #     # append one result of random pixel to the final result
    #     result = np.concatenate((result, random_pixels))
    #     # test
    #     # print(result_one.shape)
    #     # print(random_pixels.shape)
    #     # print(result.shape)
    #     # show process
    #     processing_num += 1
    #     print("process: " + str(processing_num) + "/" + str(img_num))

    feat_dir = opts.feat_dir
    dirs = os.listdir(feat_dir)

    # get all in tmp
    for file in dirs:
        random_pixels = np.load(join(opts.feat_dir, file))
        result = np.concatenate((result, random_pixels))
    # K*alpha * 3F
    print(result.shape)
    # use kmeans to get the dictionary
    filter_responses = result
    kmeans = KMeans(n_init="auto", n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    # test size: K × 3F
    print(dictionary.shape)
    # pass

    # example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)


def get_visual_words(opts, img, dictionary):
    """
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """

    # ----- TODO -----
    # get filter result
    filter_responses = visual_words.extract_filter_responses(opts, img)
    # wordmap initial
    wordmap = np.zeros((img.shape[0] * img.shape[1]))
    # wordmap = np.zeros((img.shape[0] * img.shape[1], filter_responses.shape[2]))
    # print(wordmap.shape)

    # append each pixel to wordmap
    filter_responses = filter_responses.reshape(-1, filter_responses.shape[2])
    distance = scipy.spatial.distance.cdist(filter_responses, dictionary, 'euclidean')

    # print(distance.shape)
    for i in range(img.shape[0] * img.shape[1]):
        # also can use(same as):min_index=np.argmin(distance[i])
        min_value = min(distance[i])
        for j in range(len(distance[i])):
            if distance[i][j] == min_value:
                min_index = j
        # value = mean(dictionary[min_index])
        value = min_index
        # print(min_index)
        wordmap[i] = value
        # wordmap[i, :] = dictionary[min_index]
    # print(wordmap.shape)
    wordmap = wordmap.reshape((img.shape[0], img.shape[1]))
    # wordmap = wordmap.reshape((img.shape[0], img.shape[1], filter_responses.shape[1]))
    # plt.imshow(wordmap[..., 0:3])
    # plt.show()
    return wordmap
    pass
