from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts

import os

def main():
    opts = get_opts()

    # Q1.1
    # test1
    # img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    # test for 4 channel img
    # img_path = join(opts.data_dir, 'laundromat/sun_afrrjykuhhlwiwun.jpg')
    # HW img
    # img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32) / 255
    # check float between 0-1
    # print(img)
    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # util.display_filter_responses(opts, filter_responses)

    # Q1.2
    n_cpu = util.get_num_CPU()
    # print(n_cpu)
    # visual_words.compute_dictionary(opts, n_worker=n_cpu)

    # Q1.3
    # wordmap_list = []
    # img_lists = []
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # img_read(img_lists, 'kitchen/sun_aasmevtpkslccptd.jpg', opts, wordmap_list)
    # img_read(img_lists, 'aquarium/sun_aairflxfskjrkepm.jpg', opts, wordmap_list)
    # img_read(img_lists, 'park/labelme_xgmqwzfyslqksrt.jpg', opts, wordmap_list)
    # for i in range(3):
    #     wordmap = visual_words.get_visual_words(opts, img_lists[i], dictionary)
    #     wordmap_list.append(wordmap)
    #util.visualize_wordmap(wordmap_list, opts)

    # Q2.1-2.4

    # test Q2.1
    # wordmap = visual_words.get_visual_words(opts, img_lists[0], dictionary)
    # hist = visual_recog.get_feature_from_wordmap(opts, wordmap)
    # x = np.arange(0, opts.K)
    # plt.title("normalized histogram")
    # plt.xlabel("words")
    # plt.ylabel("normalization")
    # plt.bar(x, hist)
    # plt.show()

    # test Q2.2
    # wordmap = visual_words.get_visual_words(opts, img_lists[0], dictionary)
    # hist = visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)
    # print(hist.shape)
    # print(np.sum(hist))

    # Q2.3
    # should be similar to 2nd
    # img_read(img_lists, 'aquarium/sun_auxuxzrqwkdkbbdn.jpg', opts, wordmap_list)
    # histograms=[]
    # for i in range(4):
    #     wordmap = visual_words.get_visual_words(opts, img_lists[i], dictionary)
    #     word_hist = visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)
    #     histograms.append(word_hist)
    # print(visual_recog.distance_to_set(word_hist, np.array(histograms)))

    # Q2.4
    # histograms = []
    # img_path1 = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    # h1=visual_recog.get_image_feature(opts,img_path1,dictionary)
    # histograms.append(h1)
    # img_path2 = join(opts.data_dir, 'aquarium/sun_aairflxfskjrkepm.jpg')
    # h2 = visual_recog.get_image_feature(opts, img_path2, dictionary)
    # histograms.append(h2)
    # img_path3 = join(opts.data_dir, 'park/labelme_xgmqwzfyslqksrt.jpg')
    # h3 = visual_recog.get_image_feature(opts, img_path3, dictionary)
    # histograms.append(h3)
    # img_path4 = join(opts.data_dir, 'aquarium/sun_auxuxzrqwkdkbbdn.jpg')
    # h4 = visual_recog.get_image_feature(opts, img_path4, dictionary)
    # histograms.append(h4)
    # print(visual_recog.distance_to_set(h4, np.array(histograms)))

    # Q2.4
    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    # Q2.5
    n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)

    print(conf)
    print(accuracy)
    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


def img_read(img_lists, path, opts, wordmap_list):
    img_path = join(opts.data_dir, path)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    wordmap_list.append(img)
    img_lists.append(img)


if __name__ == '__main__':
    main()
