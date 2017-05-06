# -*- coding:utf8 -*-
import sys
import os
import random
import multi_digit_reader as mdr
import svhn_data
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from PIL import Image
from svhn_model import regression_head
from svhn_data import load_svhn_data

#FILE_PREFIX = "~/Code/machine-learning/projects/capstone/data/svhn/full/test/"
FILE_PREFIX = "/home/jj/Code/machine-learning/projects/capstone/data/svhn/full/train/"
PICKLE_FILE = "digitStruct.pickle"
WEIGHTS_FILE = "regression.ckpt"


TESTSET_CNT = 8326
TEST_NUM = 16


def ReadFile(file_idxs):
    # Get the content in DigitStruct.mat
    if os.path.isfile(PICKLE_FILE):
        with open(PICKLE_FILE, 'rb') as fp:
            structs = pickle.load(fp)
    else:
        structs = svhn_data.read_digit_struct(FILE_PREFIX)
        with open(PICKLE_FILE, 'wb') as fp:
            pickle.dump(structs, fp)
        print(PICKLE_FILE + " written successfully")

    imgs = np.zeros([len(file_idxs), 64, 64, 3],
                        dtype='float32')
    labels = np.zeros([len(file_idxs), 5],
                        dtype=int)
    str_label = []
    for i in range(len(file_idxs)):
        filename = FILE_PREFIX + str(file_idxs[i]+1) + ".png"
        top = structs[file_idxs[i]]['top']
        left = structs[file_idxs[i]]['left']
        height = structs[file_idxs[i]]['height']
        width = structs[file_idxs[i]]['width'] 

        img = create_img_array(filename, top, left, height, width, 64, 64)
        lbl = create_label_array(structs[file_idxs[i]]['label'])

        imgs[i] = img
        labels[i] = lbl[1:6]
        str_label.append(prediction_to_string(labels[i]))
        

    return imgs, str_label


def create_img_array(file_name, top, left, height, width, out_height, out_width):
    img = Image.open(file_name)

    img_top = np.amin(top)
    img_left = np.amin(left)
    img_height = np.amax(top) + height[np.argmax(top)] - img_top
    img_width = np.amax(left) + width[np.argmax(left)] - img_left

    box_left = np.floor(img_left - 0.1 * img_width)
    box_top = np.floor(img_top - 0.1 * img_height)
    box_right = np.amin([np.ceil(box_left + 1.2 * img_width), img.size[0]])
    box_bottom = np.amin([np.ceil(img_top + 1.2 * img_height), img.size[1]])

    img = img.crop((box_left, box_top, box_right, box_bottom)).resize([out_height, out_width], Image.ANTIALIAS)
    pix = np.array(img)

    return pix

def create_label_array(el):
    """[count, digit, digit, digit, digit, digit]"""
    num_digits = len(el)  # first element of array holds the count
    labels_array = np.ones([5+1], dtype=int) * 10
    labels_array[0] = num_digits
    for n in range(num_digits):
        if el[n] == 10: el[n] = 0  # reassign 0 as 10 for one-hot encoding
        labels_array[n+1] = el[n]
    return labels_array

def prediction_to_string(pred_array):
    pred_str = ""
    for i in range(len(pred_array)):
        if pred_array[i] != 10:
            pred_str += str(pred_array[i])
        else:
            return pred_str
    return pred_str


# "files" is an array of numpy
def Recognition(files, saved_model_weights):
    norm_pix = (255-files)*1.0/255.0
    norm_pix -= np.mean(norm_pix, axis=0)

    # for main2()
    #norm_pix = files

    X = tf.placeholder(tf.float32, shape=(files.shape[0], 64, 64, 3))
    [logits_1, logits_2, logits_3, logits_4, logits_5] = regression_head(X)
    predict = tf.stack([tf.nn.softmax(logits_1),
                      tf.nn.softmax(logits_2),
                      tf.nn.softmax(logits_3),
                      tf.nn.softmax(logits_4),
                      tf.nn.softmax(logits_5)])
    best_prediction = tf.transpose(tf.argmax(predict, 2))

    saver = tf.train.Saver()
    with tf.Session() as session:
    	saver.restore(session, saved_model_weights)
    	print "Model restored."

    	# Feed the model and get prediction results
    	feed_dict = {X : norm_pix}
    	predictions, pred_prob = session.run([best_prediction, predict], feed_dict=feed_dict)

        str_preds = []
        for each_pred in predictions:
            str_preds.append(prediction_to_string(each_pred))

    return str_preds


# crop_files is an array of numpy
def ShowImgsAndLabels(crop_files, results):
    plt.close('all')
    fig = plt.figure()

    i = 0
    for row in range(4):
        for column in range(4):
            img = Image.fromarray(np.uint8(crop_files[i]), 'RGB')
            ax = fig.add_subplot(4, 4, i+1)
            ax.set_title(results[i])
            ax.set_axis_off()
            ax.imshow(img)

            i = i+1

    plt.show()


def main():
    file_idxs = random.sample(range(8326), TEST_NUM)
    crop_files, labels = ReadFile(file_idxs)

    if os.path.isfile(WEIGHTS_FILE):
        saved_model_weights = WEIGHTS_FILE
    else:
        raise IOError("Cannot find checkpoint file. Please run train_regressor.py")

    results = Recognition(crop_files, saved_model_weights)    

    prediction = (np.array(results) == np.array(labels))

    pred = np.sum(prediction)*1.0/prediction.shape[0]
    print('Prediction: %f' %pred)
    ShowImgsAndLabels(crop_files, results)


# 该主函数的 dataset 数据来源是 svhn_data 生成的 .npy　文件
# 测试用 —— 测试从图片读取的数据和从 .npy 文件中读取的数据之间的正确率的差别
def main2():
    file_idxs = random.sample(range(8326), TEST_NUM)
    data, labels = load_svhn_data("train", "full")
    crop_files = np.zeros([len(file_idxs), 64, 64, 3],
                        dtype='float32')
    crop_labels = np.zeros([len(file_idxs), 6],
                        dtype='float32')
    for i in range(len(file_idxs)):
        crop_files[i] = data[file_idxs[i]]
        crop_labels[i] = labels[file_idxs[i]]

    if os.path.isfile(WEIGHTS_FILE):
        saved_model_weights = WEIGHTS_FILE
    else:
        raise IOError("Cannot find checkpoint file. Please run train_regressor.py")

    results = Recognition(crop_files, saved_model_weights)
    actual_labels = []
    for each_lbl in crop_labels:
        tmp = np.zeros([5],
                        dtype=int)
        tmp[:] = each_lbl[1:6]
        actual_labels.append(prediction_to_string(tmp))

    prediction = (np.array(results) == np.array(actual_labels))
    pred = np.sum(prediction)*1.0/prediction.shape[0]
    print(prediction.shape[0])
    print('Prediction: %f' %pred)

if __name__ == '__main__':
    main()
    #main2()
    '''
    data, labels = load_svhn_data("train", "full")
    print(labels.shape)
    print(labels[0:5])
    '''