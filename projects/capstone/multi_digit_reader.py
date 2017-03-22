# -*- coding:utf8 -*-
import sys
import os
import numpy as np
import tensorflow as tf
import PIL.Image as Image
import matplotlib.pyplot as plt

from svhn_model import regression_head
from svhn_data import load_svhn_data
import time
import detect.localize as localize
from detect.detect import Box

test_dataset, test_labels = load_svhn_data("test", "full")
WEIGHTS_FILE = "regression.ckpt"


def prediction_to_string(pred_array):
    pred_str = ""
    for i in range(len(pred_array)):
        if pred_array[i] != 10:
            pred_str += str(pred_array[i])
        else:
            return pred_str
    return pred_str


def detect(img, saved_model_weights):
    sample_img = img.resize([64,64])
    plt.imshow(sample_img)
    plt.show()

    pix = np.array(sample_img)
    
    norm_pix = (255-pix)*1.0/255.0
    norm_pix = norm_pix - np.mean(norm_pix, axis=0)
    exp = np.expand_dims(norm_pix, axis=0)
    #exp = np.expand_dims(pix, axis=0)

    X = tf.placeholder(tf.float32, shape=(1, 64, 64, 3))
    [logits_1, logits_2, logits_3, logits_4, logits_5] = regression_head(X)

    predict = tf.pack([tf.nn.softmax(logits_1),
                      tf.nn.softmax(logits_2),
                      tf.nn.softmax(logits_3),
                      tf.nn.softmax(logits_4),
                      tf.nn.softmax(logits_5)])

    best_prediction = tf.transpose(tf.argmax(predict, 2))

    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, "regression.ckpt")
        print "Model restored."

        print "Initialized"
        feed_dict = {X: exp}
        start_time = time.time()
        predictions, pred_prob = session.run([best_prediction, predict], feed_dict=feed_dict)
        pred = prediction_to_string(predictions[0])
        end_time = time.time()
        print "Best Prediction", pred, "made in", end_time - start_time
        print(predictions[0])
        print(pred_prob)


def original_main():
    img_path = None
    if len(sys.argv) > 1:
        print("Reading Image file:", sys.argv[1])
        if os.path.isfile(sys.argv[1]):
            img_path = sys.argv[1]
        else:
            raise EnvironmentError("Image file cannot be opened.")
    else:
        raise EnvironmentError("You must pass an image file to process")
    if os.path.isfile(WEIGHTS_FILE):
        saved_model_weights = WEIGHTS_FILE
    else:
        raise IOError("Cannot find checkpoint file. Please run train_regressor.py")
    img = Image.open(img_path)
    detect(img, saved_model_weights)


# @func: 按照 box 中坐标指定的 area 裁剪图片
# @param: filename - 图片路径
#         box - 矩形区域的坐标信息
# @return: 裁剪后的图片
def crop_img(filename, box):
    img = Image.open(filename)
    left = int(box.left - box.width*0.1)
    right = int(box.left + box.width*1.1)
    top = int(box.top - box.height*0.1)
    bottom = int(box.top + box.height*1.1)

    img = img.crop((left, top, right, bottom))
    return img


def main_with_detect():
    if len(sys.argv) != 2:
        print('Format: python' + sys.argv[0] + 'filePath')
        exit()

    ratios = [0.5, 1]
    #ratios = [1, 2]
    ret_digit = localize.localize(sys.argv[1], ratios)
    ret_area = localize.mergeAllAreas(ret_digit)
    #show_image(sys.argv[1], [ret_area])

    img = crop_img(sys.argv[1], ret_area)
    if os.path.isfile(WEIGHTS_FILE):
        saved_model_weights = WEIGHTS_FILE
    else:
        raise IOError("Cannot find checkpoint file. Please run train_regressor.py")  
    detect(img, saved_model_weights)



if __name__ == "__main__":
    #original_main()
    main_with_detect()