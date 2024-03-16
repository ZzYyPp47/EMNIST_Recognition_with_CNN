# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2023/7/9 17:26
@version: 1.0
@File: predict.py
'''


import tensorflow as tf
from PIL import Image
import numpy as np
from train import CNN
import os


class Predict(object):
    def __init__(self):
        latest = tf.train.latest_checkpoint('./ckpt')
        self.cnn = CNN()
        # 恢复网络权重
        self.cnn.model.load_weights(latest)
    def predict(self, image_path):
        # 以黑白方式读取图片
        img = Image.open(image_path).convert('L')
        img = np.reshape(img, (28, 28, 1)) / 255.
        x = np.array([1 - img])
        y = self.cnn.model.predict(x,verbose=0)
        print(image_path,'-> Predict is', num2c(np.argmax(y[0])))


def num2c(pred_num):
    chadict = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
        10: 'A', 11: 'B', 12: 'C(c)', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I(i)', 19: 'J(j)',
        20: 'K(k)', 21: 'L(l)', 22: 'M(m)', 23: 'N(n)', 24: 'O(o)', 25: 'P(p)', 26: 'Q', 27: 'R', 28: 'S(s)', 29: 'T',
        30: 'U(u)', 31: 'V(v)', 32: 'W(w)', 33: 'X(x)', 34: 'Y(y)', 35: 'Z(z)', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
        40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
    }
    return chadict[pred_num]


def get_file_paths(folder_path):
    file_paths = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_paths.append(os.path.join(dirpath, filename))
    return file_paths


if __name__ == "__main__":
    app = Predict()
    folder_path = './test_images/'  # 图片路径
    file_paths = get_file_paths(folder_path)
    for file_path in file_paths:
        app.predict(file_path)