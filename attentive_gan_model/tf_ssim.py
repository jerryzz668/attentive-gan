# cited from:
#             @Author  : Luo Yao
#             @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
#             @File    : tf_ssim.py

import tensorflow as tf
import numpy as np


class SsimComputer(object):

    def __init__(self):
        pass

    @staticmethod
    def _tf_fspecial_gauss(size, sigma):

        x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / tf.reduce_sum(g)

    def compute_ssim(self, img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):

        assert img1.get_shape().as_list()[-1] == 1, 'Image must be gray scale'
        assert img2.get_shape().as_list()[-1] == 1, 'Image must be gray scale'

        window = self._tf_fspecial_gauss(size, sigma)  # window shape [size, size]
        K1 = 0.01  # origin parameter in paper
        K2 = 0.03  # origin parameter in paper
        L = 1  # depth of image (255 in case the image has a differnt scale)
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
        mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
        sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
        sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
        if cs_map:
            value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                  (sigma1_sq + sigma2_sq + C2)),
                     (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        else:
            value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                 (sigma1_sq + sigma2_sq + C2))

        if mean_metric:
            value = tf.reduce_mean(value)
        return value

class PSNR:
    def __init__(self, max_val=255.0):
        self.max_val = max_val

    def compute_mse(self, img1, img2):
        # RGB image
        shape = img1.get_shape().as_list()
        if len(shape) < 3:
            raise Exception('error image type')
        diff = tf.square(img1 - img2)
        mse = tf.reduce_mean(diff, [-3, -2, -1])
        return mse

    def compute_rmse(self, img1, img2):
        shape = img1.get_shape().as_list()
        if len(shape) < 3:
            raise Exception('error image type')
        diff = tf.square(img1 - img2)
        rmse = tf.sqrt(tf.reduce_mean(diff))
        return rmse

    def compute_psnr(self, img1, img2):
        # type transformation
        mse = self.compute_mse(img1, img2)

        if mse == 0:
            return 100.0, 100.0, 100.0

        psnr = 10 * ((tf.log(tf.square(self.max_val) / mse)) / tf.log(tf.constant(10.0, dtype=tf.float32)))

        return psnr