import os.path as ops
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

from attentive_gan_model import derain_drop_net
from config import global_config
import time

CFG = global_config.cfg
tf.reset_default_graph()

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default= '62.png', help='The input image path')
    parser.add_argument('--weights_path', type=str, default='./model/derain_gan_tensorflow/derain_gan_2020-10-17-23-08-12.ckpt-100000', help='The model weights path')

    return parser.parse_args()


def minmax_scale(input_arr):
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_model(image_path, weights_path):
    assert ops.exists(image_path)

    with tf.device('/gpu:0'):
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[CFG.TEST.BATCH_SIZE, CFG.TEST.IMG_HEIGHT, CFG.TEST.IMG_WIDTH, 3], name='input_tensor')

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT))
    image_vis = image
    image = np.divide(image, 127.5) - 1

    phase = tf.constant('test', tf.string)

    with tf.device('/gpu:0'):
        net = derain_drop_net.DeRainNet(phase=phase)
        output, attention_maps = net.build(input_tensor=input_tensor, name='derain_net_loss')

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    s_t = time.time()
    sess = tf.Session(config=sess_config)

    saver = tf.train.Saver()


    with tf.device('/gpu:0'):
        with sess.as_default():
            saver.restore(sess=sess, save_path=weights_path)

            output_image, atte_maps = sess.run(
                [output, attention_maps],
                feed_dict={input_tensor: np.expand_dims(image, 0)})
            # s_t = time.time()
            output_image = output_image[0]
            for i in range(output_image.shape[2]):
                output_image[:, :, i] = minmax_scale(output_image[:, :, i])

            e_t = time.time()
            total_t = e_t - s_t
            output_image = np.array(output_image, np.uint8)

            print('time:{:.4f}'.format(total_t))

            plt.figure('src_image')
            plt.imshow(image_vis[:, :, (2, 1, 0)])
            plt.figure('derain_ret')
            plt.imshow(output_image[:, :, (2, 1, 0)])
            plt.figure('atte_map_1')
            plt.imshow(atte_maps[0][0, :, :, 0], cmap='jet')
            plt.savefig('atte_map_1.png')
            plt.figure('atte_map_2')
            plt.imshow(atte_maps[1][0, :, :, 0], cmap='jet')
            plt.savefig('atte_map_2.png')
            plt.figure('atte_map_3')
            plt.imshow(atte_maps[2][0, :, :, 0], cmap='jet')
            plt.savefig('atte_map_3.png')
            plt.figure('atte_map_4')
            plt.imshow(atte_maps[3][0, :, :, 0], cmap='jet')
            plt.savefig('atte_map_4.png')
            plt.show()


if __name__ == '__main__':
    args = init_args()
    test_model(args.image_path, args.weights_path)
