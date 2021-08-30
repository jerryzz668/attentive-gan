# cited from:
#             @Author  : Luo Yao
#             @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
#             @File    : discriminative_net.py

import tensorflow as tf

from attentive_gan_model import cnn_basenet


class DiscriminativeNet(cnn_basenet.CNNBaseModel):

    def __init__(self, phase):
        super(DiscriminativeNet, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        return tf.equal(self._phase, self._train_phase)

    def _conv_stage(self, input_tensor, k_size, stride, out_dims, name):
        with tf.variable_scope(name):
            conv = self.conv2d(inputdata=input_tensor, out_channel=out_dims, kernel_size=k_size,
                               padding='SAME', stride=stride, use_bias=False, name='conv')

            relu = self.lrelu(conv, name='relu')

        return relu

    def _residual_block(self, input_tensor, name):
        output = None
        with tf.variable_scope(name):
            for i in range(3):
                if i == 0:
                    conv_1 = self.conv2d(inputdata=input_tensor,out_channel=32,kernel_size=3,padding='SAME',stride=1,
                                          use_bias=False,name='block_{:d}_conv_1'.format(i))
                    relu_1 = self.lrelu(inputdata=conv_1, name='block_{:d}_relu_1'.format(i + 1))
                    output = relu_1
                    input_tensor = output
                else:
                    conv_1 = self.conv2d(inputdata=input_tensor,out_channel=32,kernel_size=1,padding='SAME',stride=1,
                                          use_bias=False,name='block_{:d}_conv_1'.format(i))
                    relu_1 = self.lrelu(inputdata=conv_1, name='block_{:d}_conv_1'.format(i + 1))
                    conv_2 = self.conv2d(inputdata=relu_1, out_channel=32,kernel_size=1,padding='SAME',stride=1,
                                          use_bias=False,name='block_{:d}_conv_2'.format(i))
                    relu_2 = self.lrelu(inputdata=conv_2, name='block_{:d}_conv_2'.format(i + 1))

                    output = self.lrelu(inputdata=tf.add(relu_2, input_tensor), name='block_{:d}_add'.format(i))
                    input_tensor = output
        return output

    def build(self, input_tensor, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            res_dis_1 = self._residual_block(input_tensor=input_tensor, name='residual_block_dis_1')
            res_dis_2 = self._residual_block(input_tensor=res_dis_1, name='residual_block_dis_2')
            res_dis_3 = self._residual_block(input_tensor=res_dis_2, name='residual_block_dis_3')

            attention_map = self.conv2d(inputdata=res_dis_3, out_channel=1, kernel_size=5,
                                        padding='SAME', stride=1, use_bias=False, name='attention_map')
            conv_stage_7 = self._conv_stage(input_tensor=attention_map * res_dis_3, k_size=5,
                                            stride=4, out_dims=32, name='conv_stage_7')
            res_dis_4 = self._residual_block(input_tensor=conv_stage_7, name='residual_block_dis_4')

            conv_stage_9 = self._conv_stage(input_tensor=res_dis_4, k_size=5,
                                            stride=4, out_dims=32, name='conv_stage_9')
            fc_1 = self.fullyconnect(inputdata=conv_stage_9, out_dim=1024, use_bias=False, name='fc_1')
            fc_2 = self.fullyconnect(inputdata=fc_1, out_dim=1, use_bias=False, name='fc_2')
            fc_out = self.sigmoid(inputdata=fc_2, name='fc_out')

            fc_out = tf.where(tf.not_equal(fc_out, 1.0), fc_out, fc_out - 0.0000001)
            fc_out = tf.where(tf.not_equal(fc_out, 0.0), fc_out, fc_out + 0.0000001)

            return fc_out, attention_map, fc_2

    def compute_loss(self, input_tensor, label_tensor, attention_map, name):
        with tf.variable_scope(name):
            [batch_size, image_h, image_w, _] = input_tensor.get_shape().as_list()

            # 论文里的O
            zeros_mask = tf.zeros(shape=[batch_size, image_h, image_w, 1],
                                  dtype=tf.float32, name='O')
            fc_out_o, attention_mask_o, fc2_o = self.build(
                input_tensor=input_tensor, name='discriminative_inference')
            fc_out_r, attention_mask_r, fc2_r = self.build(
                input_tensor=label_tensor, name='discriminative_inference', reuse=True)

            l_map = tf.losses.mean_squared_error(attention_map, attention_mask_o) + \
                    tf.losses.mean_squared_error(attention_mask_r, zeros_mask)

            # PSNR loss
            #psnr_finer_tensor = tf.reduce_mean(tf.image.psnr(attention_map, attention_mask_o, 255)) + \
             #                   tf.reduce_mean(tf.image.psnr(attention_mask_r, zeros_mask, 255))
            #psnr_loss = 1.0 / (psnr_finer_tensor + 1e-3)

            entropy_loss = -tf.log(fc_out_r) - tf.log(-tf.subtract(fc_out_o, tf.constant(1.0, tf.float32)))
            entropy_loss = tf.reduce_mean(entropy_loss)

            loss = entropy_loss + 0.05 * l_map
            #loss = entropy_loss + 0.05 * l_map + 0.1 * psnr_loss

            return fc_out_o, loss
