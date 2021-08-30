
import tensorflow as tf

from attentive_gan_model import cnn_basenet
from attentive_gan_model import vgg16


class GenerativeNet(cnn_basenet.CNNBaseModel):

    def __init__(self, phase):

        super(GenerativeNet, self).__init__()
        self._vgg_extractor = vgg16.VGG16Encoder(phase='test')
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):

        return tf.equal(self._phase, self._train_phase)

    def _residual_block(self, input_tensor, name):

        output = None
        with tf.variable_scope(name):
            for i in range(5):
                if i == 0:
                    conv_1 = self.conv2d(inputdata=input_tensor,
                                         out_channel=32,
                                         kernel_size=3,
                                         padding='SAME',
                                         stride=1,
                                         use_bias=False,
                                         name='block_{:d}_conv_1'.format(i))
                    relu_1 = self.lrelu(inputdata=conv_1, name='block_{:d}_relu_1'.format(i + 1))
                    output = relu_1
                    input_tensor = output
                else:
                    conv_1 = self.conv2d(inputdata=input_tensor,
                                         out_channel=32,
                                         kernel_size=1,
                                         padding='SAME',
                                         stride=1,
                                         use_bias=False,
                                         name='block_{:d}_conv_1'.format(i))
                    relu_1 = self.lrelu(inputdata=conv_1, name='block_{:d}_conv_1'.format(i + 1))
                    conv_2 = self.conv2d(inputdata=relu_1,
                                         out_channel=32,
                                         kernel_size=1,
                                         padding='SAME',
                                         stride=1,
                                         use_bias=False,
                                         name='block_{:d}_conv_2'.format(i))
                    relu_2 = self.lrelu(inputdata=conv_2, name='block_{:d}_conv_2'.format(i + 1))

                    output = self.lrelu(inputdata=tf.add(relu_2, input_tensor),
                                        name='block_{:d}_add'.format(i))
                    input_tensor = output

        return output

    def _conv_lstm(self, input_tensor, input_cell_state, name):

        with tf.variable_scope(name):
            conv_i = self.conv2d(inputdata=input_tensor, out_channel=32, kernel_size=3, padding='SAME',
                                 stride=1, use_bias=False, name='conv_i')
            sigmoid_i = self.sigmoid(inputdata=conv_i, name='sigmoid_i')

            conv_f = self.conv2d(inputdata=input_tensor, out_channel=32, kernel_size=3, padding='SAME',
                                 stride=1, use_bias=False, name='conv_f')
            sigmoid_f = self.sigmoid(inputdata=conv_f, name='sigmoid_f')

            cell_state = sigmoid_f * input_cell_state + \
                         sigmoid_i * tf.nn.tanh(self.conv2d(inputdata=input_tensor,
                                                            out_channel=32,
                                                            kernel_size=3,
                                                            padding='SAME',
                                                            stride=1,
                                                            use_bias=False,
                                                            name='conv_c'))
            conv_o = self.conv2d(inputdata=input_tensor, out_channel=32, kernel_size=3, padding='SAME',
                                 stride=1, use_bias=False, name='conv_o')
            sigmoid_o = self.sigmoid(inputdata=conv_o, name='sigmoid_o')

            lstm_feats = sigmoid_o * tf.nn.tanh(cell_state)

            attention_map = self.conv2d(inputdata=lstm_feats, out_channel=1, kernel_size=3, padding='SAME',
                                        stride=1, use_bias=False, name='attention_map')
            attention_map = self.sigmoid(inputdata=attention_map)

            ret = {
                'attention_map': attention_map,
                'cell_state': cell_state,
                'lstm_feats': lstm_feats
            }

            return ret

    def build_attentive_rnn(self, input_tensor, name):

        [batch_size, tensor_h, tensor_w, _] = input_tensor.get_shape().as_list()
        with tf.variable_scope(name):
            init_attention_map = tf.constant(0.5, dtype=tf.float32,
                                             shape=[batch_size, tensor_h, tensor_w, 1])
            init_cell_state = tf.constant(0.0, dtype=tf.float32,
                                          shape=[batch_size, tensor_h, tensor_w, 32])
            init_lstm_feats = tf.constant(0.0, dtype=tf.float32,
                                          shape=[batch_size, tensor_h, tensor_w, 32])

            attention_map_list = []

            for i in range(4):
                attention_input = tf.concat((input_tensor, init_attention_map), axis=-1)
                conv_feats = self._residual_block(input_tensor=attention_input,
                                                  name='residual_block_{:d}'.format(i + 1))
                lstm_ret = self._conv_lstm(input_tensor=conv_feats,
                                           input_cell_state=init_cell_state,
                                           name='conv_lstm_block_{:d}'.format(i + 1))
                init_attention_map = lstm_ret['attention_map']
                init_cell_state = lstm_ret['cell_state']
                init_lstm_feats = lstm_ret['lstm_feats']

                attention_map_list.append(lstm_ret['attention_map'])

        ret = {
            'final_attention_map': init_attention_map,
            'final_lstm_feats': init_lstm_feats,
            'attention_map_list': attention_map_list
        }

        return ret

    def compute_attentive_rnn_loss(self, input_tensor, label_tensor, name):

        with tf.variable_scope(name):
            inference_ret = self.build_attentive_rnn(input_tensor=input_tensor,
                                                     name='attentive_inference')
            loss = tf.constant(0.0, tf.float32)
            n = len(inference_ret['attention_map_list'])
            for index, attention_map in enumerate(inference_ret['attention_map_list']):
                mse_loss = tf.pow(0.8, n - index + 1) * \
                           tf.losses.mean_squared_error(labels=label_tensor,
                                                        predictions=attention_map)
                loss = tf.add(loss, mse_loss)

        return loss, inference_ret['final_attention_map']

    def build_autoencoder(self, input_tensor, name):

        with tf.variable_scope(name):
            conv_1 = self.conv2d(inputdata=input_tensor, out_channel=64, kernel_size=5,
                                 padding='SAME',
                                 stride=1, use_bias=False, name='conv_1')
            relu_1 = self.lrelu(inputdata=conv_1, name='relu_1')

            conv_2 = self.conv2d(inputdata=relu_1, out_channel=128, kernel_size=3,
                                 padding='SAME',
                                 stride=2, use_bias=False, name='conv_2')
            relu_2 = self.lrelu(inputdata=conv_2, name='relu_2')

            conv_3 = self.conv2d(inputdata=relu_2, out_channel=128, kernel_size=3,
                                 padding='SAME',
                                 stride=1, use_bias=False, name='conv_3')
            relu_3 = self.lrelu(inputdata=conv_3, name='relu_3')

            conv_4 = self.conv2d(inputdata=relu_3, out_channel=128, kernel_size=3,
                                 padding='SAME',
                                 stride=2, use_bias=False, name='conv_4')
            relu_4 = self.lrelu(inputdata=conv_4, name='relu_4')

            conv_5 = self.conv2d(inputdata=relu_4, out_channel=256, kernel_size=3,
                                 padding='SAME',
                                 stride=1, use_bias=False, name='conv_5')
            relu_5 = self.lrelu(inputdata=conv_5, name='relu_5')

            conv_6 = self.conv2d(inputdata=relu_5, out_channel=256, kernel_size=3,
                                 padding='SAME',
                                 stride=1, use_bias=False, name='conv_6')
            relu_6 = self.lrelu(inputdata=conv_6, name='relu_6')

            dia_conv1 = self.dilation_conv(input_tensor=relu_6, k_size=3, out_dims=256, rate=2,
                                           padding='SAME', use_bias=False, name='dia_conv_1')
            relu_7 = self.lrelu(dia_conv1, name='relu_7')

            dia_conv2 = self.dilation_conv(input_tensor=relu_7, k_size=3, out_dims=256, rate=4,
                                           padding='SAME', use_bias=False, name='dia_conv_2')
            relu_8 = self.lrelu(dia_conv2, name='relu_8')

            dia_conv3 = self.dilation_conv(input_tensor=relu_8, k_size=3, out_dims=256, rate=8,
                                           padding='SAME', use_bias=False, name='dia_conv_3')
            relu_9 = self.lrelu(dia_conv3, name='relu_9')

            dia_conv4 = self.dilation_conv(input_tensor=relu_9, k_size=3, out_dims=256, rate=16,
                                           padding='SAME', use_bias=False, name='dia_conv_4')
            relu_10 = self.lrelu(dia_conv4, name='relu_10')

            conv_7 = self.conv2d(inputdata=relu_10, out_channel=256, kernel_size=3,
                                 padding='SAME', stride=1, use_bias=False,
                                 name='conv_7')
            relu_11 = self.lrelu(inputdata=conv_7, name='relu_11')

            conv_8 = self.conv2d(inputdata=relu_11, out_channel=256, kernel_size=3,
                                 padding='SAME', stride=1, use_bias=False,
                                 name='conv_8')
            relu_12 = self.lrelu(inputdata=conv_8, name='relu_12')

            deconv_1 = self.deconv2d(inputdata=relu_12, out_channel=128, kernel_size=4,
                                     stride=2, padding='SAME', use_bias=False, name='deconv_1')
            avg_pool_1 = self.avgpooling(inputdata=deconv_1, kernel_size=2, stride=1, padding='SAME',
                                         name='avg_pool_1')
            relu_13 = self.lrelu(inputdata=avg_pool_1, name='relu_13')

            conv_9 = self.conv2d(inputdata=tf.add(relu_13, relu_3), out_channel=128, kernel_size=3,
                                 padding='SAME', stride=1, use_bias=False,
                                 name='conv_9')
            relu_14 = self.lrelu(inputdata=conv_9, name='relu_14')

            deconv_2 = self.deconv2d(inputdata=relu_14, out_channel=64, kernel_size=4,
                                     stride=2, padding='SAME', use_bias=False, name='deconv_2')
            avg_pool_2 = self.avgpooling(inputdata=deconv_2, kernel_size=2, stride=1, padding='SAME',
                                         name='avg_pool_2')
            relu_15 = self.lrelu(inputdata=avg_pool_2, name='relu_15')

            conv_10 = self.conv2d(inputdata=tf.add(relu_15, relu_1), out_channel=32, kernel_size=3,
                                  padding='SAME', stride=1, use_bias=False,
                                  name='conv_10')
            relu_16 = self.lrelu(inputdata=conv_10, name='relu_16')

            skip_output_1 = self.conv2d(inputdata=relu_12, out_channel=3, kernel_size=3,
                                        padding='SAME', stride=1, use_bias=False,
                                        name='skip_ouput_1')

            skip_output_2 = self.conv2d(inputdata=relu_14, out_channel=3, kernel_size=3,
                                        padding='SAME', stride=1, use_bias=False,
                                        name='skip_output_2')

            skip_output_3 = self.conv2d(inputdata=relu_16, out_channel=3, kernel_size=3,
                                        padding='SAME', stride=1, use_bias=False,
                                        name='skip_output_3')

            # 传统GAN输出层都使用tanh函数激活
            skip_output_3 = tf.nn.tanh(skip_output_3, name='skip_output_3_tanh')

            ret = {
                'skip_1': skip_output_1,
                'skip_2': skip_output_2,
                'skip_3': skip_output_3
            }

        return ret

    def compute_autoencoder_loss(self, input_tensor, label_tensor, name):

        [_, ori_height, ori_width, _] = label_tensor.get_shape().as_list()
        label_tensor_ori = label_tensor
        label_tensor_resize_2 = tf.image.resize_bilinear(images=label_tensor,
                                                         size=(int(ori_height / 2), int(ori_width / 2)))
        label_tensor_resize_4 = tf.image.resize_bilinear(images=label_tensor,
                                                         size=(int(ori_height / 4), int(ori_width / 4)))
        label_list = [label_tensor_resize_4, label_tensor_resize_2, label_tensor_ori]
        lambda_i = [0.6, 0.8, 1.0]
        # 计算lm_loss(见公式(5))
        lm_loss = tf.constant(0.0, tf.float32)
        with tf.variable_scope(name):
            inference_ret = self.build_autoencoder(input_tensor=input_tensor, name='autoencoder_inference')
            output_list = [inference_ret['skip_1'], inference_ret['skip_2'], inference_ret['skip_3']]
            for index, output in enumerate(output_list):
                mse_loss = tf.losses.mean_squared_error(output, label_list[index]) * lambda_i[index]
                lm_loss = tf.add(lm_loss, mse_loss)

            # 计算lp_loss(见公式(6))
            src_vgg_feats = self._vgg_extractor.extract_feats(input_tensor=label_tensor,
                                                              name='vgg_feats',
                                                              reuse=False)
            pred_vgg_feats = self._vgg_extractor.extract_feats(input_tensor=output_list[-1],
                                                               name='vgg_feats',
                                                               reuse=True)

            lp_losses = []
            for index, feats in enumerate(src_vgg_feats):
                lp_losses.append(tf.losses.mean_squared_error(src_vgg_feats[index], pred_vgg_feats[index]))
            lp_loss = tf.reduce_mean(lp_losses)

            loss = tf.add(lm_loss, lp_loss)

        return loss, inference_ret['skip_3']