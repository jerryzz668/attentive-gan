import tensorflow as tf
import tensorflow.contrib.slim as slim

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

        # 修改
        self.act = None
        self._eps = 1.1e-5  # epsilon
        self.res_scale = 1  # scaling factor of res block
        self.n_res_blocks = 2  # number of residual block

    def _init_phase(self):

        return tf.equal(self._phase, self._train_phase)

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
    """
    # residual dilated conv block(RDCB)
    def RDCB(self, input_tensor, name):
        output = None
        dilate_c = []
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            for i in range(3):
                if i == 0:
                    conv_1 = self.conv2d(inputdata=input_tensor,out_channel=32,kernel_size=3,padding='SAME',stride=1,
                                          use_bias=False,name='block_{:d}_conv_1'.format(i))
                    relu_1 = self.lrelu(inputdata=conv_1, name='block_{:d}_relu_1'.format(i + 1))
                    for i in range(1, 4):
                        d1 = self.leakyRelu(slim.conv2d(relu_1, 32, 3, 1, rate=i, activation_fn=None, scope='d1'))
                        d2 = self.leakyRelu(slim.conv2d(d1, 32, 3, 1, rate=i, activation_fn=None, scope='d2'))
                        dilate_c.append(d2)
                    add = tf.add_n(dilate_c)
                    output = add
                    input_tensor = output
                else:
                    conv_1 = self.conv2d(inputdata=input_tensor,out_channel=32,kernel_size=1,padding='SAME',stride=1,
                                          use_bias=False,name='block_{:d}_conv_1'.format(i))
                    relu_1 = self.lrelu(inputdata=conv_1, name='block_{:d}_conv_1'.format(i + 1))
                    for i in range(1, 4):
                        d1 = self.leakyRelu(slim.conv2d(relu_1, 32, 3, 1, rate=i, activation_fn=None, scope='d1'))
                        d2 = self.leakyRelu(slim.conv2d(d1, 32, 3, 1, rate=i, activation_fn=None, scope='d2'))
                        dilate_c.append(d2)
                    add = tf.add_n(dilate_c)

                    output = self.lrelu(inputdata=tf.add(add, input_tensor), name='block_{:d}_add'.format(i))
                    input_tensor = output
        return output
    """
    # multi-scale aggregation and enhancement block(MAEB)
    def MAEB(self, input_tensor, scope_name):
        '''MAEB: multi-scale aggregation and enhancement block
            Params:
                input_x: input data
                scope_name: the scope name of the MAEB (customer definition)
                dilated_factor: the maximum number of dilated factors(default=3, range from 1 to 3)

            Return:
                return the output the MAEB

            Input shape:
                4D tensor with shape '(batch_size, height, width, channels)'

            Output shape:
                4D tensor with shape '(batch_size, height, width, channels)'
        '''
        dilate_c = []
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            for i in range(1, 4):
                d1 = self.leakyRelu(
                    slim.conv2d(input_tensor, 32, 3, 1, rate=i, activation_fn=None, scope='d1'))
                d2 = self.leakyRelu(slim.conv2d(d1, 32, 3, 1, rate=i, activation_fn=None, scope='d2'))
                dilate_c.append(d2)

            add = tf.add_n(dilate_c)
            shape = add.get_shape().as_list()

            output = self.SEBlock(add, shape[-1], reduce_dim=int(shape[-1] / 4))
            return output

    # global average pooling
    def globalAvgPool2D(self, input_x):
        global_avgpool2d = tf.contrib.keras.layers.GlobalAvgPool2D()
        return global_avgpool2d(input_x)

    # leaky relu
    def leakyRelu(self, input_x):
        leaky_relu = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)
        return leaky_relu(input_x)

    # squeeze-and-excitation block
    def SEBlock(self, input_x, input_dim=32, reduce_dim=8, scope='SEBlock'):
        with tf.variable_scope(scope) as scope:
            # global scale  out_dim=1024, use_bias=False, name='fc_1'activation_fn=tf.nn.relu
            global_pl = self.globalAvgPool2D(input_x)
            reduce_fc1 = self.fullyconnect(global_pl, reduce_dim, use_bias=False, name='reduce_fc_1')
            reduce_fc1 = self.leakyRelu(reduce_fc1)
            reduce_fc2 = self.fullyconnect(reduce_fc1, input_dim, use_bias=False, name='reduce_fc_2')
            g_scale = tf.nn.sigmoid(reduce_fc2)
            g_scale = tf.expand_dims(g_scale, axis=1)
            g_scale = tf.expand_dims(g_scale, axis=1)
            gs_input = input_x * g_scale
            return gs_input

    def adaptive_global_average_pool_2d(self, x):
        """
        In the paper, using gap which output size is 1, so i just gap func :)
        :param x: 4d-tensor, (batch_size, height, width, channel)
        :return: 4d-tensor, (batch_size, 1, 1, channel)
        """
        c = x.get_shape()[-1]
        return tf.reshape(tf.reduce_mean(x, axis=[1, 2]), (-1, 1, 1, c))

    def channel_attention(self, x,out_channel, name):
        """
        Channel Attention (CA) Layer
        :param x: input layer
        :param f: conv2d filter size
        :param reduction: conv2d filter reduction rate
        :param name: scope name
        :return: output layer
        """
        with tf.variable_scope("CA-%s" % name, reuse=tf.AUTO_REUSE):
            skip_conn = tf.identity(x, name='identity')

            x = self.adaptive_global_average_pool_2d(x)
            # conv_tmp1 = slim.conv2d(local_shortcut, self.channel_dim, 3, 1)
            # x = slim.conv2d(x, f=f // reduction, k=1, name="conv2d-1")
            x = slim.conv2d(x, out_channel, 3, 1)
            # x = self.act(x)

            x = slim.conv2d(x, out_channel, 3, 1)
            x = tf.nn.sigmoid(x)
            return tf.multiply(skip_conn, x)

    def residual_channel_attention_block(self, x, out_channel, use_bn, name):
        with tf.variable_scope("RCAB-%s" % name, reuse=tf.AUTO_REUSE):
            skip_conn = tf.identity(x, name='identity')

            x = slim.conv2d(x, out_channel, 3, 1)
            x = tf.layers.BatchNormalization(epsilon=self._eps, name="bn-1")(x) if use_bn else x
            # x = self.act(x)

            x = slim.conv2d(x, out_channel, 3, 1)
            x = tf.layers.BatchNormalization(epsilon=self._eps, name="bn-2")(x) if use_bn else x

            x = self.channel_attention(x, out_channel, name="RCAB-%s" % name)
            return self.res_scale * x + skip_conn  # tf.math.add(self.res_scale * x, skip_conn)

    def residual_group(self, x, use_bn, name):
        with tf.variable_scope("RG-%s" % name):
            skip_conn = tf.identity(x, name='identity')

            for i in range(self.n_res_blocks):
                x = self.residual_channel_attention_block(x, use_bn, name=str(i))

            x = slim.conv2d(x, 32, 3, 1)
            return x + skip_conn, x + skip_conn  # tf.math.add(x, skip_conn)

    # GRU with convolutional version
    # def convGRU(self, input_x, h, scope='convGRU'):
    def _conv_lstm(self, input_tensor, input_cell_state, name):
        with tf.variable_scope(name):
            if input_cell_state is None:
                self.conv_xz = self.conv2d(input_tensor, 32, 3, 1, name='conv_xz')
                self.conv_xn = self.conv2d(input_tensor, 32, 3, 1, name='conv_xn')
                z = tf.nn.sigmoid(self.conv_xz)
                f = tf.nn.tanh(self.conv_xn)
                input_cell_state = z * f
            else:
                self.conv_hz = self.conv2d(input_cell_state, 32, 3, padding='SAME', stride=1, name='conv_hz')
                self.conv_hr = self.conv2d(input_cell_state, 32, 3, padding='SAME', stride=1, name='conv_hr')

                self.conv_xz = self.conv2d(input_tensor, 32, 3, padding='SAME', stride=1, name='conv_xz')
                self.conv_xr = self.conv2d(input_tensor, 32, 3, padding='SAME', stride=1, name='conv_xr')
                self.conv_xn = self.conv2d(input_tensor, 32, 3, padding='SAME', stride=1, name='conv_xn')
                r = tf.nn.sigmoid(self.conv_hr + self.conv_xr)
                z = tf.nn.sigmoid(self.conv_hz + self.conv_xz)

                self.conv_hn = self.conv2d(r * input_cell_state, 32, 3, padding='SAME', stride=1, name='conv_hn')
                n = tf.nn.tanh(self.conv_xn + self.conv_hn)
                input_cell_state = (1 - z) * input_cell_state + z * n

            se = self.SEBlock(input_cell_state)
            h = self.leakyRelu(se)
            # h = self.residual_channel_attention_block(h, use_bn=True, name='scope')  # 修改

            attention_map = self.conv2d(inputdata=h, out_channel=1, kernel_size=3, padding='SAME',
                                        stride=1, use_bias=False, name='attention_map')
            attention_map = self.sigmoid(inputdata=attention_map)

            ret = {
                'attention_map': attention_map,
                'cell_state': self.conv_xn + self.conv_hn,
                'lstm_feats': h
            }

            return ret, h
        # shared channel attention block
        # se = self.SEBlock(h, 32, reduce_dim=int(32 / 4))
        # h = self.leakyRelu(se)
        # return h, h

    def MAEB1(self, input_x, out_channel, scope_name, dilated_factors=3):
        '''MAEB: multi-scale aggregation and enhancement block
            Params:
                input_x: input data
                scope_name: the scope name of the MAEB (customer definition)
                dilated_factor: the maximum number of dilated factors(default=3, range from 1 to 3)

            Return:
                return the output the MAEB

            Input shape:
                4D tensor with shape '(batch_size, height, width, channels)'

            Output shape:
                4D tensor with shape '(batch_size, height, width, channels)'
        '''
        dilate_c = []
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            for i in range(1, dilated_factors + 1):
                d1 = self.leakyRelu(
                    slim.conv2d(input_x, out_channel, 3, 1, rate=i, activation_fn=None, scope='d1'))
                d2 = self.leakyRelu(slim.conv2d(d1, out_channel, 3, 1, rate=i, activation_fn=None, scope='d2'))
                dilate_c.append(d2)

            add = tf.add_n(dilate_c)
            shape = add.get_shape().as_list()
            output = self.SEBlock(add, shape[-1], reduce_dim=int(shape[-1] / 4))
            # add = self.residual_channel_attention_block(add, out_channel, use_bn=True, name='scope')  # 修改
            return output

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

                #conv_feats = self.MAEB1(attention_input, out_channel=4, scope_name='d3')
                #conv_feats = self.residual_channel_attention_block(attention_input, 4, True, name="RCAB-%s" % name)
                conv_feats = self._residual_block(input_tensor=attention_input,name='residual_block_{:d}'.format(i + 1))
                #conv_feats = self.RDCB(input_tensor=attention_input,name='residual_block_{:d}'.format(i + 1))
                lstm_ret, _ = self._conv_lstm(input_tensor=conv_feats, input_cell_state=init_cell_state,
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
                # mse_loss = tf.pow(0.8, n - index + 1) * \
                #            tf.losses.mean_squared_error(labels=label_tensor,
                #                                         predictions=attention_map)
                ssim_loss = tf.image.ssim(label_tensor, attention_map, 255)  # 改动
                lm1_loss = tf.reduce_mean(ssim_loss)  # 改动
                lm1_loss = -tf.log(lm1_loss)  # 改动
                loss = tf.add(loss, lm1_loss)  # 改动
                # loss = tf.add(loss, mse_loss)

        return loss, inference_ret['final_attention_map']

    def build_autoencoder(self, input_tensor, name):
        [batch_size, tensor_h, tensor_w, _] = input_tensor.get_shape().as_list()
        with tf.variable_scope(name):
            init_cell_state = tf.constant(0.0, dtype=tf.float32,
                                          shape=[batch_size, tensor_h, tensor_w, 32])

            res1 = self._residual_block(input_tensor=input_tensor, name='residual_block1')
            #conv GRU1
            _, GRU1 = self._conv_lstm(input_tensor=res1, input_cell_state=init_cell_state, name='conv_lstm_block1')

            res2 = self._residual_block(input_tensor=GRU1, name='residual_block2')
            # conv GRU2
            _, GRU2 = self._conv_lstm(input_tensor=res2, input_cell_state=init_cell_state, name='conv_lstm_block2')

            res3 = self._residual_block(input_tensor=GRU2, name='residual_block3')
            # conv GRU3
            _, GRU3 = self._conv_lstm(input_tensor=res3, input_cell_state=init_cell_state, name='conv_lstm_block3')

            # skip1
            MAEB1 = self.MAEB1(GRU2, out_channel=32, scope_name='d1')
            conv11 = slim.conv2d(tf.add(GRU3, MAEB1), 32, 3, 1, activation_fn=tf.nn.relu, scope='conv11')
            # conv GRU4
            _, GRU4 = self._conv_lstm(input_tensor=conv11, input_cell_state=init_cell_state, name='conv_lstm_block4')

            # skip2
            MAEB2 = self.MAEB1(GRU1, out_channel=32, scope_name='d2')
            conv22 = slim.conv2d(tf.add(GRU4, MAEB2), 32, 3, 1, activation_fn=tf.nn.relu, scope='conv22')
            # conv GRU5
            _, GRU5 = self._conv_lstm(input_tensor=conv22, input_cell_state=init_cell_state, name='conv_lstm_block5')


            skip_output_1 = self.conv2d(inputdata=GRU3, out_channel=3, kernel_size=3,
                                        padding='SAME', stride=1, use_bias=False,
                                        name='skip_ouput_1')

            skip_output_2 = self.conv2d(inputdata=GRU4, out_channel=3, kernel_size=3,
                                        padding='SAME', stride=1, use_bias=False,
                                        name='skip_output_2')

            skip_output_3 = self.conv2d(inputdata=GRU5, out_channel=3, kernel_size=3,
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
        #label_list = [label_tensor_resize_4, label_tensor_resize_2, label_tensor_ori]
        label_list = [label_tensor_ori, label_tensor_ori, label_tensor_ori]
        #lambda_i = [0.6, 0.8, 1.0]
        lambda_i = [1.0, 1.0, 1.0]

        # 计算lm_loss(见公式(5))
        lm_loss = tf.constant(0.0, tf.float32)
        with tf.variable_scope(name):
            inference_ret = self.build_autoencoder(input_tensor=input_tensor, name='autoencoder_inference')
            output_list = [inference_ret['skip_1'], inference_ret['skip_2'], inference_ret['skip_3']]
            for index, output in enumerate(output_list):
                mse_loss = tf.losses.mean_squared_error(output, label_list[index]) * lambda_i[index]
                # lssim = -tf.log(tf.reduce_mean(tf.image.ssim(output, label_list[index], 255))) * lambda_i[index] # 改动
                l1 = tf.reduce_mean(tf.abs(output - label_list[index]) * lambda_i[index])
                # l_cross_entropy = (tf.nn.sigmoid_cross_entropy_with_logits(labels=label_list[index],logits=output))* lambda_j[index]#改动
                # l_cross_entropy = tf.reduce_mean(l_cross_entropy)#改动
                # lm_loss = tf.add(lm_loss,l_cross_entropy)#改动
                lm_loss = tf.add(lm_loss, l1)
                lm_loss = tf.add(lm_loss, mse_loss)
                # lm_loss = tf.add(lm_loss, lssim)

            # 计算lp_loss(见公式(6))
            src_vgg_feats = self._vgg_extractor.extract_feats(input_tensor=label_tensor,
                                                              name='vgg_feats',
                                                              reuse=False)
            pred_vgg_feats = self._vgg_extractor.extract_feats(input_tensor=output_list[-1],
                                                               name='vgg_feats',
                                                               reuse=True)
            # SSIM loss
            ssim_loss = tf.image.ssim(label_tensor, output_list[-1], 255)  # 改动
            lm1_loss = tf.reduce_mean(ssim_loss)  # 改动
            lm1_loss = -tf.log(lm1_loss)

            # L2 loss
            #final_mse_loss = tf.losses.mean_squared_error(label_tensor, output_list[-1])
            #final_mse_loss = 0.01 * final_mse_loss

            # PSNR loss
            #psnr_finer_tensor = tf.reduce_mean(tf.image.psnr(label_tensor, output_list[-1], 255))
            #psnr_loss = 1.0 / (psnr_finer_tensor + 1e-3)
            #psnr_loss = 0.1 * psnr_loss

            lp_losses = []
            for index, feats in enumerate(src_vgg_feats):
                lp_losses.append(tf.losses.mean_squared_error(src_vgg_feats[index], pred_vgg_feats[index]))

            lp_loss = tf.reduce_mean(lp_losses)
            lp_loss = tf.add(lm1_loss, lp_loss)  # 改动
            loss = tf.add(lm_loss, lp_loss)
            #loss = tf.add(loss, psnr_loss)
            # loss = tf.add(lm_loss, lm1_loss)

        return loss, inference_ret['skip_3']