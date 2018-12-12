import config
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import time
import numpy as np
from caps_layers import create_prim_conv3d_caps, create_dense_caps, layer_shape, create_conv3d_caps


def create_skip_connection(in_caps_layer, n_units, kernel_size, strides=(1, 1, 1), padding='VALID', name='skip',
                           activation=tf.nn.relu):
    in_caps_layer = in_caps_layer[0]
    batch_size = tf.shape(in_caps_layer)[0]
    _, d, h, w, ch, _ = in_caps_layer.get_shape()
    d, h, w, ch = map(int, [d, h, w, ch])

    in_caps_res = tf.reshape(in_caps_layer, [batch_size, d, h, w, ch * 16])

    return tf.layers.conv3d_transpose(in_caps_res, n_units, kernel_size=kernel_size, strides=strides, padding=padding,
                                      use_bias=False, activation=activation, name=name)


class Caps3d(object):
    def __init__(self,  input_shape=(None, 8, 112, 112, 3)):
        self.input_shape = input_shape
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.init_weights()

            # inputs to the network
            self.x_input = tf.placeholder(dtype=tf.float32, shape=self.input_shape)
            self.y_input = tf.placeholder(dtype=tf.int32, shape=[None])
            self.y_bbox = tf.placeholder(dtype=tf.float32, shape=(None, 8, 112, 112, 1))
            self.is_train = tf.placeholder(tf.bool)
            self.m = tf.placeholder(tf.float32, shape=())

            # initializes the network
            self.init_network()

            # initializes the loss
            self.cur_m = config.start_m
            self.init_loss_and_opt()

            # initializes the saver
            self.saver = tf.train.Saver()

    def init_weights(self):
        if config.use_c3d_weights:
            reader = pywrap_tensorflow.NewCheckpointReader('./c3d_pretrained/conv3d_deepnetA_sport1m_iter_1900000_TF.model')
            self.w_and_b = {
                'wc1': tf.constant_initializer(reader.get_tensor('var_name/wc1')),
                'wc2': tf.constant_initializer(reader.get_tensor('var_name/wc2')),
                'wc3a': tf.constant_initializer(reader.get_tensor('var_name/wc3a')),
                'wc3b': tf.constant_initializer(reader.get_tensor('var_name/wc3b')),
                'wc4a': tf.constant_initializer(reader.get_tensor('var_name/wc4a')),
                'wc4b': tf.constant_initializer(reader.get_tensor('var_name/wc4b')),
                'wc5a': tf.constant_initializer(reader.get_tensor('var_name/wc5a')),
                'wc5b': tf.constant_initializer(reader.get_tensor('var_name/wc5b')),
                'bc1': tf.constant_initializer(reader.get_tensor('var_name/bc1')),
                'bc2': tf.constant_initializer(reader.get_tensor('var_name/bc2')),
                'bc3a': tf.constant_initializer(reader.get_tensor('var_name/bc3a')),
                'bc3b': tf.constant_initializer(reader.get_tensor('var_name/bc3b')),
                'bc4a': tf.constant_initializer(reader.get_tensor('var_name/bc4a')),
                'bc4b': tf.constant_initializer(reader.get_tensor('var_name/bc4b')),
                'bc5a': tf.constant_initializer(reader.get_tensor('var_name/bc5a')),
                'bc5b': tf.constant_initializer(reader.get_tensor('var_name/bc5b'))
            }
        else:
            self.w_and_b = {
                'wc1': None,
                'wc2': None,
                'wc3a': None,
                'wc3b': None,
                'wc4a': None,
                'wc4b': None,
                'wc5a': None,
                'wc5b': None,
                'bc1': tf.zeros_initializer(),
                'bc2': tf.zeros_initializer(),
                'bc3a': tf.zeros_initializer(),
                'bc3b': tf.zeros_initializer(),
                'bc4a': tf.zeros_initializer(),
                'bc4b': tf.zeros_initializer(),
                'bc5a': tf.zeros_initializer(),
                'bc5b': tf.zeros_initializer()
            }

    def init_network(self):
        print('Building Caps3d Model')

        # creates the video encoder
        conv1 = tf.layers.conv3d(self.x_input, 64, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                                 activation=tf.nn.relu, kernel_initializer=self.w_and_b['wc1'],
                                 bias_initializer=self.w_and_b['bc1'], name='conv1')

        conv2 = tf.layers.conv3d(conv1, 128, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 2, 2],
                                 activation=tf.nn.relu, kernel_initializer=self.w_and_b['wc2'],
                                 bias_initializer=self.w_and_b['bc2'], name='conv2')

        conv3 = tf.layers.conv3d(conv2, 256, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                                 activation=tf.nn.relu, kernel_initializer=self.w_and_b['wc3a'],
                                 bias_initializer=self.w_and_b['bc3a'], name='conv3')

        conv4 = tf.layers.conv3d(conv3, 256, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 2, 2],
                                 activation=tf.nn.relu, kernel_initializer=self.w_and_b['wc3b'],
                                 bias_initializer=self.w_and_b['bc3b'], name='conv4')

        conv5 = tf.layers.conv3d(conv4, 512, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                                 activation=tf.nn.relu, kernel_initializer=self.w_and_b['wc4a'],
                                 bias_initializer=self.w_and_b['bc4a'], name='conv5')

        conv6 = tf.layers.conv3d(conv5, 512, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                                 activation=tf.nn.relu, kernel_initializer=self.w_and_b['wc4b'],
                                 bias_initializer=self.w_and_b['bc4b'], name='conv6')

        if config.print_layers:
            print('Conv1:', conv1.get_shape())
            print('Conv2:', conv2.get_shape())
            print('Conv3:', conv3.get_shape())
            print('Conv4:', conv4.get_shape())
            print('Conv5:', conv5.get_shape())
            print('Conv6:', conv6.get_shape())

        # creates the primary capsule layer: conv caps1
        prim_caps = create_prim_conv3d_caps(conv6, 32, kernel_size=[3, 9, 9], strides=[1, 1, 1], padding='VALID',
                                            name='prim_caps')

        # creates the secondary capsule layer: conv caps2
        sec_caps = create_conv3d_caps(prim_caps, 32, kernel_size=[3, 5, 5], strides=[1, 2, 2],
                                      padding='VALID', name='sec_caps', route_mean=True)

        # creates the final capsule layer: class caps
        pred_caps = create_dense_caps(sec_caps, config.n_classes, subset_routing=-1, route_min=0.0,
                                      name='pred_caps', coord_add=True, ch_same_w=True)
        if config.print_layers:
            print('Primary Caps:', layer_shape(prim_caps))
            print('Second Caps:', layer_shape(sec_caps))
            print('Prediction Caps:', layer_shape(pred_caps))

        # obtains the activations of the class caps layer and gets the class prediction
        self.digit_preds = tf.reshape(pred_caps[1], (-1, config.n_classes))
        self.predictions = tf.cast(tf.argmax(input=self.digit_preds, axis=1), tf.int32)

        pred_caps_poses = pred_caps[0]
        batch_size = tf.shape(pred_caps_poses)[0]
        _, n_classes, dim = pred_caps_poses.get_shape()
        n_classes, dim = map(int, [n_classes, dim])

        # masks the capsules that are not the ground truth (training) or the prediction (testing)
        vec_to_use = tf.cond(self.is_train, lambda: self.y_input, lambda: self.predictions)
        vec_to_use = tf.one_hot(vec_to_use, depth=n_classes)
        vec_to_use = tf.tile(tf.reshape(vec_to_use, (batch_size, n_classes, 1)), multiples=[1, 1, dim])
        masked_caps = pred_caps_poses * tf.cast(vec_to_use, dtype=tf.float32)
        masked_caps = tf.reshape(masked_caps, (batch_size, n_classes * dim))

        # creates the decoder network
        recon_fc1 = tf.layers.dense(masked_caps, 4 * 8 * 8 * 1, activation=tf.nn.relu, name='recon_fc1')
        recon_fc1 = tf.reshape(recon_fc1, (batch_size, 4, 8, 8, 1))

        deconv1 = tf.layers.conv3d_transpose(recon_fc1, 128, kernel_size=[1, 3, 3], strides=[1, 1, 1],
                                             padding='SAME', use_bias=False, activation=tf.nn.relu, name='deconv1')

        skip_connection1 = create_skip_connection(sec_caps, 128, kernel_size=[1, 3, 3], strides=[1, 1, 1],
                                                  padding='SAME', name='skip_1')
        deconv1 = tf.concat([deconv1, skip_connection1], axis=-1)

        deconv2 = tf.layers.conv3d_transpose(deconv1, 128, kernel_size=[3, 6, 6], strides=[1, 2, 2],
                                             padding='VALID', use_bias=False, activation=tf.nn.relu, name='deconv2')

        skip_connection2 = create_skip_connection(prim_caps, 128, kernel_size=[1, 3, 3], strides=[1, 1, 1],
                                                  padding='SAME', name='skip_2')
        deconv2 = tf.concat([deconv2, skip_connection2], axis=-1)

        deconv3 = tf.layers.conv3d_transpose(deconv2, 256, kernel_size=[3, 9, 9], strides=[1, 1, 1],
                                             padding='VALID',
                                             use_bias=False, activation=tf.nn.relu, name='deconv3')

        deconv4 = tf.layers.conv3d_transpose(deconv3, 256, kernel_size=[1, 3, 3], strides=[1, 2, 2], padding='SAME',
                                             use_bias=False, activation=tf.nn.relu, name='deconv4')

        deconv5 = tf.layers.conv3d_transpose(deconv4, 256, kernel_size=[1, 3, 3], strides=[1, 2, 2], padding='SAME',
                                             use_bias=False, activation=tf.nn.relu, name='deconv5')

        self.segment_layer = tf.layers.conv3d(deconv5, 1, kernel_size=[1, 3, 3], strides=[1, 1, 1],
                                              padding='SAME', activation=None, name='segment_layer')
        self.segment_layer_sig = tf.nn.sigmoid(self.segment_layer)

        if config.print_layers:
            print('Deconv Layer 1:', deconv1.get_shape())
            print('Deconv Layer 2:', deconv2.get_shape())
            print('Deconv Layer 3:', deconv3.get_shape())
            print('Deconv Layer 4:', deconv4.get_shape())
            print('Deconv Layer 5:', deconv5.get_shape())
            print('Segmentation Layer:', self.segment_layer.get_shape())

    def init_loss_and_opt(self):
        y_onehot = tf.one_hot(indices=self.y_input, depth=config.n_classes)

        # get a_t
        a_i = tf.expand_dims(self.digit_preds, axis=1)
        y_onehot2 = tf.expand_dims(y_onehot, axis=2)
        a_t = tf.matmul(a_i, y_onehot2)

        # calculate spread loss
        spread_loss = tf.square(tf.maximum(0.0, self.m - (a_t - a_i)))
        spread_loss = tf.matmul(spread_loss, 1. - y_onehot2)
        self.class_loss = tf.reduce_sum(tf.reduce_sum(spread_loss, axis=[1, 2]))

        # segmentation loss
        segment = tf.contrib.layers.flatten(self.segment_layer)
        y_bbox = tf.contrib.layers.flatten(self.y_bbox)
        self.segmentation_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_bbox, logits=segment))
        self.segmentation_loss = config.segment_coef * self.segmentation_loss

        # accuracy of a given batch
        correct = tf.cast(tf.equal(self.predictions, self.y_input), tf.float32)
        self.tot_correct = tf.reduce_sum(correct)
        self.accuracy = tf.reduce_mean(correct)

        self.total_loss = self.class_loss + self.segmentation_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1, name='Adam',
                                           epsilon=config.epsilon)

        self.train_op = optimizer.minimize(loss=self.total_loss)

    def train(self, sess, data_gen):
        start_time = time.time()
        # continues until no more training data is generated
        losses, batch, acc, s_losses = 0, 0, 0, 0
        while data_gen.has_data():
            x_batch, bbox_batch, y_batch = data_gen.get_batch(config.batch_size)

            # runs network on batch
            _, loss, s_loss, preds = sess.run([self.train_op, self.class_loss, self.segmentation_loss, self.digit_preds],
                                              feed_dict={self.x_input: x_batch, self.y_input: y_batch,
                                                         self.m: self.cur_m, self.is_train: True,
                                                         self.y_bbox: bbox_batch})

            # accumulates loses and accuracies
            acc += np.count_nonzero(np.argmax(preds, axis=1) == np.array(y_batch))/config.batch_size
            losses += loss
            s_losses += s_loss
            batch += 1

            # prints the loss and accuracy statistics after a certain number of batches
            if batch % config.batches_until_print == 0:
                print(preds[0][:10])  # prints activations just in case of numerical instability
                print('Finished %d batches. %d(s) since start. Avg Classification Loss is %.4f. '
                      'Avg Segmentation Loss is %.4f. Accuracy is %.4f.'
                      % (batch, time.time()-start_time, losses / batch, s_losses / batch, acc / batch))

        # prints the loss and accuracy statistics for the entire epoch
        print(preds[0][:10])  # prints activations just in case of numerical instability
        print('Epoch finished in %d(s). Avg Classification loss is %.4f. Avg Segmentation Loss is %.4f. '
              'Accuracy is %.4f.'
              % (time.time() - start_time, losses / batch, s_losses / batch,  acc / batch))

        return losses / batch, s_losses / batch, acc / batch

    def eval(self, sess, data_gen, validation=True):
        mlosses, slosses, corrs = [], [], 0
        conf_matrix = np.zeros((config.n_classes, config.n_classes), dtype=np.int32)
        start_time = time.time()
        batch = 0
        for i in range(data_gen.n_videos):
            video, bbox, label = data_gen.get_next_video()

            # gets losses and predictionfor a single video
            mloss, sloss, pred = self.eval_on_vid(sess, video, bbox, label, validation)

            # accumulates video statistics
            conf_matrix[label, pred] += 1
            mlosses.append(mloss)
            slosses.append(sloss)
            corrs += (1 if pred == label else 0)
            batch += 1

            # print statistics every 500 videos
            if batch % 500 == 0:
                print('Tested %d videos. %d(s) since start. Avg Accuracy is %.4f'
                      % (batch, time.time() - start_time, float(corrs) / batch))

        # print evaluation statistics for all evaluation videos
        print('Evaluation done in %d(s).' % (time.time() - start_time))
        print('Test Classification Loss: %.4f. Test Segmentation Loss: %.4f. Accuracy: %.4f.'
              % (float(np.mean(mlosses)), float(np.mean(slosses)), float(corrs) / data_gen.n_videos))

        return np.mean(mlosses), np.mean(slosses), float(corrs) / data_gen.n_videos, conf_matrix

    def eval_on_vid(self, sess, video, bbox, label, validation):
        losses, slosses, norms = [], [], []
        frames, _, _, _ = video.shape

        # ensures the video is trimmed and separate video into clips of 8 frames
        f_skip = config.frame_skip
        clips = []
        n_frames = video.shape[0]
        for i in range(0, video.shape[0], 8 * f_skip):
            for j in range(f_skip):
                b_vid, b_bbox = [], []
                for k in range(8):
                    ind = i + j + k * f_skip
                    if ind >= n_frames:
                        b_vid.append(np.zeros((1, 112, 112, 3), dtype=np.float32))
                        b_bbox.append(np.zeros((1, 112, 112, 1), dtype=np.float32))
                    else:
                        b_vid.append(video[ind:ind + 1, :, :, :])
                        b_bbox.append(bbox[ind:ind + 1, :, :, :])

                clips.append((np.concatenate(b_vid, axis=0), np.concatenate(b_bbox, axis=0), label))
                if clips[-1][1].sum() == 0:
                    clips.pop(-1)

        if len(clips) == 0:
            print('Video has no bounding boxes')
            return 0, 0, 0, 0, 0

        # groups clips into batches
        batches, gt_segmentations = [], []
        for i in range(0, len(clips), config.batch_size):
            x_batch, bb_batch, y_batch = [], [], []
            for j in range(i, min(i + config.batch_size, len(clips))):
                x, bb, y = clips[j]
                x_batch.append(x)
                bb_batch.append(bb)
                y_batch.append(y)
            batches.append((x_batch, bb_batch, y_batch))
            gt_segmentations.append(np.stack(bb_batch))

        # if doing validation, only do one batch per video
        if validation:
            batches = batches[:1]

        # runs the network on the clips
        n_clips = 0
        for x_batch, bbox_batch, y_batch in batches:
            loss, sloss, norm = sess.run([self.class_loss, self.segmentation_loss, self.digit_preds],
                                         feed_dict={self.x_input: x_batch, self.y_input: y_batch,
                                                    self.m: 0.9, self.is_train: False,
                                                    self.y_bbox: bbox_batch})

            n_clips_in_batch = len(x_batch)
            losses.append(loss * n_clips_in_batch)
            slosses.append(sloss * n_clips_in_batch)
            norms.append(norm)
            n_clips += n_clips_in_batch

        # calculates network prediction
        if len(norms) > 1:
            concat_norms = np.concatenate(norms, axis=0)
        else:
            concat_norms = norms[0]
        norm_mean = np.mean(concat_norms, axis=0)
        pred = np.argmax(norm_mean)

        # gets average losses
        fin_mloss = float(np.sum(losses) / n_clips)
        fin_sloss = float(np.sum(slosses) / n_clips)

        return fin_mloss, fin_sloss, pred

    def save(self, sess, file_name):
        # saves the model
        save_path = self.saver.save(sess, file_name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess, file_name):
        # loads the model
        self.saver.restore(sess, file_name)
        print('Model restored from file: %s' % file_name)

