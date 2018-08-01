"""
Unsupervised adversarial domain adaptation
"""

import tensorflow as tf
import numpy as np
import os
import cv2
# from matplotlib import pyplot as plt
import time


class DANN:

    def __init__(self):

        self.batch_size = 128
        # self.img_size = (300, 400)
        self.window_size = [1, 32, 32, 1]
        self.stride = [1, 16, 16, 1]
        self.isbuilt = False
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session()
        self.graph = tf.get_default_graph()

    def build_train(self):

        if self.isbuilt == False:

            self.img_batch = tf.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3], name="input")

            self.class_labels = tf.placeholder(
                dtype=tf.float32, shape=[None, 1], name="class_labels")
            self.domain_labels = tf.placeholder(
                dtype=tf.float32, shape=[None, 1], name="domain_labels")
            self.istraining = tf.placeholder(dtype=tf.bool, name="istraining")

            self.features = self.feature_extractor()
            self.class_output = self.classifier(self.features)
            self.domain_output = self.discriminator(self.features)

            # placeholder for domain labels
            self.domain_labels = tf.placeholder(
                dtype=tf.float32, shape=[None, 1])

            self.trainable_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope="feature_extractor")
            self.trainable_vars.extend(
                tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope="classifier"))
            self.trainable_vars.extend(
                tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator"))

            # print(self.trainable_vars)
            self.isbuilt = True

    def build_predict(self):

        if self.isbuilt == False:
            self.img_batch = tf.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3], name="input")
            self.istraining = tf.placeholder(dtype=tf.bool, name="istraining")
            self.features = self.feature_extractor()
            self.class_output = self.classifier(self.features)

            self.isbuild = True

    def feature_extractor(self, reuse=None):

        n_out = np.array([self.window_size[1], self.window_size[2]])
        #self.final_stride = np.array([self.stride[1], self.stride[2]])

        with tf.variable_scope("feature_extractor", reuse=reuse):
            initializer = tf.truncated_normal_initializer(
                mean=0, stddev=0.01, dtype=tf.float32)
            zero_initializer = tf.initializers.zeros()

            x = tf.layers.conv2d(
                self.img_batch,
                kernel_size=3,
                filters=4,
                strides=1,
                padding='VALID',
                kernel_initializer=initializer,
                bias_initializer=zero_initializer,
                name='conv1')

            n_out = (n_out - 3) + 1
            j_out = 1
            r_out = 1 + (3 - 1) * 1
            start_out = 0.5 + (3 - 1) / 2  # 1.5

            x = tf.nn.leaky_relu(x, alpha=0.2)
            #x = tf.layers.batch_normalization(x, axis=-1, training=self.istraining)
            x = tf.layers.dropout(x, rate=0.2, training=self.istraining)
            x = tf.nn.max_pool(
                x,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID',
                name='pool1')
            print(x)

            n_out = (n_out - 2) // 2 + 1
            j_out = 2
            r_out = r_out + (2 - 1) * 1
            start_out = start_out + 0.5  # 2

            # this feature map will be used to train the discriminator
            x = tf.layers.conv2d(
                x,
                kernel_size=3,
                filters=8,
                strides=1,
                padding='VALID',
                kernel_initializer=initializer,
                bias_initializer=zero_initializer,
                name='conv2')

            n_out = (n_out - 3) + 1
            j_out = 2
            r_out = r_out + (3 - 1) * 2
            start_out = start_out + ((3 - 1) / 2) * 2  # 4

            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, rate=0.2, training=self.istraining)
            x = tf.nn.max_pool(
                x,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID',
                name='pool2')
            print(x)

            n_out = (n_out - 2) // 2 + 1
            j_out = 4
            r_out = r_out + (2 - 1) * 2
            start_out = start_out + 0.5 * 2  # 5

            x = tf.layers.conv2d(
                x,
                kernel_size=3,
                filters=16,
                strides=1,
                padding='VALID',
                kernel_initializer=initializer,
                bias_initializer=zero_initializer,
                name='conv3')

            n_out = (n_out - 3) + 1
            j_out = 4
            r_out = r_out + (3 - 1) * 4
            start_out = start_out + 4  # 9

            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, rate=0.2, training=self.istraining)
            x = tf.nn.max_pool(
                x,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID',
                name='pool3')
            print(x)

            n_out = (n_out - 2) // 2 + 1
            j_out = 8
            r_out = r_out + (2 - 1) * 4
            start_out = start_out + 0.5 * 4  # 11

            print(n_out, r_out, start_out)
            #
            # print(x)

            return x

    def classifier(self, x):

        with tf.variable_scope('classifier'):
            initializer = tf.truncated_normal_initializer(
                mean=0, stddev=0.01, dtype=tf.float32)
            zero_initializer = tf.initializers.zeros()

            # weighted sum of all the filter activations, channel = 1
            x = tf.layers.conv2d(
                x,
                kernel_size=2,
                filters=1,
                strides=1,
                padding='VALID',
                kernel_initializer=initializer,
                bias_initializer=zero_initializer,
                name='conv1')

            # x = tf.nn.leaky_relu(x, alpha=0.2)
            # # x = tf.layers.batch_normalization(x, axis=-1, training=self.istraining)
            # x = tf.layers.dropout(x, rate=0.2, training=self.istraining)
            #
            # x = tf.layers.conv2d(x, kernel_size=3, filters=1, strides=3, padding='VALID',
            #                      kernel_initializer=initializer,
            #                      bias_initializer=zero_initializer,
            #                      name='classifier_layer')

            # uncomment when training
            #x = tf.reshape(x, (-1, 1))
            print("output ", x)
            return x

    def discriminator(self, x):

        with tf.variable_scope("discriminator"):
            initializer = tf.truncated_normal_initializer(
                mean=0, stddev=0.01, dtype=tf.float32)
            zero_initializer = tf.initializers.zeros()
            #regularizer =

            x = tf.layers.conv2d(
                x,
                kernel_size=2,
                filters=1,
                strides=1,
                padding='VALID',
                kernel_initializer=initializer,
                bias_initializer=zero_initializer,
                name='conv1')

            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, rate=0.2, training=self.istraining)
            print(x)

            # x = tf.layers.conv2d(x, kernel_size=2, filters=1, strides=2, padding='VALID',
            #                      kernel_initializer=initializer,
            #                      bias_initializer=zero_initializer,
            #                      name='conv2')
            x = tf.reshape(x, (-1, 1))
            print("discriminator ", x)
            print(x)

            return x

    def class_loss(self, class_logit, class_label):
        return 0.75 * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=class_label, logits=class_logit))

    def domain_loss(self, domain_logit, domain_label):
        beta = 0.5
        return beta * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=domain_label, logits=domain_logit))

    def compute_accuracy(self, logits, labels):

        correct_prediction = tf.equal(tf.round(tf.sigmoid(logits)), labels)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def prepare_dataset(self, train_val_split=0.8):

        with tf.name_scope('train_dataset'):
            # load the source data
            source_img_arr = np.load('small_target_imgs.npy').astype(
                np.float32) / 255
            source_label_arr = np.load('small_target_labels.npy').astype(
                np.float32)
            source_label_arr = np.reshape(source_label_arr, (-1, 1))

            print(source_label_arr.shape)
            print(np.count_nonzero(source_label_arr))

            # train-val split
            source_train_size = int(source_img_arr.shape[0] * train_val_split)
            source_val_size = source_img_arr.shape[0] - source_train_size

            source_train_imgs, source_val_imgs = tf.split(
                source_img_arr, [source_train_size, source_val_size], 0)
            source_train_labels, source_val_labels = tf.split(
                source_label_arr, [source_train_size, source_val_size], 0)

            print(source_train_size, source_val_size)

            # create Dataset objects
            train_class_dataset = tf.data.Dataset.from_tensor_slices(
                (source_train_imgs,
                 source_train_labels)).shuffle(source_train_size).batch(
                     self.batch_size)

            val_class_dataset = tf.data.Dataset.from_tensor_slices(
                (source_val_imgs,
                 source_val_labels)).shuffle(source_val_size).batch(
                     self.batch_size)

            # create TensorFlow Iterator object
            train_iterator = tf.data.Iterator.from_structure(
                train_class_dataset.output_types,
                train_class_dataset.output_shapes)

            val_iterator = tf.data.Iterator.from_structure(
                val_class_dataset.output_types, val_class_dataset.output_shapes)

            # iterator initializer ops
            self.class_training_init_op = train_iterator.make_initializer(
                train_class_dataset)
            self.class_val_init_op = val_iterator.make_initializer(
                val_class_dataset)

            # iterator get_next ops
            self.class_next_train_elem = train_iterator.get_next()
            self.class_next_val_elem = val_iterator.get_next()

        with tf.name_scope('domain_dataset'):
            # load the target data
            target_img_arr = np.load('small_gopro_img_arr.npy').astype(
                np.float32) / 255
            print(target_img_arr.shape)

            # train-val split
            target_train_size = int(target_img_arr.shape[0] * train_val_split)
            target_val_size = target_img_arr.shape[0] - target_train_size
            target_train_imgs, target_val_imgs = tf.split(
                target_img_arr, [target_train_size, target_val_size], 0)

            print(target_train_size, target_val_size)

            source_labels = np.zeros((source_train_imgs.shape[0], 1))
            target_labels = np.ones((target_train_imgs.shape[0], 1))

            train_domain_dataset = tf.data.Dataset.from_tensor_slices(
                (source_train_imgs[0:1000], source_labels[0:1000]))
            train_domain_dataset = train_domain_dataset.concatenate(
                tf.data.Dataset.from_tensor_slices(
                    (target_train_imgs, target_labels
                    ))).shuffle(target_train_size + source_train_size).batch(
                        self.batch_size)

            source_labels = np.zeros((source_val_imgs.shape[0], 1))
            target_labels = np.ones((target_val_imgs.shape[0], 1))

            val_domain_dataset = tf.data.Dataset.from_tensor_slices(
                (source_val_imgs, source_labels))
            val_domain_dataset = val_domain_dataset.concatenate(
                tf.data.Dataset.from_tensor_slices(
                    (target_val_imgs, target_labels
                    ))).shuffle(target_val_size + source_val_size).batch(
                        self.batch_size)

            # create TensorFlow Iterator object
            train_iterator = tf.data.Iterator.from_structure(
                train_domain_dataset.output_types,
                train_domain_dataset.output_shapes)

            val_iterator = tf.data.Iterator.from_structure(
                val_domain_dataset.output_types,
                val_domain_dataset.output_shapes)

            # iterator initializer ops
            self.domain_training_init_op = train_iterator.make_initializer(
                train_domain_dataset)
            self.domain_val_init_op = val_iterator.make_initializer(
                val_domain_dataset)

            # iterator get_next ops
            self.domain_next_train_elem = train_iterator.get_next()
            self.domain_next_val_elem = val_iterator.get_next()

    def predict(self, input_image):

        # find equivalent window size and stride
        input_image = input_image[np.newaxis, :, :, :]
        #feature_map = self.sess.run(self.features, feed_dict={self.img_batch: input_image, self.istraining: False})
        out = self.sess.run(
            tf.sigmoid(self.class_output),
            feed_dict={
                self.img_batch: input_image,
                self.istraining: False
            })
        return out

    def train(self, epochs, restore=False):

        # load data
        self.prepare_dataset()

        # loss ops
        class_loss = self.class_loss(self.class_output, self.class_labels)
        domain_loss = self.domain_loss(self.domain_output, self.domain_labels)

        # accuracy ops
        class_accuracy = self.compute_accuracy(self.class_output,
                                               self.class_labels)
        domain_accuracy = self.compute_accuracy(self.domain_output,
                                                self.domain_labels)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.00005, beta1=0.9, beta2=0.999, epsilon=1e-08)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):

            class_grads = optimizer.compute_gradients(class_loss)
            train_class_step = optimizer.apply_gradients(class_grads)

            # minimize the domain loss for the discriminator
            domain_disc_grads = optimizer.compute_gradients(
                loss=domain_loss,
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator"))
            train_domain_disc_step = optimizer.apply_gradients(
                domain_disc_grads)

            # maximize the domain loss for the feature extractor
            domain_conv_grads = optimizer.compute_gradients(
                loss=-domain_loss,
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope="feature_extractor"))
            train_domain_conv_step = optimizer.apply_gradients(
                domain_conv_grads)

            train_domain_step = tf.group(train_domain_conv_step,
                                         train_domain_disc_step)

        with self.sess as sess:

            # maximum 10 latest models are saved.
            saver = tf.train.Saver(max_to_keep=4, var_list=self.trainable_vars)
            # writer = tf.summary.FileWriter("dann_logs/", sess.graph)
            if restore == True:
                saver.restore(self.sess, 'dann_ckpts/dann.ckpt-10')
                sess.run(tf.variables_initializer(optimizer.variables()))

            else:
                sess.run(tf.global_variables_initializer())

            sess.run(self.domain_training_init_op)
            sess.run(self.class_training_init_op)
            sess.run(self.domain_val_init_op)
            sess.run(self.class_val_init_op)

            merged_summaries = tf.summary.merge_all()

            e = 0
            i_c = 0
            i_d = 0

            # reset metrics
            classifier_train_loss = 0.0
            classifier_val_loss = 0.0
            discriminator_train_loss = 0.0
            discriminator_val_loss = 0.0

            classifier_train_acc = 0.0
            classifier_val_acc = 0.0
            discriminator_train_acc = 0.0
            discriminator_val_acc = 0.0

            while e < epochs:

                # begin with one iteration of classifier optimization
                try:
                    # alternate with discriminator optimization
                    train_elem = sess.run(self.domain_next_train_elem)
                    loss, acc, _ = sess.run(
                        [domain_loss, domain_accuracy, train_domain_step],
                        feed_dict={
                            self.img_batch: train_elem[0],
                            self.domain_labels: train_elem[1],
                            self.istraining: True
                        })

                    # print("epoch/domain_iter: ", e, i_d, loss, acc)
                    discriminator_train_loss += loss
                    discriminator_train_acc += acc
                    i_d += 1

                except tf.errors.OutOfRangeError:

                    discriminator_train_loss /= i_d
                    discriminator_train_acc /= i_d
                    print("epoch domain: ", e, discriminator_train_loss,
                          discriminator_train_acc)
                    sess.run(self.domain_training_init_op)
                    i_d = 0
                    discriminator_train_loss = 0
                    discriminator_train_acc = 0

                try:
                    train_elem = sess.run(self.class_next_train_elem)
                    loss, acc, _ = sess.run(
                        [class_loss, class_accuracy, train_class_step],
                        feed_dict={
                            self.img_batch: train_elem[0],
                            self.class_labels: train_elem[1],
                            self.istraining: True
                        })

                    # print("epoch/class_iter: ", e, i_c, loss, acc)
                    classifier_train_loss += loss
                    classifier_train_acc += acc
                    # increment iteration count
                    i_c += 1

                except tf.errors.OutOfRangeError:
                    e += 1
                    # save model
                    saver.save(
                        sess,
                        os.path.join(os.getcwd(), 'fast_detection.ckpt'),
                        global_step=e,
                        write_meta_graph=False)

                    classifier_train_loss /= i_c
                    classifier_train_acc /= i_c
                    print("epoch class: ", e, classifier_train_loss,
                          classifier_train_acc)
                    sess.run(self.class_training_init_op)
                    i_c = 0
                    classifier_train_acc = 0
                    classifier_train_loss = 0

                    # validate classifier
                    try:
                        val_elem = sess.run(self.class_next_val_elem)
                        loss, acc = sess.run(
                            [class_loss, class_accuracy],
                            feed_dict={
                                self.img_batch: val_elem[0],
                                self.class_labels: val_elem[1],
                                self.istraining: False
                            })

                        print("validation class: ", e, loss, acc)

                    except tf.errors.OutOfRangeError:
                        sess.run(self.class_val_init_op)
                        val_elem = sess.run(self.class_next_val_elem)
                        loss, acc = sess.run(
                            [class_loss, class_accuracy],
                            feed_dict={
                                self.img_batch: val_elem[0],
                                self.class_labels: val_elem[1],
                                self.istraining: False
                            })

                        print("validation class: ", e, loss, acc)

                    # validate discriminator
                    try:
                        val_elem = sess.run(self.domain_next_val_elem)
                        loss, acc = sess.run(
                            [domain_loss, domain_accuracy],
                            feed_dict={
                                self.img_batch: val_elem[0],
                                self.domain_labels: val_elem[1],
                                self.istraining: False
                            })

                        print("validation domain: ", e, loss, acc)

                    except tf.errors.OutOfRangeError:
                        sess.run(self.domain_val_init_op)
                        val_elem = sess.run(self.domain_next_val_elem)
                        loss, acc = sess.run(
                            [domain_loss, domain_accuracy],
                            feed_dict={
                                self.img_batch: val_elem[0],
                                self.domain_labels: val_elem[1],
                                self.istraining: False
                            })

                        print("validation domain: ", e, loss, acc)


def test_detector():
    dann = DANN()
    dann.build()
    #dann.train(200, restore=False)

    saver = tf.train.Saver()
    saver.restore(dann.sess, '../fast_detection_ckpts/fast_detection.ckpt-98')

    from matplotlib import pyplot as plt

    test_img_path = "../resized.jpeg"
    fig = plt.figure()
    img = cv2.imread(test_img_path)
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    # start = time.time()
    # print(start)
    output = dann.predict(img / 255)
    #print(output.shape)
    #end = time.time()
    #print(end)
    #print(end - start)
    rsize = 18
    start = 9

    output = cv2.resize(output[0, :, :, 0], (img.shape[1], img.shape[0]),
                        cv2.INTER_NEAREST)
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(output[0, :, :, 0])
    # plt.show()

    map = np.uint8(np.round(output))
    # plt.imshow(map)
    # plt.show()

    map, cnt, heir = cv2.findContours(map, cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_SIMPLE)

    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)
        if w > 200 or h > 200:
            # definitely not a target
            continue
        crop = img[y:y + h, x:x + w, :]
        center = np.int16(np.mean(c, axis=0).flatten())
        print(center)
        img = cv2.circle(
            img, (center[0], center[1]),
            radius=5,
            color=[0, 0, 255],
            thickness=-1)
        plt.imshow(img)
        plt.show()

    #end = time.time()
    # print(end)
    # time = end - start
    print(time)
