# coding: utf-8

import math
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tf_utils import new_fc_layer, new_conv_layer, flatten_layer
from image_augmentor import batch_generator


def train_cv_split(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        random_state=42,
                                                        test_size=0.1)
    return X_train, y_train, X_test, y_test


def one_hot_encod(train, cv, test):
    encod = LabelBinarizer()
    encod.fit(train)
    return encod.transform(train), encod.transform(cv), encod.transform(test)


def cvt_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def normalize_image(image):
    return cv2.normalize(image, None, 0.0, 1.0,
                         cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def predict_loss_accuracy(input_layer, labels):
    predictions = tf.nn.softmax(input_layer)

    cross_entorpy = tf.nn.softmax_cross_entropy_with_logits(logits=input_layer,
                                                            labels=labels)
    loss = tf.reduce_mean(cross_entorpy)

    is_correct_prediction = tf.equal(tf.argmax(predictions, 1),
                                     tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))
    return predictions, loss, accuracy


if __name__ == '__main__':

    training_file = 'train.p'
    testing_file = 'test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']

    X_train, y_train, X_test, y_test = (np.array(X_train),
                                        np.array(y_train),
                                        np.array(X_test),
                                        np.array(y_test)
                                        )

    X_train = np.array([normalize_image(image) for image in X_train],
                       dtype=np.float32)
    X_test = np.array([normalize_image(image) for image in X_test],
                      dtype=np.float32)

    X_train = np.array([cv2.resize(image, (48, 48)) for image in X_train])

    X_test = np.array([cv2.resize(image, (48, 48)) for image in X_test])
    X_train, y_train, X_valid, y_valid = train_cv_split(X_train, y_train)

    train_labels, valid_labels, test_labels = one_hot_encod(y_train,
                                                            y_valid,
                                                            y_test
                                                            )

    features_count = 48*48*3
    labels_count = 43

    features = tf.placeholder(tf.float32, shape=[None, 48, 48, 3])
    features_flat = tf.reshape(features, shape=[-1, features_count])
    labels = tf.placeholder(tf.float32, shape=[None, labels_count])

    drop_probs = tf.placeholder(tf.float32)

    fc_layer_5_count = 300
    batch_size = 50
    epochs = 50

    layer_1, _ = new_conv_layer(input=features,
                                num_input_channels=3,
                                filter_size=7,
                                num_filters=100,
                                use_pooling=True
                                )
    layer_2, _ = new_conv_layer(input=layer_1,
                                num_input_channels=100,
                                filter_size=4,
                                num_filters=150,
                                use_pooling=True
                                )
    layer_3, _ = new_conv_layer(input=layer_2,
                                num_input_channels=150,
                                filter_size=4,
                                num_filters=250,
                                use_pooling=True
                                )
    layer_3_flat, flat_features_count = flatten_layer(layer_3)

    layer_4 = new_fc_layer(input=layer_3_flat,
                           num_inputs=flat_features_count,
                           num_outputs=fc_layer_5_count,
                           use_relu=True
                           )
    layer_4 = tf.nn.dropout(layer_4, keep_prob=drop_probs)

    layer_5 = new_fc_layer(input=layer_4,
                           num_inputs=fc_layer_5_count,
                           num_outputs=labels_count,
                           use_relu=False
                           )

    predictions, loss, accuracy = predict_loss_accuracy(input_layer=layer_5,
                                                        labels=labels
                                                        )
    train_feed_dict = {features: X_train.astype(np.float32),
                       labels: train_labels.astype(np.float32),
                       drop_probs: 0.5
                       }
    valid_feed_dict = {features: X_valid.astype(np.float32),
                       labels: valid_labels.astype(np.float32),
                       drop_probs: 1.0
                       }
    test_feed_dict = {features: X_test.astype(np.float32),
                      labels: test_labels.astype(np.float32),
                      drop_probs: 1.0
                      }
    optimizer = tf.train.AdamOptimizer(5e-4).minimize(loss)
    # optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        print('Initialized')
        batch_count = int(math.ceil(len(X_train)/batch_size))
        for epoch in range(epochs):
            print("Started epoch: {}".format(epoch))
            for batch_features, batch_labels in batch_generator(X_train,
                                                                train_labels,
                                                                batch_count,
                                                                batch_size):
                feed_dict = {features: batch_features,
                             labels: batch_labels,
                             drop_probs: 0.5
                             }
                _, l = sess.run([optimizer, loss], feed_dict=feed_dict)
            print("Finished epoch: {}".format(epoch))
            # training_accuracy = sess.run(accuracy, feed_dict=train_feed_dict)
        validation_accuracy = sess.run(accuracy, feed_dict=valid_feed_dict)
        test_accuracy = sess.run(accuracy, feed_dict=test_feed_dict)
        # print("Training accuracy: {}".format(training_accuracy))
        print("Validation accuracy: {}".format(validation_accuracy))
        print("Test accuracy: {}".format(test_accuracy))
