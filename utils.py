import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec

def normalize_image(image):
    return cv2.normalize(image, None, 0.1, 0.9, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

def pre_process(image):
    image = normalize_image(image)
    image = cv2.resize(image, (48, 48))
    return image

def random_translate(image):
    rows, cols = image.shape[:2]
    ratio = np.random.uniform(0.9, 1.1)
    trans_M = np.float32([[1,0,ratio],[0,1,ratio]])
    dst = cv2.warpAffine(image,trans_M,(cols,rows))
    return dst


def random_rotate(image):
    rows, cols = image.shape[:2]
    rot_angle = np.random.randint(-15,15)
    rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),rot_angle,1)
    dst = cv2.warpAffine(image,rot_M,(cols,rows))
    return dst


def random_scale(image):
    rows, cols = image.shape[:2]
    ratio = np.random.uniform(0.9, 1.1)
    size = np.array((rows*ratio,cols*ratio), dtype=np.float32).astype(np.int32)
    size = tuple(size)

    dst = cv2.resize(image, size)
    return dst


def preturb_image(image):
    dst = random_translate(image)
    dst = random_scale(dst)
    dst = random_rotate(dst)
    dst = cv2.resize(dst,(48, 48))
    dst = normalize_image(dst)
    return dst


def preturb_images(images):
    out_images = np.zeros(shape=[images.shape[0], 48, 48, 3])
    for i,image in enumerate(images):
        out_images[i,:,:,:] = preturb_image(image)
    return out_images


def batch_generator(images, labels, batch_count=None,batch_size=50):
    imshape = images.shape
    idxs = np.arange(imshape[0])
    if not batch_count:
        batch_count = 1
    else:
        pass
    for i in range(batch_count):
        batch_idx = np.random.choice(idxs, size=batch_size, replace=False)
        batch_images = images[batch_idx]
        batch_images = preturb_images(batch_images)
        batch_labels = labels[batch_idx]
        batch_images = flatten_images(batch_images)
        yield batch_images, batch_labels


####Tensorflow functions####

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def fc_layer(input_layer,
            num_inputs,
            num_outputs,
            use_relu=True):

    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input_layer, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def conv_layer(input_layer,
               num_input_channels,
               kernel_size,
               num_filters,
               padding = 'VALID',
               use_pooling=True):

    shape = [kernel_size, kernel_size, num_input_channels, num_filters]

    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input_layer,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding=padding)

    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding=padding)
    layer = tf.nn.relu(layer)
    return layer


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


#### Misc Functions ###


def get_label_map(f_name, labels):
    mapper = np.genfromtxt(f_name, delimiter=',', usecols=(1,), dtype=str, skip_header=1)
    crapper = np.vectorize(lambda x : mapper[x])
    return crapper(labels)


def get_unique_images(images, labels):
    class_id, idx = np.unique(labels, return_index=True)
    return images[idx], class_id


def group(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return np.asarray((unique, counts)).T


def get_label_map(f_name, labels):
    mapper = np.genfromtxt(f_name, delimiter=',', usecols=(1,), dtype=str, skip_header=1)
    crapper = np.vectorize(lambda x : mapper[x])
    return crapper(labels)


def plot_class_frequencies(labels):
    freqs = group(labels)
    plt.figure(figsize=(15,5))
    plt.bar(freqs[:,0], freqs[:,1])
    plt.xlabel('ClassID')
    plt.ylabel('Frequency')
    ind = np.arange(0.5,43.5)
    plt.xticks(ind, get_label_map('signnames.csv', np.unique(labels)),  ha='right', rotation=45)
    plt.show()
    

def get_unique_images(images, labels):
    class_id, idx = np.unique(labels, return_index=True)
    return images[idx], class_id


def dense_to_one_hot(labels, n_classes=2):
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot


def get_accuracy(predictions, acutals):
    prediction_labels = np.argmax(predictions, axis=1)
    actual_labels = np.argmax(acutals, axis=1)
    return np.mean(np.equal(prediction_labels, actual_labels).astype(np.float32))


def flatten_images(images):
    imshape = images.shape
    features_count = imshape[1]*imshape[2]*imshape[3]
    return images.reshape(-1, features_count)


def plot_images(images, labels):
    gs = GridSpec(6, 7)
    gs.update(wspace=0.03, hspace=0.03) # set the spacing between axes. 
    fig = plt.figure(figsize=(12,12))
    image_titles = get_label_map('signnames.csv', labels)
    for i in range(len(images)-1):
        ax = fig.add_subplot(gs[i])
        img = images[i]
        ax.imshow(img)
        ax.set_aspect('equal')
        ax.set_title(image_titles[i],fontsize=7)
        plt.axis('off')
    gs.tight_layout(fig)
    plt.show()