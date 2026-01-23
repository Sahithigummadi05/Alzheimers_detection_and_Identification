import numpy as np
from scipy import linalg
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.image import smooth_img
import cv2
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split

n_subjects = 416

oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
gray_matter_map_filenames = oasis_dataset.gray_matter_maps
gm_imgs = gray_matter_map_filenames

cdr = oasis_dataset.ext_vars['cdr'].astype(float)
cdr_numpy_arr = np.array(cdr)

for i in range(len(cdr_numpy_arr)):
    if np.isnan(cdr_numpy_arr[i]):
        cdr_numpy_arr[i] = 1
    elif cdr_numpy_arr[i] > 0.0:
        cdr_numpy_arr[i] = 1

imgArr = []

for imgUrl in gray_matter_map_filenames:
    result_img = smooth_img(imgUrl, fwhm=1)
    imgArr.append(result_img.get_data())

x_train = []
x_test = []
y_train = []
y_test = []

rshapedImgArr = []

for img in imgArr:
    newImg = [cv2.resize(each_slice, (50, 50)) for each_slice in img]
    newImg = np.array(newImg)
    rshapedImgArr.append(newImg)

label = keras.utils.to_categorical(cdr_numpy_arr, 2)

much_data = []

for num, img in enumerate(rshapedImgArr):
    much_data.append([img, label[num]])

IMG_SIZE_PX_X = 50
IMG_SIZE_PX_Y = 50
SLICE_COUNT = 91

n_classes = 2
batch_size = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8

def conv3d(x, W):
    conv = tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')
    conv = tf.nn.dropout(conv, 0.5)
    return conv

def maxpool3d(x):
    return tf.nn.max_pool3d(
        x,
        ksize=[1, 2, 2, 2, 1],
        strides=[1, 2, 2, 2, 1],
        padding='SAME'
    )

def convolutional_neural_network(x):
    weights = {
        'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
        'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
        'W_fc': tf.Variable(tf.random_normal([248768, 1024])),
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    x = tf.reshape(x, [-1, IMG_SIZE_PX_X, IMG_SIZE_PX_Y, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2, [-1, 248768])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']
    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=prediction,
            labels=y
        )
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    file = open("output.txt", "w")

    hm_epochs = 1000

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        successful_runs = 0
        total_runs = 0

        for epoch in range(hm_epochs):
            epoch_loss = 0
            train_data, validation_data = train_test_split(much_data, train_size=0.8)

            for data in train_data:
                total_runs += 1
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run(
                        [optimizer, cost],
                        feed_dict={x: X, y: Y}
                    )
                    epoch_loss += c
                    successful_runs += 1
                except:
                    pass

            print("Epoch", epoch + 1, "loss:", epoch_loss)
            file.write(str(epoch + 1) + " " + str(epoch_loss) + "\n")

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            acc_val = accuracy.eval({
                x: [i[0] for i in validation_data],
                y: [i[1] for i in validation_data]
            })

            print("Accuracy:", acc_val)
            file.write("Accuracy: " + str(acc_val) + "\n")

        final_acc = accuracy.eval({
            x: [i[0] for i in validation_data],
            y: [i[1] for i in validation_data]
        })

        print("Final Accuracy:", final_acc)
        print("Fitment percent:", successful_runs / total_runs)

        file.write("Final Accuracy: " + str(final_acc) + "\n")
        file.write("Fitment percent: " + str(successful_runs / total_runs))

    file.close()

train_neural_network(x)
