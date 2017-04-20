import struct
import os
from array import array as pyarray

from numpy import *
from pylab import *
from sklearn import svm


def read(dataset="training", path="."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)


def show(image):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = plt.imshow(image, cmap=plt.cm.gray)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()


def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = 't10k-images.idx3-ubyte'  # os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = 't10k-labels.idx1-ubyte'  # os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


# Load training data
# ,digits=[3,8,5,6,9])
training_images, training_labels = load_mnist('training',
                                              path='/media/afali/Project/Projects/Projects/Python-workspace/hello-scikitlearn/Dataset/')

# print(training_images[1])
# print(training_labels[1])

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(training_images)
dataImage = training_images.reshape((n_samples, -1))

n_labels = len(training_labels)
dataLabels = training_labels.reshape((n_labels, -1))
# show(images[1])

x_data = dataImage[:5000]
y_data = dataLabels[:5000]
print(x_data[1].shape)
print(y_data[1].shape)
# Applying classification
classifier = svm.SVC(gamma=0.0001, C=100)
classifier.fit(x_data, y_data)

predict = classifier.predict(training_images[6000])
print('result label:', predict)
show(training_images[6000])

""" 
#Working code
training_data = list(read(dataset='training',
                          path='/media/afali/Project/Projects/Projects/Python-workspace/hello-scikitlearn/Dataset/'))
training_label, training_pixels = training_data[3]
print(training_label)
print(training_pixels.shape)
"""

"""
testing_data = list(
    read(dataset='testing',
         path='/media/afali/Project/Projects/Projects/Python-workspace/hello-scikitlearn/Dataset/'))
testing_label, testing_pixels = testing_data[1]
print(testing_label)
print(testing_pixels.shape)

clf = svm.SVC(gamma=0.0001, C=100)
clf.fit(training_label, training_pixels)
predict = clf.predict(testing_pixels)
print('result label:', predict)
"""

# show(testing_pixels)
