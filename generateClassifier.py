import struct
from array import array as pyarray
import os
from numpy import *
from pylab import *
from sklearn import svm
from sklearn.externals import joblib

PATH_DATASET = '/media/afali/Project/Projects/Projects/Python-workspace/hello-scikitlearn/Dataset/'


def load_mnist(dataset="training", digits=np.arange(10), path=PATH_DATASET):
    """
    Loads MNIST files into 3D numpy arrays
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
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

    return images, labels, rows, cols


# Load training data
training_images, training_labels, rows, cols = load_mnist('training')

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(training_images)
dataImage = training_images.reshape((n_samples, -1))

n_labels = len(training_labels)
dataLabels = training_labels.reshape((n_labels, -1))

x_data = dataImage[:-1]
y_data = dataLabels[:-1]

# Create an SVC object
classifier = svm.SVC(gamma=0.0001, C=100)

# Perform the training
classifier.fit(x_data, y_data.ravel())

# Save the classifier
joblib.dump((classifier, rows, cols), "digits_classifier.pkl", compress=3)

# print
print('classifier file created')
