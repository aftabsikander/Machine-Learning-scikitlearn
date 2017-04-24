import struct
from array import array as pyarray
import os
import matplotlib.pyplot as plt
from numpy import *
from scipy import misc

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
PATH_DATASET = os.path.join(PROJECT_ROOT, 'Dataset')
PATH_TEST_IMAGES = os.path.join(PROJECT_ROOT, 'Test', '')
CURRENT_WORKING_DIRECTORY = os.getcwd()
PATH_CLASSIFIER = os.path.join(CURRENT_WORKING_DIRECTORY, '')


def show(image):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = plt.imshow(image, cmap=plt.cm.gray)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.show()


def show(image, prediction):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = plt.imshow(image, cmap=plt.cm.gray)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.title('Prediction: %i' % prediction)
    plt.show()


def load_mnist(dataset="training", digits=arange(10), path=PATH_DATASET):
    """
    Loads MNIST files into 3D numpy arrays
    """
    print(path)
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

    images = zeros((N, rows, cols), dtype=float_)
    labels = zeros((N, 1), dtype=float_)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape(rows, cols)
        labels[i] = lbl[ind[i]]

    return images, labels


def load_image_locally(filepath):
    # Read the input image
    # Load an color image in grayscale
    # load image from test folder
    img = misc.imread(filepath, 'rb', 'L')

    return img


# show image ascii
def ascii_show(image):
    for eachRow in image:
        row = ""
        for eachPixels in eachRow:
            row += '{0: <4}'.format(eachPixels)
        print(row)
