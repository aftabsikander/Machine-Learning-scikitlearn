import sys

import struct
from scipy.misc import imsave
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

sample_size = 10000000  # sys.maxint # for test

train_label_fn = '/media/afali/Project/Projects/Projects/Python-workspace/hello-scikitlearn/Dataset/train-labels.idx1-ubyte'
train_image_fn = '/media/afali/Project/Projects/Projects/Python-workspace/hello-scikitlearn/Dataset/train-images.idx3-ubyte'
test_label_fn = '/media/afali/Project/Projects/Projects/Python-workspace/hello-scikitlearn/Dataset/t10k-labels.idx1-ubyte'
test_image_fn = '/media/afali/Project/Projects/Projects/Python-workspace/hello-scikitlearn/Dataset/t10k-images.idx3-ubyte'


# read the labels
def read_labels(fn):
    labels = None
    # Load everything in some numpy arrays
    with open(fn, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

        get_img = lambda idx: (lbl[idx])

        # Create an iterator which returns each image in turn
        for i in range(len(lbl)):
            yield get_img(i)

        for i in range(n):
            if i > sample_size: break
            b = f.read(1)
            labels[i] = int(b.encode('hex'), 16)
    return labels


# read the images
def read_images(fn):
    images = None
    with open(fn) as f:
        b = f.read(4)  # magic mumber
        b = f.read(4)  # number of items
        n = int(b.encode('hex'), 16)
        b = f.read(4)  # number of rows
        r = int(b.encode('hex'), 16)
        b = f.read(4)  # number of columns
        c = int(b.encode('hex'), 16)

        if n > sample_size: n = sample_size
        images = np.zeros((n, r * c), dtype=np.uint8)

        for i in range(n):
            if i > sample_size: break
            for j in range(r * c):
                b = f.read(1)
                images[i][j] = int(b.encode('hex'), 16)
    return images


train_labels = read_labels(train_label_fn)
train_images = read_images(train_image_fn)
test_labels = read_labels(test_label_fn)
test_images = read_images(test_image_fn)

# imsave('img/'+str(i)+'.png', train_images[i])

print('train labels:', train_labels.shape)
print('train images', train_images.shape)
print('test labels', test_labels.shape)
print('test images', test_images.shape)

'''
C_range = 2.0 ** np.arange(-2, 2.5, 0.5)
gamma_range = 2.0 ** np.arange(-10, -2, 0.5)
param_rbf_grid = dict(gamma=gamma_range, C=C_range)
rbf = GridSearchCV(SVC(kernel='rbf'), param_grid=param_rbf_grid)
rbf.fit(train_images, train_labels)
predict = rbf.predict(test_images)
print 'result shape:', predict.shape
print predict
print test_labels
print 'correct count', sum(predict == test_labels)

sys.exit(0)
'''

# clf = SVC(kernel="rbf", C=2.0, gamma=.0625)
clf = svm.LinearSVC()
# clf = SVC(kernel="poly", degree=3, C=0.35, coef0=0.125, gamma=0.0625)
clf.fit(train_images, train_labels)
predict = clf.predict(test_images)
print('result shape:', predict.shape)
# print predict
# print test_labels
print('correct count', sum(predict == test_labels))
print(clf.score(test_images, test_labels))
