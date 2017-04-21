# Import the modules
import argparse as ap
import cv2

import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

PATH_CLASSIFIER = '/media/afali/Project/Projects/Projects/Python-workspace/hello-scikitlearn/digits_classifier.pkl'
PATH_TEST_FILES = '/media/afali/Project/Projects/Projects/Python-workspace/hello-scikitlearn/Test/'


def show(image, prediction):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = plt.imshow(image, cmap=plt.cm.gray)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.title('Prediction: %i' % prediction)
    plt.show()


parser = ap.ArgumentParser()
# parser.add_argument("-c", "--classifierLocation", help="Location of Classifier File", required="True")
# parser.add_argument("-i", "--image", help="Path to Image", required="True")
args = vars(parser.parse_args())

# Load the classifier
clf, rows, cols = joblib.load(PATH_CLASSIFIER)  # args["classifierLocation"]
print(rows)
print(cols)
# Read the input image
# Load an color image in grayscale
img = cv2.imread(PATH_TEST_FILES + 'test_2.jpg', 0)
image_data = np.array(img).astype(float)
n_samples = len(image_data)
dataImage = image_data.reshape((n_samples, -1))
print(len(dataImage))
print(dataImage.shape)
predict = clf.predict(dataImage)
print('result label:', predict)
predict = clf.score(dataImage)
show(img, predict[0])
