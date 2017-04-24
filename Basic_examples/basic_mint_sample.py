from sklearn import svm
from sklearn.metrics import accuracy_score

import utils as ut

# Load training data
# ,digits=[3,8,5,6,9])
training_images, training_labels = ut.load_mnist('training', path=ut.PATH_DATASET)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(training_images)
dataImage = training_images.reshape((n_samples, -1))

n_labels = len(training_labels)
dataLabels = training_labels.reshape((n_labels, -1))
# show(images[1])

x_data = dataImage[:1000]
y_data = dataLabels[:1000]

# Applying classification
classifier = svm.SVC(gamma=0.0001, C=100)
classifier.fit(x_data, y_data.ravel())

predict = classifier.predict(dataImage[900])
print('result label:', predict)
# get the accuracy
print('Accuracy score:', accuracy_score(dataLabels[900], predict) * 100, '%')

ut.show(training_images[900], predict)
