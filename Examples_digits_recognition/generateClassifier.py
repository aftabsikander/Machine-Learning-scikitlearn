from pylab import *
from sklearn import metrics
from sklearn import svm
from sklearn.externals import joblib

import utils as ut

# Load training data and testing data
training_images, training_labels = ut.load_mnist('training')

testing_images, testing_labels = ut.load_mnist('testing')

training_samples_count = len(training_images)
testing_sample_count = len(testing_images)

# a support vector classifier
classifier = svm.LinearSVC()

# using two thirds for training
# ans one third for testing
split_point_training = int(training_samples_count * 0.80)
split_point_test = int(testing_sample_count * 0.66)

# flat training images to 1D array
n_training_samples = len(training_images)
flat_training_Images = training_images.reshape((n_training_samples, -1))

# flat testing images to 1D array
n_testing_samples = len(testing_images)
flat_testing_Images = testing_images.reshape((n_testing_samples, -1))

#
labels_learn = training_labels[:split_point_training]
data_learn = flat_training_Images[:split_point_training]

labels_test = testing_labels[split_point_test:]
data_test = flat_testing_Images[split_point_test:]

# normalize the images [0 to 1]
data_learn /= 255
data_test /= 255

print("Training: " + str(len(labels_learn)) + " Test: " + str(len(labels_test)))

# Learning Phase
classifier.fit(data_learn, labels_learn)

# Predict Test Set
predicted = classifier.predict(data_test)

# classification report
print("Classification report for classifier %s:\n%s" %
      (classifier, metrics.classification_report(labels_test, predicted))
      )

# confusion matrix
print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels_test, predicted))

# Save the classifier
joblib.dump(classifier, "digits_classifier.pkl", compress=3)

# print
print('\nclassifier file created')
