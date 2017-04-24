# Import the modules
import argparse as ap

import numpy as np
from scipy import misc
from sklearn.externals import joblib

import utils as ut


def convert_image_classifier(convert_image):
    n_training_samples = len(convert_image)
    image = []
    for eachRow in convert_image:
        for eachPixel in eachRow:
            image.append(eachPixel)

    image = np.array(image)
    flat_training_images = image.reshape((n_training_samples, -1))

    # normalize data
    # mat /= 255

    print(flat_training_images)
    print(flat_training_images.shape)

    return image


parser = ap.ArgumentParser()
# parser.add_argument("-c", "--classifierLocation", help="Location of Classifier File", required="True")
# parser.add_argument("-i", "--image", help="Path to Image", required="True")
args = vars(parser.parse_args())

clf = joblib.load(ut.PATH_CLASSIFIER + "digits_classifier.pkl")  # args["classifierLocation"]

file_path = ut.PATH_TEST_IMAGES

img = ut.load_image_locally(file_path + 'sample_2.png')
img = misc.imresize(img, (28, 28))

# now convert input image to match feature image
# matrix i.e (64 pixels max) meaning 8x8 image
image_data = np.array(img).astype(float)

# normalize data
image_data /= 255

predictedImage = convert_image_classifier(image_data)

predict = clf.predict(predictedImage)

print(predict)
if len(predict) >= 2:
    ut.show(img, predict[0])
else:
    ut.show(img, predict)
