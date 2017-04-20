# Machine-Learning-scikitlearn
Various example and experiment playground for scikit learn using python. We would be using [Mnist Dataset](http://yann.lecun.com/exdb/mnist/)

# Understanding Yann LeCun's MNIST IDX file format
**File format:**
Each file has 1000 training examples. Each training example is of size 28x28 pixels. The pixels are stored as unsigned chars (1 byte) and take values from 0 to 255. The first 28x28 bytes of the file correspond to the first training example, the next 28x28 bytes correspond to the next example and so on. [For more detail](http://stackoverflow.com/questions/39969045/parsing-yann-lecuns-mnist-idx-file-format)

## Helper Utility Script
Following are few helper scripts which can be used to convert mnist dataset into viewable images, or create our very own mnist formatted dataset using custom images.

- [Convert mnist data into PNG image](https://github.com/myleott/mnist_png)

- [Create custom mnist format using own images](https://github.com/gskielian/JPG-PNG-to-MNIST-NN-Format)
