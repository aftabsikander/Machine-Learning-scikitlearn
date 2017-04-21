import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
clf = svm.SVC(gamma=0.0001, C=100)
print(len(digits.data))
print(digits.data[1].shape)
x, y = digits.data[:-1], digits.target[:-1]
print(digits.data[-6])
clf.fit(x, y)

print('prediction:', clf.predict(digits.data[-9]))
plt.imshow(digits.images[-9], cmap=plt.cm.gray, interpolation="nearest")
plt.show()
