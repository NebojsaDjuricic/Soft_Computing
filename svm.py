import cv2
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from keras.datasets import mnist

from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_mldata


def svm_model():
    # print("Loading dataset...")
    # mndata = mnist.load_data()
    # images, labels = mndata.data, mndata.target
    #
    # clf = LinearSVC()
    #
    # # Train on the first 10000 images:
    # train_x = images[:10000]
    # train_y = labels[:10000]
    #
    # print("Train model")
    # clf.fit(train_x, train_y)

    mnist = fetch_mldata('MNIST original', data_home='./')

    mnist.keys()

    images = mnist.data
    targets = mnist.target

    X_data = images / 255.0
    Y = targets

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.15, random_state=42)

    ################ Classifier with good params ###########
    # Create a classifier: a support vector classifier

    param_C = 5
    param_gamma = 0.05
    classifier = svm.SVC(C=param_C, gamma=param_gamma)

    classifier.fit(X_train, y_train)

    return classifier



# # Import datasets, classifiers and performance metrics
# from sklearn import datasets, svm, metrics
#
# # The digits dataset
# digits = datasets.load_digits()
#
# # The data that we are interested in is made of 8x8 images of digits, let's
# # have a look at the first 4 images, stored in the `images` attribute of the
# # dataset.  If we were working from image files, we could load them using
# # matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# # images, we know which digit they represent: it is given in the 'target' of
# # the dataset.
# images_and_labels = list(zip(digits.images, digits.target))
# for index, (image, label) in enumerate(images_and_labels[:4]):
#     plt.subplot(2, 4, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Training: %i' % label)
#
# # To apply a classifier on this data, we need to flatten the image, to
# # turn the data in a (samples, feature) matrix:
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))
#
# # Create a classifier: a support vector classifier
# classifier = svm.SVC(gamma=0.001)
#
# # We learn the digits on the first half of the digits
# classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
#
# # Now predict the value of the digit on the second half:
# expected = digits.target[n_samples // 2:]
# predicted = classifier.predict(data[n_samples // 2:])

# pokusaj sa SVM
def SVM():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_train, x_test = x_train / 255.0, x_test / 255.0

    clf = svm.SVC(kernel='rbf', gamma=0.50625, C=12.5)      # gamma=0.001, C=100

    clf.fit(x_train, y_train)

    # svm_nn = cv2.ml.SVM_create()
    # svm_nn.setType(cv2.ml.SVM_C_SVC)
    # svm_nn.setKernel(cv2.ml.SVM_RBF)
    # svm_nn.setC(12.5)
    # # Set parameter Gamma controls the stretching of data in the third dimension.
    # # It helps in classification but it also distorts the data
    # svm_nn.setGamma(0.50625)
    #
    # svm_nn.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

    return clf


def HOG():
    winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                            derivAperture, winSigma, histogramNormType,
                            L2HysThreshold, gammaCorrection, nlevels, signedGradients)

    return hog

# descriptor = hog.compute(frame)