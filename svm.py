import cv2
import numpy as np

# pokusaj sa SVM
def deskew(img, SZ):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11'] / m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def SVM(trainData, trainLabels, testData):
    # Set up SVM for OpenCV 3
    svm = cv2.ml.SVM_create()
    # Set SVM type
    svm.setType(cv2.ml.SVM_C_SVC)
    # Set SVM Kernel to Radial Basis Function (RBF)
    svm.setKernel(cv2.ml.SVM_RBF)
    # Set parameter C
    svm.setC(12.5)
    # Set parameter Gamma controls the stretching of data in the third dimension.
    # It helps in classification but it also distorts the data
    svm.setGamma(0.50625)

    # Train SVM on training data
    svm.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)

    # Save trained model
    model = svm.save("digits_svm_model.yml")

    # Test on a held out test set
    testResponse = svm.predict(testData)[1].ravel()

    return model


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