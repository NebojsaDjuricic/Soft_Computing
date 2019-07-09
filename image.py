
import cv2
from keras.datasets import mnist
from pracenje_brojeva import Broj
import numpy as np


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)


#pomeranje slike u gornji levi ugao
def move(image,x,y):
    img = np.zeros((28,28))
    img[:(28-x),:(28-y)] = image[x:,y:]

    return img


# dopunjavanje slike do 28x28
def fill(image):
    if(np.shape(image)!=(28,28)):
        img = np.zeros((28,28))
        x = 28 - np.shape(image)[0]
        y = 28 - np.shape(image)[1]
        img[:-x,:-y] = image
        return img
    else:
        return image


def select_roi(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28.
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika

        if w < 7 and h < 7:
            continue

        else:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return sorted_regions


def select_roi(bin_img):

    _, contours, _ = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    konture = []

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)

        if h > 7:
            konture.append(Broj((x, y, w, h)))

    return konture


def KNN():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_train, x_test = x_train / 255.0, x_test / 255.0

    knn = cv2.ml.KNearest_create()
    knn.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

    return knn
