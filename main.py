
import os
os.environ['THEANO_FLAGS'] = "device=cpu"
import cv2
import numpy as np
from keras.models import load_model
from image import *
from copy import copy, deepcopy
from numpy.linalg import norm
from svm import *
from pracenje_brojeva import Broj, Tracker

# istrenirana neuronska mreza
nn = load_model('neuronska.h5')

videos = ['video-0.avi','video-1.avi','video-2.avi','video-3.avi','video-4.avi','video-5.avi','video-6.avi','video-7.avi','video-8.avi','video-9.avi',]


def nadji_konture(frejm):

    hsv_slika = cv2.cvtColor(frejm, cv2.COLOR_BGR2HSV)
    maska_za_plavu = cv2.inRange(hsv_slika, np.array([110, 200, 50]), np.array([130, 255, 255]))
    maska_za_zelenu = cv2.inRange(hsv_slika, np.array([50, 70, 50]), np.array([120, 255, 255]))

    frejm_copy = deepcopy(frejm)
    frejm_copy[maska_za_plavu > 0] = 0
    frejm_copy[maska_za_zelenu > 0] = 0

    gray_slika = cv2.cvtColor(frejm_copy, cv2.COLOR_BGR2GRAY)
    dilation_slika = cv2.dilate(gray_slika, np.ones((3,3)), iterations=1)

    dilation_slika[dilation_slika >= 128] = 255
    dilation_slika[dilation_slika < 128] = 0

    # cv2.imshow('d', dilation_slika)

    img, contours, hierarchy = cv2.findContours(dilation_slika.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frejm_copy_2 = deepcopy(frejm)
    konture = []

    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        x_centar = x + w // 2
        y_centar = y + h // 2

        if w < 7 and h < 7:
            continue
        else:
            # cv2.rectangle(frejm_copy_2, (x_centar - 15, y_centar - 15), (x_centar + 15, y_centar + 15), (50, 50, 255), 2)
            # cv2.imshow('img', frejm_copy_2)
            konture.append(([x_centar - 15, y_centar - 15], [w, h]))

    return konture


def da_li_je_prosao_broj(linija, broj):
    x, y, w, h = broj.get_granice_broja()
    X = int(x + (w / 2.0))
    Y = int(y + (h / 2.0))
    centar = X, Y
    donji_desni = broj.donji_desni_ugao()
    levo_teme, desno_teme = linija

    if levo_teme[0] < centar[0] < desno_teme[0] and desno_teme[1] < centar[1] < levo_teme[1]:
        tacka1 = np.array(levo_teme)
        tacka2 = np.array(desno_teme)
        tacka3 = np.array(centar)
        # tacka3 = np.array(donji_desni)
        distance = norm(np.cross(tacka2 - tacka1, tacka1 - tacka3)) / norm(tacka2 - tacka1)
        # distance = norm(np.cross(tacka2 - tacka1, tacka1 - tacka3)) / norm(tacka2 - tacka1)   # 17

        if distance < 30:
            return True

    return False

# digits = load_digits()
# X, y = digits.data, digits.target
#
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# knn = kNN()
# knn.fit(X_train, y_train)

print('treniranje modela...')
model = KNN()
# model = SVM()
print('model istreniran')

fajl = open('out.txt','w')
fajl.write('RA 110/2014 Nebojsa Djuricic\n')
fajl.write('file\t\tsum\n')

for video in videos:
    print('\nFajl: ', video)
    cap = cv2.VideoCapture('videos/%s' % (video))
    ret, frame = cap.read()

    frame_plava = frame.copy()
    frame_zelena = frame.copy()

    if not ret:
        print('Video nije ucitan')
        break

    tracker = Tracker()

    # PLAVA LINIJA

    # maska za plavu liniju
    maska_za_plavu = cv2.inRange(frame, np.array([180, 0, 0]), np.array([255, 50, 50]))
    frame_plava = cv2.bitwise_and(frame_plava, frame_plava, mask=maska_za_plavu)
    #cv2.imshow('Plava linija', frame_plava)

    # 76%
    erosion_plava = cv2.erode(frame_plava, np.ones((3, 3), np.uint8), iterations=1)
    dilation_plava = cv2.dilate(erosion_plava, np.ones((3, 3)), iterations=1)
    gray_plava = cv2.cvtColor(dilation_plava, cv2.COLOR_BGR2GRAY)

    # 75%
    # gray_plava = cv2.cvtColor(frame_plava, cv2.COLOR_BGR2GRAY)

    # 66%
    # ret, thresh_plava = cv2.threshold(gray_plava, 0, 255, cv2.THRESH_BINARY)
    # erosion_plava = cv2.erode(thresh_plava, np.ones((3, 3), np.uint8), iterations=1)

    edges_plava = cv2.Canny(gray_plava, 50, 150, apertureSize=3)
    # gblur_plava = cv2.GaussianBlur(edges_plava, (3, 3), 1)
    # cv2.imshow('Plava linija', gblur_plava)
    plava_linija = cv2.HoughLinesP(edges_plava, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=40)

    xmin, ymin, xmax, ymax = [], [], [], []
    for linija in plava_linija:
        for x1, y1, x2, y2 in linija:
            xmin.append(x1)
            ymin.append(y2)
            xmax.append(x2)
            ymax.append(y1)
    xmin_p, ymin_p, xmax_p, ymax_p = min(xmin), min(ymin), max(xmax), max(ymax)

    plava_levo_teme = (xmin_p, ymax_p)      # ymax zato sto y osa ide nadole
    plava_desno_teme = (xmax_p, ymin_p)

    pla_lin = plava_levo_teme, plava_desno_teme

    # ZELENA LINIJA

    # uklanjanje suma posto je zelene boje
    frame_zelena = cv2.erode(frame_zelena, np.ones((3,3)), iterations=1)
    frame_zelena = cv2.dilate(frame_zelena, np.ones((3, 3)), iterations=1)

    # maska za zelenu liniju
    maska_za_zelenu = cv2.inRange(frame, np.array([0, 180, 0]), np.array([80, 255, 80]))
    frame_zelena = cv2.bitwise_and(frame_zelena, frame_zelena, mask=maska_za_zelenu)
    #cv2.imshow('Zelena linija', frame_zelena)

    erosion_zelena = cv2.erode(frame_zelena, np.ones((3, 3), np.uint8), iterations=1)
    dilation_zelena = cv2.dilate(erosion_zelena, np.ones((3,3)), iterations=1)
    gray_zelena = cv2.cvtColor(dilation_zelena, cv2.COLOR_BGR2GRAY)

    # gray_zelena = cv2.cvtColor(frame_zelena, cv2.COLOR_BGR2GRAY)

    # ret, thresh_zelena = cv2.threshold(gray_zelena, 0, 255, cv2.THRESH_BINARY)
    # erosion_zelena = cv2.erode(thresh_zelena, np.ones((3, 3), np.uint8), iterations=1)

    edges_zelena = cv2.Canny(gray_zelena, 50, 150, apertureSize=3)
    # gblur_zelena = cv2.GaussianBlur(edges_zelena, (3, 3), 1)
    # cv2.imshow('Zelena linija', edges_zelena)
    zelena_linija = cv2.HoughLinesP(edges_zelena, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=40)

    xmin, ymin, xmax, ymax = [], [], [], []
    for linija in zelena_linija:
        for x1, y1, x2, y2 in linija:
            xmin.append(x1)
            ymin.append(y2)
            xmax.append(x2)
            ymax.append(y1)
    xmin_z, ymin_z, xmax_z, ymax_z = min(xmin), min(ymin), max(xmax), max(ymax)

    zelena_levo_teme = (xmin_z, ymax_z)
    zelena_desno_teme = (xmax_z, ymin_z)

    zel_lin = zelena_levo_teme, zelena_desno_teme

    print('Plava: ' + str(plava_levo_teme) + ', ' + str(plava_desno_teme))
    print('Zelena: ' + str(zelena_levo_teme) + ', ' + str(zelena_desno_teme) + '\n')

    print('================================')

    fgbg = cv2.createBackgroundSubtractorMOG2()  # create background subtractor

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return_val, bin_image = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    konacna_suma = 0
    suma = 0
    brojevi = []

    while cap.isOpened():

        ret, frame = cap.read()

        if ret is True:

            frame_copy = frame.copy()
            maska_za_belu = cv2.inRange(frame_copy, np.array([127, 127, 127]), np.array([255, 255, 255]))
            frame_bela = cv2.bitwise_and(frame_copy, frame_copy, mask=maska_za_belu)
            gray_bela = cv2.cvtColor(frame_bela, cv2.COLOR_BGR2GRAY)
            ret, image_bin_bela = cv2.threshold(gray_bela, 127, 255, cv2.THRESH_BINARY)
            # problem prilikom prepoznavanja kad se izvrsi dilate
            # image_bin_bela = cv2.dilate(image_bin_bela, np.ones((3,3)), iterations=1)
            # cv2.imshow('Bela', image_bin_bela)
            konture_brojeva = select_roi(image_bin_bela)
            brojevi = tracker.update(konture_brojeva)

            for broj in brojevi:

                if not broj.da_li_je_prosao_plavu_liniju and da_li_je_prosao_broj(pla_lin, broj):
                    slika_br = np.zeros((28, 28)).astype('float32')
                    x, y, w, h = broj.get_granice_broja()
                    x -= 3
                    y -= 3
                    w += 3
                    h += 3
                    x_mid = int((28.0 - w) / 2.0)
                    y_mid = int((28.0 - h) / 2.0)

                    for i in range(0, w):
                        for j in range(0, h):
                            if (y + j >= 0 and y + j < image_bin_bela.shape[0]) and (
                                    x + i >= 0 and x + i < image_bin_bela.shape[1]):
                                temp = image_bin_bela[y + j, x + i]
                                slika_br[y_mid + j, x_mid + i] = temp / 255.0

                    # cv2.imshow('broj plava', slika_br)
                    ulaz_za_knn = slika_br.reshape(1, 784).astype('float32')

                    # knn
                    prepoznat_broj = int(model.findNearest(ulaz_za_knn, k=1)[0])

                    # svm
                    # prepoznat_broj = int(model.predict(ulaz_za_knn))

                    print('Preko plave prelazi: ' + str(prepoznat_broj))
                    broj.da_li_je_prosao_plavu_liniju = True
                    suma += prepoznat_broj

                if not broj.da_li_je_prosao_zelenu_liniju and da_li_je_prosao_broj(zel_lin, broj):
                    slika_br = np.zeros((28, 28)).astype('float32')
                    x, y, w, h = broj.get_granice_broja()
                    x -= 3
                    y -= 3
                    w += 3
                    h += 3
                    x_mid = int((28.0 - w) / 2.0)
                    y_mid = int((28.0 - h) / 2.0)

                    for i in range(0, w):
                        for j in range(0, h):
                            if (y + j >= 0 and y + j < image_bin_bela.shape[0]) and (
                                    x + i >= 0 and x + i < image_bin_bela.shape[1]):
                                temp = image_bin_bela[y + j, x + i]
                                slika_br[y_mid + j, x_mid + i] = temp / 255.0

                    # cv2.imshow('broj zelena', slika_br)
                    ulaz_za_knn = slika_br.reshape(1, 784).astype('float32')

                    # knn
                    prepoznat_broj = int(model.findNearest(ulaz_za_knn, k=1)[0])

                    # svm
                    # prepoznat_broj = int(model.predict(ulaz_za_knn))

                    print('Preko zelene prelazi: ' + str(prepoznat_broj))
                    broj.da_li_je_prosao_zelenu_liniju = True
                    suma -= prepoznat_broj

            boundingBoxes = nadji_konture(frame)

            for bb in boundingBoxes:
                x, y = bb[0]
                w, h = bb[1]

                # cv2.rectangle(frame, (x, y), (x + 30, y + 30), (0, 255, 255), 2)
                cv2.circle(frame, (x+14, y+16), 18, (0, 255, 255), 2)

            cv2.line(frame, plava_levo_teme, plava_desno_teme, [0, 255, 255])
            cv2.line(frame, zelena_levo_teme, zelena_desno_teme, [255, 0, 255])

            cv2.putText(frame, 'SUMA: ' + str(suma), (430, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)
            cv2.imshow(video, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    print('\nSuma za ' + video + ' je: ' + str(suma))
    fajl.write(video + '\t' + str(suma) + '\n')

    cap.release()
    cv2.destroyAllWindows()
