import numpy as np
from dist import distance


class Broj:

    def __init__(self, granice_br):
        self.da_li_je_prosao_plavu_liniju = False
        self.da_li_je_prosao_zelenu_liniju = False
        self.granice_broja = granice_br

    def get_centar(self):
        x, y, w, h = self.granice_broja
        x = int(x + (w / 2.0))
        y = int(y + (h / 2.0))

        return x, y

    def set_granice_broja(self, granice_br):
        self.granice_broja = granice_br

    def get_granice_broja(self):
        return self.granice_broja

    def donji_desni_ugao(self):
        x = self.granice_broja[0]
        y = self.granice_broja[1]
        w = self.granice_broja[2]
        h = self.granice_broja[3]

        xw = x + w
        yh = y + h

        return (xw, yh)


class Tracker:

    def __init__(self):
        self.__numbers = []

    def update(self, regioni):
        for region in regioni:
            self.update_regione(region)

        return self.__numbers

    def update_regione(self, region):
        centar = region.get_centar()
        granice_broja = region.get_granice_broja()

        if self.__numbers.__len__() == 0:
            number = Broj(granice_broja)
            self.__numbers.append(number)

            return number

        distances = []

        for number in self.__numbers:
            distances.append(distance(centar, number.get_centar()))

        number = None

        i = np.argmin(distances)

        if distances[i] < 20:
            number = self.__numbers[i]

        if number is None:
            number = Broj(granice_broja)
            self.__numbers.append(number)
        else:
            number.set_granice_broja(granice_broja)

        return number
