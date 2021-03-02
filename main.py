# https://www.kaggle.com/harlfoxem/housesalesprediction
from collections import Counter

from House import House
import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sn


def getData(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = False
        houses = []
        for row in csv_reader:
            if not header:
                header = True
                continue
            else:
                house = House(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7],
                              row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15],
                              row[16], row[17], row[18], row[19], row[20])
                houses.append(house)
    return houses


def checkAtributes(houses):
    for a in ["date", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
              "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built",
              "yr_renovated", "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]:
        total = 0
        for obj in houses:
            total = obj.checkAtribute(a, total)
        print(f'{a} atributu truksta: {100 * total / len(houses)} %')
    print(f'viso reiksmiu {len(houses)} \n')


def getSingleAtributes(houses, name):
    list = []
    for obj in houses:
        temp = getattr(obj, name)
        if name == "lat":
            list.append(float(temp))
        else:
            if temp:
                list.append(int(float(temp)))
    list.sort()
    return list


# 2-3 punktai
def getSingleAtributeCount(name):
    c = 0
    for obj in houses:
        if getattr(obj, name):
            c = c + 1
    return c


# 2-3 punktai
def getFirstQuartile(array):
    i = len(array) * 0.25
    if i == math.floor(i):
        return array[math.floor(i)]
    else:
        i = math.floor(i)
        return (array[i] + array[i + 1]) / 2


# 2-3 punktai
def getThirdQuartile(array):
    i = len(array) * 0.75
    if i == math.floor(i):
        return array[math.floor(i)]
    else:
        i = math.floor(i)
        return (array[i] + array[i + 1]) / 2


# 2-3 punktai
def getAverage(array):
    sum = 0
    for obj in array:
        sum += obj
    return sum / len(array)


# 2-3 punktai
def getMedian(array):
    i = len(array) * 0.5
    if i == math.floor(i):
        return array[math.floor(i)]
    else:
        i = math.floor(i)
        return (array[i] + array[i + 1]) / 2


# 2-3 punktai
def getSD(array):
    vid = getAverage(array)
    total = 0
    for i in array:
        total += (vid - i) ** 2
    return math.sqrt(total / len(array))


# 2-3 punktai
def getModa(array):
    counter = 0
    uniques = set(array)
    num = 0

    for i in uniques:
        curr = array.count(i)
        if curr > counter:
            counter = curr
            num = i
    proc = (counter * 100) / len(array)
    return num, counter, proc


# 2-3 punktai
def getSecondModa(array, moda):
    counter = 0
    uniques = set(array) - set([moda])
    num = 0

    for i in uniques:
        curr = array.count(i)
        if curr > counter:
            counter = curr
            num = i
    proc = (counter * 100) / len(array)
    return num, counter, proc


# 3.Atlikti duomenų rinkinio kokybės analizę
def calculateCatStuff(name, array, total):
    count = getSingleAtributeCount(name)
    print(f'{name}: viso: {count}')
    print(f'{name}: naudojama: {len(array)}')
    print(f'{name}: trukstamu: {((total - count) * 100) / total} %')
    print(f'{name}: kardinalumas: {len(set(array))}')
    modaEtc = getModa(array)
    secondModaEtc = getSecondModa(array, modaEtc[0])
    print(f'{name}: moda: "{modaEtc[0]}", dažnumas: {modaEtc[1]}, procentai: {modaEtc[2]}')
    print(f'{name}: antra moda: "{secondModaEtc[0]}", dažnumas: {secondModaEtc[1]}, procentai: {secondModaEtc[2]} \n')


# 2.Atlikti duomenų rinkinio kokybės analizę
def calculateNumStuff(name, fixedArr, total):
    count = getSingleAtributeCount(name)
    print(f'{name}: viso: {count}')
    print(f'{name}: naudojama: {len(fixedArr)}')
    print(f'{name}: trukstamu: {((total - count) * 100) / total} %')
    print(f'{name}: kardinalumas: {len(set(fixedArr))}')
    print(f'{name}: Min: {min(fixedArr)}')
    print(f'{name}: Max: {max(fixedArr)}')
    print(f'{name}: 25 procentilis: {getFirstQuartile(fixedArr)}')
    print(f'{name}: 75 procentilis: {getThirdQuartile(fixedArr)}')
    print(f'{name}: vidurkis: {getAverage(fixedArr)}')
    print(f'{name}: mediana: {getMedian(fixedArr)}')
    print(f'{name}: standartinis nuokrypis: {getSD(fixedArr)}\n')


# 1-2.Atlikti duomenų rinkinio kokybės analizę
def printAllStuff():
    calculateNumStuff("price", price, totalHouses)
    calculateNumStuff("sqft_living", sqft_living, totalHouses)
    calculateNumStuff("sqft_lot", sqft_lot, totalHouses)
    calculateNumStuff("sqft_above", sqft_above, totalHouses)
    calculateNumStuff("sqft_basement", sqft_basement, totalHouses)
    calculateNumStuff("sqft_living15", sqft_living15, totalHouses)
    calculateNumStuff("sqft_lot15", sqft_lot15, totalHouses)

    calculateCatStuff('waterfront', waterfront, totalHouses)
    calculateCatStuff('bedrooms', bedrooms, totalHouses)
    calculateCatStuff('floors', floors, totalHouses)
    calculateCatStuff('yearBin', yearBin, totalHouses)


# 4.Nupaišyti atributų histogramas
def drawHistogram(arr, name):
    if name == 'yearBin':
        bins = (np.arange(min(arr), max(arr) + 6, 5))
    else:
        bins = math.floor(1 + 3.22 * np.log(len(arr)))
        t = np.arange(min(arr), max(arr), (max(arr) - min(arr)) / bins)
        labels = [math.floor(i) for i in t]
        plt.gca().set_xticks(labels)
        plt.gca().set_xticklabels(labels, rotation=45)
    plt.style.use('bmh')
    plt.title(name)
    plt.hist(arr, bins=bins)
    plt.show()


# 4.Nupaišyti atributų histogramas
def drawCategoryHistogram(arr, name):
    histdic = {x: arr.count(x) for x in arr}
    x = []
    y = []
    for key, value in histdic.items():
        x.append(key)
        y.append(value)

    plt.figure()
    plt.title(name)
    plt.style.use('bmh')
    barwidth = 0.5
    plt.bar(np.arange(len(y)), y, barwidth)
    plt.gca().set_xticks(np.arange(len(y)))
    plt.gca().set_xticklabels(x)
    plt.show()


# 4 Nupaišyti atributų histogramas
def drawAllHistograms():
    drawHistogram(price, "price")
    drawHistogram(sqft_living, "sqft_living")
    drawHistogram(sqft_lot, "sqft_lot")
    drawHistogram(sqft_above, "sqft_above")
    drawHistogram(sqft_basement, "sqft_basement")
    drawHistogram(sqft_living15, "sqft_living15")
    drawHistogram(sqft_lot15, "sqft_lot15")

    drawCategoryHistogram(waterfront, "waterfront")
    drawCategoryHistogram(bedrooms, "bedrooms")
    drawCategoryHistogram(bathrooms, "bathrooms")
    drawCategoryHistogram(floors, "floors")
    drawHistogram(yearBin, "yearBin")


# 5-6 Identifikuoti  duomenų  kokybės  problemas, salinimas
def deleteNotValidRows(arr):
    fixed = []
    for h in arr:
        if h.isValid():
            fixed.append(h)
    return fixed


# 5-6 Identifikuoti  duomenų  kokybės  problemas, tolydusis atributas keiciamas i kategorinius, METAI -> PENKMECIAI
def rewriteYears(houses):
    mi = int(min([x.yr_built for x in houses if x.yr_built != '']))
    ma = int(max([x.yr_built for x in houses if x.yr_built != '']))
    bins = (np.arange(mi, ma + 6, 5)).tolist()
    sz = len(bins)
    for h in houses:
        for i in range(0, sz):
            if bins[i] <= int(h.yr_built) < bins[i + 1]:
                h.yearBin = int(bins[i])
                break
    return houses


# 5-6 Identifikuoti  duomenų  kokybės  problemas, salinam ekstremalias reiksmes
def checkExtremes(item, arr):
    lower = getFirstQuartile(arr) - 1.5 * (getThirdQuartile(arr) - getFirstQuartile(arr))
    upper = getThirdQuartile(arr) + 1.5 * (getThirdQuartile(arr) - getFirstQuartile(arr))
    if item < lower or item > upper:
        return True
    else:
        return False


# 5-6 Identifikuoti  duomenų  kokybės  problemas, salinam ekstremalias reiksmes
def fixExtremes():
    for h in fixedHouses:
        for attr, value in h.__dict__.items():
            if attr == 'price':
                value = int(float(value))
                if value < 0 or checkExtremes(value, price):
                    fixedHouses.remove(h)
                    break
            if attr == 'sqft_living':
                value = int(value)
                if value < 0 or checkExtremes(value, sqft_living):
                    fixedHouses.remove(h)
                    break
            if attr == 'sqft_lot':
                value = int(float(value))
                if value < 0 or checkExtremes(value, sqft_lot):
                    fixedHouses.remove(h)
                    break
            if attr == 'sqft_above':
                value = int(value)
                if value < 0 or checkExtremes(value, sqft_above):
                    fixedHouses.remove(h)
                    break
            if attr == 'sqft_basement':
                value = int(value)
                if value < 0 or checkExtremes(value, sqft_basement):
                    fixedHouses.remove(h)
                    break
            if attr == 'sqft_living15':
                value = int(value)
                if value < 0 or checkExtremes(value, sqft_living15):
                    fixedHouses.remove(h)
                    break
            if attr == 'sqft_lot15':
                value = int(value)
                if value < 0 or checkExtremes(value, sqft_lot15):
                    fixedHouses.remove(h)
                    break


# 7.1 Tolydinio tipo atributams: naudojant „scatter plot“tipo diagramą
def drawScatterPlot():
    plt.style.use('bmh')
    plt.scatter(price2, sqft_living2)
    plt.xlabel('price')
    plt.ylabel('sqft_living')
    plt.title("Price- sqft_living")
    plt.show()

    plt.scatter(sqft_above2, sqft_living2)
    plt.xlabel('sqft_above')
    plt.ylabel('sqft_living')
    plt.title("Sqft_above- sqft_living")
    plt.show()

    plt.scatter(sqft_living2, sqft_living152)
    plt.xlabel('sqft_living')
    plt.ylabel('sqft_living15')
    plt.title("Sqft_living- sqft_living15")
    plt.show()

    plt.scatter(sqft_above2, sqft_lot152)
    plt.xlabel('sqft_above')
    plt.ylabel('sqft_lot15')
    plt.title("Sqft_above- sqft_lot15")
    plt.show()

    plt.scatter(zipcode, sqft_basement2)
    plt.xlabel('zipcode')
    plt.ylabel('sqft_basement2')
    plt.title("zipcode- sqft_basement2")
    plt.show()


# 7.2 Pateikti SPLOMdiagramą
def SplomMatrix():
    price2 = getSingleAtributes(fixedHouses, "price")
    sqft_living2 = getSingleAtributes(fixedHouses, "sqft_living")
    sqft_lot2 = getSingleAtributes(fixedHouses, "sqft_lot")
    sqft_basement2 = getSingleAtributes(fixedHouses, "sqft_basement")
    sqft_above2 = getSingleAtributes(fixedHouses, "sqft_above")
    sqft_living152 = getSingleAtributes(fixedHouses, "sqft_living15")
    sqft_lot152 = getSingleAtributes(fixedHouses, "sqft_lot15")

    data = {'price': price2, 'sqft_living2': sqft_living2, 'sqft_lot2': sqft_lot2, 'sqft_basement2': sqft_basement2,
            'sqft_above2': sqft_above2, 'sqft_living152': sqft_living152, 'sqft_lot152': sqft_lot152}
    df = pd.DataFrame(data=data, )

    fig = px.scatter_matrix(data, dimensions=["price", "sqft_living2", "sqft_lot2", "sqft_basement2", "sqft_above2",
                                              "sqft_living152", "sqft_lot152", ], color='price')
    fig.show()


def filterHouses(houses, atribute, value):
    filtered = []
    for h in houses:
        if h.returnAtribute(atribute) == value:
            filtered.append(h)
    return filtered


# 7.3 Kategorinio  tipo  atributams:  naudojant  „bar  plot“  tipo  diagramą  pateikti keletą
def barPlot():
    floors2 = getSingleAtributes(fixedHouses, "floors")
    drawCategoryHistogram(floors2, 'floors')

    filtered = filterHouses(fixedHouses, 'bathrooms', '0')
    floors3 = getSingleAtributes(filtered, "floors")
    drawCategoryHistogram(floors3, "floors, su 0 vonios kambariu")

    filtered = filterHouses(fixedHouses, 'bathrooms', '1')
    floors3 = getSingleAtributes(filtered, "floors")
    drawCategoryHistogram(floors3, "floors, su 1 vonios kambariu")

    filtered = filterHouses(fixedHouses, 'bathrooms', '2')
    floors3 = getSingleAtributes(filtered, "floors")
    drawCategoryHistogram(floors3, "floors, su 2 vonios kambariu")

    filtered = filterHouses(fixedHouses, 'bathrooms', '3')
    floors3 = getSingleAtributes(filtered, "floors")
    drawCategoryHistogram(floors3, "floors, su 3 vonios kambariu")

    filtered = filterHouses(fixedHouses, 'bathrooms', '4')
    floors3 = getSingleAtributes(filtered, "floors")
    drawCategoryHistogram(floors3, "floors, su 4 vonios kambariu")

    filtered = filterHouses(fixedHouses, 'bathrooms', '5')
    floors3 = getSingleAtributes(filtered, "floors")
    drawCategoryHistogram(floors3, "floors, su 5 vonios kambariu")


# 7.3 Kategorinio  tipo  atributams:  naudojant  „bar  plot“  tipo  diagramą  pateikti keletą
def stackedBar1():
    filtered = filterHouses(fixedHouses, 'floors', '1')
    f1 = list(Counter(getSingleAtributes(filtered, "bathrooms")).values())
    f1.append(0)

    # f1 = len(getSingleAtributes(filtered, "floors"))

    filtered = filterHouses(fixedHouses, 'floors', '2')
    f2 = list(Counter(getSingleAtributes(filtered, "bathrooms")).values())

    filtered = filterHouses(fixedHouses, 'floors', '3')
    f3 = list(Counter(getSingleAtributes(filtered, "bathrooms")).values())
    f3.append(0)
    f3.append(0)

    labels = ['0 baths', '1 baths', '2 baths', '3 baths', '4 baths', '5 baths', '6 baths']
    width = 0.35  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()
    ax.bar(labels, f1, width, label='1 floor')
    ax.bar(labels, f2, width, bottom=f1, label='2 floors')
    ax.bar(labels, f3, width, bottom=f2, label='3 floors')
    ax.set_ylabel('Bustu kiekis')
    ax.set_title('Vonios kambariu pasiskirstymas, pagal busto aukstu kieki')
    ax.legend()
    plt.show()


# 7.4 Pateikti keletą (2-3)histogramų(žr. 3paskaita, 12-14skaidres)ir „box plot“ diagramų
# pavyzdžių(žr. 3paskaita, 15skaidrę),vaizduojančiųsąryšius tarp kategorinio(pavyzdys pateiktas pav.3)
# ir tolydiniotipo kintamųjų.
def stackedBar2():
    mi = int(min([x.yr_built for x in houses if x.yr_built != '']))
    ma = int(max([x.yr_built for x in houses if x.yr_built != '']))
    bins = (np.arange(mi, ma + 1, 5)).tolist()

    for b in bins:
        filtered = filterHouses(fixedHouses, 'yearBin', b)
        f1 = getSingleAtributes(filtered, "sqft_living")
        drawHistogram(f1, b)


# 7.4 Pateikti keletą (2-3)histogramų(žr. 3paskaita, 12-14skaidres)ir „box plot“ diagramų
# pavyzdžių(žr. 3paskaita, 15skaidrę),vaizduojančiųsąryšius tarp kategorinio(pavyzdys pateiktas pav.3)
# ir tolydiniotipo kintamųjų.
def boxPlot():
    mi = int(min([x.yr_built for x in houses if x.yr_built != '']))
    ma = int(max([x.yr_built for x in houses if x.yr_built != '']))
    bins = (np.arange(mi, ma + 1, 5)).tolist()

    data = []
    for b in bins:
        filtered = filterHouses(fixedHouses, 'yearBin', b)
        f1 = getSingleAtributes(filtered, "sqft_living")
        data.append(f1)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    bp = ax.boxplot(data, labels=bins)

    plt.show()


# 8. Paskaičiuoti  kovariacijos  ir  koreliacijos  reikšmes  tarp  tolydinio  tipo  atributų
def cov(arr, arr2, avg_a, avg_b):
    cov = 0
    if len(arr) == len(arr2):
        if arr == arr2:
            return 1
        else:
            for a, b in zip(arr, arr2):
                cov += (a - avg_a) * (b - avg_b)
            return cov / (len(arr) - 1)
    else:
        return None


# 8. Paskaičiuoti  kovariacijos  ir  koreliacijos  reikšmes  tarp  tolydinio  tipo  atributų
def calcCov():
    arr = [price2, sqft_living2, sqft_lot2, sqft_above2, sqft_basement2, sqft_living152, sqft_lot152]
    # arr = [price2, sqft_living2]

    covs = np.zeros((len(arr), len(arr)))
    corls = np.zeros((len(arr), len(arr)))
    i = j = 0
    for a in arr:
        avg_a = getAverage(a)
        for b in arr:
            if i == j:
                covs[i, j] = 1
                corls[i, j] = 1
            else:
                avg_b = getAverage(b)
                temp = cov(a, b, avg_a, avg_b)
                covs[i, j] = temp
                corls[i, j] = temp / (getSD(a) * getSD(b))
            j += 1
        i += 1
        j = 0

    sn.heatmap(corls, annot=True, xticklabels=["price", "sqft_living", "sqft_lot", "sqft_above", "sqft_basement",
                                               "sqft_living15", "sqft_lot15"],
               yticklabels=["price", "sqft_living", "sqft_lot", "sqft_above", "sqft_basement",
                            "sqft_living15", "sqft_lot15"])
    plt.style.use('bmh')
    plt.title("Tolydiniu atributu koreliacija")
    plt.show()


# 9. Atlikti duomenų normalizaciją
def normalization():
    arr = [price2, sqft_living2, sqft_lot2, sqft_above2, sqft_basement2, sqft_living152, sqft_lot152]
    normalized = []
    for a in arr:
        mi = min(a)
        ma = max(a)
        new_a = []
        for i in a:
            new_a.append((i - mi) / (ma - mi))
        normalized.append(new_a)
    return normalized




houses = getData('kc_house_data.csv')
fixedHouses = deleteNotValidRows(houses)
fixedHouses = rewriteYears(fixedHouses)

totalHouses = len(houses)
totalfixedHouses = len(fixedHouses)

price = getSingleAtributes(fixedHouses, "price")
sqft_living = getSingleAtributes(fixedHouses, "sqft_living")
sqft_lot = getSingleAtributes(fixedHouses, "sqft_lot")
sqft_above = getSingleAtributes(fixedHouses, "sqft_above")
sqft_basement = getSingleAtributes(fixedHouses, "sqft_basement")
sqft_living15 = getSingleAtributes(fixedHouses, "sqft_living15")
sqft_lot15 = getSingleAtributes(fixedHouses, "sqft_lot15")

waterfront = getSingleAtributes(fixedHouses, "waterfront")
bedrooms = getSingleAtributes(fixedHouses, "bedrooms")
bathrooms = getSingleAtributes(fixedHouses, "bathrooms")
floors = getSingleAtributes(fixedHouses, "floors")
yearBin = getSingleAtributes(fixedHouses, "yearBin")

# pakoreguoti duomenys- be extremaliu reiksmiu
price2 = getSingleAtributes(fixedHouses, "price")
sqft_living2 = getSingleAtributes(fixedHouses, "sqft_living")
sqft_lot2 = getSingleAtributes(fixedHouses, "sqft_lot")
sqft_basement2 = getSingleAtributes(fixedHouses, "sqft_basement")
sqft_above2 = getSingleAtributes(fixedHouses, "sqft_above")
sqft_living152 = getSingleAtributes(fixedHouses, "sqft_living15")
sqft_lot152 = getSingleAtributes(fixedHouses, "sqft_lot15")
zipcode = getSingleAtributes(fixedHouses, "zipcode")

printAllStuff()
drawAllHistograms()
print(max(price))

fixExtremes()
drawAllHistograms()
printAllStuff()
print("Pasichekinam, maksimali kaina skiriasi", max(price))

drawScatterPlot()
SplomMatrix()
barPlot()
stackedBar1()
stackedBar2()
boxPlot()
calcCov()
normalized = normalization()