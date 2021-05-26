import cv2
import csv
import sys
import extcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def camera():
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        s = cv2.waitKey(1)
        if s == 32:
            cv2.imwrite("IS/1.png", frame)
        elif s == 27:
            break
    vid.release()
    cv2.destroyAllWindows()

def detection():
    # test image
    image = cv2.imread('coconut/4.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # data1 image
    image = cv2.imread('im/img9.png')
    Rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    histogram1 = cv2.calcHist([Rgb_image], [0], None, [256], [0, 256])

    # data2 image
    image = cv2.imread('IS/3.png')
    if image is not None:
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        histogram2 = cv2.calcHist([image], [0], None, [256], [0, 256])
    elif image is None:
        print("There is no image!!!")
        sys.exit()

    c1, c2 = 0, 0

    # Euclidean Distace between data1 and test
    i = 0
    while i < len(histogram) and i < len(histogram1):
        c1 += (histogram[i] - histogram1[i]) ** 2
        i += 1
    c1 = c1 ** (1 / 2)

    # Euclidean Distace between data2 and test
    i = 0
    while i < len(histogram) and i < len(histogram2):
        c2 += (histogram[i] - histogram2[i]) ** 2
        i += 1
    c2 = c2 ** (1 / 2)

    if c1 < c2:
        print("image is similar")
    else:
        print("image is not similar")
        sys.exit()

def adjust_brightness():
    im = Image.open("IS/3.png")
    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    im_output.save('brightened-image.png')
    
def K_Means(im, K):
    global Z
    if len(im.shape) < 3:
        Z = im.reshape((-1, 1))
    elif len(im.shape) == 3:
        Z = im.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    CI = res.reshape(im.shape)
    return CI

def main():
    Input_Image = cv2.imread("brightened-image.png")
    cv2.imshow('brightened-image', Input_Image)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    Clusters = 8
    Clustered_Image = K_Means(Input_Image, Clusters)
    cv2.imwrite("Cluster_Image.png", Clustered_Image)
    cv2.imshow('Cluster_Image', Clustered_Image)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        
def Threshold():
    max_value = 255
    max_value_H = 99
    low_H = 0
    low_S = 0
    low_V = 0
    high_H = max_value_H
    high_S = max_value
    high_V = max_value
    low1 = min(high_H - 1, low_H)
    high1 = max(high_H, low_H + 1)
    low2 = min(high_S - 1, low_S)
    high2 = max(high_S, low_S + 1)
    low3 = min(high_V - 1, low_V)
    high3 = max(high_V, low_V + 1)
    im = cv2.imread('Cluster_Image.png')
    HSV = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    threshold = cv2.inRange(HSV, (low1, low2, low3), (high1, high2, high3))
    cv2.imwrite("threshold.png", threshold)
    cv2.imshow("threshold", threshold)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()

def percent_color():
    global c
    colors, pixel_count = extcolors.extract_from_path("threshold.png")
    for color in colors:
       c1 = (int(color[1])/pixel_count)*80
       c2 = (int(color[1])/pixel_count)*20
       c3 = (int(color[1])/pixel_count)*60
       c4 = (int(color[1])/pixel_count)*40
       c5 = (int(color[1])/pixel_count)*100
       avg = (c1+c2+c3+c4+c5)/5

    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        print("\n")
        writer.writerow(["sno", "Ring No.", "Percentage"])
        writer.writerow(["1", "1", c1])
        writer.writerow(["2", "1.5", c2])
        writer.writerow(["3", "2", c4])
        writer.writerow(["4", "2.5", c3])
        writer.writerow(["5", "3", c5])
        print("The Coconut is:", avg, "% Aromatic")

def poly():
    datas = pd.read_csv('data.csv')
    X = datas.iloc[:, 1:2].values
    y = datas.iloc[:, 2].values
    lin = LinearRegression()
    lin.fit(X, y)
    poly = PolynomialFeatures(degree=4)
    X_poly = poly.fit_transform(X)
    poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)
    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color='green')
    plt.plot(X, lin2.predict(poly.fit_transform(X)), color='blue')
    plt.title('Polynomial Regression')
    plt.xlabel('Ring No.')
    plt.ylabel('Percentage')
    plt.savefig('poly.png')
    i = cv2.imread('poly.png')
    cv2.imshow('frame', i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    camera()
    detection()
    adjust_brightness()
    main()
    Threshold()
    percent_color()
    poly()