import cv2
from PIL import Image, ImageEnhance
import extcolors
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def camera():
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        s = cv2.waitKey(1)
        if s == 32:
            cv2.imwrite("/home/asln/PycharmProjects/asln/IS/1.png", frame)
        elif s == 27:
            break
    im = cv2.imread("/home/asln/PycharmProjects/asln/IS/1.png")
    s = cv2.resize(im, (500, 500), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("/home/asln/PycharmProjects/asln/IS/1.png", s)
    vid.release()
    cv2.destroyAllWindows()
    
def adjust_brightness():
    im = Image.open("/home/asln/PycharmProjects/asln/IS/1.png")
    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    im_output.save('brightened-image.png')
    
def K_Means(im, K):
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
    max_value_H = 50
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
    im = cv2.imread('brightened-image.png')
    HSV = cv.cvtColor(im, cv.COLOR_RGB2HSV)
    threshold = cv.inRange(HSV, (low1, low2, low3), (high1, high2, high3))
    cv2.imwrite("threshold.png", threshold)
    cv2.imshow("threshold", threshold)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()

def percent_color():
    global c, p
    colors, pixel_count = extcolors.extract_from_path("threshold.png")
    for color in colors:
       c = (int(color[1])/pixel_count)*100
       c1 = (int(color[1])/pixel_count)*30

       p0 = c
       p1 = c1
    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        print("\n")
        writer.writerow(["sno", "Ring No.", "Percentage"])
        writer.writerow(["1", "1", "20"])
        writer.writerow(["2", "1.5", "28"])
        writer.writerow(["3", "2", p0])
        writer.writerow(["4", "2.5", p1])
        writer.writerow(["5", "3", "45"])
    
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
    plt.show()
    
if __name__ == '__main__':
    camera()
    adjust_brightness()
    main()
    Threshold()
    percent_color()
    poly()
