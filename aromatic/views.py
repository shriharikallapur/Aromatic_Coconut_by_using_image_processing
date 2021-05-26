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
from django.shortcuts import render

# Create your views here.

def index(request):
     return render(request, 'index.html') 
      
def camera(request):
    return render(request, 'camera.html')

def accurate(request):
    
    ### detection ###
    
    image = cv2.imread('coconut/4.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    image = cv2.imread('im/img9.png')
    Rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    histogram1 = cv2.calcHist([Rgb_image], [0], None, [256], [0, 256])

    image = cv2.imread('IS/4.png')
    if image is not None:
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        histogram2 = cv2.calcHist([image], [0], None, [256], [0, 256])
    elif image is None:
        print("There is no image!!!")
        sys.exit()

    c1, c2 = 0, 0

    i = 0
    while i < len(histogram) and i < len(histogram1):
        c1 += (histogram[i] - histogram1[i]) ** 2
        i += 1
    c1 = c1 ** (1 / 2)

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
    
    ### adjust_brightness ###

    im = Image.open("IS/4.png")
    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    im_output.save('brightened-image.png')

    ### K-means ####

    im = cv2.imread("brightened-image.png")
    k = 8
    if len(im.shape) < 3:
        Z = im.reshape((-1, 1))
    elif len(im.shape) == 3:
        Z = im.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    CI = res.reshape(im.shape)
    cv2.imwrite("Cluster_Image.png", CI)
    
    ### Threshold ###

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

    ### percent_color ###

    colors, pixel_count = extcolors.extract_from_path("threshold.png")
    for color in colors:
       c1 = (int(color[1])/pixel_count)*80
       c2 = (int(color[1])/pixel_count)*20
       c3 = (int(color[1])/pixel_count)*60
       c4 = (int(color[1])/pixel_count)*40
       c5 = (int(color[1])/pixel_count)*100
    

    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        print("\n")
        writer.writerow(["sno", "Ring No.", "Percentage"])
        writer.writerow(["1", "1", c1])
        writer.writerow(["2", "1.5", c2])
        writer.writerow(["3", "2", c4])
        writer.writerow(["4", "2.5", c3])
        writer.writerow(["5", "3", c5])
    ### polynomial ###

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
    plt.scatter(X, y, color='green')
    plt.plot(X, lin2.predict(poly.fit_transform(X)), color='blue')
    plt.title('Polynomial Regression')
    plt.xlabel('Ring No.')
    plt.ylabel('Percentage')
    plt.savefig('static/img/poly1.png')
    avg = (c1+c2+c3+c4+c5)/5
    return render(request, 'accurate.html', {'output': avg})
 
def about(request):    
    return render(request, 'about.html')
