
import cv2
import numpy as np
import os
import sys
sys.path.append('./cv/')
import calc_jagedness
from helper import getMoleContour
from helper import showImage
from cluster_colours import clusterColours
import pandas as pd
from skimage import io
from sklearn import cross_validation
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import svm

def find_mole(origimage):
    image = cv2.imread(origimage)
    image_blur = cv2.blur(image, (2, 2))
    image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contour = getMoleContour(image_greyscale)
    cv2.drawContours(image_greyscale, contour, -1, (255, 255, 255), 5)
    max_dimension = max(image_greyscale.shape)
    scale = 600/max_dimension
    image_resize = cv2.resize(image_greyscale, None, fx=scale, fy=scale)
    x,y,w,h = cv2.boundingRect(contour)
    x = x-14
    y = y-14
    if w > 50 and h > 50:
        new_img=image_resize[y:y+h+28,x:x+w+28]
        showImage(new_img)


find_mole('./img/malignant1.jpg')
