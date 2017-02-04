
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

def test(origimage):
    image = cv2.imread(origimage)
    image_blur = cv2.blur(image, (2, 2))
    image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contour = getMoleContour(image_greyscale)
    cv2.drawContours(image_greyscale, contour, -1, (255, 255, 255), 5)
    max_dimension = max(image_greyscale.shape)
    x,y,w,h = cv2.boundingRect(contour)
    x = x-25
    y = y-25
    new_img=image[y:y+h+50,x:x+w+50]
    showImage(new_img)
    clusterColours(image)




test('./img/malignant2.jpg')
