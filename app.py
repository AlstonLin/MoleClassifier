
import cv2
import numpy as np
import os
import sys
sys.path.append('./cv/')
import calc_jagedness
from helper import getMoleContour
from helper import showImage
import pandas as pd
from skimage import io
from sklearn import cross_validation
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import svm

def find_mole(origimage):
    image = cv2.imread(origimage, 0)
    # # image_blur = cv2.blur(image, (5, 5))
    # image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_contour = getMoleContour(image)
    # max_dimension = max(image.shape)
    # scale = 700/max_dimension
    # image = cv2.resize(image, None, fx=scale, fy=scale)
    # R = boundingRect(image_contour); ## Get bounding box for contour i
    # ROI = cv2.minAreaRect(R)
    # showImage(image)


find_mole('./img/malignant1.jpg')
