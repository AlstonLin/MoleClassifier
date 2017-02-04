import cv2
import numpy as np
import os
import sys
from helper import showImage
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import argparse
import cv2

def clusterColours(image):
    # Resize it
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape
    w_new = int(100 * w / max(w, h) )
    h_new = int(100 * h / max(w, h) )

    image = cv2.resize(image, (w_new, h_new));


    # Reshape the image to be a list of pixels
    image_array = image.reshape((image.shape[0] * image.shape[1], 3))
    # Clusters the pixels
    clt = KMeans(n_clusters = 3)
    clt.fit(image_array)
