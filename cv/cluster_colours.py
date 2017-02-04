import cv2
import numpy as np
import os
import sys
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2

clusterColours(image):
    image_colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
