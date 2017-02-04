import cv2
import itertools
import numpy as np
from helper import showImage
from crop import crop
from sklearn.cluster import KMeans

def calculateColoring(image, debug=False):
    # Reshapes the image to a giant vector of pixels & the 3 GRB channels
    pixelChannels = image.reshape((-1, 3))
    pixelChannels = np.float32(pixelChannels)
    # Clusters the channels for all the pixels
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixelChannels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Debugging / Testing stuff
    if debug:
        centersInt = np.uint8(centers)
        clusterImg = centersInt[labels.flatten()]
        clusterImg = clusterImg.reshape((image.shape))
        showImage(clusterImg)
    # Estimates the skin color by averaging the colors in the 4 corners
    w, h, _ = image.shape
    skinColor = [0, 0, 0]
    skinColor += image[0][0]
    skinColor += image[0][h - 1]
    skinColor += image[w - 1][0]
    skinColor += image[w - 1][h - 1]
    skinColor /= 3
    # Euclidean distances between each cluster center
    meanSquareErr = 0
    for (c1, c2) in itertools.combinations(centers, 2):
        for k in range(3):
            dist = abs(c1[k] - c2[k])
            # TODO: Normalize against skin color
            meanSquareErr += dist
    return meanSquareErr

if __name__ == "__main__":
    # Tests
    img = cv2.imread("./img/malignant1.jpg")
    print(calculateColoring(crop(img), debug=True))
    img = cv2.imread("./img/malignant2.jpg")
    print(calculateColoring(crop(img), debug=True))
    img = cv2.imread("./img/benign1.jpg")
    print(calculateColoring(crop(img), debug=True))
    img = cv2.imread("./img/benign2.jpg")
    print(calculateColoring(crop(img), debug=True))
