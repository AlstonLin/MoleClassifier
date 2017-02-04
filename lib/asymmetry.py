import cv2
import helper
from crop import crop
import numpy

def calculateAsymmetry(grayscale):
    h, w = grayscale.shape[:2]
    # Splits the image into 2 halfs
    half = w / 2 - 1 if w % 2 == 1 else w / 2
    rotation = cv2.getRotationMatrix2D((half / 2, h / 2), 180, 1)
    left = grayscale[0:h, 0:half]
    # right = cv2.warpAffine(grayscale[0:h, half:w], rotation, (half, h))
    right = cv2.flip(grayscale[0:h, (w - half):w], 1)
    # Returns the mean square difference between each pixel
    numPixels = half * h
    return numpy.sum((left.astype("float") - right.astype("float")) ** 2) / numPixels

if __name__ == "__main__":
    # Tests
    files = [
        "./img/malignant1.jpg",
        "./img/malignant2.jpg",
        "./img/benign1.jpg",
        "./img/benign2.jpg",
    ]
    for filename in files:
        image = crop(cv2.imread(filename))
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(calculateAsymmetry(grayscale))
