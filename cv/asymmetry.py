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
    cv2.imshow('Rotation', left)
    cv2.waitKey()
    cv2.imshow('Rotation', right)
    cv2.waitKey()
    # Returns the mean square difference between each pixel
    numPixels = half * h
    return numpy.sum((left.astype("float") - right.astype("float")) ** 2) / numPixels

if __name__ == "__main__":
    # Tests

    grayscale = cv2.imread("./img/malignant1.jpg")
    print(calculateAsymmetry(crop(grayscale)))
    grayscale = cv2.imread("./img/malignant2.jpg")
    print(calculateAsymmetry(crop(grayscale)))
    grayscale = cv2.imread("./img/benign1.jpg")
    print(calculateAsymmetry(crop(grayscale)))
    grayscale = cv2.imread("./img/benign2.jpg")
    print(calculateAsymmetry(crop(grayscale)))
