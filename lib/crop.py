import cv2
import numpy as np
from helper import getMoleContour
from helper import showImage

def crop(image):
    PADDING = 25
    image = cv2.GaussianBlur(image, (1, 1), 0)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contour = getMoleContour(grayscale)
    x, y, w, h = cv2.boundingRect(contour)
    x -= PADDING
    y -= PADDING
    if w < PADDING * 2 and h < PADDING * 2:
        raise Error("Picture is waay too small!!!")
    return image[y:y+h+PADDING*2,x:x+w+PADDING*2]

if __name__ == "__main__":
    image = cv2.imread('./img/malignant1.jpg')
    showImage(crop(image))
    image = cv2.imread('./img/malignant2.jpg')
    showImage(crop(image))
    image = cv2.imread('./img/benign1.jpg')
    showImage(crop(image))
    image = cv2.imread('./img/benign2.jpg')
    showImage(crop(image))
