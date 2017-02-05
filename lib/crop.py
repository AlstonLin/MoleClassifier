import cv2
import numpy as np
from helper import getMoleContour
from helper import showImage

def crop(image):
    imageH, imageW, _ = image.shape
    MAX_PADDING = 7
    # image = cv2.GaussianBlur(image, (1, 1), 0)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contour = getMoleContour(grayscale)
    cv2.drawContours(grayscale, contour, -1, (255, 255, 255), 5)
    x, y, w, h = cv2.boundingRect(contour)
    padding = MAX_PADDING if x > MAX_PADDING and y > MAX_PADDING else min(x, y)
    print(padding)
    padding = padding if x + w + padding * 2 > imageW and y + h + padding * 2 > imageH else 0
    x -= padding
    y -= padding
    print(padding)
    print((x, y))
    if w < padding * 2 and h < padding * 2:
        raise Error("Picture is waay too small!!!")
    return image[y:y+h+padding*2,x:x+w+padding*2]

if __name__ == "__main__":
    image = cv2.imread('./img/dataset/IMD348.bmp')
    showImage(image)
    showImage(crop(image))
    image = cv2.imread('./img/malignant1.jpg')
    showImage(image)
    showImage(crop(image))
    image = cv2.imread('./img/malignant2.jpg')
    showImage(image)
    showImage(crop(image))
    image = cv2.imread('./img/benign1.jpg')
    showImage(image)
    showImage(crop(image))
    image = cv2.imread('./img/benign2.jpg')
    showImage(image)
    showImage(crop(image))
