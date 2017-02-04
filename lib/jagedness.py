import cv2
import math
import helper
from crop import crop

def calculateJagedness(image, debug=False):
    # Contours
    contour = helper.getMoleContour(image)
    # Fits an ellipse around the image
    ellipse = cv2.fitEllipse(contour)
    # Shows the contour and ellipse for debugging
    if debug:
        cv2.drawContours(image, contour, -1, (255, 255, 255), 5)
        cv2.ellipse(image, ellipse, (0, 0, 0), 5)
        helper.showImage(image)
    # Calculates the mean square distance between the ellipse and contour
    center = ellipse[0]
    dim = ellipse[1]
    theta = ellipse[2]
    meanSquare = 0
    for i in range(1, len(contour)):
        diffX = (contour[i][0][0] - center[0]) * math.cos(-theta) - \
                (contour[i][0][1] - center[1]) * math.sin(-theta)
        diffY = (contour[i][0][0] - center[0]) * math.sin(-theta) - \
                (contour[i][0][1] - center[1]) * math.cos(-theta)
        meanSquare += abs(diffX ** 2 + diffY ** 2)
    return meanSquare / (dim[0] * dim[1])

if __name__ == "__main__":
    files = [
        "./img/malignant1.jpg",
        "./img/malignant2.jpg",
        "./img/benign1.jpg",
        "./img/benign2.jpg"
    ]
    for filename in files:
        image = crop(cv2.imread(filename))
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(calculateJagedness(grayscale))
