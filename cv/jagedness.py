import cv2
import math
import helper

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
        meanSquare += abs((diffX / dim[0]) ** 2 + (diffY / dim[1]) ** 2)
    return meanSquare

if __name__ == "__main__":
    # Tests
    grayscale = cv2.imread("./img/malignant1.jpg", 0)
    print(calculateJagedness(grayscale, debug=True))
    grayscale = cv2.imread("./img/malignant2.jpg", 0)
    print(calculateJagedness(grayscale, debug=True))
    grayscale = cv2.imread("./img/benign1.jpg", 0)
    print(calculateJagedness(grayscale, debug=True))
    grayscale = cv2.imread("./img/benign2.jpg", 0)
    print(calculateJagedness(grayscale, debug=True))
