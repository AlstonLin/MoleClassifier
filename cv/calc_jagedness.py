import cv2
import helper

def calculateJagedness(image, debug=False):
    # Contours    
    contour = helper.getMoleContour(image)
    if debug:
        cv2.drawContours(image, contour, -1, (255, 255, 255), 5)
        helper.showImage(image)

if __name__ == "__main__":
    grayscale = cv2.imread("./img/malignant.jpg", 0)
    calculateJagedness(grayscale, debug=True)
