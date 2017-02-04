import cv2

def getMoleContour(image):
    ret, thresh = cv2.threshold(image, 150, 200, cv2.THRESH_BINARY_INV) # Binary mask
    showImage(thresh)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Check to make sure there actually is a contour
    if len(contours) == 0:
        raise Error("No contours found!")
    # Only find the longest one
    longestContour = []
    for i in range(len(contours)):
        if len(contours[i]) > len(longestContour):
            longestContour = contours[i]
    return longestContour

def showImage(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
