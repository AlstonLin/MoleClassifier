import cv2

def calculateJagedness(image):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img = cv2.imread("./img/malignant.jpg", 0)
    calculateJagedness(img)

