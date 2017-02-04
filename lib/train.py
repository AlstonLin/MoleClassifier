import cv2
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from crop import crop
from ml import ML
from asymmetry import calculateAsymmetry
from jagedness import calculateJagedness
from coloring import calculateColoring

if __name__ == "__main__":
    # Testing

    # images = [
    #     "./img/malignant1.jpg",
    #     "./img/malignant2.jpg",
    #     "./img/benign1.jpg",
    #     "./img/benign2.jpg"
    # ]

    images = glob.glob('./tests/')

    X = []
    f = open('./training.txt')
    f.readline()
    data = np.loadtxt(f)
    Y = data[:, 0]
    for filename in images:
        img = crop(cv2.imread(filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append([
            calculateJagedness(gray),
            calculateAsymmetry(gray),
            calculateColoring(img)
        ])
    cml = ML()
    cml.train(X, Y)
    cml.dump("test.dat")
    cml2 = ML.load("dat/test.dat")
    print(cml2.predict([X[2]]))
