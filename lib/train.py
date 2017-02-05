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
    images = glob.glob('./img/dataset/*')
    images = sorted(images)
    X = []
    Y = np.genfromtxt('./dat/training.csv', dtype=float)
    for filename in images:
        try:
            img = crop(cv2.imread(filename))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append([
                calculateJagedness(gray),
                calculateAsymmetry(gray),
                calculateColoring(img)
            ])
        except Exception:
            print("ERROR processing ", filename)
    cml = ML()
    print(X)
    cml.train(X, Y)
    cml.dump("./dat/trained.dat")
    cml2 = ML.load("./dat/trained.dat")
    print(cml2.predict([X[2]]))
