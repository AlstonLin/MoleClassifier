import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from crop import crop
from asymmetry import calculateAsymmetry
from jagedness import calculateJagedness
from coloring import calculateColoring

class ML:
    def __init__(self):
        self.scaler = StandardScaler()
        self.ann = None
    
    """
    The format of X should be a 2D array.
    Eg. [[a1, b1], [a2, b2], [a3, b3]] is the input for a dataset with 3 samples and 2 features
    Y should be in a similar format as well
    """
    def train(self, X, y):
        # Data sets
        TEST_PCT = 0.3
        trainSize = int(len(X) * (1 - TEST_PCT))
        trainData = X[0:trainSize]
        trainTarget = y[0:trainSize]
        testData = X[trainSize:-1]
        testOutput = y[trainSize:-1]
        # Scales data
        self.scaler.fit(trainData)
        trainData = self.scaler.transform(trainData)
        testData = self.scaler.transform(testData)
        # Neural Network
        self.ann = MLPClassifier(hidden_layer_sizes=(10))
        self.ann.fit(trainData, trainTarget)
        # Test
        testPredictions = self.ann.predict(testData)
        print(classification_report(testOutput, testPredictions))

    def predict(self, X):
        if self.ann == None:
            raise Error("Need to either load or train the ANN before using predict!")
        y = self.ann.predict(X)
        return y[0][0]


if __name__ == "__main__":
    # Testing
    images = [
        "./img/malignant1.jpg",
        "./img/malignant2.jpg",
        "./img/benign1.jpg",
        "./img/benign2.jpg"
    ]
    X = []
    for filename in images:
        img = crop(cv2.imread(filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append([
            calculateJagedness(gray),
            calculateAsymmetry(gray),
            calculateColoring(img)
        ])
    ml = ML()
    ml.train(X, [1, 1, 0, 0])
