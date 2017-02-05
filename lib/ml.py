import cv2
import random
import pickle
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
        self.numTrained = 0
        self.totalJagedness = 0
        self.totalAsymmetry = 0
        self.totalColoring = 0

    """
    The format of X should be a 2D array.
    Eg. [[a1, b1], [a2, b2], [a3, b3]] is the input for a dataset with 3 samples and 2 features
    Y should be in a similar format as well
    """
    def train(self, X, y):
        # Data sets
        TEST_PCT = 0.25 # Of the non-cv data
        CV_PCT = 0.1
        trainData = []
        trainTarget = []
        testData = [] 
        testTarget = []
        cvData = [] 
        cvTarget = []
        # Assigns data points randomly to each data set
        for i in range(len(X)):
            if random.random() >= CV_PCT:
                trainData.append(X[i])
                trainTarget.append(y[i])
                if random.random() <= TEST_PCT:
                    testData.append(X[i])
                    testTarget.append(y[i])
            else:
                cvData.append(X[i])
                cvTarget.append(y[i])
        # Saves sums
        self.numTrained = len(trainData)
        for x in trainData:
            self.totalJagedness += x[0]
            self.totalAsymmetry += x[1]
            self.totalColoring += x[2]
        # Scales data
        self.scaler.fit(trainData)
        trainData = self.scaler.transform(trainData)
        testData = self.scaler.transform(testData)
        cvData = self.scaler.transform(cvData)
        # Neural Network
        self.ann = MLPClassifier(
            activation="relu",
            hidden_layer_sizes=(18),
            alpha=0.00125,
            learning_rate_init=0.02,
            verbose=True,
            max_iter=1000
        )
        self.ann.fit(trainData, trainTarget)
        # Test
        print(">>>>>>>>>>>>>>> TEST RESULTS <<<<<<<<<<<<<<<<<<<")
        testPredictions = self.ann.predict(testData)
        print(classification_report(testTarget, testPredictions))
        print(">>>>>>>>>>> CROSS VALIDATION RESULTS <<<<<<<<<<<")
        cvPredictions = self.ann.predict(cvData)
        print(classification_report(cvTarget, cvPredictions))
    def predict(self, X):
        if self.ann == None:
            raise Error("Need to either load or train the ANN before using predict!")
        X = self.scaler.transform(X)
        y = self.ann.predict(X)
        return y[0]

    def predictProbability(self, X):
        if self.ann == None:
            raise Error("Need to either load or train the ANN before using predict!")
        X = self.scaler.transform(X)
        y = self.ann.predict_proba(X)
        return y[0][1]
    
    def dump(self, filepath):
        file = open(filepath, 'w+')
        pickle.dump(self, file)

    @staticmethod
    def load(filepath):
        file = open(filepath, 'r')
        loaded = pickle.load(file)
        return loaded

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
    ml.dump("test.dat")
    ml2 = ML.load("dat/test.dat")
