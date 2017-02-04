from cluster_colours import clusterColours
import pandas as pd
from skimage import io
from sklearn import cross_validation
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import svm
