import sys
sys.path.insert(0, "../Classifier")
import importTS
import LearningClassifier
import lime_timeseries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pyplot as plt
import math


"""Importation de la ST Trace
X_train et Y_train servent à l'entrainement du classifieur. Les ST non étiqueté sont dans X_test."""
X_train, Y_train, X_test, Y_test = importTS.dataImport("Trace")

for i in range(30):
    s = "b"
    if(Y_train[i]==4):
        s="r"
    if(Y_train[i]==1):
        s="g"
    if(Y_train[i]==2):
        s="c"
    if(Y_train[i]==3):
        s="y"
    plt.plot(X_train[i].ravel(),s)
plt.show()


"""Construction classifieur 1NN-DTW. En vrai, on va le loader"""
cl = LearningClassifier.NN1_Classifier(X_train, Y_train)

cl1 = LearningClassifier.learningShapeletClassifier(X_train, Y_train)


num_cuts = 24
num_features = 5
num_samples = 1000
myTs = X_test[0].ravel()
myTSexp=lime_timeseries.TSExplainer()
exp = myTSexp.explain_instance(myTs,cl,X_train, num_cuts, num_features, num_samples)
print(exp.as_list())
exp.domain_mapper.plot(exp, myTs, num_features)
