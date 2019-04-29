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

coffee_train = pd.read_csv('coffee_train.csv', sep=',', header=None).astype(float)
coffee_train_y = coffee_train.loc[:, 0]
coffee_train_x = coffee_train.loc[:, 1:]
coffee_test = pd.read_csv('coffee_test.csv', sep=',', header=None).astype(float)
coffee_test_y = coffee_test.loc[:, 0]
coffee_test_x = coffee_test.loc[:, 1:]
knn = KNN()
knn.fit(coffee_train_x, coffee_train_y)
"Accès à la première ST"
"Le .ravel est important à faire !!!"


num_cuts = 24
num_features = 5
num_samples = 1000
myTs = X_test[8].ravel()
myindexedTS = lime_timeseries.IndexedTS(myTs)

myTSexp=lime_timeseries.TSExplainer()
series = pd.Series(myTs)
exp = myTSexp.explain_instance(myTs,cl,X_train, num_cuts, num_features, num_samples)
print(exp.as_list())
exp.domain_mapper.plot(exp, series, num_features)
