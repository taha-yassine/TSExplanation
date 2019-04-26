import argparse

import sys

sys.path.insert(0, "../Classifier")
import importTS
import LearningClassifier
import lime_timeseries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pyplot as plt
import math

parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=argparse.FileType('r'), help="")
parser.add_argument("classifier", help="")

# default = pas dans la ligne de commande
# const = dans la ligne mais pas renseigné
parser.add_argument("-i", "--indice", type=int,  default=0, help="indice of the ts explained (default : %(default)s)")
parser.add_argument("-f", "--features", type=int,  default=10,
                    help="max of features present in explanation (default : %(default)s)")
parser.add_argument("-c", "--cuts", type=int, default=24, help="(default : %(default)s)")
parser.add_argument("-s", "--samples", type=int,  default=1000,
                    help="size of the neighborhood to learn the linear model (default : %(default)s)")

args = parser.parse_args()

print(args)

"""Importation de la ST Trace
X_train et Y_train servent à l'entrainement du classifieur. Les ST non étiqueté sont dans X_test."""
X_train, Y_train, X_test, Y_test = importTS.dataImport("Trace")

for i in range(30):
    s = "b"
    if (Y_train[i] == 4):
        s = "r"
    if (Y_train[i] == 1):
        s = "g"
    if (Y_train[i] == 2):
        s = "c"
    if (Y_train[i] == 3):
        s = "y"
    # plt.plot(X_train[i].ravel(), s)
# plt.show()

"""Construction classifieur 1NN-DTW. En vrai, on va le loader"""
cl = LearningClassifier.NN1_DTWClassifier(X_train, Y_train)

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
"J'ai repris tes tests"

myTs = X_test[0].ravel()
myindexedTS = lime_timeseries.IndexedTS(myTs)
print(myindexedTS.raw.shape)

myTSexp = lime_timeseries.TSExplainer(class_names=['1', '2', '3', '4'])
#series = pd.Series(X_test[args.indice].ravel())
series = pd.Series(X_test[1].ravel())
#exp = myTSexp.explain_instance(myTs, cl1, X_train, num_cuts=args.cuts, num_features=args.features, num_samples=args.samples)
exp = myTSexp.explain_instance(myTs, cl1, X_train)
print(exp.as_list())
#values_per_slice = math.ceil(len(series) / args.cuts)
values_per_slice = math.ceil(len(series) / 24)

fig, _ = plt.subplots()
fig.canvas.set_window_title('Explanation')

plt.plot(series, color='b', label='Explained instance')

# plt.suptitle('sous titre')

plt.legend(loc='lower left')
for i in range(10):
    feature, weight = exp.as_list()[i]
    start = feature * values_per_slice
    end = start + values_per_slice
    plt.axvspan(start, end, color='green', alpha=abs(weight * 100))

plt.show()

"""test inverseremoving3"""
