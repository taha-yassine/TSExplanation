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

"""Construction classifieur 1NN-DTW. En vrai, on va le loader"""
#cl = LearningClassifier.NN1_DTWClassifier(X_train, Y_train)


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
"""
mysegts = myindexedTS.tsSegmentation()
#TS brute
print ("TS BRUTE:" , myindexedTS.raw_timeSeries())
#TS segmentation
print ("TS Segmentation:", mysegts)
#longueur TS
print ("Longeur TS:", myindexedTS.num_timeSubSeries())
#rendre une valeur en fonction de son id
print ("valeur de l'id 2:", myindexedTS.timeSubSeries(2))
#rendre toutes les positions de l'indice passe en param
print ("positions de l'indice 1: ", myindexedTS.timeSeries_position(1))
#Enlever les mots aux indices donnes
print ("enleve mots indice 0 et 1:", myindexedTS.inverse_removing([0,1]))
print(type(myindexedTS.inverse_removing([0,1])))
"""
"Pour indexed ça a l'air de marcher"

"""Pour TSexplainer et utiliser le classifier, faudra que l'on code la fonction data_label_distance
car c'est celle la qui utilise le classifier. Mais à mon avis yaura juste à le passer en paramètre comme ca :"""

myTSexp=lime_timeseries.TSExplainer(class_names=['0', '1'])
series = coffee_test_x.iloc[5, :]
exp = myTSexp.explain_instance(series,knn)
print(exp.as_list())
values_per_slice = math.ceil(len(series) / 300)
plt.plot(series, color='b', label='Explained instance')
plt.plot(coffee_test_x.iloc[15:,:].mean(), color='green', label='Mean of other class')
plt.legend(loc='lower left')
for i in range(10):
    feature, weight = exp.as_list()[i]
    start = feature * values_per_slice
    end = start + values_per_slice
    plt.axvspan(start , end, color='green', alpha=abs(weight*100))
plt.show()

"""test inverseremoving3"""

