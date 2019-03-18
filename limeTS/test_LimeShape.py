import sys
sys.path.insert(0, "../Classifier")
import importTS
import LearningClassifier
import lime_timeseries


"""Importation de la ST Trace
X_train et Y_train servent à l'entrainement du classifieur. Les ST non étiqueté sont dans X_test."""
X_train, Y_train, X_test, Y_test = importTS.dataImport("Trace")

"""Construction classifieur 1NN-DTW. En vrai, on va le loader"""
cl = LearningClassifier.NN1_DTWClassifier(X_train, Y_train)


"Accès à la première ST"
"Le .ravel est important à faire !!!"
"J'ai repris tes tests"
myTs = X_test[0].ravel()
myindexedTS = lime_timeseries.IndexedTS(myTs)
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

"Pour indexed ça a l'air de marcher"

"""Pour TSexplainer et utiliser le classifier, faudra que l'on code la fonction data_label_distance
car c'est celle la qui utilise le classifier. Mais à mon avis yaura juste à le passer en paramètre comme ca :"""
myTSexp=lime_timeseries.TSExplainer(myindexedTS.raw_timeSeries())
data, labels, distances = myTSexp.data_labels_distances(myindexedTS, cl, 2)
