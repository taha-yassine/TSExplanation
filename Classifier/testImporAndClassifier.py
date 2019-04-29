import importTS
import LearningClassifier
import numpy as np
import pandas as pd

"Fichier de test"


X_train, Y_train, X_test, Y_test = importTS.dataImport("Trace")
print("Résultat attendu : " + str(Y_test[1]))


"Test LS"

clLS = LearningClassifier.learningShapeletClassifier(X_train, Y_train)
print("Résultat LS : " + str(clLS.predict(X_test[1].ravel().tolist())))

LearningClassifier.saveClassifierLS(clLS, "LS")
loadclLS, labelsave = LearningClassifier.loadClassifieurLS("LS")
newlabel = loadclLS.predict(X_test)
print("Résulat classifieur LS save : " + str(labelsave.inverse_transform(newlabel)[1]))


"Test 1NN-DTW"
cl = LearningClassifier.NN1_DTWClassifier(X_train, Y_train)
print("Résultat 1NN-DTW : " + str(cl.predict(X_test[1].ravel().tolist())))

LearningClassifier.saveClassifier1NN(cl, "1NN")

loadcl = LearningClassifier.loadClassifieur1NN("1NN")
print("Résulat classifieur 1NN-DTW save : " + str(loadcl.predict(X_test[1].ravel().tolist())))



"Test 1NN"
x = np.zeros((X_train.shape[0],X_train.shape[1]))
for i in range(X_train.shape[0]):
    for y in range(X_train.shape[1]):
        x[i][y] = X_train[i][y].ravel()
dt = pd.DataFrame(x)
cl = LearningClassifier.NN1_Classifier(X_train, Y_train)


print("Résultat 1NN : " + str(cl.predict(dt)))

LearningClassifier.saveClassifier1NN(cl, "1NN")

loadcl = LearningClassifier.loadClassifieur1NN("1NN")
print("Résulat classifieur 1NN save : " + str(loadcl.predict(dt)))



"Test import file"
tsx, tsy = importTS.fileImportTS("TimeSeriesFiles/myTS.txt")
print(tsx)
print(tsy)
