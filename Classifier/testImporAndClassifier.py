import importTS
import LearningClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

"Fichier de test"


X_train, Y_train, X_test, Y_test = importTS.dataImport("ArrowHead")
print("Résultat attendu : " + str(Y_test[1]))


"Test LS"

clLS = LearningClassifier.learningShapeletClassifier(X_train, Y_train)
print("Résultat LS : " + str(clLS.predict(X_test[1].ravel().tolist())))
predicted_labels = clLS.predict(X_test)
print("Correct classification rate :", accuracy_score(Y_test, predicted_labels))


LearningClassifier.saveClassifierLS(clLS, "LS", X_train, Y_train)

loadclLS, labelsave,_,_ = LearningClassifier.loadClassifieurLS("_LS_.sav")
newlabel = loadclLS.predict(X_test)
print("Résulat classifieur LS save : " + str(labelsave.inverse_transform(newlabel)[1]))
print("Correct classification rate :", accuracy_score(Y_test, labelsave.inverse_transform(newlabel)))


"""
"Test 1NN-DTW"
cl = LearningClassifier.NN1_DTWClassifier(X_train, Y_train)
print("Résultat 1NN-DTW : " + str(cl.predict(X_test[1].ravel().tolist())))

LearningClassifier.saveClassifier1NN(cl, "1NN")

loadcl = LearningClassifier.loadClassifieur1NN("1N_1NN.sav")
print(loadcl.predict_proba(X_test))
print("Résulat classifieur 1NN-DTW save : " + str(loadcl.predict(X_test[1].ravel().tolist())))

"""


"Test 1NN"
x = np.zeros((X_train.shape[0],X_train.shape[1]))
for i in range(X_train.shape[0]):
    for y in range(X_train.shape[1]):
        x[i][y] = X_train[i][y].ravel()
dt = pd.DataFrame(x)
cl = LearningClassifier.NN1_Classifier(X_train, Y_train)

x = np.zeros((X_test.shape[0],X_test.shape[1]))
for i in range(X_test.shape[0]):
    for y in range(X_test.shape[1]):
        x[i][y] = X_test[i][y].ravel()
dtt = pd.DataFrame(x)

print("Résultat 1NN : " + str(cl.predict(dt)[1]))

LearningClassifier.saveClassifier1NN(cl, "1NN")

loadcl = LearningClassifier.loadClassifieur1NN("1N_1NN.sav")
print("Résulat classifieur 1NN save : " + str(loadcl.predict(dt)[1]))
predicted_labels = loadcl.predict(dtt)
print("Correct classification rate :", accuracy_score(Y_test, predicted_labels))


"""
"Test import file"
tsx, tsy = importTS.fileImportTS("TimeSeriesFiles/myTS.txt")
print(tsx)
print(tsy)
"""