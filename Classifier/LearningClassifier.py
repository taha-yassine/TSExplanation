from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from keras.optimizers import Adagrad
from sklearn.externals import joblib
from tslearn.preprocessing import TimeSeriesScalerMinMax
import pickle
from tslearn import shapelets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd


def NN1_DTWClassifier(X_train,Y_train):

    X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
    knn1_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
    knn1_clf.fit(X_train, Y_train)
    return knn1_clf

def learningShapeletClassifier(X_train,Y_train):

    X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=X_train.shape[0],ts_sz=X_train.shape[1],n_classes=len(set(Y_train)),l=0.1,r=2)
    shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                        optimizer=Adagrad(lr=.1),
                        weight_regularizer=.01,
                        max_iter=200,
                        verbose_level=0)

    shp_clf.fit(X_train, Y_train)
    return shp_clf

def NN1_Classifier(X_train, Y_train):
    x = np.zeros((X_train.shape[0],X_train.shape[1]))
    for i in range(X_train.shape[0]):
        for y in range(X_train.shape[1]):
            x[i][y] = X_train[i][y].ravel()
    dt = pd.DataFrame(x)
    s = pd.Series(Y_train)
    knn = KNeighborsClassifier(1)
    knn.fit(dt,s)
    return knn


def saveClassifierLS(classifier, name):
    name = name[:len(name)-4] + "_LS_.sav"
    classifier.save(name)
    pickle.dump(classifier.label_binarizer ,open(name[0:len(name)-4] + "label.sav",'wb'))


def saveClassifier1NN(classifier, name):
    name = name[:len(name) - 4] + "_1NN.sav"
    print(name)
    joblib.dump(classifier, open(name, 'wb'))

def loadClassifieur1NN(name):
    return joblib.load(name)

def loadClassifieurLS(name):
    return shapelets.load_model(name), pickle.load(open(name[0:len(name)-4] + "label.sav",'rb'))

    




