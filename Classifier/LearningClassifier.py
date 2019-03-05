from tslearn import shapelets
from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMinMax
from keras.optimizers import Adagrad
import numpy
import pickle
from sklearn.externals import joblib
from keras.models import load_model
import pickle
from tslearn import shapelets


def NN1_DTWClassifier(X_train,Y_train):

    knn1_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
    knn1_clf.fit(X_train, Y_train)
    return knn1_clf

def learningShapeletClassifier(X_train,Y_train):

    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=X_train.shape[0],ts_sz=X_train.shape[1],n_classes=len(set(Y_train)),l=0.1,r=2)
    shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                        optimizer=Adagrad(lr=.1),
                        weight_regularizer=.01,
                        max_iter=200,
                        verbose_level=0)

    shp_clf.fit(X_train, Y_train)
    return shp_clf

def saveClassifierLS(classifier, name):
    classifier.save("SaveClassifierFiles/" + name + ".sav")
    pickle.dump(classifier.label_binarizer ,open("SaveClassifierFiles/" + name + "label.sav",'wb'))

def saveClassifier1NN(classifier, name):
    joblib.dump(classifier, open("SaveClassifierFiles/" + name + ".sav", 'wb'))

def loadClassifieur1NN(name):
    return joblib.load("SaveClassifierFiles/" + name + ".sav")

def loadClassifieurLS(name):
    return shapelets.load_model("SaveClassifierFiles/" + name + ".sav"), pickle.load(open("SaveClassifierFiles/" + name + "label.sav",'rb'))

    




