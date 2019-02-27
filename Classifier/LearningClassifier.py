from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict
from tslearn.neighbors import KNeighborsTimeSeriesClassifier


def NN1_DTWClassifieur(X_train,Y_train):

    knn1_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
    knn1_clf.fit(X_train, Y_train)
    return knn1_clf

def learningShapeletClassifieur(X_train,Y_train):

    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=X_train.shape[0],ts_sz=X_train.shape[1],n_classes=len(set(y_train)),l=0.1,r=2)
    shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                        optimizer=Adagrad(lr=.1),
                        weight_regularizer=.01,
                        max_iter=200,
                        verbose_level=0)

    shp_clf.fit(X_train, Y_train)
    return shp_clf

"""def saveClassifieur(classifieur, name):"""



