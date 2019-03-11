from tslearn.utils import save_timeseries_txt, load_timeseries_txt
from tslearn.datasets import CachedDatasets
from tslearn.utils import to_time_series_dataset
from tslearn.datasets import extract_from_zip_url
import numpy
import os


def fileImportTS(filePath):
    data_train = numpy.loadtxt(filePath)
    X_train = to_time_series_dataset(data_train[:, 1:])
    y_train = data_train[:, 0].astype(numpy.int)
    return X_train, y_train


def dataImport(name):
    
    if not os.path.exists("TimeSeriesFiles/"+name):
        url = "http://www.timeseriesclassification.com/Downloads/%s.zip" % name
        extract_from_zip_url(url, "TimeSeriesFiles/"+name +"/", verbose=False)

    data_train = numpy.loadtxt("TimeSeriesFiles/"+name +"/"+name+"_TRAIN.txt")
    data_test = numpy.loadtxt("TimeSeriesFiles/"+name +"/"+name+"_TEST.txt")
    X_train = to_time_series_dataset(data_train[:, 1:])
    y_train = data_train[:, 0].astype(numpy.int)
    X_test = to_time_series_dataset(data_test[:, 1:])
    y_test = data_test[:, 0].astype(numpy.int)
    return X_train, y_train, X_test, y_test
