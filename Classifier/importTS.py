from tslearn.utils import save_timeseries_txt, load_timeseries_txt
from tslearn.datasets import UCR_UEA_datasets


def fileImportTS(fileName):
    time_series_dataset = load_timeseries_txt(fileName)
    return time_series_dataset

def dataImport(name):
    X_train, y_train = UCR_UEA_datasets().load_dataset(name)
    return X_train, y_train
