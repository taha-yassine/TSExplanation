from tslearn.utils import save_timeseries_txt, load_timeseries_txt
from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMinMax


def fileImportTS(fileName):
    X_train = load_timeseries_txt("TimeSeriesFiles/" + fileName + ".txt")
    return X_train

def dataImport(name):
    X_train, y_train = UCR_UEA_datasets(True).load_dataset(name)
    X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
    return X_train, y_train

