"""
Functions for explaining time series classifiers.
"""
import itertools
import numpy as np
import scipy as sp
import sklearn
from sklearn.utils import check_random_state
"""
from . import explanation
from . import lime_base

class TSDomainMapper(explanation.DomainMapper):

    def __init__(self, ts_seg):
        # sous serie de base ou sous serie segmentee ?
        self.ts_seg = ts

    def map_exp_ids(ts, positions=False):
        if positions:
            exp = [(self.mTS[x[0]],x[0], x[1])
                   for x in exp]
        else:
            exp = [(self.mTS[x[0]], x[1]) for x in exp]
        return exp

    def visualize_instance_html(ts):
        return 0

"""
class IndexedTS(object):

    def __init__(self, raw_ts, bow=True):
        self.raw = raw_ts
        self.as_list = list(self.raw)
        self.as_np = np.array(self.as_list)
        self.timeseries_start = np.arange(len(self.raw))
        values = {}
        self.inverse_values = []
        self.positions = []
        self.bow = bow
        non_values = set()
        for i, value in enumerate(self.as_np):
            if value in non_values:
                continue
            if bow:
                if value not in values:
                    values[value] = len(values)
                    self.inverse_values.append(value)
                    self.positions.append([])
                idx_value = values[value]
                self.positions[idx_value].append(i)
            else:
                self.inverse_value.append(value)
                self.positions.append(i)
        if not bow:
            self.positions = np.array(self.positions)


    def raw_timeSeries(self):
        """Returns the original raw time series"""
        return self.raw

    def num_timeSubSeries(self):
        """Returns the number of different values in the time series."""
        return len(self.inverse_values)

    def timeSubSeries(self, id_):
        """Returns the sub-time-series that corresponds to id_ (int)"""
        return self.inverse_values[id_]
   
    def timeSeries_position(self, id_):
        if self.bow:
            return self.timeseries_start[self.positions[id_]]
        else:
            return self.timeseries_start[[self.positions[id_]]]

    def tsSegmentation(self, seg_length=5, segmentationType=None):
        res = [self.raw[x:x+seg_length] for x in range(0,len(self.raw),seg_length)]
        return res

    def inverse_removing(self, values_to_remove):
        """Returns a time series after removing the appropriate values.
        If self.bow is false, replaces sub-time-series with UNKWNOW_TS instead of removing
        it.
        Args:
            values_to_remove: list of ids (ints) to remove
        Returns:
            original raw time series with appropriate values removed.
        """
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self.__get_idxs(values_to_remove)] = False
        if not self.bow:
            return ''.join(str([self.as_list[i] if mask[i]
                            else 'UNKNOW_TS' for i in range(mask.shape[0])]))
        return ''.join(str([self.as_list[v] for v in mask.nonzero()[0]]))

    def __get_idxs(self, values):
        """Returns indexes to appropriate values."""
        if self.bow:
            return list(itertools.chain.from_iterable(
                [self.positions[z] for z in values]))
        else:
            return self.positions[values]


class TSExplainer(object):

    def __init__():
        return 0

    def explain_instance(tsToExplain, classifier, label, numFeatures, numSamples, distance_metric):
        return 0

    def data_labels_distances(index, classifier, numSamples, distance_metric):
        return 0

""" OTHER USEFULL FUNCTIONS """

def generateTS(size=100,min=0,max=10):
    mlist = []
    for i in range(size):
        mlist.append(round(uniform(min,max),2))
    return mlist

def generateMockExp(ts):
    res = [(x,uniform(0,1)) for x in range(0,len(ts))]
    return res


"""TEST"""
myTS = [0.0, 0.21, 1.24, 1.21, 0.21 , 0.85, 1.96]
print (myTS)
myindexedTS = IndexedTS(myTS)
mysegts = myindexedTS.tsSegmentation()
#TS brute
print ("TS BRUTE" , myindexedTS.raw_timeSeries())
#TS segmentation
print ("TS Segmentation", mysegts)
#longueur TS
print ("Longeur TS", myindexedTS.num_timeSubSeries())
#rendre une valeur en fonction de son id 
print ("valeur de l'id 2", myindexedTS.timeSubSeries(2))
#rendre toutes les positions de l'indice passe en param
print ("positions de l'indice 1 ", myindexedTS.timeSeries_position(1))
#Enlever les mots aux indices donnes
print ("enleve mots indice 0 et 1", myindexedTS.inverse_removing([0,1]))
