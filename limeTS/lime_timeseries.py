"""
Functions for explaining time series classifiers.
"""

import numpy as np
import scipy as sp
import sklearn
from sklearn.utils import check_random_state

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




class IndexedTS(object):

    def __init__(self, raw_ts, bow=True):
         """Initializer.

        Args:
            raw_ts: time series with raw time series in it
            bow: if True, a sub-time-series is the same everywhere in the time series
            - i.e. we will index multiple occurrences of the same sub-time-series.
            If False, order matters, so that the same sub-time-series will have
            different ids according to position.
        """
        self.raw = raw_ts
        self.as_list = list(self.raw)
        self.inverse_vocab = []
        self.as_np = np.array(self.as_list)
        self.timeseries_start = np.arange(len(self.raw))
        vocab = {}
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        non_vocab = set()






    def raw_timeSeries(self):
        """Returns the original raw time series"""
        return self.raw

    def num_timeSubSeries(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def timeSubSeries(self, id_):
        """Returns the sub-time-series that corresponds to id_ (int)"""
        return self.inverse_vocab[id_]
   
    def timeSeries_position(self, id_):
         """Returns a np array with indices to id_ (int) occurrences"""
        if self.bow:
            return self.timeseries_start[self.positions[id_]]
        else:
            return self.timeseries_start[[self.positions[id_]]]


    def tsSegmentation(self, seg_length=5, segmentationType=None):
        """ Fait-on plusieurs types de segmentation?"""
        res = [self.raw[x:x+seg_length] for x in range(0,len(self.raw),seg_length)]
        return res

    def inverse_removing(self, sub_ts_to_remove):
        """Returns a time series after removing the appropriate sub-time-series.

        If self.bow is false, replaces sub-time-series with UNKWNOW_TS instead of removing
        it.

        Args:
            sub_ts_to_remove: list of ids (ints) to remove

        Returns:
            original raw time series with appropriate sub-time-series removed.
        """
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self.__get_idxs(sub_ts_to_remove)] = False
        if not self.bow:
            return ''.join([self.as_list[i] if mask[i]
                            else chr(0) for i in range(mask.shape[0])])
        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])

        def __get_idxs(self, timeSubSeries):
        """Returns indexes to appropriate timeSubSeries."""
        if self.bow:
            return list(itertools.chain.from_iterable(
                [self.positions[z] for z in timeSubSeries]))
        else:
            return self.positions[timeSubSeries]


class TSExplainer(object):

    def __init__():

    def explain_instance(tsToExplain, classifier, label, numFeatures, numSamples, distance\_metric):

    def data_labels_distances(index, classifier, numSamples, distance_metric):

""" OTHER USEFULL FUNCTIONS """

def generateTS(size=100,min=0,max=10):
    mlist = []
    for i in range(size):
        mlist.append(round(uniform(min,max),2))
    return mlist

def generateMockExp(ts):
    res = [(x,uniform(0,1)) for x in range(0,len(ts))]
    return res