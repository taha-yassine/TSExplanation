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

    def __init__():

    def map_exp_ids(ts, positions=False):

    def visualize_instance_html(ts):


class IndexedTS(object):

    def __init__():

    def raw_timeSeries():

    def num_timeSubSeries():

    def timeSubSeries(id):
   
    def timeSeries_position():

    def inverse_removing(sub_ts):

class TSExplainer(object):

    def __init__():

    def explain_instance(tsToExplain, classifier, label, numFeatures, numSamples, distance\_metric):

    def data_labels_distances(index, classifier, numSamples, distance_metric):