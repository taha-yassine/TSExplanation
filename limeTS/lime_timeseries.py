"""
Functions for explaining time series classifiers.
"""
from functools import partial
import re

import itertools
import numpy as np
import scipy as sp
import sklearn
import pandas as pd
from sklearn.utils import check_random_state
import math
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import copy
import shutil
import os



from matplotlib import pyplot as plt

from random import *
from decimal import Decimal



import explanation
import lime_base

class TSDomainMapper(explanation.DomainMapper):

    """def __init__(self, raw, ts_seg):
        # sous serie de base ou sous serie segmentee ?
        self.ts_seg = ts_seg
        self.raw = raw

    def map_exp_ids(ts, positions=False):
        
        if positions:
            exp = [(self.mTS[x[0]],x[0], x[1])
                   for x in exp]
        else:
            exp = [(self.mTS[x[0]], x[1]) for x in exp]
        return exp
    """

    def map_exp_ids(self, exp, **kwargs):
        """Maps the feature ids to concrete names.

        Default behaviour is the identity function. Subclasses can implement
        this as they see fit.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]
            kwargs: optional keyword arguments

        Returns:
            exp: list of tuples [(name, weight), (name, weight)...]
        """
        return exp

    """
    def visualize_instance_html(self):
        #Adds textimeseries with highlighted sub_timeseries to visualization.
        plt.plot(self.raw)
        #plt.show()
        plt.savefig('temp.png')
        return 0
    """

    def save_to_file(self, file_path, exp, myTs, num_cuts, result_class):
        res = "Resultat : " + result_class
        sorted_weights = sorted(exp.as_list(), key=lambda tup: tup[1], reverse=True)
        weights = " Poids : " + (str(sorted_weights))[1:(len(str(sorted_weights))-1)]
        name = os.path.basename(file_path)
        os.makedirs(file_path, exist_ok=True)
        shutil.copy2("../GUI/icons/TSExplanation.ico", file_path + "/TSExplanation.ico")
        shutil.copy2("../GUI/icons/TSExplanation_long.png", file_path + "/TSExplanation_long.png")
        _, figure = exp.domain_mapper.as_pyplot(exp, myTs, num_cuts)
        size = figure.get_size_inches()
        figure.set_size_inches(10.5, 3.0)
        figure.savefig(file_path + "/" + name + ".png")
        figure.set_size_inches(size)
        canvas = FigureCanvas(figure)
        canvas.draw()
        fichier = open(file_path + "/" + name + ".html", "a")
        fichier.write('''
        <!doctype html>
        <html>
            <head>
                <meta charset="utf-8">
                <title>''' + name + '''</title>
                <link rel="icon" type="image/x-icon" href="TSExplanation.ico">
            </head>
            <body style="text-align: center;">
                <br>
                <img src="TSExplanation_long.png" alt="TSExplanation" width="600" /><br><br>
                <p style="font-family: sans-serif;">''' + res +'''</p>
                <img src="''' + name + '''.png" alt="Explication" /><br><br>
                <p style="font-family: sans-serif;">''' + weights +'''</p><br>
            </body>
        </html>
        ''')
        fichier.close()

    def as_pyplot(self, exp, ts, num_cut):
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        series = pd.Series(ts)
        segment_length = math.ceil(len(series) / num_cut)
        #colors = {-3:"black", 0:"#ff0000", 0.1:"#ff2801", 0.2:"#ff5101", 0.3:"#ff7900", 0.4:"#ffa100", 0.5:"#ffc900", 0.6:"#d2bd06", 0.7:"#a4b10b", 0.8:"#76a310", 0.9:"#489816", 1:"#1b8b1b"}
        colors = {-3: "#e3e3e2", 0: "#cfdcce", 0.1: "#bdd6bd", 0.2: "#98c998", 0.3: "#72bc72", 0.4: "#61b761",
                  0.5: "#51b151", 0.6: "#41ac41", 0.7: "#31a731", 0.8: "#21a121", 0.9: "#109b11", 1: "#019600"}

        feature = []
        oldweights = []
        for y in range(0, len(exp.as_list())):
            f, w = exp.as_list()[y]
            feature.append(f)
            oldweights.append(abs(w))
        print(feature)
        print(oldweights)
        """Normalize : """
        weights = []
        for z in range(0, len(oldweights)):
            weights.append((oldweights[z]-min(oldweights))/(max(oldweights)-min(oldweights)))
        print(weights)
        for i in range(0, num_cut):
            weight = -3
            if i in feature:
                weight = round(weights[feature.index(i)],1)
            start = int(i*segment_length)
            if i==(num_cut - 1):
                segment_length = len(series) - 1 - i*segment_length
            x = np.arange(0.0, len(series), 1)
            y1 = np.ma.masked_where((x<start), ts)
            curve = np.ma.masked_where((x>(start + segment_length)), y1)
            ax.plot(curve, linestyle='-', color=colors[weight])
            plt.axvline(x=start, linewidth=1, color="#D3D3D3")
        return canvas, fig


class IndexedTS(object):

    def __init__(self, raw_ts,bow=True):
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

    def length_timeSeries(self):
        """Returns the length of the raw time series"""
        return len(self.raw)

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
        """Returns a np array with indices to id_ (int) occurrences"""
        if self.bow:
            return self.timeseries_start[self.positions[id_]]
        else:
            return self.timeseries_start[[self.positions[id_]]]

    def tsSegmentation(self, seg_length=5, segmentationType=None):
        res = [self.raw[x:x+seg_length] for x in range(0,len(self.raw),seg_length)]
        return res

    def inverse_removing(self, first_value,second_value):
        """Returns a time series after removing the appropriate values in the interval.
        Args:
            first_value: beginning of the interval
            second_value: end of the interval
        Returns:
            original raw time series with appropriate values removed.
        """
        for i in range(first_value, second_value):
            self.raw[i] = 0


    def __get_idxs(self, values):
        """Returns indexes to appropriate values."""
        if self.bow:
            return list(itertools.chain.from_iterable(
                [self.positions[z] for z in values]))
        else:
            return self.positions[values]


class TSExplainer(object):
    """Explains time series classifiers."""
    def __init__(self,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 bow=True,
                 random_state=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            bow: if True (bag of words), will perturb input data by removing
                all occurrences of individual sub_timeseries. Explanations will be in
                terms of these sub_timeseries. Otherwise, will explain in terms of
                sub_timeseries-positions, so that a sub_timeseries may be important the first time
                it appears and unimportant the second. Only set to false if the
                classifier uses sub_timeseries order in some way (bigrams, etc).
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(kernel_fn, verbose)
        self.class_names = class_names
        self.feature_selection = feature_selection
        self.bow = bow

    def data_labels_distances(self, indexed_ts, classifier_fn, num_cuts, num_samples, training_set, distance_metric='cosine'):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly removing sub time series from
        the instance, and predicting with the classifier. Uses cosine distance
        to compute distances between original and perturbed instances.
        Args:
            indexed_ts: time series (IndexedTS) to be explained,
            classifier_fn: classifier prediction probability function, which
                takes a time series and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity.


        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of tokens in indexed_string. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: cosine distance between the original instance and
                    each perturbed instance (computed in the binary 'data'
                    matrix), times 100.
        """
        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100



        nbvalues_by_cut = math.ceil(indexed_ts.length_timeSeries() / num_cuts)
        sample = np.random.randint(1, num_cuts + 1, num_samples - 1)
        data = np.ones((num_samples, num_cuts))
        features_range = range(num_cuts)
        timeseries = pd.Series(indexed_ts.raw_timeSeries()).copy()
        inverse_data = [timeseries]
        for i, size in enumerate(sample, start=1):
            inactive = np.random.choice(features_range, size,
                                                replace=False)
            data[i, inactive] = 0
            tmp_timeseries = timeseries.copy()
            for i, inac in enumerate(inactive, start=1):
                index = inac * nbvalues_by_cut
                #tmp_timeseries.loc[index:(index + nbvalues_by_cut)] = np.mean(training_set.mean())
                tmp_timeseries.loc[index:(index + nbvalues_by_cut)] = 0
            inverse_data.append(tmp_timeseries)
        labels = classifier_fn.predict_proba(inverse_data)
        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, labels, distances

    def explain_instance(self,
                        tsToExplain,
                        classifier_fn,
                        training_set,
                        num_cuts=24,
                        num_features=10,
                        num_samples=1000,
                        labels=(1,),
                        top_labels=None,
                        distance_metric='cosine',
                        model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).

        Args:
            tsToExplain: raw time series to be explained.
            classifier_fn: classifier prediction probability function, which
                takes a list of d time series and outputs a (d, k) numpy array with
                prediction probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations. 
        """

        indexed_ts = IndexedTS(tsToExplain, bow=self.bow)
        domain_mapper = TSDomainMapper()
        data, yss, distances = self.data_labels_distances(indexed_ts, classifier_fn, num_cuts, num_samples, training_set)
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper, class_names=self.class_names)
        ret_exp.predict_proba = yss[0]
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(data, yss, distances, label,
                                                                                       num_features,
                                                                                       feature_selection=self.feature_selection)
        return ret_exp


""" OTHER USEFULL FUNCTIONS """

def generateTS(size=100,min=0,max=10):
    mlist = []
    for i in range(size):
        mlist.append(round(uniform(min,max),2))
    return mlist

def generateMockExp(ts):
    res = [(x,uniform(0,1)) for x in range(0,len(ts))]
    return res
