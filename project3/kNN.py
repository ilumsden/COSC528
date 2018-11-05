import pandas as pd
import numpy as np
from numpy.linalg import norm

def generate_confusion_matrix(correct_labels, generated_labels):
    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for c, g in zip(correct_labels, generated_labels):
        if c == g and c == 0:
            tn += 1
        elif c == g and c == 1:
            tp += 1
        elif c != g and c == 0:
            fp += 1
        elif c != g and c == 1:
            fn += 1
        else:
            raise ValueError("Invalid class ID passed in correct_labels")
    return tn, fn, tp, fp

def get_accuracy(tn, fn, tp, fp):
    return (tn + tp) / (tn + tp + fn + fp)

def get_true_positive_rate(tp, fn):
    return tp / (tp + fn)

def get_precision(tp, fp):
    return tp / (tp + fp)

def get_true_negative_rate(tn, fp):
    return tn / (tn + fp)

def get_f1_score(ppv, tpr):
    return (2 * ppv * tpr) / (ppv + tpr)

class kNearestNeighbors:

    def __init__(self, k, train_data, train_labels):
        self.k = k
        if isinstance(train_data, pd.DataFrame):
            self.train_data = train_data.values
        elif isinstance(train_data, np.array):
            self.train_data = train_data
        else:
            raise TypeError("kNearestNeighbors only supports Pandas DataFrames \
                             or Numpy Arrays.")
        if isinstance(train_labels, (pd.DataFrame, pd.Series)):
            self.train_labels = train_labels.values
        elif isinstance(train_data, np.array):
            self.train_labels = train_labels
        else:
            raise TypeError("kNearestNeighbors only supports Pandas DataFrames \
                             or Numpy Arrays.")

    def _classify_one(self, v1):
        neighbors = []
        for i, row in enumerate(self.train_data):
            dist = norm(v1 - row)
            if dist == 0:
                return self.train_labels[i]
            neighbors.append([dist, i])
        neighbors.sort(key=lambda x: x[0])
        nearest = neighbors[:self.k]
        wsum = 0
        for i, _ in enumerate(nearest):
            nearest[i][0] = 1 / nearest[i][0]
            wsum += nearest[i][0]
        for i, _ in enumerate(nearest):
            nearest[i][0] /= wsum
        score0 = 0
        score1 = 0
        for elem in nearest:
            if self.train_labels[elem[1]] == 0:
                score0 += elem[0]
            else:
                score1 += elem[0]
        return 0 if score0 > score1 else 1

    def classify(self, data):
        if isinstance(data, pd.DataFrame):
            cdata = data.values
            cdata.astype(dtype=np.float64, copy=False)
        elif isinstance(data, np.array):
            cdata = data
        else:
            raise TypeError("kNearestNeighbors only supports Pandas DataFrames \
                             or Numpy Arrays.")
        classes = []
        for i, row in enumerate(cdata):
            classes.append(self._classify_one(row))
        return classes

    def updateK(self, newk):
        self.k = newk
