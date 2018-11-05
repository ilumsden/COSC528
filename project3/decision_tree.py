import pandas as pd
import numpy as np
from math import log

def entropy(p, q):
    if p == 0:
        pe = 0
    else:
        pe = -p * log(p, 2)
    if q == 0:
        qe = 0
    else:
        qe = -q * log(q, 2)
    return pe + qe

def gini(p, q):
    return 2 * p * (1-p)

def misclasification_error(p, q):
    return 1 - max(p, 1-p)

class TreeNode:

    def __init__(self, feature, splitter, value=-1):
        self.feature = feature
        self.splitter = splitter
        self.left = None
        self.right = None
        self.value = value

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right

class DecisionTree:

    def __init__(self, train_data, train_labels, impurity="gini",
                 threshold=0.0, max_depth=10):
        self.threshold = threshold
        self.max_depth = max_depth
        self.impurity = impurity
        if isinstance(train_data, pd.DataFrame):
            self.train_data = train_data.values
        elif isinstance(train_data, np.array):
            self.train_data = train_data
        else:
            raise TypeError("DecisionTree only supports Pandas DataFrames \
                             or Numpy Arrays.")
        if isinstance(train_labels, (pd.DataFrame, pd.Series)):
            strain_data = train_data.values
        elif isinstance(train_labels, np.array):
            strain_data = train_data
        else:
            raise TypeError("DecisionTree only supports Pandas DataFrames \
                             or Numpy Arrays.")
        strain_data = np.resize(strain_data, (strain_data.size, 1))
        self.train_data = np.append(self.train_data, strain_data, axis=1)
        self.tree = []
        self.train()

    def get_impurity(self, p, q):
        if self.impurity == "gini":
            return gini(p, q)
        elif self.impurity == "entropy":
            return entropy(p, q)
        elif self.impurity == "misclasification error":
            return misclasification_error(p, q)
        else:
            raise ValueError("Invalid Impurity type. Must be \"gini,\" \"entropy,\" \
                              or \"misclasification error\"")

    def _check_if_split_exists(self, feature, split):
        for elem in self.tree:
            if feature == elem.feature and split == elem.splitter:
                return True
        return False

    def _find_best_split(self, feature, data):
        best_impurity = 11
        best_split_point = 0
        best_p = -1
        best_q = -1
        for i in range(1.5, 10, 0.5):
            s1 = data[(np.where(data[:, feature] < i))[0], :]
            c0s1 = s1[(np.where(s1[:, -1]) == 0)[0], :]
            p = c0s1.shape[0] / s1.shape[0]
            s2 = data[(np.where(data[:, feature] >= i))[0], :]
            c0s2 = s2[(np.where(s2[:, -1]) == 0)[0], :]
            q = c0s2.shape[0] / s2.shape[0]
            imp = self.get_impurity(p, q)
            if self._check_if_split_exists(feature, imp):
                continue
            if imp < best_impurity:
                best_impurity = imp
                best_split_point = i
                best_p = p
                best_q = q
        if best_impurity == 11:
            return [-1, -1, -1, -1, -1]
        else:
            return [feature, best_impurity, best_split_point, best_p, best_q]

    def _train(self, data):
        split_data = []
        for i in range(data.shape[1] - 1):
            split = self._find_best_split(i, data)
            if not [i == -1 for i in split].all():
                split_data.append(split)
        split_data.sort(key=lambda x: x[1])
        s1 = data[(np.where(data[:, split_data[0][0]] < split_data[0][2]))[0], :]
        s2 = data[(np.where(data[:, split_data[0][0]] >= split_data[0][2]))[0], :]
        if split_data[0][1] > self.threshold and split_data[0][1] != 0:
            node = TreeNode(split_data[0][0], split_data[0][2])
            if split_data[0][3] != 0:
                node.left = self._train(s1)
            else:
                node.left = TreeNode(-1, -1, s1[0, -1])
            if split_data[0][4] != 0:
                node.right = self._train(s2)
            else:
                node.right = TreeNode(-1, -1, s2[0, -1])
        else:
            node = TreeNode(split_data[0][0], split_data[0][2])
            node.left = TreeNode(-1, -1, s1[0, -1])
            node.right = TreeNode(-1, -1, s2[0, -1])
        self.tree.append(node)
        return node

    def train(self):
        self._train(self.train_data[:, :])

    def _classify(self, row, node):
        if row[node.feature] < node.splitter:
            if node.left.value != -1:
                return node.left.value
            else:
                return self._classify(row, node.left)
        else:
            if node.right.value != -1:
                return node.right.value
            else:
                return self._classify(row, node.right)

    def classify(self, data):
        classes = []
        for row in data:
            classes.append(self._classify(row, self.tree[0]))
        return classes
