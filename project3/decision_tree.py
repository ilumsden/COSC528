import pandas as pd
import numpy as np
from collections import deque
from functools import partial
from math import ceil, floor, log

def entropy(p, q):
    if p in (0, 1):
        pe = 0
    else:
        pe = -p * log(p, 2) - (1-p)*log((1-p),2)
    if q in (0, 1):
        qe = 0
    else:
        qe = -q * log(q, 2) - (1-q)*log((1-q),2)
    return pe + qe

def gini(p, q):
    return 2 * p * (1-p) + 2 * q * (1-q)

def misclasification_error(p, q):
    return 1 - max(p, 1-p) + 1 - max(q, 1-q)

class TreeNode:

    def __init__(self, feature, splitter, parent=None, value=-1):
        self.feature = feature
        self.splitter = splitter
        self.parent = parent
        self.left = None
        self.right = None
        self.value = value

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right

    def print_node(self):
        print("Feature #:", self.feature)
        if self.value == -1:
            print("    splitter:", self.splitter)
            print("    Left Link:")
            self.left.print_node()
            print("    Right Link:")
            self.right.print_node()
        else:
            print("    Leaf Value:", self.value)

class DecisionTree:

    def __init__(self, train_data, train_labels, impurity="entropy",
                 threshold=0.0, max_depth=10):
        self.threshold = threshold
        self.max_depth = max_depth
        self.impurity = impurity
        if isinstance(train_data, pd.DataFrame):
            self.train_data = train_data.values[:, :]
        elif isinstance(train_data, np.ndarray):
            self.train_data = train_data[:, :]
        else:
            raise TypeError("DecisionTree only supports Pandas DataFrames \
                             or Numpy Arrays.")
        if isinstance(train_labels, (pd.DataFrame, pd.Series)):
            strain_data = train_labels.values[:]
        elif isinstance(train_labels, np.ndarray):
            strain_data = train_labels[:]
        else:
            raise TypeError("DecisionTree only supports Pandas DataFrames \
                             or Numpy Arrays.")
        strain_data = np.reshape(strain_data, (strain_data.size, 1))
        self.train_data = np.append(self.train_data, strain_data, axis=1)
        self.tree = []
        self.train()

    def get_impurity(self, p, q):
        if self.impurity == "gini":
            return gini(p, q)
        elif self.impurity == "entropy":
            return entropy(p, q)
        elif self.impurity == "misclassification error":
            return misclasification_error(p, q)
        else:
            raise ValueError("Invalid Impurity type. Must be \"gini,\" \"entropy,\" \
                              or \"misclassification error\"")

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
        mins = np.amin(data, axis=0)
        feature_min = mins[feature]
        maxs = np.amax(data, axis=0)
        feature_max = maxs[feature]
        range_min = floor(feature_min) + 0.5
        range_max = ceil(feature_max)
        for i in np.arange(range_min, range_max, 0.5):
            s1 = data[(data[:, feature] < i)]
            if s1.shape[0] == 0:
                continue
            c0s1 = s1[(s1[:, -1] == 0)]
            p = c0s1.shape[0] / s1.shape[0]
            s2 = data[(data[:, feature] >= i)]
            if s2.shape[0] == 0:
                continue
            c0s2 = s2[(s2[:, -1] == 0)]
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

    def _get_curr_depth(self, node):
        if node is None:
            return 0
        ldepth = self._get_curr_depth(node.left)
        rdepth = self._get_curr_depth(node.right)
        if ldepth > rdepth:
            return ldepth + 1
        return rdepth + 1

    def _check_max_depth(self, parent):
        if len(self.tree) == 0:
            return True
        curr_depth = self._get_curr_depth(self.tree[0])
        elem_depth = 0
        curr_parent = parent
        while curr_parent is not None and curr_parent.parent is not None:
            elem_depth += 1
            curr_parent = curr_parent.parent
        elem_depth += 2
        if elem_depth == curr_depth:
            return (curr_depth + 1 < self.max_depth)
        elif elem_depth < curr_depth:
            return True
        else:
            return (curr_depth + 2 < self.max_depth)

    def _add_leaf(self, value, parent, isleft, queue):
        leaf = TreeNode(-1, -1, parent, value)
        if isleft:
            parent.set_left(leaf)
        else:
            parent.set_right(leaf)
        self.tree.append(leaf)
        return queue

    def _get_node(self, data, parent, isleft, queue):
        split_data = []
        for i in range(data.shape[1] - 1):
            split = self._find_best_split(i, data)
            if not all([i == -1 for i in split]):
                split_data.append(split)
        split_data.sort(key=lambda x: x[1])
        if len(split_data) == 0:
            num0 = data[(data[:, -1] == 0)].shape[0]
            num1 = data[(data[:, -1] == 1)].shape[0]
            p0 = num0 / data.shape[0]
            p1 = num1 / data.shape[1]
            if p0 > p1:
                self._add_leaf(0, parent, isleft, queue)
            else:
                self._add_leaf(1, parent, isleft, queue)
            return queue
        s1 = data[(data[:, split_data[0][0]] < split_data[0][2])]
        s2 = data[(data[:, split_data[0][0]] >= split_data[0][2])]
        if self._check_max_depth(parent) and split_data[0][1] > self.threshold \
                and split_data[0][1] != 0:
            node = TreeNode(split_data[0][0], split_data[0][2], parent)
            if parent is not None:
                if isleft:
                    parent.set_left(node)
                else:
                    parent.set_right(node)
            self.tree.append(node)
            if split_data[0][3] != 0 and split_data[0][3] != 1:
                queue.appendleft(partial(self._get_node, s1, node, True))
            elif split_data[0][3] == 0:
                queue.appendleft(partial(self._add_leaf, 1, node, True))
            else:
                queue.appendleft(partial(self._add_leaf, 0, node, True))
            if split_data[0][4] != 0 and split_data[0][4] != 1:
                queue.appendleft(partial(self._get_node, s2, node, False))
            elif split_data[0][4] == 0:
                queue.appendleft(partial(self._add_leaf, 1, node, False))
            else:
                queue.appendleft(partial(self._add_leaf, 0, node, False))
        else:
            node = TreeNode(split_data[0][0], split_data[0][2], parent)
            if parent is not None:
                if isleft:
                    parent.set_left(node)
                else:
                    parent.set_right(node)
            self.tree.append(node)
            both_500 = False
            if split_data[0][3] < 0.5:
                queue.appendleft(partial(self._add_leaf, 1, node, True))
            elif split_data[0][3] > 0.5:
                queue.appendleft(partial(self._add_leaf, 0, node, True))
            else:
                if split_data[0][4] < 0.5:
                    queue.appendleft(partial(self._add_leaf, 0, node, True))
                elif split_data[0][4] > 0.5:
                    queue.appendleft(partial(self._add_leaf, 1, node, True))
                else:
                    both_500 = True
            if not both_500:
                if split_data[0][4] < 0.5:
                    queue.appendleft(partial(self._add_leaf, 1, node, False))
                elif split_data[0][4] > 0.5:
                    queue.appendleft(partial(self._add_leaf, 0, node, False))
                else:
                    if split_data[0][3] < 0.5:
                        queue.appendleft(partial(self._add_leaf, 0, node, False))
                    elif split_data[0][3] > 0.5:
                        queue.appendleft(partial(self._add_leaf, 1, node, False))
                    else:
                        both_500 = True
            else:
                queue.appendleft(partial(self._add_leaf, 0, node, True))
                queue.appendleft(partial(self._add_leaf, 1, node, False))
        return queue

    def _train(self):
        queue = deque()
        start_func = partial(self._get_node, self.train_data[:, :], None, False)
        queue.appendleft(start_func)
        while len(queue) > 0:
            func = queue.pop()
            queue = func(queue)

    def train(self):
        self._train()

    def _classify(self, row, node):
        if node is None:
            raise RuntimeError("Classification reached a padding None, but it should have \
                                stopped on the parent leaf node.")
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
        if isinstance(data, (pd.DataFrame, pd.Series)):
            cdata = data.values
            if cdata.ndim == 1:
                cdata = cdata[:]
            else:
                cdata = cdata[:, :]
        else:
            cdata = data
            if cdata.ndim == 1:
                cdata = cdata[:]
            else:
                cdata = cdata[:, :]
        classes = []
        if cdata.ndim == 1:
            classes.append(self._classify(cdata, self.tree[0]))
        else:
            for row in cdata:
                classes.append(self._classify(row, self.tree[0]))
        return classes

    def print_tree(self):
        self.tree[0].print_node()
