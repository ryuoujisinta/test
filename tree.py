#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:32:57 2020

@author: daichi_morita
"""

import numpy as np


class Node:
    def __init__(self, feature_ind, split_val):
        self.feature_ind = feature_ind
        self.split_val = split_val
        self.left = None
        self.right = None

    def next(self, x):
        if x[self.feature_ind] < self.split_val:
            return self.left, True
        else:
            return self.right, False

    def set_leaf(self, x, y):
        x_j = x[:, self.feature_ind]
        self.left = y[x_j < self.split_val].mean()
        self.right = y[x_j >= self.split_val].mean()


class Tree:
    def __init__(self):
        self.root = None

    def _predict(self, data):
        leaf_ind = []
        node, ind = self.root.next(data)
        leaf_ind.append(ind)
        while isinstance(node, Node):
            node, ind = node.next(data)
            leaf_ind.append(ind)
        return node, leaf_ind

    def predict(self, x):
        leaves = []
        for data in x:
            leaf, _ = self._predict(data)
            leaves.append(leaf)
        return np.array(leaves)

    def fit(self, x, y, n, n_min=10):
        for i in range(n):
            if self._fit(x, y, n_min):
                print(i)
                break

    def _fit(self, x, y, n_min):
        d = x.shape[1]
        # grouping
        if self.root is None:
            group_inds = [np.arange(len(y))]
            leaf_inds = [None]
        else:
            leaf_inds = []
            group_inds = []
            for i, data in enumerate(x):
                _, leaf_ind = self._predict(data)
                if leaf_ind in leaf_inds:
                    ind = leaf_inds.index(leaf_ind)
                    group_inds[ind].append(i)
                else:
                    group_inds.append([i])
                    leaf_inds.append(leaf_ind)

        max_gain = -np.inf
        opt_split = None
        total_loss = 0
        for i, group in enumerate(group_inds):
            if len(group) < n_min:
                continue
            x_group = x[group]
            y_group = y[group]

            L_group = (y_group**2).mean() - y_group.mean()**2
            L_group *= len(group)
            total_loss += L_group

            L_min_group = np.inf
            ind_min_group = None
            # each feature
            for j in range(d):
                x_j = x_group[:, j]
                uni = np.unique(x_j)
                L_min = np.inf
                s_min = None
                # each split
                for s in np.convolve(uni, [0.5, 0.5], "valid"):
                    y_left = y_group[x_j < s]
                    y_right = y_group[x_j >= s]
                    if len(y_left) < n_min or len(y_right) < n_min:
                        continue
                    L_left = (y_left**2).mean() - y_left.mean()**2
                    L_right = (y_right**2).mean() - y_right.mean()**2
                    L = L_left * len(y_left) + L_right * len(y_right)
                    if L < L_min:
                        L_min = L
                        s_min = s

                if L_min < L_min_group:
                    L_min_group = L_min
                    ind_min_group = [j, s_min]
            if L_min_group == np.inf:
                continue
            gain = L_group - L_min_group
            if gain > max_gain:
                max_gain = gain
                opt_split = [group, leaf_inds[i], ind_min_group]

        if max_gain == -np.inf:
            return True

        total_loss -= max_gain
        total_loss /= len(y)

        # add node
        group, leaf_ind, [feature_ind, split_val] = opt_split
        print(feature_ind, split_val, total_loss, max_gain / len(group))
        added_node = Node(feature_ind, split_val)
        added_node.set_leaf(x[group], y[group])
        if self.root is None:
            self.root = added_node
        else:
            node = self.root
            for is_left in leaf_ind[:-1]:
                if is_left:
                    node = node.left
                else:
                    node = node.right
            if leaf_ind[-1]:
                node.left = added_node
            else:
                node.right = added_node

        return False
