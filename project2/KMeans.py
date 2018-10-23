import math
import matplotlib.pyplot as plt
import numpy as np
import sys

def _initialize_k_means(data, k):
    centroids = np.empty((k, data.shape[1]+1))
    for i in range(k):
        for col in range(data.shape[1]):
            bound = np.amax(data[:, col])
            c = np.random.rand() * bound
            centroids[i, col] = c
    for i in range(k):
        centroids[i, -1] = i
    zero_data = np.zeros((data.shape[0], data.shape[1]+1))
    zero_data[:, :-1] = data[:, :]
    return zero_data, centroids

def calc_distance(p1, p2):
    return np.linalg.norm(p1-p2)

def _calc_new_centroid(cluster, cent):
    if cluster.ndim == 1:
        if cluster.size == 0:
            return cent
        new_cent = cluster
    else:
        new_cent = np.mean(cluster[:, :-1], axis=0)
        new_cent = np.append(new_cent, cluster[0, -1])
    return new_cent

def k_means_train(data, k):
    train_data, centroids = _initialize_k_means(data, k)
    thresh = 0.001
    delta = 1
    while delta > thresh:
        tmp = []
        for elem in train_data:
            min_distance = sys.float_info.max
            label = -1
            for cent in centroids:
                dist = calc_distance(cent, elem)
                if dist < min_distance:
                    min_distance = dist
                    label = cent[-1]
            elem[-1] = label
            tmp.append(elem)
        train_data = np.array(tmp)
        labelled = []
        for i in range(k):
            labelled.append(np.array([]))
        for elem in train_data:
            if labelled[int(elem[-1])].size == 0:
                labelled[int(elem[-1])] = elem
            else:
                labelled[int(elem[-1])] = np.vstack((labelled[int(elem[-1])], elem))
        delta = 0
        cents = np.array([])
        for i, cluster in enumerate(labelled):
            if cents.size == 0:
                cents = _calc_new_centroid(cluster, centroids[i, :])
                delta += (calc_distance(centroids[i, :], cents))**2
            else:
                cents = np.vstack((cents, _calc_new_centroid(cluster, centroids[i, :])))
                tmpn = (calc_distance(centroids[i, :], cents[-1, :]))**2
                delta += tmpn
        centroids = cents
        delta /= k
    return centroids, train_data

def get_maximum_intra_dist(data, k):
    labelled = []
    for i in range(k):
        labelled.append(np.array([]))
    for elem in data:
        if labelled[int(elem[-1])].size == 0:
            labelled[int(elem[-1])] = elem
        else:
            labelled[int(elem[-1])] = np.vstack((labelled[int(elem[-1])], elem))
    dists = []
    for cluster in labelled:
        max_dist = 0
        if cluster.size == 0:
            dists.append(0)
            continue
        for i in range(cluster.shape[0]-1):
            for j in range(i+1, cluster.shape[0]):
                dist = calc_distance(cluster[i], cluster[j])
                if dist > max_dist:
                    max_dist = dist
        dists.append(max_dist)
    return np.amax(np.array(dists))

def get_minimum_inter_dist(centroids):
    dists = []
    if centroids.size == 0:
        return -1
    for i in range(centroids.shape[0]):
        for j in range(i+1, centroids.shape[0]):
            dists.append(calc_distance(centroids[i, :], centroids[j, :]))
    return np.amin(np.array(dists))

def get_dunn(centroids, data):
    intra = get_maximum_intra_dist(data, centroids.shape[0])
    inter = get_minimum_inter_dist(centroids)
    return inter / intra

def plot_dunns(dunns):
    plt.plot(range(2, 10), dunns, "ro-")
    plt.title("Dunn Indices vs k")
    plt.xlabel("k")
    plt.ylabel("Dunn Index")
    plt.grid()
    plt.show()

def find_UT(data, df):
    cluster = data[0, -1]
    inds = []
    for i in range(1, data.shape[0]):
        if data[i, -1] == cluster:
            inds.append(i)
    univs = []
    for i in inds:
        univs.append(df.loc[i, "Name"])
    return univs

def plot_k_means_2D(data, k):
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    labelled = []
    for i in range(k):
        labelled.append(np.array([]))
    for elem in data:
        if labelled[int(elem[-1])].size == 0:
            labelled[int(elem[-1])] = elem
        else:
            labelled[int(elem[-1])] = np.vstack((labelled[int(elem[-1])], elem))
    cind = 0
    for elem in labelled:
        if elem.size == 0:
            continue
        plt.scatter(elem[:, 0], elem[:, 1], c=colors[cind])
        cind += 1
        if cind == len(colors):
            cind = 0
    plt.title("K Means Clustering")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()
    plt.show()
