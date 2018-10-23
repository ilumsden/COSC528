import numpy as np
from scipy.stats import norm
from KMeans import get_maximum_intra_dist

def calc_prob(val, mu, sig, lam):
    p = lam
    for i in range(len(val)):
        if sig[i][i] == 0:
            continue
        p *= norm.pdf(val[i], mu[i], sig[i][i])
    return p

def expectation(data, params, k):
    for i in range(data.shape[0]):
        probs = []
        for j in range(k):
            probs.append(calc_prob(data[i], params[2*j], params[2*j+1], params[-1][j]))
        probs = np.array(probs)
        key = np.argmax(probs)
        data[i, -1] = key
    return data

def maximization(data, params, k):
    clusters = []
    percents = []
    for i in range(k):
        clusters.append(np.array([r for r in data if r[-1] == i]))
        percents.append(len(clusters[-1])/data.shape[0])
    new_params = params[:]
    for i in range(k):
        if clusters[i].size == 0:
            continue
        # Mu
        new_params[2*i] = [np.mean(a) for a in clusters[i].T]
        # Sig
        tmp = [np.std(a) for a in clusters[i].T]
        new_params[2*i+1] = np.diag(tmp)
    # Lambda
    new_params[-1] = [p for p in percents]
    return new_params

def calc_distance_params(old_params, new_params, k):
    dist = 0
    for i in range(k):
        for j in range(len(old_params[2*i])):
            dist += (old_params[2*i][j] - new_params[2*i][j])**2
    return dist ** 0.5

def em_clustering(data, k):
    params = []
    num_features = data.shape[1]
    for i in range(k):
        params.append([i for j in range(num_features)])
        tmp = [i for j in range(num_features)]
        params.append(np.diag(tmp))
    params.append([1 for i in range(k)])
    train_data = np.zeros((data.shape[0], data.shape[1]+1))
    train_data[:, :-1] = data[:, :]
    epsilon = 0.01
    delta = 1
    while delta > epsilon:
        data = expectation(data, params, k)
        new_params = maximization(data, params, k)
        delta = calc_distance_params(params, new_params, k)
        params = new_params
    return data, params

def calc_distance(mu1, mu2):
    dist = 0
    for i in range(len(mu1)):
        dist += (mu1[i] - mu2[i])**2
    return dist ** 0.5

def get_minimum_inter_em(params, k):
    dists = []
    if len(params) == 0:
        return -1
    for i in range(k-1):
        for j in range(i+1, k):
            dists.append(calc_distance(params[2*i], params[2*j]))
    return np.amin(np.array(dists))

def get_dunn_em(data, params, k):
    return get_minimum_inter_em(params, k) / get_maximum_intra_dist(data, k)
