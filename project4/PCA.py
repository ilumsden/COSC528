import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

def normalize(data):
    return zscore(data, axis=0)

def get_pc(V, num_comps):
    return V.T[:, :num_comps]

def decomp(data):
    U, S, V = np.linalg.svd(data)
    return S, V

def graph_scree(S):
    eigvals = S**2 / np.cumsum(S)[-1]
    sing_vals = np.arange(S.shape[0]) + 1
    plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Percentage of Variance Covered')
    plt.show()

def reduce_data(data, pcs):
    return np.dot(data, pcs)

def plot_2PC(data):
    plt.scatter(data[:, 0], data[:, 1])
    plt.title("2 Component PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()
    plt.show()
