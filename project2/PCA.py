import numpy as np
import matplotlib.pyplot as plt

def get_svd(data):
    return np.linalg.svd(data)

def graph_scree(S):
    eigvals = S**2 / np.cumsum(S)[-1]
    sing_vals = np.arange(S.shape[0]) + 1
    plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Percentage of Variance Covered')
    plt.show()

def reduce_data(data, V, k):
    V = V.T
    return np.matmul(data, V[:, :k])

def plot_2PC(data):
    X = data[:, 0]
    Y = data[:, 1]
    plt.scatter(X, Y)
    plt.title("2 Component PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()
    plt.show()
