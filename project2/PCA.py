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

def reduce_data(U, S, k):
    return np.matmul(U[:, :k], np.diag(S[:k]))

def plot_2PC(data, df, annotate=False):
    plt.scatter(data[:, 0], data[:, 1])
    if annotate:
        for i in range(data.shape[0]):
            plt.annotate(df.loc[i, "Name"], (data[i, 0], data[i, 1]))
    plt.title("2 Component PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()
    plt.show()
