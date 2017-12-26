

import progressbar
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np

def calculate_covariance_matrix(X, Y=None):
    """ Calculate the covariance matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = X.shape[0]

    covariance_matrix = (1.0 / (n_samples-1.0)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    print('(X - X.mean(axis=0))***1234',(1.0 / (n_samples-1.00)))

    return np.array(covariance_matrix, dtype=float)

def _transform( X, dim):
    covariance = calculate_covariance_matrix(X)

    print('covariance_matrix******123', covariance)

    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    # Sort eigenvalues and eigenvector by largest eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx][:dim]
    eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :dim]
    # Project the data onto principal components
    X_transformed = X.dot(eigenvectors)

    return X_transformed

def plot_in_2d_(X, y=None, title=None, accuracy=None, legend_labels=None):
    cmap = plt.get_cmap('viridis')

    X_transformed = _transform(X, dim=2)
    x1 = X_transformed[:, 0]
    x2 = X_transformed[:, 1]
    class_distr = []

    y = np.array(y).astype(int)

    colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

    # Plot the different class distributions
    for i, l in enumerate(np.unique(y)):
        _x1 = x1[y == l]
        _x2 = x2[y == l]
        _y = y[y == l]
        class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

    # Plot legend
    if not legend_labels is None:
        plt.legend(class_distr, legend_labels, loc=1)

    # Plot title
    if title:
        if accuracy:
            perc = 100 * accuracy
            plt.suptitle(title)
            plt.title("Accuracy: %.1f%%" % perc, fontsize=10)
        else:
            plt.title(title)

    # Axis labels
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.show()