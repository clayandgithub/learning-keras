import time
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def generate_data(random_seed, n_samples):
    np.random.seed(random_seed)
    X, y = datasets.make_moons(n_samples, noise=0.20)
    return X, y

def visualize(X, y, pred_func):
    plt.title("nn_theano_gpu_classification")
    plot_decision_boundary(lambda x:pred_func(x), X, y)

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()