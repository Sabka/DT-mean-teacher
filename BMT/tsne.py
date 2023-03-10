import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def visualize(X, labels, filename):

    colors = ["blue", "red", "green", "yellow"]

    labels_set = set(labels)
    labels_cols_dict = {}
    for i, lab in enumerate(labels_set):
        labels_cols_dict[lab] = colors[i]

    label_colors = []
    for l in labels:
        label_colors.append(labels_cols_dict[l])


    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
    x, y = [], []
    for dx, dy in X_embedded:
        x.append(dx)
        y.append(dy)

    plt.scatter(x, y, c=label_colors)
    plt.savefig(filename)

visualize(np.array([[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 0, 0], [1, 1, 1, 1], [1,1,1,0]]), [0,1,1,1,1], "xx.png")