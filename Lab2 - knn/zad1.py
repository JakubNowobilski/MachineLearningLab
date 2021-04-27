import make_dataset
import matplotlib.pyplot as plt
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def plot_boundary(classifier, ds):
    X, labels, classes = ds
    X = StandardScaler().fit_transform(X)

    h = .02

    plt.figure(figsize=(12, 12))
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
    ax = plt.subplot()
    ax.set_title("Custom dataset")

    classifier.fit(X, labels)

    Z = classifier.predict(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2)

    ax.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.8, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    plt.show()


ds = make_dataset.make_dataset()
X = StandardScaler().fit_transform(ds[0])
classifiers = [KNeighborsClassifier(1),
               KNeighborsClassifier(13),
               KNeighborsClassifier(1, metric='mahalanobis', metric_params={'V': numpy.cov(X.T)}),
               KNeighborsClassifier(9, weights='distance')]

for c in classifiers:
    plot_boundary(c, ds)
