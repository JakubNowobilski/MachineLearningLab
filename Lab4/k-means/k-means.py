import pandas
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from functools import reduce
from sklearn.metrics import davies_bouldin_score
from collections import Counter
from sklearn.decomposition import PCA

random_state = np.random.RandomState(0)


def random_centers(df: pandas.DataFrame, n_centers: int):
    centers = []
    for i in range(n_centers):
        center = []
        for c in df.columns:
            center.append(np.random.uniform(low=df[c].min(), high=df[c].max()))
        centers.append(center)
    return np.array(centers)


def mean_std_describe(all_scores):
    # calculate longest list
    # m - max length, l - next list
    max_len = reduce(lambda m, l: m if m > len(l) else len(l), all_scores, 0)
    # create data frame and extend all lists
    df = pandas.DataFrame()
    for i, scores in enumerate(all_scores):
        scores = np.hstack([scores, [None] * (max_len - len(scores))])
        df[i] = scores

    means = df.mean(axis=1)
    stds = df.std(axis=1)
    stds = stds.fillna(0)
    return np.array(means), np.array(stds)


def prepare_data(verbose=0):
    data_frame = pandas.read_csv("./menu.csv", ",")
    # remove unnecessary columns
    for c in data_frame.columns:
        if "%" in c:
            data_frame.pop(c)

    if verbose:
        pandas.set_option("display.max_rows", None, "display.max_columns", None)
        corr = data_frame.corr()
        print(corr)

    data_frame.pop("Category")
    data_frame.pop("Item")
    data_frame.pop("Serving Size")
    data_frame.pop("Total Fat")

    # normalize features
    data_frame = data_frame.apply(lambda c: scipy.stats.zscore(c))
    return data_frame


def various_centers(df: pandas.DataFrame, n_tests=10):
    X = df.values

    estimators = [
        ('k-means++',  KMeans, 5, 'k-means++', {}),
        ('random', KMeans, 5, 'random', {}),
        ('custom random', KMeans, 5, random_centers, {'df': df, 'n_centers': 5}),
    ]

    x = []
    y = []
    err = []
    for name, factory, n_clusters, init, init_params in estimators:
        all_scores = []
        for i in range(n_tests):
            if callable(init):
                est = factory(n_clusters=n_clusters, init=init(**init_params))
            else:
                est = factory(n_clusters=n_clusters, init=init)
            est.fit(X)
            all_scores.append(est.scores_)

        means, stds = mean_std_describe(all_scores)
        x.append(range(1, len(means) + 1))
        y.append(means)
        err.append(stds)

        # individual plot
        plt.figure()
        plt.title('Initialization method: ' + name + '\nNumber of tests: ' + str(n_tests))
        plt.xlabel('Iteration number')
        plt.ylabel('Davies - Bouldin score')
        plt.errorbar(range(1, len(means) + 1), means, yerr=stds, marker='o')

    plt.show()

    # combined plot
    plt.figure()
    plots = []
    legends = []
    for i, (name, _, _, _, _) in enumerate(estimators):
        p = plt.errorbar(x[i], y[i], err[i], marker='o')
        plots.append(p)
        legends.append(name)
    plt.title('Comparison of different initialization methods\nNumber of tests: ' + str(n_tests))
    plt.xlabel('Iteration number')
    plt.ylabel('Davies - Bouldin score')
    plt.legend(plots, legends)
    plt.show()


def best_k(df: pandas.DataFrame, n_tests=10):
    X = df.values
    k_range = range(2, 26)
    mean_scores = []
    for k in k_range:
        scores = []
        est = KMeans(n_clusters=k, init="k-means++")
        for i in range(n_tests):
            est.fit(X)
            scores.append(davies_bouldin_score(X, est.labels_))
        mean_scores.append([k, np.mean(scores), np.std(scores)])

    mean_scores = np.array(mean_scores)

    plt.figure()
    plt.title('Average Davies - Bouldin score with standard deviation for various k\nNumber of tests: ' + str(n_tests))
    plt.xlabel('Number of clusters')
    plt.ylabel('Davies - Bouldin score')
    plt.errorbar(mean_scores[:, 0], mean_scores[:, 1], yerr=mean_scores[:, 2], marker='o')
    plt.show()


def describe(df: pandas.DataFrame, n_clusters):
    X = df.values
    est = KMeans(n_clusters=n_clusters, init="k-means++")
    est.fit(X)
    surveys_per_cluster = Counter(est.labels_)
    print(surveys_per_cluster)
    print(est.cluster_centers_)


def visualize_clustering(df: pandas.DataFrame, n_clusters):
    X = df.values
    est = KMeans(n_clusters=n_clusters, init="k-means++")
    est.fit(X)
    pca = PCA(n_components=2)
    pca.fit(X)
    projected_2d = pca.transform(X)
    projected_2d = np.hstack((projected_2d, est.labels_.reshape(len(est.labels_), 1)))
    centers = pca.transform(est.cluster_centers_)
    centers = np.array(centers)

    plt.figure()

    for i in range(n_clusters):
        of_label_i = np.array([v for v in projected_2d if v[2] == i])
        plt.scatter(of_label_i[:, 0], of_label_i[:, 1], label=i, s=10)

    plt.scatter(centers[:, 0], centers[:, 1], marker='^', c='black', s=20, label='Centers')

    plt.title('2D projection using PCA dimensionality reduction')
    plt.legend()
    plt.show()


def visualize_data():
    data_frame = pandas.read_csv("./menu.csv", ",")
    X = prepare_data().values
    pca = PCA(n_components=2)
    pca.fit(X)
    projected_2d = pca.transform(X)
    categories = np.array(data_frame["Category"])
    projected_2d = np.hstack((projected_2d, categories.reshape(len(categories), 1)))

    plt.figure()

    for category in set(categories):
        of_category = np.array([v for v in projected_2d if v[2] == category])
        plt.scatter(of_category[:, 0], of_category[:, 1], label=category, s=10)

    plt.title('2D projection using PCA dimensionality reduction')
    plt.legend()
    plt.show()


df = prepare_data()
