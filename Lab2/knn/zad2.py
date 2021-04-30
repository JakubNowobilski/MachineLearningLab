import make_dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def plot_results(results, classifier_name):
    plt.title("Classifier: " + classifier_name)
    plt.xlabel("K parameter")
    plt.ylabel("Average and standard deviation of performance of classifier")
    plt.errorbar(results[:, 0], results[:, 1], results[:, 2], linestyle='None', capsize=6, capthick=2, marker='s', ecolor='red')
    plt.show()


def test_classifier(metric, ds):
    X, y, classes = ds
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    avg_scores = []
    for k in range(1, 21):
        if metric == "euclid":
            classifier = KNeighborsClassifier(k)
        elif metric == "mahalanobis":
            classifier = KNeighborsClassifier(k, metric='mahalanobis', metric_params={'V': numpy.cov(X.T)})
        else:
            return

        scores = []
        for i in range(10):
            X_r_train, X_validate, y_r_train, y_validate = train_test_split(X_train, y_train, train_size=0.8)
            classifier.fit(X_r_train, y_r_train)
            score = classifier.score(X_validate, y_validate)
            scores.append(score)

        avg_scores.append([k, numpy.mean(scores)])

    best_score = max(avg_scores, key=lambda sc: sc[1])

    print("Result for k-nn classifier: metric = " + metric + ", k = " + str(best_score[0]))

    if metric == "euclid":
        classifier = KNeighborsClassifier(best_score[0])
    elif metric == "mahalanobis":
        classifier = KNeighborsClassifier(best_score[0], metric='mahalanobis', metric_params={'V': numpy.cov(X.T)})
    else:
        return

    scores = []
    for i in range(10):
        X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(X, y, train_size=0.8)
        classifier.fit(X_f_train, y_f_train)
        score = classifier.score(X_f_test, y_f_test)
        scores.append(score)

    print("Mean: " + str(numpy.mean(scores)))
    print("Std: " + str(numpy.std(scores)))
    print("\n")


ds = make_dataset.make_dataset()
metrices = ['euclid', 'mahalanobis']

for m in metrices:
    test_classifier(m, ds)
