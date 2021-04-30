import numpy
import matplotlib.pyplot as plt
import imageio


def read_image(path):
    im = imageio.imread(path)
    X = []
    labels = []
    classes = {}

    for i_idx, i in enumerate(im):
        for j_idx, j in enumerate(i):
            if any(el != 255 for el in j):
                X.append([i_idx, j_idx])
                j_tuple = tuple(j)
                if j_tuple not in classes:
                    classes[j_tuple] = len(classes)
                labels.append(classes[j_tuple])

    X = numpy.array(X)
    labels = numpy.array(labels)
    return X, labels, list(classes.values())


def thin_out_dataset(X, labels, fraction):
    n = int(1 / fraction)
    X_res = numpy.array([v for i, v in enumerate(X) if i % n == 0])
    labels_res = numpy.array([v for i, v in enumerate(labels) if i % n == 0])

    return X_res, labels_res


def noise_dataset(X):
    ran_x = numpy.random.normal(0, 0.5, len(X))
    ran_y = numpy.random.normal(0, 0.5, len(X))
    X_res = numpy.array([[v[0] + ran_x[i], v[1] + ran_y[i]] for i, v in enumerate(X)])

    return X_res


def plot_dataset(X, labels, classes):
    colors = 'bgrcmyk'
    for l in classes:
        my_members = labels == l
        plt.plot(X[my_members, 0], X[my_members, 1], colors[l % len(colors)] + '.')
    plt.show()


def make_dataset():
    path = 'dataset2.bmp'
    X, labels, classes = read_image(path)
    X, labels = thin_out_dataset(X, labels, 0.1)
    X = noise_dataset(X)

    return X, labels, classes


def run_ex():
    X, labels, classes = make_dataset()
    plot_dataset(X, labels, classes)
