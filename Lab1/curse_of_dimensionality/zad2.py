import numpy as np
import math
import matplotlib.pyplot as plt


def hyper_cube(dim, points):
    # Hypercube: center - [0]^dim, edge length - 2
    p = [np.random.uniform(-1, 1, dim) for _ in range(points)]
    d = [math.dist(p[i], p[j]) for i in range(points) for j in range(i + 1, points)]
    return np.mean(d), np.std(d)


def dist_with_std(n, x_axis):
    cube_res = [hyper_cube(x, n) for x in x_axis]
    y_axis = [x[0] for x in cube_res]
    std = [x[1] for x in cube_res]

    plt.title("Number of points: " + str(n))
    plt.xlabel("Dimension")
    plt.ylabel("Average and standard deviation of\ndistance between 2 points")
    plt.errorbar(x_axis, y_axis, std, linestyle='None', capsize=6, capthick=2, marker='s', ecolor='red')
    plt.show()


def dist_std_radio(n, x_axis):
    y_axis = list(map(lambda res: 100 * (res[1] / res[0]), [hyper_cube(x, n) for x in x_axis]))

    plt.title("Number of points: " + str(n))
    plt.xlabel("Dimension")
    plt.ylabel("Ratio of standard deviation to mean (%)")
    plt.scatter(x_axis, y_axis)
    plt.show()


def dist_with_std_repeated(n, x_axis, r):
    y_axis_means = []
    std_means = []
    y_axis_stds = []
    std_stds = []
    for x in x_axis:
        cube_res = [hyper_cube(x, n) for _ in range(r)]
        x_means = [x[0] for x in cube_res]
        y_axis_means.append(np.mean(x_means))
        std_means.append(np.std(x_means))

        x_stds = [x[1] for x in cube_res]
        y_axis_stds.append(np.mean(x_stds))
        std_stds.append(np.std(x_stds))

    plt.title("Number of points: " + str(n) + "\nNumber of repetitions: " + str(r))
    plt.xlabel("Dimension")
    plt.ylabel("Average and standard deviation of\naverage distance between 2 points")
    plt.errorbar(x_axis, y_axis_means, std_means, linestyle='None', capsize=6, capthick=2, marker='s', ecolor='red')
    plt.show()

    plt.title("Number of points: " + str(n) + "\nNumber of repetitions: " + str(r))
    plt.xlabel("Dimension")
    plt.ylabel("Average and standard deviation of\nstandard deviation of distance between 2 points")
    plt.errorbar(x_axis, y_axis_stds, std_stds, linestyle='None', capsize=6, capthick=2, marker='s', ecolor='red')
    plt.show()


def dist_std_radio_repeated(n, x_axis, r):
    y_axis = []
    std = []
    for x in x_axis:
        cube_res = list(map(lambda res: 100 * (res[1] / res[0]), [hyper_cube(x, n) for _ in range(r)]))
        y_axis.append(np.mean(cube_res))
        std.append(np.std(cube_res))

    plt.title("Number of points: " + str(n) + "\nNumber of repetitions: " + str(r))
    plt.xlabel("Dimension")
    plt.ylabel("Average and standard deviation of\nratio of standard deviation to mean (%)")
    plt.errorbar(x_axis, y_axis, std, linestyle='None', capsize=6, capthick=2, marker='s', ecolor='red')
    plt.show()


def single_survey():
    n = 1000
    dims = 15
    x_axis = range(1, dims + 1)

    dist_with_std(n, x_axis)
    dist_std_radio(n, x_axis)


def repeated_survey():
    n = 1000
    dims = 15
    r = 10
    x_axis = range(1, dims + 1)

    dist_with_std_repeated(n, x_axis, r)
    dist_std_radio_repeated(n, x_axis, r)


def main():
    single_survey()
    repeated_survey()


main()
