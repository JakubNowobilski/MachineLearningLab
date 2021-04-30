import numpy as np
import matplotlib.pyplot as plt


def hyp_sphere_in_hyp_cube(dim, points):
    # Hypersphere: center - [0]^dim, range - 1
    # Hypercube: center - [0]^dim, edge length - 2
    # Because of above np.linalg.norm is sufficient to calculate distance
    points_in = 0
    for _ in range(points):
        p = np.random.uniform(-1, 1, dim)
        norm = np.linalg.norm(p)
        if norm <= 1:
            points_in += 1
    return 100 * (points_in / points)


def single_survey():
    n = 10000
    dims = 15
    x_axis = range(1, dims + 1)
    y_axis = [hyp_sphere_in_hyp_cube(x, n) for x in x_axis]

    plt.title("Number of points: " + str(n))
    plt.xlabel("Dimension")
    plt.ylabel("Ratio of points inside the hypersphere to all points in (%)")
    plt.scatter(x_axis, y_axis)
    plt.show()


def repeated_survey():
    n = 10000
    dims = 15
    r = 10
    x_axis = range(1, dims + 1)
    y_axis = []
    std = []
    for x in x_axis:
        sphere_in_cube = [hyp_sphere_in_hyp_cube(x, n) for _ in range(r)]
        y_axis.append(np.mean(sphere_in_cube))
        std.append(np.std(sphere_in_cube))

    plt.title("Number of points: " + str(n) + "\nNumber of repetitions: " + str(r))
    plt.xlabel("Dimension")
    plt.ylabel("Average and standard deviation of\nratio of points inside the hypersphere to all points (%)")
    plt.errorbar(x_axis, y_axis, std, linestyle='None', capsize=6, capthick=2, marker='s', ecolor='red')
    plt.show()


def main():
    single_survey()
    repeated_survey()


main()
