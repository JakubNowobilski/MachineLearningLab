import numpy as np
import matplotlib.pyplot as plt
import math


def hyper_cube(dim, points):
    # Hypersphere: center - [0]^dim, range - 1
    p = [np.random.uniform(-1, 1, dim) for _ in range(points)]
    return p


def get_angle(cube):
    if len(cube) < 4:
        return [], None
    s1 = cube.pop(np.random.randint(len(cube)))
    e1 = cube.pop(np.random.randint(len(cube)))
    s2 = cube.pop(np.random.randint(len(cube)))
    e2 = cube.pop(np.random.randint(len(cube)))
    v1 = e1 - s1
    v2 = e2 - s2
    v1_n = v1 / np.linalg.norm(v1)
    v2_n = v2 / np.linalg.norm(v2)
    return cube, np.arccos(np.clip(np.dot(v1_n, v2_n), -1.0, 1.0))


def get_angles(cube):
    angles = []
    while True:
        cube, angle = get_angle(cube)
        if angle is None:
            break
        else:
            angles.append(angle)
    return angles


def single_dimension_distribution():
    n = 10000
    dim = 10
    cube = hyper_cube(dim, n)
    angles = get_angles(cube)

    plt.title("Number of points: " + str(n) + "\nDimension: " + str(dim))
    plt.xlabel("Angle between 2 random vectors (radians)")
    plt.ylabel("Frequency")
    plt.hist(angles, bins=np.arange(0.0, math.pi, 0.1))
    plt.show()


def single_survey():
    n = 10000
    dims = 15
    x_axis = range(1, dims + 1)
    y_axis = []
    std = []
    for x in x_axis:
        angles = get_angles(hyper_cube(x, n))
        y_axis.append(np.mean(angles))
        std.append(np.std(angles))

    plt.title("Number of points: " + str(n))
    plt.xlabel("Dimension")
    plt.ylabel("Average and standard deviation of\nangle between 2 random vectors")
    plt.errorbar(x_axis, y_axis, std, linestyle='None', capsize=6, capthick=2, marker='s', ecolor='red')
    plt.show()


def main():
    single_dimension_distribution()
    single_survey()


main()
