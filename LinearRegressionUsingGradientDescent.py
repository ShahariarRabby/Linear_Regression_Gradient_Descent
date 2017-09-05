# coding: utf-8


import numpy as np


def run():
    points = np.genfromtxt('data.csv', delimiter=',')
    learning_rate = 0.0001
    # y = mx + b (slope Formula)
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    [b, m] = gradient_decent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print('b: ', b)
    print('m: ', m)


def gradient_decent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
    b = initial_b
    m = initial_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]


def step_gradient(b, m, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((m * x) + b))
        m_gradient += -(2 / N) * x * (y - ((m * x) + b))

    new_b = b - (learning_rate * b_gradient)
    new_m = m - (learning_rate * m_gradient)

    return [new_b, new_m]


def compute_error_for_given_points(b, m, points):
    totalError = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2

    return totalError / len(points)


if __name__ == '__main__':
    run()
