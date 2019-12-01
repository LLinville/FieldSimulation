import numpy as np
from math import pi
import pylab as plt


def grad(field):
    return np.roll(field, -1) - np.roll(field, 1) / 2.0


def conv(field, kernel = np.array([1, 1, 1]) / 3):
    return np.roll(field, -1) * kernel[0] + field * kernel[1] + np.roll(field, 1) * kernel[2]


def conv_by_grad(field):
    field_grad = grad(field)

    return grad(field_grad)


SIZE = 200
field = np.linspace(-2*pi, 2*pi, SIZE)
field = np.exp(-1 * (field * field)) + np.exp(-1 * (field - 2.0) ** 2)
for i in range(10000000):
    print(i)
    # kernel = [1/3] * 3
    alpha = 0.1
    field = np.pad(field, 1, mode='edge')
    field = (1 - alpha) * field + alpha * conv(field)
    field -= 0.1 * conv_by_grad(field)
    field = field[1:-1]
    plt.subplot(1, 2, 1)
    plt.plot(field)
    plt.subplot(1, 2, 2)
    plt.plot(conv(field))
    plt.draw_all()
    plt.pause(0.001)
    plt.clf()

