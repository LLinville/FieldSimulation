import numpy as np
from math import pi
import pylab as plt

def grad(field):
    return np.roll(field, -1) - np.roll(field, 1)

def conv(field):
    kernel = [1, -2, 1]
    return np.roll(field, -1) * kernel[0] + field * kernel[1] + np.roll(field, 1) * kernel[2]

SIZE = 200
field = np.linspace(-pi, pi, SIZE)
field = np.cos(field)
for i in range(10000000):
    print(i)
    # kernel = [1/3] * 3
    alpha = 0.01
    field = np.pad(field, 1, mode='edge')
    field = (1 - alpha) * field + alpha * conv(field)
    field = field[1:-1]
    plt.subplot(1, 2, 1)
    plt.plot(field)
    plt.subplot(1, 2, 2)
    plt.plot(conv(field))
    plt.draw_all()
    plt.pause(0.001)
    plt.clf()

