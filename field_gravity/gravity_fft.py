import numpy as np
from math import pi
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
from matplotlib import pyplot as plt
from scipy.signal.windows import gaussian
from opensimplex import OpenSimplex
from random import randint

width = 1000

noisegen = OpenSimplex(seed=2)

# state = np.sin(np.linspace(0, 2*pi*1, width))
# state = gaussian(width, width//30) + np.roll(gaussian(width, width//30), width//5)
# state = np.array([noisegen.noise2d(1, x) for x in np.linspace(0, 10, width)])
state = np.zeros(width)
# state = np.ones(width)
# state[width//4:3*width//4] = 1
# state = ifft(fftshift(state))

# state = fftshift(ifft(np.roll(gaussian(width, width//100), 0)))

density = np.zeros_like(state)
density[200:220] = 0.0001
density[600:620] = 0.0001

# plt.subplot(1,2,1)
# plt.plot(np.abs(state))
# plt.plot(state)
# plt.show()
# # plt.plot(fftshift(fft(np.sin(np.linspace(0, 2*pi*5, 100)))))
# plt.subplot(1,2,2)

neighbor_diff = np.roll(state, -1) + np.roll(state, 1) - 2 * state
# plt.plot(neighbor_diff)
# plt.plot(fftshift(fft(state)))
# plt.show()

diffuse_operator = np.exp(-1 * np.abs(np.linspace(-1, 1, width)) * np.abs(np.linspace(-1, 1, width)) * 1000)
# diffuse_operator[width//2] = 1

state_fft = fftshift(fft(state))
# state_fft = np.array([noisegen.noise2d(1, x) for x in np.linspace(0, 10, width)])

mix_ratio = 0.5
for i in range(1000):
    for sub in range(100):
        # state_fft *= diffuse_operator
        # neighbor_diff = np.roll(state, -1) + np.roll(state, 1) - 2 * state
        # new_state = (1.0 - mix_ratio) * state + (state + neighbor_diff) * mix_ratio
        # fft_diff = fftshift((np.abs(fft(state)) - np.abs(fft(new_state))))# / (np.abs(fft(state)) + np.abs(fft(new_state))))
        state = ifft(fftshift(fftshift(fft(state)) * diffuse_operator))
        state += density
    plt.clf()
    plt.subplot(1,3,1)
    # plt.plot(np.abs(state))
    plt.plot(np.abs(fftshift(fft(state)))[460:540])
    plt.subplot(1,3,2)
    # plt.plot(np.abs(fftshift(fft(state))))
    plt.plot(state_fft[480:520])
    plt.subplot(1,3,3)
    plt.plot(state)
    plt.pause(0.001)