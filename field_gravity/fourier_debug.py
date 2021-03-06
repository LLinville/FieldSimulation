import numpy as np
from numpy.fft import fft2, ifft2, fft, ifft, fftshift, ifftshift
from util import add_packet
# from util import add_point_zero_origin_smooth_tail as add_point
from util import add_point_vortex as add_point
from scipy.ndimage.filters import gaussian_filter
import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from math import pi
from opensimplex import OpenSimplex
from coloring import colorize

noisegen = OpenSimplex(seed=2)

width = 105
x = np.array(np.linspace(-6*pi, 6*pi, width), dtype=complex)

def laplacian(position):
    return position * 2 - np.roll(position, -1) - np.roll(position, 1)

def multiplier_at_time(t):
    return np.cos(t * x)

def plot(field, zoom_width=None):
    if zoom_width is None:
        zoom_width = width
    if zoom_width is not None:
        field = field[width // 2 - zoom_width // 2:width // 2 + zoom_width // 2]
    plt.imshow(colorize(np.repeat([field], zoom_width, axis=0)))




vel = np.zeros_like(x)
position = np.zeros_like(x)
# position = np.sin(x/3)*0 + np.sin(1*x)
# position = np.array([noisegen.noise2d(i/pi, 10) for i in x])
position = np.exp(-1*x**2) + 0*np.exp(-1*np.roll(x, 10)**2)
# position *= 0.0
# position = np.sin(x)
# position = ifft(ifftshift(position))
initial = np.copy(position)
dt = 1.1

# vel += np.exp(-1*x**2)*0.1
# vel -= np.exp(-1*(x-10)**2)*0.1

momentum_initial = fftshift(fft(position))
momentum_positive = np.copy(momentum_initial)
momentum_negative = np.copy(momentum_initial)

momentum_initial = np.zeros_like(momentum_initial)
momentum_initial[50]=10
# momentum_initial[54]=10
position = ifft(ifftshift(momentum_initial))

for i in range(10000):
    vel -= laplacian(position)*dt*1
    nonlinear = np.sin(position*2*pi)*dt*0.0
    vel -= nonlinear
    position += vel * dt
    # print('vel, position, nonlinear')
    # print(np.max(np.abs(vel)))
    # print(np.max(np.abs(position)))
    # print(np.max(np.abs(nonlinear)))

    view_width = 50

    # wave_relation = np.abs(x**1)
    # momentum_positive *= np.exp(1j * wave_relation * dt)
    # momentum_negative *= np.exp(-1j * wave_relation * dt)
    # momentum = momentum_positive + 0*momentum_negative
    # position = ifft(ifftshift(momentum))

    # momentum = momentum_positive * np.exp(1j*i*dt * wave_relation) * 1 + momentum_negative * np.exp(-1j*i*dt * wave_relation)
    # momentum = fftshift(fft(position))
    # momentum=np.imag(momentum)
    # momentum = np.angle(momentum)
    # momentum = momentum_initial * multiplier_at_time(i*dt)
    # position = ifft(ifftshift(momentum))
    # plot(momentum - fftshift(fft(np.abs(position))))
    # plt.imshow(colorize(np.repeat([momentum[width // 2 - view_width // 2:width // 2 + view_width // 2]/10], view_width, axis=0)))
    plt.imshow(colorize(np.repeat([position], width, axis=0)))
    # plot(momentum)
    # plt.clf()
    # plt.plot(np.abs(momentum))
    plt.pause(0.001)

plt.show()
position2 = np.exp(-1*x**2)
momentum = fftshift(fft(position))
# momentum *= np.exp(-1j*x) ** 200
position2 = ifft(ifftshift(momentum))

# plt.imshow(colorize(np.repeat([momentum[width//2-view_width//2:width//2+view_width//2]], view_width, axis=0)))
# plt.imshow(np.repeat([np.abs(fftshift(fft(position))) / np.abs(fftshift(fft(initial)))], width, axis=0))

