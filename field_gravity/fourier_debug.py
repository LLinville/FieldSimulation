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
from coloring import colorize

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
position = np.exp(-1*x**2) + 1*np.exp(-1*np.roll(x, 15)**2)
# position = np.sin(x)
# position = ifft(ifftshift(position))
initial = np.copy(position)
dt = 0.1

momentum_initial = fftshift(fft(position))

for i in range(10000):
    vel -= laplacian(position)*dt
    nonlinear = np.sin(position)*dt*10
    vel -= nonlinear
    position += vel * dt
    print(np.max(np.abs(vel)))
    print(np.max(np.abs(position)))
    print(np.max(np.abs(nonlinear)))

    view_width = 50
    # momentum = fftshift(fft(position))
    # momentum=np.imag(momentum)
    # momentum = np.angle(momentum)
    # momentum = momentum_initial * multiplier_at_time(i*dt)
    # position = ifft(ifftshift(momentum))
    # plot(momentum - fftshift(fft(np.abs(position))))
    # plt.imshow(colorize(np.repeat([momentum[width // 2 - view_width // 2:width // 2 + view_width // 2]/10], view_width, axis=0)))
    plt.imshow(colorize(np.repeat([position], width, axis=0)))
    plt.pause(0.001)

plt.show()
position2 = np.exp(-1*x**2)
momentum = fftshift(fft(position))
# momentum *= np.exp(-1j*x) ** 200
position2 = ifft(ifftshift(momentum))

# plt.imshow(colorize(np.repeat([momentum[width//2-view_width//2:width//2+view_width//2]], view_width, axis=0)))
# plt.imshow(np.repeat([np.abs(fftshift(fft(position))) / np.abs(fftshift(fft(initial)))], width, axis=0))

