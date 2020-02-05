import numpy as np
from numpy.fft import fft2, ifft2, fft, ifft, fftshift, ifftshift
from util import add_packet
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from math import pi
from coloring import colorize

width = 400
position = np.zeros((width, width), dtype=complex)

# position = np.array([[np.sin(x/width*10*pi) * np.sin(y/width*10*pi) for x in range(width)] for y in range(width)], dtype=complex)

# position = np.array([[np.sin(x/width*10*pi + y/width*10*pi) for x in range(width)] for y in range(width)], dtype=complex)

# position = np.array([[np.exp(1j * (x+y)/width*2*pi*10) for x in range(width)] for y in range(width)], dtype=complex)


# add_packet(position, 100, 140, width=5, momentum=10)
add_packet(position, 200, 150, width=30, momentum=15)

dt = 100

coords = np.array(np.linspace(-1, 1, width))
# coords = np.roll(coords, width//2)

momentum_map = np.array([coords]).repeat(width, axis=0)
# np.roll(momentum_map, width//2, axis=0)
momentum_map = momentum_map * momentum_map
momentum_map = momentum_map + momentum_map.transpose()
momentum_map *= 1

momentum_op = np.exp(-1j * momentum_map * dt / 2)

ground_potential = np.abs(np.array([[x**2 + y**2 for x in np.array(np.linspace(-1, 1, width))] for y in np.array(np.linspace(-1, 1, width))], dtype=complex)) * 0.50

potential_op = np.exp(-1j * ground_potential * dt / 2)
# ground_potential = np.zeros_like(position, dtype=float)
# potential = np.ones_like(position)*1000000000
# potential[10:width-10,10:width-10] = 0

# submap = momentum_map[0:width//2, 0:width//2]
# momentum_map[width//2:width, width//2:width] = submap[::-1, ::-1]
# momentum_map[0:width//2, width//2:width] = submap[::, ::-1] * -1
# momentum_map[width//2:width, 0:width//2] = submap[::-1, ::] * -1

# momentum_map[width//2:width, ...] = 0
# momentum_map[..., width//2:width] = 0
# momentum_map = np.roll(momentum_map, width // 2, axis=0)

m = np.zeros_like(position)
x, y = 6,6
m[x,x] = 1
m[-y, -y] = 1
m[-y, x] = -1
m[x, -y] = -1

# position = psi0
for iter in range(1000):
    print(iter)
    to_compare = np.copy(position)
    for substep in range(10):
        momentum = fftshift(fft2(position))
        momentum *= momentum_op
        position = ifft2(ifftshift(momentum))
        # potential = ground_potential# + 0.00001*1.0 / gaussian_filter(np.abs(np.maximum(position, 0.01)*1), sigma=15)**2
        position *= potential_op
    plt.subplot(1, 3, 1)
    # plt.imshow(colorize(momentum))
    plt.imshow(momentum.real)
    # plt.imshow(np.sqrt(momentum.real ** 2 + momentum.imag ** 2))
    plt.subplot(1, 3, 2)
    # plt.imshow(colorize(position/100))
    plt.imshow(np.sqrt(position.real ** 2 + position.imag ** 2))
    # plt.subplot(1, 3, 3)
    # plt.imshow(ground_potential + 0.00001*1.0 / gaussian_filter(np.abs(np.maximum(position, 0.01)*1), sigma=15)**2)
    plt.draw_all()
    plt.pause(0.001)
