import numpy as np
from numpy.fft import fft2, ifft2, fft, ifft, fftshift, ifftshift
from util import add_packet
from util import add_point_zero_origin_smooth_tail as add_point
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from math import pi
from coloring import colorize
from util import first_unoccupied
import imageio



width = 512
position = np.zeros((width, width), dtype=complex)

# position = np.array([[np.sin(x/width*10*pi) * np.sin(y/width*10*pi) for x in range(width)] for y in range(width)], dtype=complex)

# position = np.array([[np.sin(x/width*10*pi + y/width*10*pi) for x in range(width)] for y in range(width)], dtype=complex)

# position = np.array([[np.exp(1j * (x+y)/width*2*pi*10) for x in range(width)] for y in range(width)], dtype=complex)

# add_point(position, 256, 256, width=20, turns=1)
add_packet(position, 256, 256, width=30, momentum=3)
add_packet(position, 350, 350, width=30, momentum=3, direction=2*pi/4)
position *= 0.25
# add_packet(position, 130, 100, width=10, momentum=-3, direction=3*pi/4)

dt = 100

coords = np.array(np.linspace(-1, 1, width))
# coords = np.roll(coords, width//2)

momentum_map = np.array([coords]).repeat(width, axis=0)
# np.roll(momentum_map, width//2, axis=0)
momentum_map = momentum_map * momentum_map
# momentum_map = momentum_map + momentum_map.transpose()
# momentum_map *=1.11
momentum_map = np.sqrt(momentum_map + momentum_map.transpose())

# larger mass means wave of given size travels slower
# lower mass means larger wave scale on average
mass = 2
momentum_op = np.exp(-1j * momentum_map * dt / 2 / mass)

ground_potential = np.abs(np.array([[x**2 + y**2 for x in np.array(np.linspace(-1, 1, width))] for y in np.array(np.linspace(-1, 1, width))], dtype=complex)) * 0.350
ground_potential = np.zeros_like(position)
# potential = np.ones_like(position)*1000000000
# potential[10:width-10,10:width-10] = 0

potential_op = np.exp((1 * -1j - 0*1)/np.sqrt(2) * ground_potential * dt / 2)

# diffuse_op = np.exp(-1000 * np.array([[x**2 + y**2 for x in np.array(np.linspace(-1, 1, width))] for y in np.array(np.linspace(-1, 1, width))], dtype=complex))

# diffuse_op = np.exp(-1000 * np.array([[np.sqrt(x**2 + y**2)*0+1 for x in np.array(np.linspace(-1, 1, width))] for y in np.array(np.linspace(-1, 1, width))], dtype=complex))
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
all_outputs = []
for iter in range(500):
    print(iter)
    to_compare = np.copy(position)
    # if iter % 10 == -10:
    #     for substep in range(1):
    #         potential -= np.abs(position)*1.1
    #         potential = ifft2(fftshift(fftshift(fft2(potential)) * diffuse_op)) * 1

        # potential_op = np.exp(-1j * potential * 0.00001 * dt / 2)
    for substep in range(1):
        momentum = fftshift(fft2(position))
        momentum *= momentum_op
        position = ifft2(ifftshift(momentum))
        # potential = ground_potential# + 0.00001*1.0 / gaussian_filter(np.abs(np.maximum(position, 0.01)*1), sigma=15)**2
        pos_mag = np.abs(position)
        potential_op = np.exp(-1j * (1*ground_potential - 1*0.11*(pos_mag*pos_mag)) * 0.3 * dt / 2)
        position *= potential_op

        # position -= 0.1*position * (1-pos_mag)
        # position *= np.exp(-1j * potential * dt / 1000)

    all_outputs.append(colorize(position / 1))
    if iter % 3 == 0:
        plt.subplot(1, 2, 1)
        # plt.imshow(np.abs(potential))
        # plt.imshow(colorize(momentum))
        # plt.imshow(momentum.real)
        plt.imshow(np.minimum(10, np.sqrt(momentum.real ** 2 + momentum.imag ** 2)))#[200:300,200:300])
        plt.subplot(1, 2, 2)
        # plt.imshow(all_outputs[-1])
        plt.imshow(np.sqrt(position.real ** 2 + position.imag ** 2))
        # plt.subplot(1, 3, 3)
        # plt.imshow(ground_potential + 0.00001*1.0 / gaussian_filter(np.abs(np.maximum(position, 0.01)*1), sigma=15)**2)
        # plt.imshow(colorize(fftshift(fft2(position))))
        plt.draw_all()
        plt.pause(0.001)

print("normalizing and reshaping")
# from 0 - 1 (nIter, width, width, 3) to 0 - 255 (nIter, 3, width, width)
all_outputs = np.array(all_outputs) * 255
all_outputs = np.array(np.floor(all_outputs), dtype=int)
# all_outputs = np.transpose(all_outputs, (0, 3, 1, 2))
print("Writing to video")
# array2gif.write_gif(all_outputs, "schrodinger2dOut.gif")

imageio.mimwrite(first_unoccupied("schrodinger2dOut%s.mp4"), all_outputs, "mp4", fps=20)

