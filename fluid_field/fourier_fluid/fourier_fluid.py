import numpy as np
from numpy.fft import fft2, ifft2, fft, ifft, fftshift, ifftshift
# from util import add_packet, plot_vec, loop_noise2, decompose, divergence, fftfreq2, is_conj_sym
from util import *
# from util import add_point_zero_origin_smooth_tail as add_point
from util import add_point_vortex as add_point
from scipy.ndimage.filters import gaussian_filter
import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from math import pi
from opensimplex import OpenSimplex
from coloring import colorize

width = 95
x = np.array(np.linspace(-6*pi, 6*pi, width), dtype=complex)

def laplacian(position):
    return position * 2 - np.roll(position, -1) - np.roll(position, 1)

def plot(field, zoom_width=None):
    if zoom_width is None:
        zoom_width = width
    if zoom_width is not None:
        field = field[width // 2 - zoom_width // 2:width // 2 + zoom_width // 2]
    plt.imshow(colorize(np.repeat([field], zoom_width, axis=0)))


fig, axs = plt.subplots(4,3)


unit = [
    [
        row + 1j*col for col in np.linspace(-2, 2, width)
    ] for row in np.linspace(-2, 2, width)
]
unit = np.array(unit)
unit /= np.abs(unit)


# vel = np.zeros_like(x)
dt = 1.1

# vel += np.exp(-1*x**2)*0.1
# vel -= np.exp(-1*(x-10)**2)*0.1
noise_real = OpenSimplex(seed=1)
noise_imag = OpenSimplex(seed=0)
arrow_scale = 0.08
noise_scale = 0.51
arrow_settings = {
    "shape": "right",
    "head_width": 0.01
}
vel = [
    [
        loop_noise2(x, y, noise_scale, noise_real) + 1j * loop_noise2(x, y, noise_scale, noise_imag) for x in np.linspace(-1, 1, width)
    ] for y in np.linspace(-1, 1, width)
]
# vel = [
#     [
#         noise_real.noise2(noise_scale*y, noise_scale*x) for x in np.linspace(-1, 1, width)
#     ] for y in np.linspace(-1, 1, width)
# ]
# vel = [
#     [
#         np.exp(-1 * (x*x*8+y*y*8)) for x in np.linspace(-1, 1, width)
#     ] for y in np.linspace(-1, 1, width)
# ]
initial = np.copy(vel)

# for x in np.linspace(-1, 1, width):
#     for y in np.linspace(-1, 1, width):
#         # plt.arrow(x, y, -0.01*x, -0.01*y)
#
#         plt.arrow(x, y, -1 * arrow_scale * noise.noise2(noise_scale * x, 1), -1 * arrow_scale * noise.noise2(noise_scale * x, noise_scale * y), **arrow_settings)

# plt.arrow(np.linspace(-1, 1, width), np.linspace(-1, 1, width), np.linspace(-1, 1, width), np.linspace(-1, 1, width))
# plot_vec(initial, ax=axs[0], scale=arrow_scale)


# axs[0][0].imshow(initial)

# k = fftfreq2(width)
#
# z = np.array([
#     [1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0],
# ])
# z = np.array([
#     [0, 0, 1, 0, 0],
#     [0, 1+1j, 1, 0, 1+2j],
#     [1, 1, 1, 1, 1],
#     [1-2j, 0, 1, 1-1j, 0],
#     [0, 0, 1, 0, 0]
# ])
#
# p = np.imag(ifft2(ifftshift(z)))
# print(is_conj_sym(z))



transformed = fft2(initial)
# transformed /= np.max(np.abs(transformed))*10

# axs[0][1].set_xlim(0, 1)
# axs[0][1].set_ylim(0, 1)
# axs[1][0].set_xlim(0, 1)
# axs[1][0].set_ylim(0, 1)
# axs[1][1].set_xlim(0, 1)
# axs[1][1].set_ylim(0, 1)

rotational_ft, divergent_ft = decompose(transformed)

x_rot_ft, x_div_ft = decompose(fft2(np.real(initial)))
y_rot_ft, y_div_ft = decompose(fft2(np.imag(initial)))
x_rot, x_div = ifft2(x_rot_ft), ifft2(x_div_ft)
y_rot, y_div = ifft2(y_rot_ft), ifft2(y_div_ft)
rotational = x_rot + 1j * y_rot
divergent = x_div + 1j * y_div

# rotational = ifft2(ifftshift(rotational_ft))
# divergent = ifft2(ifftshift(divergent_ft))

plot_vec(initial, ax=axs[0][0])
axs[0][1].imshow(np.real(initial))
axs[0][2].imshow(np.imag(initial))
plot_vec(np.real(initial), ax=axs[0][1])
plot_vec(-1j * np.imag(initial), ax=axs[0][2])

plot_vec(transformed, ax=axs[1][0])
plot_vec(fftshift(fft2(np.real(initial))), ax=axs[1][1])
plot_vec(fftshift(fft2(np.imag(initial))), ax=axs[1][2])

plot_vec(rotational, ax=axs[2][0])
plot_vec(x_rot, ax=axs[2][1])
plot_vec(y_rot, ax=axs[2][2])
# #
plot_vec(divergent, ax=axs[3][0])
plot_vec(x_div, ax=axs[3][1])
plot_vec(y_div, ax=axs[3][2])

plt.show()
plt.show(block=False)
plt.pause(0.001)




# plt.imshow(np.cos(10*np.abs(vel)))
# plt.imshow(colorize(fftshift(fft(vel))))
# momentum_initial = fftshift(fft(vel))
# momentum_positive = np.copy(momentum_initial)
# momentum_negative = np.copy(momentum_initial)
#
# momentum_initial = np.zeros_like(momentum_initial)
# momentum_initial[50]=10
# # momentum_initial[54]=10
# position = ifft(ifftshift(momentum_initial))
#
# for i in range(10000):
#     vel -= laplacian(position)*dt*1
#     nonlinear = np.sin(position*2*pi)*dt*0.0
#     vel -= nonlinear
#     position += vel * dt
#     # print('vel, position, nonlinear')
#     # print(np.max(np.abs(vel)))
#     # print(np.max(np.abs(position)))
#     # print(np.max(np.abs(nonlinear)))
#
#     view_width = 50
#
#     # wave_relation = np.abs(x**1)
#     # momentum_positive *= np.exp(1j * wave_relation * dt)
#     # momentum_negative *= np.exp(-1j * wave_relation * dt)
#     # momentum = momentum_positive + 0*momentum_negative
#     # position = ifft(ifftshift(momentum))
#
#     # momentum = momentum_positive * np.exp(1j*i*dt * wave_relation) * 1 + momentum_negative * np.exp(-1j*i*dt * wave_relation)
#     # momentum = fftshift(fft(position))
#     # momentum=np.imag(momentum)
#     # momentum = np.angle(momentum)
#     # momentum = momentum_initial * multiplier_at_time(i*dt)
#     # position = ifft(ifftshift(momentum))
#     # plot(momentum - fftshift(fft(np.abs(position))))
#     # plt.imshow(colorize(np.repeat([momentum[width // 2 - view_width // 2:width // 2 + view_width // 2]/10], view_width, axis=0)))
#     # plt.imshow(colorize(np.repeat([vel], width, axis=0)))
#     # plot(momentum)
#     # plt.clf()
#     # plt.plot(np.abs(momentum))
#     plt.pause(0.001)
#
# plt.show()
# position2 = np.exp(-1*x**2)
# momentum = fftshift(fft(position))
# # momentum *= np.exp(-1j*x) ** 200
# position2 = ifft(ifftshift(momentum))
#
# # plt.imshow(colorize(np.repeat([momentum[width//2-view_width//2:width//2+view_width//2]], view_width, axis=0)))
# # plt.imshow(np.repeat([np.abs(fftshift(fft(position))) / np.abs(fftshift(fft(initial)))], width, axis=0))

