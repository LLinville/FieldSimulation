import numpy as np

from numpy import pi
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
from scipy import signal
from numba import jit
from numpy.fft import fft2, ifft2
from timeit import default_timer as timer
import cmath

def colorize(z):
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + pi)  / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.5)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    return c

# N=1000
# x,y = np.ogrid[-5:5:N*1j, -5:5:N*1j]
# z = x + 1j*y
#
# w = 1/(z+1j)**2 + 1/(z-2)**2
# img = colorize(w)
# plt.imshow(img)
# plt.show()
#


SIZE = 200

# @jit(nopython=True, parallel=True)
def np_fftconvolve(A):
    # kernel = [[1/9]*3]*3
    kernel = [
        [-1, 2, -1]*3
    ]
    return np.real(ifft2(fft2(A)*fft2(kernel, s=A.shape)))

# @jit(parallel=True,forceobj=True, nopython=True)
def step(field):
    field = np.pad(field,1, mode='edge')
    # return signal.convolve2d(field, np.array([[1/25]*5]*5), mode='same', boundary='fill', fillvalue=0)
    out = field + np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
    #
    return out[1:-1,1:-1] / 5
    # out = np.zeros_like(field, dtype=complex)
    # s = SIZE
    # for x in range(s):
    #     for y in range(s):
    #         out[x,y] = (out[x,y] + out[(x+1)%s,y] + out[(x-1)%s,y] + out[x,(y+1)%s] + out[x, (y-1)%s]) / 5
    # return out

def conv(field):
    kernel = [[-1, 2, -1]] * 3
    return signal.convolve2d(field, kernel, mode='same', boundary='fill', fillvalue=0)

def grad(field):
    magnitude = np.abs(field)
    d_x = np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)
    d_y = np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)
    return d_x + 1j * d_y


def add_point(field, location, size=10, polarity=1, rotation = 0, turns=1):
    patch_input_x, patch_input_y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    patch = np.zeros((size, size), dtype=complex)
    for u in range(size):
        for v in range(size):
            input_point = patch_input_x[u,v] + 1j * patch_input_y[u,v]
            r, theta = np.abs(input_point), np.angle(input_point)
            theta *= turns
            r = 1 / np.abs(r * 10)
            input_point = cmath.rect(r, theta)
            dist2 = (u) ** 2 + (v) ** 2 + 1
            patch[u,v] += input_point

    field[location[0] - size // 2 : location[0] + size // 2, location[1] - size // 2 : location[1] + size // 2] = patch * (np.cos(rotation) + np.sin(rotation) * 1j)


x, y = np.meshgrid(np.linspace(0, 1, SIZE), np.linspace(0, 1, SIZE))

z = x + 1j * y

field = np.zeros_like(z)

# field[50:100,100:150] = np.array([[0.1 * j + 0.1j * n for n in range(-25,25)] for j in range(-25, 25)])
# field[100:150,150:200] = np.array([[0.1 * j + 0.1j * n for n in range(-25,25)] for j in range(-25, 25)])
plt.show()

# add_point(field, (30, 30), 10)
polarity = 4
for x in [40, 160]:
    for y in [40, 160]:
        # add_point(field, (x,y), 16, turns=polarity)
        polarity *= -1

# for i in range(1,8)[::-1]:
#     add_point(field, (60, 60), i*2, 0.1 ** (i))
# add_point(field, (140, 140), 20, turns=1)
# add_point(field, (80, 140), 20, turns=1, rotation=pi)
add_point(field, (90, 90), 90, turns=1)#, pi/2)
# add_point(field, (160, 160), 50, turns=7)#, pi/2)
# field[0,0] = 1 + 1j

time = timer()
prev_time = time
vel = np.zeros_like(field)
for i in range(1000000):
    prev_field = field.copy()
    # vel -= field * 0.001
    vel *= 0.999
    alpha = 0.0001
    field = (1 - alpha) * field + alpha * conv(field) + vel * 0.01

    if i%100 == 0:
        prev_time = time
        time = timer()
        print(f'{i}) Time: {(time - prev_time)}')

    if i%100 == 0:

        plt.subplot(1,3,1)
        f_img = colorize(field)# / np.max(np.abs(field)))
        plt.imshow(f_img)
        plt.subplot(1,3,2)
        vel = field - prev_field
        v_img = colorize(vel / np.max(np.abs(vel)))
        plt.imshow(v_img)
        plt.subplot(1,3,3)
        grad_img = colorize(grad(field))
        plt.imshow(grad_img)
        plt.draw_all()
        plt.pause(0.0001)
        plt.clf()



