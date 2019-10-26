import numpy as np

from numpy import pi
import pylab as plt
from colorsys import hls_to_rgb
from scipy import signal
from numba import jit
from numpy.fft import fft2, ifft2

def colorize(z):
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + pi)  / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
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
# @jit(nopython=True, parallel=True)
def np_fftconvolve(A):
    return np.real(ifft2(fft2(A)*fft2([[1/9]*3]*3, s=A.shape)))

def conv(field):
    return signal.convolve2d(field,  [[1/9]*3]*3, mode='same')

def add_point(field, location, size=10, polarity=1):
    field[location[0] - size // 2 : location[0] + size // 2, location[1] - size // 2 : location[1] + size // 2] = np.array(
        [[polarity / size * u + 1j / size * v for u in range(-1 * size // 2, size // 2)] for v in range(-1*size//2, size//2)]
    )

SIZE = 100
x, y = np.meshgrid(np.linspace(0, 1, SIZE), np.linspace(0, 1, SIZE))

z = x + 1j * y

field = np.zeros_like(z)

# field[50:100,100:150] = np.array([[0.1 * j + 0.1j * n for n in range(-25,25)] for j in range(-25, 25)])
# field[100:150,150:200] = np.array([[0.1 * j + 0.1j * n for n in range(-25,25)] for j in range(-25, 25)])
plt.show()

# add_point(field, (30, 30), 10)
add_point(field, (40, 40), 10, -10)
add_point(field, (60, 60), 10, 10)

for i in range(1000000):
    field = conv(field) * 1.01
    img = colorize(field)
    plt.imshow(img)
    plt.draw()
    plt.pause(0.0001)
    plt.cla()



