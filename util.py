import numpy as np
import cmath
from os import path
# from numba import cuda, float32
from matplotlib import pyplot as plt
from numpy import pi
from colorsys import hls_to_rgb
from opensimplex import OpenSimplex
from math import pi
from numpy.fft import fft2, ifft2, fft, ifft, fftshift, ifftshift, fftfreq

noisegen = OpenSimplex(seed=2)

def add_point_parabolic(data, cx, cy, width=100, turns=0, mag=1):
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            dx, dy = x-cx, y-cy
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if (0 < dist < 100):
                mag = dist / 100
                mag = -4 * mag * (mag - 1)
                phase = turns * cmath.phase(dx + dy*1j)
                out = cmath.rect(mag, phase)
                if len(data.shape) == 2:
                    data[x, y] += out
                else:
                    data[x, y, 0] += out.real
                    data[x, y, 1] += out.imag

def add_point_zero_origin_smooth_tail(data, cx, cy, width=100, turns=0, mag=1):
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            dx, dy = x-cx, y-cy
            dist_squared = dx ** 2 + dy ** 2
            if (0 < dist_squared < width**2*100):
                dist_squared /= width**2
                out = (dx + 1j*dy)**turns / cmath.exp(dist_squared) * mag
                if len(data.shape) == 2:
                    data[x, y] += out
                else:
                    data[x, y, 0] += out.real
                    data[x, y, 1] += out.imag


def add_point_vortex(data, cx, cy, width=100, turns=0, mag=1):
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            dx, dy = x-cx, y-cy
            dist_squared = dx ** 2 + dy ** 2
            if True or (0 < dist_squared < width**2*100):
                dist_squared /= width**2
                out = (dx + 1j*dy)**turns
                try:
                    out /= np.abs(out)
                except Exception as ex:
                    pass
                out *= (1 - 1/(1+np.sqrt(dist_squared/width/10)))
                if len(data.shape) == 2:
                    data[x, y] += out
                else:
                    data[x, y, 0] += out.real
                    data[x, y, 1] += out.imag


# var cp = Math.cos(phi), sp = Math.sin(phi);
#    var pixels = [],  h = 1/n1;
#    for(var i = 0; i<n; i++)
#      for(var j = 0; j<n; j++){
#        var x = h*(j-n/2),  y = h*(i-n/2);
#        var exp = Math.exp(-20*(x*x + y*y));
#        pixels.push( exp*Math.cos(ka*(cp*x + sp*y)),
#          exp*Math.sin(ka*(cp*x + sp*y)), 0, 0 );
#      }
def add_packet(data, cx, cy, width=100, momentum=1, direction=0):
    cosdir, sindir = np.cos(direction), np.sin(direction)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            # if x == 0 and y == 0:
            #     continue
            dx, dy = x - cx, y - cy
            dx, dy = dx/width, dy/width
            dx, dy = dx*cosdir - dy*sindir, dx*sindir + dy*cosdir

            dist = np.sqrt(dx ** 2 + dy ** 2)
            out = cmath.exp(-1*dist**2 + 1j * dx * momentum)
            if len(data.shape) == 2:
                data[x, y] += out
            else:
                data[x, y, 0] += out.real
                data[x, y, 1] += out.imag

def add_singordon_packet(data, cx, cy, width=100, momentum=0, direction=0, mag=1):
    cosdir, sindir = np.cos(direction), np.sin(direction)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            # if x == 0 and y == 0:
            #     continue
            dx, dy = x - cx, y - cy
            dx, dy = dx / width, dy / width
            dx, dy = dx * cosdir - dy * sindir, dx * sindir + dy * cosdir

            dist = np.sqrt(dx ** 2 + dy ** 2)
            # out = cmath.exp(-1 * dist ** 2 + 1j * dx * momentum)
            mass=1
            delta = -0.0001
            gamma = np.sqrt(1/(1-momentum**2))
            out = 4*np.arctan(np.exp(mass*gamma*(-1*dist) + delta)) * mag
            if len(data.shape) == 2:
                data[x, y] += out
            else:
                data[x, y, 0] += out.real
                data[x, y, 1] += out.imag

def total_mag(buffer):
    points = np.reshape(np.frombuffer(buffer.read(), dtype="f4"), (-1, 1024, 4))
    return np.sum(np.sqrt(points[:,:,0]*points[:,:,0] + points[:,:,1]*points[:,:,1]))

def grad(field):
    return (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)) / 4 - field


def divergence(field):
    return (np.roll(field, 1, axis=0) - np.roll(field, -1, axis=0)) / 2 + (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)) / 2


def runge_kutta_force(field, stepper, dt=0.01):
    stepper(field, dt)
    k1 = stepper(field, dt)
    k2 = grad(field + k1 / 2) * dt
    k3 = grad(field + k2 / 2) * dt
    k4 = grad(field + k3) * dt
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


def first_unoccupied(pattern):
    for i in range(1, 10000):
        if not path.exists(pattern % i):
            return pattern % i
    return None

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


# x and y are in the range [0,1] where 0 wraps back to 1
def loop_noise2(x, y, scale=0.1, noisegen=noisegen):
    return noisegen.noise4(
        scale * np.sin(2 * pi * x),
        scale * np.cos(2 * pi * x),
        scale * np.sin(2 * pi * y),
        scale * np.cos(2 * pi * y)
    )


def xplot_vec(field, ax=plt, scale=None):
    width = field.shape[0]
    if not scale:
        scale = 0.11/np.max(np.abs(field))
    for x, row in enumerate(field):
        for y, cell in enumerate(row):
            ax.arrow(x/width, y/width, scale * np.real(cell), scale * np.imag(cell))


def plot_vec(field, ax=plt, scale=None):
    width = field.shape[0]
    X, Y = np.meshgrid(range(width), range(width))
    ax.quiver(X, Y, np.real(field), np.imag(field))

def fftfreq2(width):
    row = fftfreq(width)
    x, y = np.meshgrid(row, row)
    return x + 1j * y


def conj_asym(field):
    largest_difference = (0, '')
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            if field[x][y].real - field[-x-1][-y-1].real > largest_difference[0]:
                print(f"Failed at ({x},{y}), ({-x-1},{-y-1}) with {field[x][y]}, {field[-x-1][-y-1]}")
                largest_difference = (field[x][y].real - field[-x-1][-y-1].real, f"Failed at ({x},{y}), ({-x-1},{-y-1}) with {field[x][y]}, {field[-x-1][-y-1]}")

            if field[x][y].imag + field[-x-1][-y-1].imag > largest_difference[0]:
                print(f"Failed at ({x},{y}), ({-x-1},{-y-1}) with {field[x][y]}, {field[-x-1][-y-1]}")
                largest_difference = (field[x][y].imag + field[-x-1][-y-1].imag,f'mag: {np.sqrt(field[x][y]**2 + field[x-1][y-1])}', f"Failed at ({x},{y}), ({-x-1},{-y-1}) with {field[x][y]}, {field[-x-1][-y-1]}")

    return largest_difference



# Returns rotational(field), divergent(field) in fourier space
def decompose_ft(field):
    width = field.shape[0]
    # unit = [
    #     [
    #         row + 1j*col for col in np.linspace(-2, 2, width)
    #     ] for row in np.linspace(-2, 2, width)
    # ]

    unit = np.fft.fftfreq(width).reshape(width, 1)

    # unit = np.array(unit)
    unit /= np.abs(unit)
    unit = np.nan_to_num(unit)

    parallel_mag = np.real(unit) * np.real(field) + np.imag(unit) * np.imag(field)
    return unit * parallel_mag, field - unit * parallel_mag


def decompose(field):
    field = np.array(field)

    width = field.shape[0]
    vx, vy = np.real(field), np.imag(field)
    vx_f = fft2(vx)
    vy_f = fft2(vy)
    kx = np.real(fftfreq2(width))
    ky = np.imag(fftfreq2(width))
    k2 = kx**2 + ky**2
    k2[0,0] = 1.

    div_Vf_f = (vx_f * kx +  vy_f * ky) #*1j
    V_compressive_overk = div_Vf_f / k2
    V_compressive_x = np.fft.ifftn(V_compressive_overk * kx)
    V_compressive_y = 1*np.fft.ifftn(V_compressive_overk * ky)

    V_solenoidal_x = vx - V_compressive_x
    V_solenoidal_y = vy - V_compressive_y
    V_solenoidal_y *= 1

    return V_compressive_x + 1j * V_compressive_y, V_solenoidal_x + 1j * V_solenoidal_y
