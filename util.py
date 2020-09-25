import numpy as np
import cmath
from os import path
from numba import cuda, float32



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

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16
@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp
