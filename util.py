import numpy as np
import cmath

def add_point(data, cx, cy, width=100, turns=0, mag=1):

    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            dx, dy = x-cx, y-cy
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if (0 < dist < 100):
                mag = dist / 100
                mag = -4 * mag * (mag - 1)
                phase = turns * cmath.phase(dx + dy*1j)
                out = cmath.rect(mag, phase)
                data[x, y, 0] = out.real
                data[x, y, 1] = out.imag

def add_packet(data, cx, cy, width=100, momentum=1):
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            dx, dy = x - cx, y - cy
            dist = np.sqrt(dx ** 2 + dy ** 2) / (width/2)
            out = cmath.exp(1j * momentum * dist - (dist - dx/(width/2))**2)
            data[x, y, 0] = out.real
            data[x, y, 1] = out.imag

def total_mag(buffer):
    points = np.reshape(np.frombuffer(buffer.read(), dtype="f4"), (-1, 1024, 4))
    return np.sum(np.sqrt(points[:,:,0]*points[:,:,0] + points[:,:,1]*points[:,:,1]))