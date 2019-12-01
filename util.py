import numpy as np
import cmath

def add_point(data, cx, cy, width=100, turns=0, mag=1):

    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            dx, dy = x-cy, y-cy
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if (0 < dist < 100):
                mag = dist / 100
                mag = -4 * mag * (mag - 1)
                phase = turns * cmath.phase(dx + dy*1j)
                out = cmath.rect(mag, phase)
                data[x, y, 0] = out.real
                data[x, y, 1] = out.imag