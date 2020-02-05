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

def total_mag(buffer):
    points = np.reshape(np.frombuffer(buffer.read(), dtype="f4"), (-1, 1024, 4))
    return np.sum(np.sqrt(points[:,:,0]*points[:,:,0] + points[:,:,1]*points[:,:,1]))


def first_unoccupied