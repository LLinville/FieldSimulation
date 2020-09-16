import numpy as np
from numpy.fft import fft2, ifft2, fft, ifft, fftshift, ifftshift
from util import add_packet, grad
from util import add_point_zero_origin_smooth_tail as add_point
from util import add_singordon_packet as add_packet
# from util import add_point_vortex as add_point
from scipy.ndimage.filters import gaussian_filter
import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from math import pi
from coloring import colorize
from util import first_unoccupied
import imageio

from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg


# width = 11
width = 251


fields = {}


position = np.zeros((width,width), dtype=complex)

# position = np.array([[np.sin(x/width*10*pi) * np.sin(y/width*10*pi) for x in range(width)] for y in range(width)], dtype=complex)

# position = np.array([[np.sin(x/width*10*pi + y/width*10*pi) for x in range(width)] for y in range(width)], dtype=complex)

# position = np.array([[np.exp(1j * (x+y)/width*2*pi*10) for x in range(width)] for y in range(width)], dtype=complex)

# add_point(position, 168, 168, width=10, turns=0)
# add_point(position, 107, 107, width=30, turns=0, mag=-1)
# add_point(position, 207, 207, width=20, turns=0, mag=1)
# add_point(position, 50, 50, width=10, turns=1, mag=1)
# add_point(position, 180, 180, width=10, turns=2, mag=1)
# add_packet(position, 150,150, width=20, momentum=10, direction=0)
# add_packet(position, 350,150, width=30, momentum=1, direction=pi/-2)
add_packet(position, 107,107, width=8, momentum=0, direction=0)
# add_packet(position, 147,147, width=10, momentum=0, direction=0, mag=-1)
# add_packet(position, 350, 350, width=30, momentum=3, direction=2*pi/4)
# add_packet(position, 5,5,width=3)
# position *= 1j
# add_packet(position, 130, 100, width=10, momentum=-3, direction=3*pi/4)
# position /= np.sum(np.abs(position))
position /= np.max(np.abs(position))
position *= 1.0
# position = np.abs(position)

vel = np.zeros_like(position)
nonlinear_vel = np.zeros_like(position)

# position[3,3] = 1

dt = 10

coords = np.array(np.linspace(-1, 1, width))
# coords = np.roll(coords, width//2)



momentum_map = np.array([coords]).repeat(width, axis=0)
# np.roll(momentum_map, width//2, axis=0)
# momentum_map = momentum_map * momentum_map
momentum_map = momentum_map + momentum_map.transpose()
# momentum_map *=1.11
# momentum_map = np.sqrt(momentum_map + momentum_map.transpose())
momentum_map = momentum_map ** 1



# larger mass means wave of given size travels slower
# lower mass means larger wave scale on average
mass = 1
momentum_op = np.exp(-1j * momentum_map**1 * dt / 2 / mass)
# momentum_op = np.exp(-20*momentum_map**2*dt/mass)


ground_potential = np.abs(np.array([[x**2 + y**2 for x in np.array(np.linspace(-1, 1, width))] for y in np.array(np.linspace(-1, 1, width))], dtype=complex)) * 0.950
# ground_potential = np.zeros_like(position)
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

pad_width = 10

def handleClick():
    pass



app = QtGui.QApplication([])

## Create window with GraphicsView widget
win = pg.GraphicsLayoutWidget()
win.show()  ## show widget alone in its own window
win.setWindowTitle('pyqtgraph example: ImageItem')
view = win.addViewBox()
view2 = win.addViewBox()

## lock the aspect ratio so pixels are always square
view.setAspectLocked(True)
view2.setAspectLocked(True)

## Create image item
img = pg.ImageItem(border='w')
view.addItem(img)
img2 = pg.ImageItem(border='w')
view2.addItem(img2)

# position = psi0
all_outputs = []
positive_momentum = fftshift(fft2(position))
negative_momentum = np.copy(positive_momentum)

for iter in range(10000):
    print(iter)
    to_compare = np.copy(position)

    if iter % 10 == -10:
        position /= np.sum(np.abs(position))
    # if iter % 10 == 0:
    #     position /= np.max(np.abs(position))
    # if iter % 10 == -10:
    #     for substep in range(1):
    #         potential -= np.abs(position)*1.1
    #         potential = ifft2(fftshift(fftshift(fft2(potential)) * diffuse_op)) * 1

        # potential_op = np.exp(-1j * potential * 0.00001 * dt / 2)
    for substep in range(1):
        # momentum_transfer_strength = 0.91*0
        # positive_momentum_new = fftshift(fft(position))
        # negative_momentum_new = np.copy(positive_momentum)
        # positive_momentum += (positive_momentum_new - positive_momentum) * momentum_transfer_strength
        # negative_momentum += (negative_momentum_new - negative_momentum) * momentum_transfer_strength
        positive_momentum *= momentum_op
        negative_momentum /= momentum_op
        momentum = positive_momentum + 0*negative_momentum
        # # momentum = fftshift(fft2(position))
        # # momentum *= momentum_op ** 1
        position = ifft2(ifftshift(momentum))
        # potential = ground_potential# + 0.00001*1.0 / gaussian_filter(np.abs(np.maximum(position, 0.01)*1), sigma=15)**2
        pos_mag = np.abs(position)

        # position


        # TODO:
        # store velocity, adding nonlinear term to velocity, then add velocity to position
        # nonlinear_vel += (1-pos_mag**2)
        # position += nonlinear_vel * position * 0.03


        # position *= ground_potential
        # position /= np.sum(np.abs(position))

        nonlinear_strength = -0.01*0
        # nonlinear = position * (1-pos_mag**2)
        # nonlinear = np.pad(nonlinear[pad_width:-1 * pad_width, pad_width:-1 * pad_width], pad_width)
        nonlinear = (np.sin(pos_mag*2*pi)) * position/pos_mag * nonlinear_strength
        # nonlinear = pos_mag**2*position*nonlinear_strength

        # vel += nonlinear * dt
        # vel += grad(position) * dt
        # position += vel * dt

        # position *= np.exp(-1j*nonlinear)

        # position *= np.exp(1*(1-pos_mag**2)*1.0)
        # position += nonlinear
        # position /= np.exp(-1j*pos_mag**2 * 0.08)
        # position -= 0.1*position * (1-pos_mag)
        # position *= np.exp(-1j * potential * dt / 1000)
        # position *= potential_op

    # padding = np.pad(position[pad_width:-1*pad_width, pad_width:-1*pad_width], pad_width, mode='edge')
    # padding[pad_width:-1*pad_width, pad_width:-1*pad_width] = 0
    # position = np.pad(position[pad_width:-1*pad_width, pad_width:-1*pad_width], pad_width) + padding*0


    # all_outputs.append(colorize(position))
    if iter % 1 == 0:
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.abs(potential))
        # plt.imshow(colorize(momentum))
        # plt.imshow(momentum.real)
        # plt.imshow(np.minimum(10, np.sqrt(momentum.real ** 2 + momentum.imag ** 2)))#[200:300,200:300])
        # plt.subplot(1, 2, 2)
        # plt.imshow(all_outputs[-1])
        # plt.imshow(np.sqrt(position.real ** 2 + position.imag ** 2))
        # plt.subplot(1, 3, 3)
        # plt.imshow(ground_potential + 0.00001*1.0 / gaussian_filter(np.abs(np.maximum(position, 0.01)*1), sigma=15)**2)
        # plt.imshow(colorize(fftshift(fft2(position))))
        # plt.draw_all()
        # plt.pause(0.001)]]
        # img.setImage(np.abs(momentum))
        # img.setImage(np.maximum(0,np.abs(position)))

        # img2.setImage(colorize(positive_momentum_new - positive_momentum))

        max_pos = np.max(pos_mag)
        max_vel = np.max(np.abs(vel))
        # pos_mag[0,0] = 1
        # img2.setImage(pos_mag % 1)
        # img2.setImage(colorize(np.exp(2j*pi*pos_mag)))
        energy = 0*pos_mag**2 + np.abs(vel)**2
        img2.setImage(energy)

        # img2.setImage(colorize(momentum[2*width//5:3*width//5,2*width//5:3*width//5]))
        img.setImage(colorize(position.transpose()/np.ceil(max_pos+0.1)))
        # img2.setImage(colorize(nonlinear))
        print(f'max position: {max_pos}')
        print(f'max nonlinear: {np.max(np.abs(nonlinear))}')
        print(f'total energy: {np.sum(energy)}')
        # img.setImage(colorize(position - to_compare))
        # img.setImage(colorize(position.imag))
        # img.setImage(colorize(momentum_op))
        # img2.setImage(colorize(position))
        app.processEvents()




# print("normalizing and reshaping")
# # from 0 - 1 (nIter, width, width, 3) to 0 - 255 (nIter, 3, width, width)
# all_outputs /= np.max(all_outputs)
# all_outputs = np.array(all_outputs) * 255
# all_outputs = np.array(np.floor(all_outputs), dtype=int)
# # all_outputs = np.transpose(all_outputs, (0, 3, 1, 2))
# print("Writing to video")
# # array2gif.write_gif(all_outputs, "schrodinger2dOut.gif")
# # print(np.max(all_outputs), np.min(all_outputs))
# imageio.mimwrite(first_unoccupied("../images/schrodinger2dOut%s.mp4"), all_outputs, "mp4", fps=20)


# n_particles = 1
# field_res = 100
# particle_pos = np.random.random((n_particles,2))
# particle_pos = np.zeros_like(particle_pos)
# field_pos = np.zeros((field_res, field_res))
#
# potential = np.zeros_like(field_pos)
# potential = np.zeros_like(potential)
#
#
# p_pos = particle_pos[0]
# x,y = np.linspace(-1-p_pos[0], 1-p_pos[0],field_res), np.linspace(-1-p_pos[1], 1-p_pos[1],field_res)
# x,y = np.meshgrid(x,y)
# r2 = x*x+y*y
#
# while True:
#     img.setImage(r2)
#     app.processEvents()

