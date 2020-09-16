import numpy as np
from numpy.fft import fft2, ifft2, fft, ifft, fftshift, ifftshift
from util import add_packet, grad
from util import add_point_zero_origin_smooth_tail as add_point
from util import add_singordon_packet as add_packet
# from util import add_point_vortex as add_point
import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from math import pi
from coloring import colorize
from util import first_unoccupied
from itertools import count

from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg


# width = 11
width = 251

n_particles = 1
particle_pos = [(0,0)]

field_pos = np.zeros((n_particles, width,width))







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
for iter in count():
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

