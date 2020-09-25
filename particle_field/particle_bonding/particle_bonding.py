import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from itertools import count
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from particle_bonding import ParticleGrapher
from util import fast_matmul

pos = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64) * 1
pos = pos[0:3]
# pos = np.random.random((n_part,2))*6-3
grid_min, grid_max = -10,10
spacing = 1.5
part_per_row = (grid_max - grid_min) // spacing
pos = np.array([[x,y] for x in np.linspace(grid_min, grid_max, part_per_row) for y in np.linspace(grid_min, grid_max, part_per_row)])

n_part = len(pos)

vel = np.zeros_like(pos)

eneg = np.ones(n_part)
# charge = np.ones(n_part)
# charge = np.array([1,1,-1,-1], dtype=np.float64)
charge = np.ones(n_part, dtype=np.float64)
charge = np.random.random(n_part) * 2 - 1

elec_shared = np.diag(np.ones(n_part))

spread_speed = 0.1

base_spread = (np.ones_like(elec_shared) * spread_speed) / n_part + np.diag(np.ones(n_part) * (1 - spread_speed))


app = QtGui.QApplication([])

## Create window with GraphicsView widget
win = pg.GraphicsLayoutWidget()
win.show()  ## show widget alone in its own window
win.setWindowTitle('pyqtgraph example: ImageItem')
view = win.addViewBox()
# view2 = win.addViewBox()

## lock the aspect ratio so pixels are always square
view.setAspectLocked(True)
# view2.setAspectLocked(True)

cmap = pg.ColorMap([-1,1], np.array([[0,0,0,255],[255,255,255,255]]))

g1 = pg.GraphItem()

view.addItem(g1)
# p.addItem(g2)



# dist = squareform(pdist(pos))







view_window = (-15,15)
eq_dist = 1

dt = 0.05
max_vel = 1.1

def graph(g_item, position, vel, charge, view_dist):
    brush = cmap.map(np.sum(vel*vel, axis=1), 'qcolor') * 10
    brush = cmap.map(charge, 'qcolor')
    g1.setData(pos=position, pen=0.5, brush=brush, symbol='o', size=eq_dist, pxMode=False)
    view.setXRange(-1*view_dist[0], view_dist[0])
    view.setYRange(-1*view_dist[1], view_dist[1])


def jones(dist, eq_dist):
    dist = dist / eq_dist
    dist2 = dist * dist
    dist4 = dist2 * dist2
    dist8 = dist4 * dist4
    return 1 / (dist8) - 1 / (dist4 * dist2)


grav_pull = np.array([0,-1]) * 0.00001

for iter in count():

    dist = pos - pos[:,None,:]
    dist2 = dist * dist
    dist_mag = np.sum(dist2, axis=2)
    dist2 = dist_mag*dist_mag

    particle_charge_attraction = 1.5 * charge[None, :] * charge[None, :].T / (0.05 + dist2)
    attraction = 1 * jones(dist_mag + 0.000, eq_dist) + particle_charge_attraction

    force = attraction[:,:,None] * dist / dist_mag[:,:,None]
    force = np.sum(np.nan_to_num(force), axis=0)

    neg_embed = view_window[0] - np.minimum(view_window[0], pos)
    pos_embed = view_window[1] - np.maximum(view_window[1], pos)
    force += neg_embed * neg_embed
    force -= pos_embed * pos_embed

    # x_force = attraction * x_dist / dist
    # x_force = np.sum(np.nan_to_num(x_force), axis=1)
    # x_left_embed = view_window[0] -  np.minimum(view_window[0], pos[:, 0])
    # # x_right_embed = np.maximum(view_window[0], pos[:, 0] + )
    # x_force += x_left_embed * x_left_embed
    #
    #
    # y_force = attraction * y_dist / dist
    # y_force = np.sum(np.nan_to_num(y_force), axis=1)



    vel += force * dt + grav_pull
    vel *= 0.9999

    vel /= np.maximum(max_vel, np.abs(vel)) / max_vel
    pos += vel * dt

    if iter%100 == 0:
        graph(g1, pos, vel, charge, view_dist=view_window)
        app.processEvents()
        # plt.clf()
        # plt.scatter(pos[:,0],pos[:,1], c=charge)
        # axes = plt.axes()
        # axes.set_xlim([-2, 2])
        # axes.set_ylim([-2, 2])
        # plt.pause(0.001)
        print(iter)
        # print(f'Ekin = {np.sum(vel*vel)}')
        # print(np.abs(vel))


print(dist)