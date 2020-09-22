import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from itertools import count
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

def graphics_setup():
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

    return app, win, [view, view2], [img, img2]


app, win, [view, view2], [img, img2] = graphics_setup()

n_part = 1000

pos = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
pos = np.random.random((n_part,2))*2-1

vel = np.zeros_like(pos)

eneg = np.ones(n_part)
# charge = np.ones(n_part)
# charge = np.array([1,1,-1,-1], dtype=np.float64)
charge = np.ones(n_part, dtype=np.float64)
charge = np.random.random(n_part)

elec_shared = np.diag(np.ones(n_part))

spread_speed = 0.1

base_spread = (np.ones_like(elec_shared) * spread_speed) / n_part + np.diag(np.ones(n_part) * (1 - spread_speed))


# dist = squareform(pdist(pos))



# spread charge to others by dist?

# nonlinear scaling of electronegativity with distance
# particle getting close to another more electronegative one causes an even more imbalanced charge
# polar covalent bond will become more polar with less distance
# potential for interaction at the scale of transferring charge from thermal motion alone
# polar covalent bond pushed closer together will cause loss of negative charge on less electronegative particle
# balancing effect pushing away? What is the attraction that makes it a balanced force?


# ---- distance not involved
# spread by bond strength minus difference in electronegativity
# should be symmetric to preserve charge


# ---- distance involved

# function of equilibrium of a bond's charge distribution given a distance and two electronegativities
# little to no difference in eneg means balanced covalent bond
# moderate difference in eneg means polar covalent bond
# large difference in eneg means ionic bond

# calculate equilibrium charge at current distance with charge difference
# charge distribution approaches that value


dt = 0.001


# plt.scatter(pos[:,0],pos[:,1], c=charge)


for iter in count():

    x_dist = pos[:, 0] - pos[:, 0][None, :].T
    y_dist = pos[:, 1] - pos[:, 1][None, :].T
    dist2 = x_dist * x_dist + y_dist * y_dist
    dist = np.sqrt(dist2)

    particle_charge_attraction = 1 * charge[None, :] * charge[None, :].T / (0.001 + dist2)

    x_force = particle_charge_attraction * x_dist / dist
    x_force = np.sum(np.nan_to_num(x_force), axis=1)

    y_force = particle_charge_attraction * y_dist / dist
    y_force = np.sum(np.nan_to_num(y_force), axis=1)

    vel += np.stack([x_force, y_force], axis=1) * dt
    pos += vel * dt

    if iter%1 == 0:
        plt.clf()
        plt.scatter(pos[:,0],pos[:,1], c=charge)
        axes = plt.axes()
        axes.set_xlim([-2, 2])
        axes.set_ylim([-2, 2])
        plt.pause(0.001)
        print(iter)
        # print(np.abs(vel))


print(dist)