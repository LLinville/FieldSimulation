import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from itertools import count
from numba import jit
from util import colorize
from matplotlib import pyplot as plt
import time

@jit(nopython=True)
def jones(dist, eq_dist):
    dist = dist / eq_dist
    dist2 = dist * dist
    dist4 = dist2 * dist2
    dist8 = dist4 * dist4
    return 1 / (dist8 + 0.3) - 1 / (dist4 * dist2+0.4)


def particle_grid(width, spacing):
    grid_min, grid_max = -1*width/2, width/2
    part_per_row = (grid_max - grid_min) // spacing
    pos = np.array([[x, y] for x in np.linspace(grid_min, grid_max, part_per_row) for y in
                    np.linspace(grid_min, grid_max, part_per_row)])
    return pos


def charge_oscillation(charges, dist2):
    # A catalyzes -B -> B
    # B catalyzes A -> -A
    # -A catalyzes B -> -B
    # -B catalyzes -A -> A

    # Unit change from presence of each charge type
    # Diagonal zeros
    induction_strengths = np.array([[0.0, -1], [-1, 0.0]])

    dcharge_dt = np.sum(charges * induction_strengths, axis=0)
    return dcharge_dt

class ParticleGraph(pg.GraphItem):
    def __init__(self, pos, vel, charges, eneg, idempot, dt):
        pg.setConfigOptions(antialias=True)

        self.app = QtGui.QApplication([])
        self.w = pg.GraphicsWindow()
        self.w.setWindowTitle('Particles')
        self.g = pg.GraphItem()
        self.v = self.w.addPlot()
        self.v.getViewBox().setAspectLocked(True)
        self.v.getViewBox().disableAutoRange()
        self.v.addItem(self.g)
        self.view_window = (-11,11)
        self.wall_dist = (-10,10)
        self.v.setXRange(-1 * self.view_window[0], self.view_window[0])
        self.v.setYRange(-1 * self.view_window[1], self.view_window[1])
        self.dragged_point_index = None
        self.dragOffset = None

        self.dt = dt
        self.position = pos
        self.velocity = vel
        self.charges = charges
        self.eneg = eneg
        self.idempot = idempot
        self.n_part = len(pos)
        self.eq_dist = 1
        self.grav_pull = np.array([0,-1]) * 0.00000
        self.max_vel = 1
        self.cmap = pg.ColorMap([-0.5,0,0.5], np.array([[255,0,0,255],[255,255,255,255],[0,0,255,255]]))
        pg.GraphItem.__init__(self)


    def bond_order(self, dist2):
        a = -0.2
        return np.exp(a * dist2)

    def force(self, position):
        dist = position - position[:, None, :]
        dist2 = dist * dist
        dist_mag = np.sum(dist2, axis=2)
        # dist2 = dist_mag * dist_mag
        # dist4 = dist2 * dist2

        # bond_order = self.bond_order(dist2)
        #
        # atomic_coulomb_grad = -1 * dist_mag / (np.power(dist2 + 1, 1.5))
        #
        # charge_grad = self.eneg + self.idempot * self.charges + 1*np.sum(bond_order * atomic_coulomb_grad, axis=1)
        # charge_transfer = charge_grad - charge_grad[None, :].T

        # self.charges += np.sum(np.nan_to_num(charge_transfer * bond_order * bond_order * bond_order), axis=1) * 0.01
        # self.charges += charge_oscillation(self.charges, dist2) * 0.01

        # particle_charge_attraction = 1.0 * self.charges[None, :] * self.charges[None, :].T / (0.05 + dist2)

        attraction = 0.009 * jones(dist_mag * 1.00, self.eq_dist)  # + particle_charge_attraction

        force = attraction[:, :, None] * dist / dist_mag[:, :, None]
        force = np.sum(np.nan_to_num(force), axis=0)
        return force

    def step(self):
        neg_embed = self.wall_dist[0] - np.minimum(self.wall_dist[0], self.position)
        pos_embed = self.wall_dist[1] - np.maximum(self.wall_dist[1], self.position)
        # self.velocity += neg_embed * neg_embed * self.dt * 0.01
        # self.velocity -= pos_embed * pos_embed * self.dt * 0.01

        self.velocity += self.grav_pull * self.dt
        # self.velocity *= 0.99995
        self.velocity /= np.maximum(self.max_vel, np.abs(self.velocity)) / self.max_vel

        kp1 = self.velocity
        kv1 = self.force(self.position)
        kp2 = self.velocity + kv1 * self.dt / 2
        kv2 = self.force(self.position + kp1 * self.dt / 2)
        kp3 = self.velocity + kv2 * self.dt / 2
        kv3 = self.force(self.position + kp2 * self.dt / 2)
        kp4 = self.velocity + kv3 * self.dt
        kv4 = self.force(self.position + kp3 * self.dt)

        self.position += (kp1 + 2*kp2 + 2*kp3 + kp4) * self.dt / 6
        self.velocity += (kv1 + 2*kv2 + 2*kv3 + kv4) * self.dt / 6


    def redraw(self):
        brush = self.cmap.map(np.sum(self.velocity * self.velocity, axis=1), 'qcolor')
        # brush = self.cmap.map(self.max_bond_order - self.total_bond_order, 'qcolor')
        # brush = self.cmap.map(np.sum(self.charge * self.charge, axis=1), 'qcolor')
        # brush = self.cmap.map(self.charge, 'qcolor')
        # brush = [pg.mkColor(self.charge[i][0] * 255, self.charge[i][1] * 255, self.charge[i][2] * 255) for i in range(n_part)]

        self.g.setData(pos=np.nan_to_num(self.position),
                        pen=0.5,
                        symbol='o',
                        size=self.eq_dist,
                        brush=brush,
                        pxMode=False)




if __name__ == '__main__':

    dt = 6.6125

    pos = np.array([[-1, -1], [1, 1]]).astype(np.float64)
    pos = particle_grid(22, 1)
    # pos = np.array([[0,0]]).astype(np.float64)
    vel = np.zeros_like(pos)
    n_part = len(pos)
    charge = np.random.random(n_part) * 2 - 1
    charges = np.linspace(-2,2,n_part)
    charges = np.zeros_like(charge)
    charges = np.array([[2, -1]]).astype(np.float64)

    eneg = np.array([1, 1]).astype(np.float64)
    eneg = np.ones_like(charge)*1
    # eneg = np.random.choice([1,1])
    idempot = np.array([1, 1]).astype(np.float64)
    idempot = np.ones_like(charge)

    grapher = ParticleGraph(pos=pos, vel=vel, charges=charges, eneg=eneg, idempot=idempot, dt=dt)

    history = []
    iter_per_redraw = 10
    start = time.time()
    for iter in count():
        grapher.step()
        history.append(grapher.charges.copy())
        if iter % iter_per_redraw == 0:
            print(charges)
            grapher.redraw()
            grapher.app.processEvents()
            end = time.time()
            try:
                print(f'{iter_per_redraw / (end - start)} iter/s')
                start = end
            except Exception as e:
                print(e)
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     QtGui.QApplication.instance().exec_()