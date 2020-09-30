import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from itertools import count
from numba import jit
import time

@jit(nopython=True)
def jones(dist, eq_dist):
    dist = dist / eq_dist
    dist2 = dist * dist
    dist4 = dist2 * dist2
    dist8 = dist4 * dist4
    return 1 / (dist8) - 1 / (dist4 * dist2)

def particle_grid(width, spacing):
    grid_min, grid_max = -1*width/2, width/2
    part_per_row = (grid_max - grid_min) // spacing
    pos = np.array([[x, y] for x in np.linspace(grid_min, grid_max, part_per_row) for y in
                    np.linspace(grid_min, grid_max, part_per_row)])
    return pos


class ParticleGraph(pg.GraphItem):
    def __init__(self, pos, vel, charge, eneg, idempot, dt):
        pg.setConfigOptions(antialias=True)

        self.app = QtGui.QApplication([])
        self.w = pg.GraphicsWindow()
        self.w.setWindowTitle('Particles')
        self.g = pg.GraphItem()
        self.v = self.w.addPlot()
        self.v.addItem(self.g)
        self.view_window = (-11,11)
        self.wall_dist = (-10,10)
        self.dragged_point_index = None
        self.dragOffset = None

        self.dt = dt
        self.position = pos
        self.vel = vel
        self.charge = charge
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

    # def bond_energy(self, dist2):


    def step(self):
        dist = self.position - self.position[:, None, :]
        dist2 = dist * dist
        dist_mag = np.sum(dist2, axis=2)
        dist2 = dist_mag * dist_mag
        dist4 = dist2 * dist2

        bond_order = self.bond_order(dist2)

        atomic_coulomb_grad = -1 * dist_mag / (np.power(dist2 + 1, 1.5))

        charge_grad = self.eneg + self.idempot * self.charge + 1*np.sum(bond_order * atomic_coulomb_grad, axis=1)
        charge_transfer = charge_grad - charge_grad[None, :].T

        self.charge += np.sum(np.nan_to_num(charge_transfer * bond_order * bond_order * bond_order), axis=1) * 0.01

        particle_charge_attraction = 4.5 * self.charge[None, :] * self.charge[None, :].T / (0.05 + dist2)

        attraction = 0.011 * jones(dist_mag * 1.00, self.eq_dist) + particle_charge_attraction

        force = attraction[:, :, None] * dist / dist_mag[:, :, None]
        force = np.sum(np.nan_to_num(force), axis=0)

        neg_embed = self.wall_dist[0] - np.minimum(self.wall_dist[0], self.position)
        pos_embed = self.wall_dist[1] - np.maximum(self.wall_dist[1], self.position)
        force += neg_embed * neg_embed
        force -= pos_embed * pos_embed

        self.vel += force * self.dt + self.grav_pull
        self.vel *= 0.99995

        self.vel /= np.maximum(self.max_vel, np.abs(self.vel)) / self.max_vel
        self.position += self.vel * self.dt

    def redraw(self):
        # brush = self.cmap.map(np.sum(self.vel * self.vel, axis=1), 'qcolor') * 10
        brush = self.cmap.map(self.charge, 'qcolor')
        self.g.setData(pos=self.position, pen=0.5, brush=brush, symbol='o', size=self.eq_dist, pxMode=False)
        self.v.setXRange(-1 * self.view_window[0], self.view_window[0])
        self.v.setYRange(-1 * self.view_window[1], self.view_window[1])

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind][1] - pos[1]
        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        ind = self.dragPoint.data()[0]
        self.data['pos'][ind][1] = ev.pos()[1] + self.dragOffset
        self.updateGraph()
        ev.accept()


if __name__ == '__main__':

    dt = 0.1

    pos = np.array([[-1, -1], [1, 1]]).astype(np.float64)
    pos = particle_grid(12, 1)
    vel = np.zeros_like(pos)
    n_part = len(pos)
    charge = np.random.random(n_part) * 2 - 1
    charge = np.linspace(-2,2,n_part)
    charge = np.zeros_like(charge)

    eneg = np.array([1, 1]).astype(np.float64)
    eneg = np.ones_like(charge)*1
    eneg = np.random.choice([1,1])
    idempot = np.array([1, 1]).astype(np.float64)
    idempot = np.ones_like(charge)

    grapher = ParticleGraph(pos=pos, vel=vel, charge=charge, eneg=eneg, idempot=idempot, dt=dt)

    iter_per_redraw = 10
    start = time.time()
    for iter in count():
        grapher.step()
        if iter % iter_per_redraw == 0:
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