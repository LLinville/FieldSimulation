import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from itertools import count
from numba import jit
from util import colorize
from matplotlib import pyplot as plt
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from PyQt5.QtCore import *
from pyqtgraph.Qt import QtCore, QtGui



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
    return pos.astype(np.float32)


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
        pg.GraphItem.__init__(self)
        #self.scatter.sigClicked.connect(self.clicked)
        self.v = self.w.addPlot()
        self.v.addItem(self.g)
        self.v.getViewBox().setAspectLocked(True)
        self.view_window = (-81,81)
        self.wall_dist = (-30,30)
        self.dragged_point_index = None
        self.dragOffset = None

        self.dt = dt
        self.position = pos
        self.vel = vel
        self.charges = charges
        self.debug = np.zeros_like(self.position)
        self.eneg = eneg
        self.idempot = idempot
        self.n_part = np.array([pos.shape[0]]).astype(np.int32)
        self.eq_dist = 1
        self.grav_pull = np.array([0,-1]) * 0.00000
        self.max_vel = 1
        self.cmap = pg.ColorMap([-0.1,0,0.1], np.array([[255,0,0,255],[255,255,255,255],[0,0,255,255]]))
        pg.GraphItem.__init__(self)

        self.position_gpu = cuda.mem_alloc(pos.nbytes)
        self.vel_gpu = cuda.mem_alloc(vel.nbytes)
        self.charges_gpu = cuda.mem_alloc(charges.nbytes)
        self.dt_gpu = cuda.mem_alloc(dt.nbytes)
        self.debug_gpu = cuda.mem_alloc(pos.nbytes)
        self.n_gpu = cuda.mem_alloc(self.n_part.nbytes)
        cuda.memcpy_htod(self.position_gpu, pos)
        cuda.memcpy_htod(self.vel_gpu, vel)
        cuda.memcpy_htod(self.charges_gpu, charges)
        cuda.memcpy_htod(self.dt_gpu, dt)
        cuda.memcpy_htod(self.n_gpu, self.n_part)

        with open("nbody.c", "r") as kernel_file:
            source_module = SourceModule(kernel_file.read())

        self.update_vel = source_module.get_function("applyForce")

        self.BLOCK_SIZE = 256
        self.nBlocks = (self.n_part + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE

        pg.GraphItem.__init__(self)
        self.scatter.mouseClickEvent = self.mouseClickEvent

    def bond_order(self, dist2):
        a = -0.2
        return np.exp(a * dist2)

    def step(self):

        self.update_vel(self.position_gpu, self.vel_gpu,  self.dt_gpu, self.n_part, block=(self.BLOCK_SIZE, 1, 1), grid=(int(self.nBlocks), 1, 1))


        # dist = self.position - self.position[:, None, :]
        # dist2 = dist * dist
        # dist_mag = np.sum(dist2, axis=2)
        # # dist2 = dist_mag * dist_mag
        # # dist4 = dist2 * dist2
        #
        # # bond_order = self.bond_order(dist2)
        # #
        # # atomic_coulomb_grad = -1 * dist_mag / (np.power(dist2 + 1, 1.5))
        # #
        # # charge_grad = self.eneg + self.idempot * self.charges + 1*np.sum(bond_order * atomic_coulomb_grad, axis=1)
        # # charge_transfer = charge_grad - charge_grad[None, :].T
        #
        #
        #
        # # self.charges += np.sum(np.nan_to_num(charge_transfer * bond_order * bond_order * bond_order), axis=1) * 0.01
        # self.charges += charge_oscillation(self.charges, dist2) * 0.01
        #
        # # particle_charge_attraction = 1.0 * self.charges[None, :] * self.charges[None, :].T / (0.05 + dist2)
        #
        # attraction = 0.911 * jones(dist_mag * 1.00, self.eq_dist)# + particle_charge_attraction
        #
        # force = attraction[:, :, None] * dist / dist_mag[:, :, None]
        # force = np.sum(np.nan_to_num(force), axis=0)
        #
        # neg_embed = self.wall_dist[0] - np.minimum(self.wall_dist[0], self.position)
        # pos_embed = self.wall_dist[1] - np.maximum(self.wall_dist[1], self.position)
        # force += neg_embed * neg_embed
        # force -= pos_embed * pos_embed
        #
        # self.vel += force * self.dt + self.grav_pull
        # # self.vel *= 0.99995
        #
        # self.vel /= np.maximum(self.max_vel, np.abs(self.vel)) / self.max_vel
        # self.position += self.vel * self.dt

    def redraw(self):
        brush = self.cmap.map(np.sum(self.vel * self.vel, axis=1), 'qcolor')

        cuda.memcpy_dtoh(self.position, self.position_gpu)
        cuda.memcpy_dtoh(self.vel, self.vel_gpu)
        cuda.memcpy_dtoh(self.debug, self.debug_gpu)

        self.g.setData(pos=np.nan_to_num(self.position), pen=0.5, symbol='o', size=self.eq_dist, brush=brush, pxMode=False, c=colorize(self.charges[None,:,0] + 1j * self.charges[None,:,1]))
        self.v.setXRange(-1 * self.view_window[0], self.view_window[0])
        self.v.setYRange(-1 * self.view_window[1], self.view_window[1])

    def clicked(self, pts):
        print("clicked: %s" % pts)

    def mouseClickEvent(self, ev):
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

    dt = np.float32(0.0)

    # pos = np.array([[0, -1], [0, 1], [-1, 0], [1, 0], [2,2]]).astype(np.float32)
    # pos = np.random.random((100,2)).astype(np.float32)*20-10
    pos = particle_grid(33, 1.0).astype(np.float32)
    vel = np.zeros_like(pos)
    # vel = np.random.random((5,2)).astype(np.float32)*0.1-0.05
    # vel = np.array([[-1, 0], [1, 0], [0, 1], [0, -1], [0,0]]).astype(np.float32)
    n_part = len(pos)
    charge = np.random.random(n_part) * 2 - 1
    charges = np.linspace(-2,2,n_part)
    charges = np.zeros_like(charge)
    charges = np.array([[2, -1]]).astype(np.float32)

    eneg = np.array([1, 1]).astype(np.float32)
    eneg = np.ones_like(charge)*1
    eneg = np.random.choice([1,1])
    idempot = np.array([1, 1]).astype(np.float32)
    idempot = np.ones_like(charge)

    grapher = ParticleGraph(pos=pos, vel=vel, charges=charges, eneg=eneg, idempot=idempot, dt=dt)

    history = []
    iter_per_redraw = 1000
    start = time.time()
    for iter in count():
        grapher.step()
        history.append(grapher.charges.copy())
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