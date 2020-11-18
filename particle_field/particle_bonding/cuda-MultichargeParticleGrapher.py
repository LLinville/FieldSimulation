#!/c/Users/Eracoy/virtualenv/graphicsMisc/Scripts python

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from itertools import count
from numba import jit
# from util import colorize
from matplotlib import pyplot as plt
import time
import pycuda.driver as cuda
import pycuda.driver
pycuda.driver.set_debugging()
import pycuda.autoinit

# import pycuda.debug
from pycuda.compiler import SourceModule
from PyQt5.QtCore import *
from pyqtgraph.Qt import QtCore, QtGui
import random



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
    pos = np.array([[x, y] for x in np.linspace(grid_min, grid_max, part_per_row+2) for y in
                    np.linspace(grid_min, grid_max, part_per_row)])
    return pos.astype(np.float32)


def charge_oscillation(charge, dist2):
    # A catalyzes -B -> B
    # B catalyzes A -> -A
    # -A catalyzes B -> -B
    # -B catalyzes -A -> A

    # Unit change from presence of each charge type
    # Diagonal zeros
    induction_strengths = np.array([[0.0, -1], [-1, 0.0]])

    dcharge_dt = np.sum(charge * induction_strengths, axis=0)
    return dcharge_dt

class ParticleGraph(pg.GraphItem):
    def __init__(self, pos, vel, charge, charge_attractions, eneg, max_bond_order, idempot, dt):
        pg.setConfigOptions(antialias=True)

        self.app = QtGui.QApplication([])
        self.w = pg.GraphicsWindow()
        self.w.setWindowTitle('Particles')
        self.g1 = pg.GraphItem()
        pg.GraphItem.__init__(self)
        #self.scatter.sigClicked.connect(self.clicked)
        self.v1 = self.w.addPlot()
        # self.v2 = self.w.addPlot()
        self.v1.addItem(self.g1)
        self.v1.getViewBox().setAspectLocked(True)
        self.v1.getViewBox().disableAutoRange()
        self.view_window = (-31,31)
        self.wall_dist = (-30,30)
        self.v1.setXRange(-1 * self.view_window[0], self.view_window[0])
        self.v1.setYRange(-1 * self.view_window[1], self.view_window[1])
        self.dragged_point_index = None
        self.dragOffset = None

        self.dt = dt
        self.position = pos
        self.vel = vel
        self.charge = charge
        self.charge_attractions = charge_attractions
        self.debug = np.zeros_like(self.position)
        self.eneg = eneg
        self.total_bond_order = np.zeros_like(self.charge)
        self.max_bond_order = max_bond_order
        self.idempot = idempot
        self.n_part = np.array([pos.shape[0]]).astype(np.int32)
        self.eq_dist = 1
        self.grav_pull = np.array([0,-1]) * 0.00000
        self.max_vel = 1
        self.cmap = pg.ColorMap([1, 0, -1], np.array([[255,0,0,255],[255,255,255,255],[0,0,255,255]]))
        pg.GraphItem.__init__(self)

        self.history = np.zeros((1, n_part))

        self.position_gpu = cuda.mem_alloc(pos.nbytes)
        self.vel_gpu = cuda.mem_alloc(vel.nbytes)
        self.charge_gpu = cuda.mem_alloc(charge.nbytes)
        self.charge_attractions_gpu = cuda.mem_alloc(charge_attractions.nbytes)
        self.dt_gpu = cuda.mem_alloc(dt.nbytes)
        self.debug_gpu = cuda.mem_alloc(pos.nbytes)
        self.eneg_gpu = cuda.mem_alloc(eneg.nbytes)
        self.total_bond_order_gpu = cuda.mem_alloc(self.total_bond_order.nbytes)
        self.max_bond_order_gpu = cuda.mem_alloc(self.max_bond_order.nbytes)
        self.n_gpu = cuda.mem_alloc(self.n_part.nbytes)
        cuda.memcpy_htod(self.position_gpu, self.position)
        cuda.memcpy_htod(self.vel_gpu, self.vel)
        cuda.memcpy_htod(self.charge_gpu, self.charge.flatten())
        cuda.memcpy_htod(self.charge_attractions_gpu, self.charge_attractions.flatten())
        cuda.memcpy_htod(self.eneg_gpu, eneg)
        cuda.memcpy_htod(self.total_bond_order_gpu, self.total_bond_order)
        cuda.memcpy_htod(self.max_bond_order_gpu, self.max_bond_order)
        cuda.memcpy_htod(self.dt_gpu, dt)
        cuda.memcpy_htod(self.n_gpu, self.n_part)

        with open("nbody.c", "r") as kernel_file:
            source_module = SourceModule(kernel_file.read(), options=['-g'])

        self.update_vel = source_module.get_function("applyForce")

        self.BLOCK_SIZE = 256
        self.nBlocks = (self.n_part + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE

        pg.GraphItem.__init__(self)
        # pg.sigClicked.connect(self.handleMouseClickEvent)

    def bond_order(self, dist2):
        a = -0.2
        return np.exp(a * dist2)

    def step(self):
        # cuda.memcpy_dtoh(self.vel, self.vel_gpu)
        # self.history = np.concatenate((self.history, np.sum(grapher.vel * grapher.vel, axis=1)[None, :]))
        # if self.history.shape[0] > 1000:
        #     self.history = self.history[:900]

        self.update_vel(self.position_gpu,
                        self.vel_gpu,
                        self.charge_gpu,
                        self.charge_attractions_gpu,
                        self.eneg_gpu,
                        self.total_bond_order_gpu,
                        self.max_bond_order_gpu,
                        self.dt_gpu,
                        self.n_part,
                        block=(self.BLOCK_SIZE, 1, 1),
                        grid=(int(self.nBlocks), 1, 1))

    def redraw(self):
        # brush = self.cmap.map(np.sum(self.vel * self.vel, axis=1), 'qcolor')

        cuda.memcpy_dtoh(self.position, self.position_gpu)
        cuda.memcpy_dtoh(self.vel, self.vel_gpu)
        cuda.memcpy_dtoh(self.charge, self.charge_gpu)
        cuda.memcpy_dtoh(self.total_bond_order, self.total_bond_order_gpu)
        # brush = self.cmap.map(np.sum(self.charge * self.charge, axis=1), 'qcolor')
        # brush = self.cmap.map(self.charge, 'qcolor')


        self.g1.setData(pos=np.nan_to_num(self.position),
                        pen=0.5,
                        symbol='o',
                        size=self.eq_dist,
                        brush=[pg.mkColor(self.charge[i][0]*255,self.charge[i][1]*255,self.charge[i][2]*255) for i in range(n_part)],
                        pxMode=False)#, c=colorize(self.n_part[None,:]))
        # y,x = np.histogram(self.history, bins=np.linspace(0.000000, 1.5, 100))
        # self.v2.clear()
        # self.v2.plot(x[:-1],y)
        # self.v1.setXRange(-1 * self.view_window[0], self.view_window[0])
        # self.v1.setYRange(-1 * self.view_window[1], self.view_window[1])

    def clicked(self, pts):
        print("clicked: %s" % pts)

    def handleMouseClickEvent(self, ev):
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
    pos = particle_grid(15, 1.5).astype(np.float32)
    vel = np.zeros_like(pos)
    # vel = np.random.random((vel.shape[0],2)).astype(np.float32)*0.1-0.05
    # vel = np.array([[-1, 0], [1, 0], [0, 1], [0, -1], [0,0]]).astype(np.float32)
    n_part = len(pos)
    n_charges = 2
    # charge = np.random.random(n_part) * 2 - 1
    # charge = np.linspace(-1,1,n_part).astype(np.float32)
    charge = np.ones((n_part, n_charges)).astype(np.float32)
    # charge = np.array([[2, -1]]).astype(np.float32)
    # charge = np.random.choice([1,0,-1], size=charge.shape).astype(np.float32)
    charge = np.array([[[1,0,0], [0,1,0], [0,0,1]][random.choice([0,1,1,1,2])] for i in range(n_part)]).astype(np.float32)


    pair_attractions = [
        [1, 5, 1],
        [5, 1, 5],
        [1, 5, 1]
    ]
    pair_attractions = np.array([
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 10]
    ]) * -1.1
    charge_attractions = np.matmul(charge, pair_attractions).astype(np.float32)
    # charge_attractions = np.ones_like(charge)

    eneg = np.array([1, 1]).astype(np.float32)
    # eneg = np.ones_like(charge)*1
    eneg = np.random.choice([1,2], size=charge.size).astype(np.float32)
    idempot = np.array([1, 1]).astype(np.float32)
    idempot = np.ones_like(charge)

    # max_bond_order = np.random.choice([1,4], size=charge.size).astype(np.float32)
    max_bond_order = eneg * 1

    grapher = ParticleGraph(pos=pos, vel=vel, charge=charge, charge_attractions=charge_attractions, eneg=eneg, max_bond_order=max_bond_order, idempot=idempot, dt=dt)

    # history = []

    iter_per_redraw = 100
    start = time.time()
    for iter in count():
        grapher.step()
        # history.append(grapher.charge.copy())
        if iter % iter_per_redraw == 0:
            grapher.redraw()
            grapher.app.processEvents()
            end = time.time()
            try:
                print(f'{iter_per_redraw / (end - start)} iter/s')
                start = end
            except Exception as e:
                print(e)
        pycuda.driver.stop_profiler()
        # pycuda.autoinit.context.detach()

    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     QtGui.QApplication.instance().exec_()

