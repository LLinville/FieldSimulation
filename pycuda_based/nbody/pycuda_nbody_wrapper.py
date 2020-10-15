import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from PyQt5.QtCore import *
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg

from pycuda.elementwise import ElementwiseKernel

if __name__ == "__main__":
    n = np.int(2)
    BLOCK_SIZE = 256
    nBlocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    pos = np.array([
        [0.5,0.5],
        [1.0,1.0]
    ], dtype=np.float32)

    # pos = np.random.random((n*2)) * 100
    # pos = np.array(pos, dtype=np.float32)
    # pos = np.zeros((n), dtype=np.float32)
    vel = np.zeros_like(pos, dtype=np.float32)
    # vel = np.random.random((n*2)) * 100
    # vel = np.array(vel, dtype=np.float32)
    debug = np.zeros_like(pos, dtype=np.float32)
    dt = np.float32(10)

    pos_gpu = cuda.mem_alloc(pos.nbytes)
    vel_gpu = cuda.mem_alloc(vel.nbytes)
    debug_gpu = cuda.mem_alloc(debug.nbytes)
    dt_gpu = cuda.mem_alloc(dt.nbytes)
    n_gpu = cuda.mem_alloc(len(bytes(n)))

    cuda.memcpy_htod(pos_gpu, pos)
    cuda.memcpy_htod(vel_gpu, vel)
    cuda.memcpy_htod(dt_gpu, dt)
    cuda.memcpy_htod(n_gpu, bytes(n))
    cuda.memcpy_htod(debug_gpu, debug)

    with open("nbody.c", "r") as kernel_file:
        source_module = SourceModule(kernel_file.read())

    update_vel = source_module.get_function("applyForce")

    app = QtGui.QApplication([])

    ## Create window with GraphicsView widget
    win = pg.GraphicsLayoutWidget()
    win.show()  ## show widget alone in its own window
    win.setWindowTitle('pyqtgraph example: ImageItem')
    view = win.addPlot()
    plot = view.plot()
    view.setXRange(0, 100, padding=0)
    view.setYRange(0, 100, padding=0)

    for iter in range(100):
        for i in range(1):
            # print(i)
            update_vel(pos_gpu, vel_gpu, debug_gpu, dt_gpu, n_gpu, block=(nBlocks,BLOCK_SIZE,1), grid=(1,1,1))

        cuda.memcpy_dtoh(pos, pos_gpu)
        cuda.memcpy_dtoh(vel, vel_gpu)
        cuda.memcpy_dtoh(debug, debug_gpu)
        print(np.max(vel))
        plot.setData(pos[::2], pos[1::2],pen=None, symbol='o')
        # img.setImage(vel.reshape((32,32)))

        app.processEvents()

    out = vel
    # out = out.reshape((n))
    out = out


