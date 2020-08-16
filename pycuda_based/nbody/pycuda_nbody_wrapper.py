import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg

from pycuda.elementwise import ElementwiseKernel



if __name__ == "__main__":
    n = np.int(1024)
    BLOCK_SIZE = 256
    nBlocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    pos = np.random.random((n*2))
    pos = np.array(pos, dtype=np.float32)
    # pos = np.zeros((n), dtype=np.float32)
    vel = np.zeros_like(pos, dtype=np.float32)
    dt = np.float32(0.001)

    pos_gpu = cuda.mem_alloc(pos.nbytes)
    vel_gpu = cuda.mem_alloc(vel.nbytes)
    # out_gpu = cuda.mem_alloc(out.nbytes)
    dt_gpu = cuda.mem_alloc(dt.nbytes)
    n_gpu = cuda.mem_alloc(len(bytes(n)))

    cuda.memcpy_htod(pos_gpu, pos)
    cuda.memcpy_htod(vel_gpu, vel)
    cuda.memcpy_htod(dt_gpu, dt)
    cuda.memcpy_htod(n_gpu, bytes(n))
    # cuda.memcpy_htod(out_gpu, out)

    with open("nbody.cu", "r") as kernel_file:
        source_module = SourceModule(kernel_file.read())

    update_vel = source_module.get_function("applyForce")

    app = QtGui.QApplication([])

    ## Create window with GraphicsView widget
    win = pg.GraphicsLayoutWidget()
    win.show()  ## show widget alone in its own window
    win.setWindowTitle('pyqtgraph example: ImageItem')
    view = win.addPlot()
    plot = view.plot()

    while True:
        for i in range(1):
            print(i)
            update_vel(pos_gpu, vel_gpu, dt_gpu, n_gpu, block=(nBlocks,BLOCK_SIZE,1), grid=(1,1,1))

        cuda.memcpy_dtoh(pos, pos_gpu)
        cuda.memcpy_dtoh(vel, vel_gpu)
        print(np.max(vel))
        plot.setData(pos[::2], pos[1::2],pen=None, symbol='o')
        # img.setImage(vel.reshape((32,32)))

        app.processEvents()

    out = vel
    # out = out.reshape((n))
    out = out


