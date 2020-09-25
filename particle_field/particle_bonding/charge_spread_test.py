import numpy as np
from matplotlib import pyplot as plt
from numba import cuda
from util import fast_matmul
from itertools import count

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.cublas as cublas
import skcuda.linalg as linalg
import skcuda
skcuda.misc.init()

coupling = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
]).astype(np.float64)
coupling = np.random.choice([0,1],(100,100))
coupling_img = plt.imshow(coupling)
coupling_gpu = gpuarray.to_gpu(coupling)
tmp_gpu = gpuarray.to_gpu(np.zeros_like(coupling))

n_part = len(coupling)

spread_speed = 0.0001

base_spread = (np.ones_like(coupling) * spread_speed) / n_part + np.diag(np.ones(n_part) * (1 - spread_speed))
base_spread_gpu = gpuarray.to_gpu(base_spread)

app = QtGui.QApplication([])

## Create window with GraphicsView widget
win = pg.GraphicsLayoutWidget()
win.show()  ## show widget alone in its own window
win.setWindowTitle('pyqtgraph example: ImageItem')
view = win.addViewBox()
view.setAspectLocked(True)
g1 = pg.ImageItem()
view.addItem(g1)

for iter in count():

    # coupling_gpu = linalg.dot(coupling_gpu, base_spread_gpu)
    # coupling_gpu = tmp_gpu
    coupling = coupling * (1-spread_speed) + np.sum(coupling, axis=1) * spread_speed / n_part / 2 + np.sum(coupling, axis=1)[None,:].T * spread_speed / n_part / 2

    if iter % 10000 == 0:
        print(iter)
        # coupling = coupling_gpu.get()
        # print(coupling)
        g1.setImage(coupling)
        g1.setLevels([0,1])
        app.processEvents()