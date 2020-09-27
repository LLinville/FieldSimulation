import numpy as np
from matplotlib import pyplot as plt
from numba import cuda
from util import fast_matmul
from itertools import count

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.cublas as cublas
import skcuda.linalg as linalg
import skcuda
skcuda.misc.init()

position = np.array([[0,0],[1,1]]).astype(np.float64)
charge = np.array([-1,1]).astype(np.float64)
eneg = np.array([1,2]).astype(np.float64)
idempot = np.array([2,2]).astype(np.float64)

dist = position - position[:, None, :]
dist2 = np.sum(dist * dist, axis=2)
dist_mag = np.sqrt(dist2)
dist4 = dist2 * dist2

for iter in count():
    # atomic_coulomb_energy = charge * charge[None,:].T / np.sqrt(dist2 + 0.51)
    atomic_coulomb_grad = -1 * dist_mag / (np.power(dist2 + 1, 1.5))

    charge_grad = eneg + idempot*charge + np.sum(atomic_coulomb_grad, axis=1)
    charge_transfer = charge_grad - charge_grad[None,:].T



    charge += np.sum(charge_transfer, axis=1) * 0.01

    print(charge)