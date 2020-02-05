import pickle
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
import numpy as np

with open("../output/out3.p", 'rb') as pickle_file:
    out = pickle.load(pickle_file)

rows = np.array([np.convolve(val, gaussian(2311, 260))[2000:20000] for key, val in out.items()])
rows = np.repeat(rows, 10, axis=0)
plt.imshow(rows)
plt.pause(0.001)
while True:
    plt.pause(0.001)
