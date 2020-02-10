import pickle
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
import numpy as np

with open("../output/out8.p", 'rb') as pickle_file:
    out = pickle.load(pickle_file)

length = len([item for key, item in out.items()][0])
kernel = gaussian(length//4, length//16)
rows = np.array([np.convolve(val, kernel)[length//8:length - length//8] for key, val in out.items()])
# rows = np.array([val for key, val in out.items()])
rows = np.repeat(rows, 100, axis=0)
plt.imshow(rows)
plt.pause(0.001)
while True:
    plt.pause(0.001)
