import numpy as np
from matplotlib import pyplot as plt

# import cupy

def pot_f_mag(x1, x2):
    a,b = 1,1
    c,d = 12,6

    dx = x2-x1
    dist = dx
    dist2 = dx*dx
    dist4 = dist2*dist2
    dist8 = dist4*dist4


    #a(x^-b)-c(x^-d)
    pot = a/(dist8*dist4) - c/(dist4*dist2)
    f = c*d/(dist8/dist) - a*b/(dist8*dist4*dist)
    return np.clip(pot,-5,5), np.clip(f,-5,5)


nprot1 = 1
nprot2 = 1
p1 = {"nprot":nprot1, "nelec":nprot1, "xpos": 0}
p2 = {"nprot":nprot2, "nelec":nprot2, "xpos": 3}
particles = [p1, p2]

x = np.linspace(-10, 10, 1000)

pot, f = pot_f_mag(np.array(p1["xpos"]),x)
plt.plot(pot)
plt.show()