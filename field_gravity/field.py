import numpy as np
import matplotlib.pyplot as plt
from runge_kutta_integrate import runge_kutta_force as force
# from runge_kutta_integrate import simple_force as force
from coloring import colorize
import cplot

def add_mass(current, cx, cy, width=15, amount=1):
    for dx in range(width//-2, width//2):
        for dy in range(width//-2, width//2):
            current[cx+dx, cy+dy] += amount / max(1, (10*dx/width)**2+(10*dy/width)**2)

def mass_to_gain(mass, pull):
    # return (
    #                (np.roll(pull, 1, axis=1)+pull)/2).real - ((np.roll(pull, -1, axis=1)+pull)/2).real +\
    #        (
    #                (np.roll(pull, -1, axis=0)+pull)/2).imag - ((np.roll(pull, 1, axis=0)+pull)/2).imag
    from_left = ((np.roll(pull, 1, axis=1) + pull) / 2).real * ((np.roll(mass, 1, axis=1) + mass) / 2).real
    to_right = ((np.roll(pull, -1, axis=1) + pull) / 2).real * ((np.roll(mass, -1, axis=1) + mass) / 2).real
    from_down = ((np.roll(pull, 1, axis=0) + pull) / 2).imag * ((np.roll(mass, 1, axis=0) + mass) / 2).real
    to_up = ((np.roll(pull, -1, axis=0) + pull) / 2).imag * ((np.roll(mass, -1, axis=0) + mass) / 2).real
    return from_left - to_right + from_down - to_up


        # (
        #            (np.roll(pull, 1, axis=1) + pull) / 2).real - ((np.roll(pull, -1, axis=1) + pull) / 2).real + \
        #    (
        #            (np.roll(pull, 1, axis=0) + pull) / 2).imag - ((np.roll(pull, -1, axis=0) + pull) / 2).imag

SIZE = 70

# field g
g = np.zeros((SIZE, SIZE), dtype=complex)

# strength of gravity
G = 0.1

# diffusion speed of mass
MASS_SPREAD_RATE = 11

# rate of change of g
dg_dt = np.zeros_like(g)

mass = np.zeros_like(g)
# dg_dt[20:22,20:22] = 1

# mass[20:22,20:22] = 1
# mass[30:32,30:32] = 1


add_mass(mass, 30,30, width=30, amount=1)
add_mass(mass, 40,40, width=20, amount=5.5)
# mass[20:35, 20:35] = 1
# mass[35, 30] = 1


# timestep
dt = 0.1

total_iterations = 1000000
substeps = 10
for i in range(total_iterations):
    print(f'Iteration: {i}')

    # if i % 20 < 10:
    #     mass = np.roll(mass, 1, axis=0)
    # else:
    #     mass = np.roll(mass, -1, axis=0)
    print(np.sum(mass))
    # print(np.sum(mass_to_lose(mass, g)))
    for substep in range(substeps):
        # dg_dt += force(g,dt=dt)
        mass_grad = (np.roll(mass, -1, axis=1) - np.roll(mass, 1, axis=1)) / 2 + 1j * (np.roll(mass, -1, axis=0) - np.roll(mass, 1, axis=0)) / 2
        g += mass_grad * dt
        # g += force(g, dt=dt)
        mass += force(mass, dt=dt) * MASS_SPREAD_RATE
        mass += mass_to_gain(mass, g) * dt * G


        # g += dg_dt * dt + mass * dt
        # dg_dt += mass * dt
        g *= 0.999
        # g += force(g,dt=dt)


    # plt.imshow(colorize(g))
    plt.subplot(2, 2, 1)
    plt.imshow(mass.real)
    plt.subplot(2, 2, 2)
    plt.imshow(colorize(g.transpose()))
    plt.subplot(2, 2, 3)
    plt.imshow((mass_to_gain(mass, g) * dt * G).real)
    plt.subplot(2, 2, 4)
    plt.imshow((force(mass, dt=dt) * MASS_SPREAD_RATE).real + (mass_to_gain(mass, g) * dt * G).real)
    plt.draw_all()
    plt.pause(0.001)
    plt.clf()
