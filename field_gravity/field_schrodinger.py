import numpy as np
import matplotlib.pyplot as plt
from runge_kutta_integrate import runge_kutta_force as force
# from runge_kutta_integrate import simple_force as force
from coloring import colorize
from util import add_packet
import math

def add_field(current, cx, cy, width=15, amount=1):
    for dx in range(width//-2, width//2):
        for dy in range(width//-2, width//2):
            current[cx+dx, cy+dy] += amount / max(1, (10*dx/width)**2+(10*dy/width)**2) / width**2

def field_to_gain(field, pull):
    # return (
    #                (np.roll(pull, 1, axis=1)+pull)/2).real - ((np.roll(pull, -1, axis=1)+pull)/2).real +\
    #        (
    #                (np.roll(pull, -1, axis=0)+pull)/2).imag - ((np.roll(pull, 1, axis=0)+pull)/2).imag
    avg_field_left = ((np.roll(field, 1, axis=1) + field) / 2).real
    avg_field_right = ((np.roll(field, -1, axis=1) + field) / 2).real
    avg_field_up = ((np.roll(field, 1, axis=0) + field) / 2).real
    avg_field_down = ((np.roll(field, -1, axis=0) + field) / 2).real

    from_left = ((np.roll(pull, 1, axis=1) + pull) / 2).real * np.maximum(0, avg_field_left)
    to_right = ((np.roll(pull, -1, axis=1) + pull) / 2).real * np.maximum(0, avg_field_right)
    from_down = ((np.roll(pull, 1, axis=0) + pull) / 2).imag * np.maximum(0, avg_field_up)
    to_up = ((np.roll(pull, -1, axis=0) + pull) / 2).imag * np.maximum(0, avg_field_down)
    return from_left - to_right + from_down - to_up


        # (
        #            (np.roll(pull, 1, axis=1) + pull) / 2).real - ((np.roll(pull, -1, axis=1) + pull) / 2).real + \
        #    (
        #            (np.roll(pull, 1, axis=0) + pull) / 2).imag - ((np.roll(pull, -1, axis=0) + pull) / 2).imag

SIZE = 70

# field g
g = np.zeros((SIZE, SIZE), dtype=complex)

pot = np.zeros((SIZE, SIZE), dtype=complex)

# strength of gravity
G = 10.5

# diffusion speed of field
field_SPREAD_RATE = 1.1

# rate of change of g
dg_dt = np.zeros_like(g)

field = np.zeros_like(g)
# dg_dt[20:22,20:22] = 1

# field[20:22,20:22] = 1
# field[30:32,30:32] = 1


# add_field(field, 30,30, width=40, amount=0.05)
# add_field(field, 40,40, width=40, amount=0.05)
add_packet(field, 20, 20, width=10, momentum=10)

# field[20:35, 20:35] = 1
# field[35, 30] = 1


# timestep
dt = 0.5

total_iterations = 1000000
substeps = 30
for i in range(total_iterations):
    print(f'Iteration: {i}')

    # if i % 40 < 10:
    #     field = np.roll(field, 1, axis=0)
    # elif i%40 < 20:
    #     field = np.roll(field, -1, axis=1)
    # elif i % 40 < 30:
    #     field = np.roll(field, -1, axis=0)
    # else:
    #     field = np.roll(field, 1, axis=1)

    # if i%40 < 20:
    #     field = np.roll(field, 2, axis=0)
    # else:
    #     field = np.roll(field, -2, axis=0)

    # field[30, 30] = math.sin(i / 10 * math.pi)
    # field[40, 30] = math.sin(i / -10 * math.pi)
    # field[30,30] = 10
    # field[40, 40] = 30
    print(np.sum(field))
    # print(np.sum(field_to_lose(field, g)))
    for substep in range(substeps):
        # dg_dt += force(g,dt=dt)
        # field_grad = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / 2 + 1j * (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / 2
        
        # for g_spread_iter in range(10):
        #     g += field_grad * dt
        #     g += force(g, dt=dt)
        #     g *= 0.999
        # dg_dt += field_grad * dt
        # g += dg_dt * dt
        # dg_dt += force(dg_dt, dt=dt*1)

        field += force(field, dt=dt) * field
        # field += field_to_gain(field, g) * dt * G


        # g += dg_dt * dt + field * dt
        # dg_dt += field * dt
        # dg_dt *= 0.999

        # g += force(g,dt=dt)


    # plt.imshow(colorize(g))
    plt.subplot(2, 2, 1)
    plt.imshow(colorize(field.transpose()))
    plt.subplot(2, 2, 2)
    plt.imshow(colorize(g.transpose()))
    plt.subplot(2, 2, 3)
    plt.imshow((field_to_gain(field, g) * dt * G).real)
    # plt.imshow(colorize(dg_dt.transpose()))
    plt.subplot(2, 2, 4)
    plt.imshow((force(field, dt=dt) * field_SPREAD_RATE * field_SPREAD_RATE * field * field).real)# + (field_to_gain(field, g) * dt * G).real)
    plt.draw_all()
    plt.pause(0.001)
    plt.clf()
