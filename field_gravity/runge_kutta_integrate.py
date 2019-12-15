import numpy as np

def neighbor_diff(field):
    return (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)) / 4 - field

def runge_kutta_force(field, dt = 0.01):
    k1 = neighbor_diff(field) * dt
    k2 = neighbor_diff(field + k1 / 2) * dt
    k3 = neighbor_diff(field + k2 / 2) * dt
    k4 = neighbor_diff(field + k3) * dt
    return (k1 + 2*k2 + 2*k3 + k4) / 6

def simple_force(field, dt=0.01):
    return neighbor_diff(field) * dt