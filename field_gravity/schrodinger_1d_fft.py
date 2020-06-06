import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from util import first_unoccupied
import pickle
from coloring import colorize
from numpy import exp, dot, mean, fft, cos,sin,cosh,sqrt,absolute, array, transpose, var, conj

import math
from math import pi
import time

from numpy.fft import fft, ifft, fft2, ifft2, fftshift

L = 700
dt = 5
naxis = 10000
k0 = 0.1
M0 =1

# Define position vector
xvector = np.linspace(-L,L,naxis,retstep=True)
dx = xvector[1]
xvector = xvector[0]


nt = 100000
wall_thickness = 1000
psi0 = np.roll(M0*exp(1j*k0*(xvector)*0.00*1 - xvector*xvector/10000), 1000)
psi0 += np.roll(M0*exp(-1j*k0*(xvector)*0.0*1 - xvector*xvector/10000), -1000)
# psi0 = M0*np.sin(xvector/L/2 * (naxis/2) / (naxis/2 - wall_thickness-1) * pi + pi/2)
psi0[:wall_thickness+1] = 0
psi0[naxis-wall_thickness-1:] = 0
# psi0 = np.zeros_like(xvector)
# psi0 = np.roll(psi0, 200)
# psi0[220:820] = 1
psi0 /= np.sum(np.abs(psi0))
potential = np.linspace(-1.0, 1.0, naxis)
# potential = potential*potential*potential*potential - potential*potential + 1
potential = potential*potential
potential = potential*potential
potential = potential*potential
# potential *= 0.01

changing_potential = np.linspace(-1, 1, naxis)
drive_frequency = 0.037 * 1
# potential = np.zeros(naxis, dtype=complex) * 0
# potential[4000:4010] = 0.0001
# potential[100:900] = 0

# potential[0:wall_thickness] = 100
# potential[naxis-wall_thickness:] = 100


klist = np.array([y for y in range(0,naxis//2)] + [0] + [a for a in range(-naxis//2+1,0)])
mu_k = pi * (naxis-1)*klist/(L*naxis)
expmu = exp(-1j*(mu_k**2)*(dt)/2/0.1)
m2 = np.linspace(-1, 1, naxis) ** 2

mode = "multiple"
# mode = "single"
save_path = first_unoccupied("../output/out%s.p")

total_momentum_sums = {}
momentum_save_frequency = 2

lower_bound, upper_bound = 0.00020, 0.00050
lower_bound_ratio, upper_bound_ratio = 0.4, 0.7
lower_bound, upper_bound = (upper_bound - lower_bound) * lower_bound_ratio + lower_bound, (upper_bound - lower_bound) * upper_bound_ratio + lower_bound
# for d_i, drive_frequency in enumerate(np.linspace(lower_bound*dt, upper_bound*dt, 1000)):
# for d_i, drive_frequency in enumerate(np.linspace(0.000711*dt, 0.001273*dt, 10)):
# for d_i, drive_frequency in [(1, 0.0033693693693693694)]:
# for d_i, drive_frequency in enumerate(np.linspace(0.003370, 0.003376, 100)):
momentum_sum_list = []
psit_list = []
psit = psi0
psit_list.append(psit)

drive_list = []
momentum_list = []
# Computation
ti = time.time()
for i in range(1,nt):
    drive_phase = math.sin(i * drive_frequency) * 0.2
    # drive_potential = - 2*drive_phase*changing_potential + drive_phase**2
    drive_potential = changing_potential * drive_phase
    # drive_potential *= 1 if i < 70000 else 0
    psitft = fft(psit)
    psitft = expmu*psitft
    psit = ifft(psitft)
    composite_potential = potential + 0*0.09*drive_potential - np.abs(psit)*1.0*1
    composite_potential *= 0.2
    # composite_potential *=
    # composite_potential += 0.002j
    psit = exp(-1j*composite_potential*dt) * psit

    if i % 100 == 0:
        total_mag = np.sum(psit.real * psit.real + psit.imag * psit.imag)
        psit /= sqrt(total_mag)
        print(f'normalizing at step {i} by factor of {total_mag}')


    if i % momentum_save_frequency == 0:
        psit_list.append(psit)
        # drive_list.append(composite_potential)
        # momentum_list.append(fftshift(psitft))
        # momentum_sum_list.append(np.sum(m2*fftshift(psitft)))

    tf = time.time()
    # psit_list_abs = map(absolute,psit_list)
    # psit_list_phase = map(np.angle,psit_list)
    # print(["Time of computation: ",tf-ti," seconds"])

    # fig = plt.figure(figsize=(9,8))
    if i % 30 == 0:
        # ax = plt.subplot(1, 2, 1)
        plt.imshow((np.absolute(psit_list)), origin='lower', aspect='auto')
    # plt.xlabel("Position",fontsize=20)
    # plt.ylabel("Time",fontsize=20)
    # plt.colorbar()
    # plt.suptitle('Wavefunction amplitude $|\psi|$',fontsize=20)

    # plt.subplot(1, 2, 2)
    # drive_list = np.minimum(0.0001, drive_list)
    # plt.imshow(drive_list[:100000],origin='lower',aspect='auto')
    # plt.plot(np.convolve(np.absolute(momentum_sum_list), gaussian((nt/momentum_save_frequency)//8, (nt/momentum_save_frequency)//32)))
    # plt.plot(np.absolute(momentum_sum_list))
    # plt.imshow(np.abs(momentum_list), origin='lower', aspect='auto')
    # plt.subplot(1,3,3)
    # plt.imshow(np.abs(np.fft.fftshift(np.fft.fft(np.array(psit_list)))),origin='lower',aspect='auto')


    # else:
    #     momentum_list = np.minimum(1, momentum_list)
    #     for m_stretch in range(1):
    #         # total_momentum_sums[drive_frequency] = np.convolve(np.absolute(momentum_sum_list), gaussian((nt/momentum_save_frequency)//8, (nt/momentum_save_frequency)//32))
    #         total_momentum_sums[drive_frequency] = np.absolute(momentum_sum_list)
    #
    #     # plt.imshow(np.array(total_momentum_sums))
    #     # plt.savefig(f"../output/fig_out_{d_i}.png")
    plt.pause(0.001)
