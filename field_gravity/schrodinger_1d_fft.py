import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
import pickle
from coloring import colorize
from numpy import exp, dot, mean, fft, cos,sin,cosh,sqrt,absolute, array, transpose, var, conj

import math
from math import pi
import time

from numpy.fft import fft, ifft, fft2, ifft2, fftshift

L = 700
dt = 10
naxis = 500
k0 = 0.01
M0 =1

# Define position vector
xvector = np.linspace(-L,L,naxis,retstep=True)
dx = xvector[1]
xvector = xvector[0]


nt = 500000
wall_thickness = 100
# psi0 = M0*exp(1j*k0*(xvector)*0*1 - xvector*xvector/100000)
psi0 = M0*np.sin(xvector/L/2 * (naxis/2) / (naxis/2 - wall_thickness-1) * pi + pi/2)
psi0[:wall_thickness+1] = 0
psi0[naxis-wall_thickness-1:] = 0
# psi0 = np.roll(psi0, 200)
# psi0 = np.zeros_like(xvector)
# psi0[220:820] = 1
potential = np.linspace(-1.0, 1.0, naxis)
# potential = potential*potential*potential*potential - potential*potential + 1
potential = potential*potential
# potential *= 0.01

changing_potential = np.linspace(-1, 1, naxis)
drive_frequency = 0.037 * 1
potential = np.zeros(naxis) * 0
# potential[4000:4010] = 0.0001
# potential[100:900] = 0

potential[0:wall_thickness] = 100
potential[naxis-wall_thickness:] = 100


klist = np.array([y for y in range(0,naxis//2)] + [0] + [a for a in range(-naxis//2+1,0)])
mu_k = pi * (naxis-1)*klist/(L*naxis)
expmu = exp(-1j*(mu_k**2)*(dt)/2/0.1)
m2 = np.linspace(-1, 1, naxis) ** 2

mode = "single"

total_momentum_sums = {}

lower_bound, upper_bound = 0.00020, 0.00050
lower_bound_ratio, upper_bound_ratio = 0.4, 0.7
lower_bound, upper_bound = (upper_bound - lower_bound) * lower_bound_ratio + lower_bound, (upper_bound - lower_bound) * upper_bound_ratio + lower_bound
# for d_i, drive_frequency in enumerate(np.linspace(lower_bound*dt, upper_bound*dt, 1000)):
# for d_i, drive_frequency in enumerate(np.linspace(0.000711*dt, 0.001273*dt, 10)):
# for d_i, drive_frequency in [(1, 0.0033693693693693694)]:
for d_i, drive_frequency in enumerate(np.linspace(0.003372, 0.003378, 10)):
    momentum_sum_list = []
    psit_list = []
    psit = psi0
    psit_list.append(psit)

    drive_list = []
    momentum_list = []
    print(f'{d_i}: {drive_frequency}')
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
        composite_potential = potential + 1*0.05*drive_potential
        composite_potential *= 0.02
        psit = exp(-1j*composite_potential*dt) * psit

        if i % 10000 == 0:
            print(i)

        if i % 5 == 0:
            psit_list.append(psit)
            drive_list.append(composite_potential)
            momentum_list.append(fftshift(psitft))
            momentum_sum_list.append(np.sum(m2*fftshift(psitft)))

    tf = time.time()
    # psit_list_abs = map(absolute,psit_list)
    # psit_list_phase = map(np.angle,psit_list)
    # print(["Time of computation: ",tf-ti," seconds"])

    # fig = plt.figure(figsize=(9,8))
    if mode == 'single':
        # ax = plt.subplot(1, 3, 1)
        # gpeplot = plt.imshow((np.absolute(psit_list[:10000])),origin='lower',aspect='auto')
        plt.imshow((np.absolute(psit_list[:100000])), origin='lower', aspect='auto')
        # plt.xlabel("Position",fontsize=20)
        # plt.ylabel("Time",fontsize=20)
        # plt.colorbar()
        # plt.suptitle('Wavefunction amplitude $|\psi|$',fontsize=20)

        # plt.subplot(1, 3, 2)
        # drive_list = np.minimum(0.0001, drive_list)
        # plt.imshow(drive_list[:100000],origin='lower',aspect='auto')
        # plt.plot(np.convolve(np.absolute(momentum_sum_list), gaussian(5311, 560)))
        # plt.imshow(np.abs(momentum_list), origin='lower', aspect='auto')
        # plt.subplot(1,3,3)
        # plt.imshow(np.abs(np.fft.fftshift(np.fft.fft(np.array(psit_list)))),origin='lower',aspect='auto')


    else:
        momentum_list = np.minimum(1, momentum_list)
        for m_stretch in range(1):
            total_momentum_sums[drive_frequency] = np.absolute(momentum_sum_list)


        if d_i % 15 == 0:
            with open(f"../output/out4.p", 'wb+') as pickle_file:
                pickle.dump(total_momentum_sums, pickle_file)

        # plt.imshow(np.array(total_momentum_sums))
        # plt.savefig(f"../output/fig_out_{d_i}.png")
    plt.pause(0.001)
