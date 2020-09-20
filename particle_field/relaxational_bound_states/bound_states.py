import numpy as np
from math import pi
from matplotlib import pyplot as plt

def energy(field, potential):
    neighbor_average = (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1))/4
    return -1/2


def fundamental(nx, ny, width):
    x,y = np.mgrid[0:pi:1j*width,0:pi:1j*width]
    return np.sin(nx * x) * np.sin(ny * y) + (np.random.rand(width,width) - 0.5)/10


width = 50
width2 = width * width
n_eigen = 4
eigenstates = np.array([
    fundamental(1,1,width),
    fundamental(1,2,width),
    fundamental(2,1,width),
    fundamental(2,2,width)
])
print(eigenstates)
x,y = np.mgrid[-1:1:1j*width, -1:1:1j*width]
potential = x*x+2*y*y

relax = True

for eig_index in range(len(eigenstates)):
    eigenstate = eigenstates[eig_index]
    prev_energy = 0
    field_mag = np.sum(np.abs(eigenstate))
    field_mag2 = field_mag * field_mag
    neighbor_avg = (np.roll(eigenstate, 1, axis=0) + np.roll(eigenstate, -1, axis=0) + np.roll(eigenstate, 1,axis=1) + np.roll(eigenstate, -1, axis=1)) / 4
    e_kinetic = -1/2 * eigenstate * (neighbor_avg - 4*eigenstate)
    e_pot = (potential * eigenstate * eigenstate)
    e_total = e_kinetic + e_pot
    for i in range(20000):
        # Relax
        eigenstate[0,:] = 0
        eigenstate[-1,:] = 0
        eigenstate[:,0] = 0
        eigenstate[:,-1] = 0
        neighbor_avg = (np.roll(eigenstate, 1, axis=0) + np.roll(eigenstate, -1, axis=0) + np.roll(eigenstate, 1, axis=1) + np.roll(eigenstate, -1, axis=1))/4


        e_total_sum = np.sum(e_total)
        print(f'Energy: {e_total_sum} = kinetic {np.sum(e_kinetic)} + potential {np.sum(e_pot)}')


        if relax:
            new_eigenstate = neighbor_avg / (1 - 0.5*(e_total/field_mag2 - potential)/width2)
            mag2_diff = new_eigenstate*new_eigenstate - eigenstate*eigenstate
            eigenstate += 1.0*(new_eigenstate-eigenstate)
            field_mag2 += mag2_diff/width2
            e_total += (2 + potential/(width2)) * mag2_diff - 4*neighbor_avg*(new_eigenstate-eigenstate)

        if abs(prev_energy - e_total) < 0.0000001:
            pass
            #break
        else:
            prev_energy = e_total

        if i%1 == 0:
            eigenstate /= np.sum(field_mag)



        for orthogonal_index, orthogonal_state in enumerate(eigenstates):
            if orthogonal_index < eig_index:
                orthogonal_component = orthogonal_state * eigenstate
                eigenstate -= np.sum(orthogonal_component) * orthogonal_state * 0.0


        if i%10 == 0:
            plt.imshow(eigenstate)
            plt.pause(0.001)
    eigenstates[eig_index] = eigenstate / np.sum(np.abs(eigenstate))


    print(f'Final Energy: {np.sum(e_total/width2)} = kinetic {np.sum(e_kinetic)} + potential {np.sum(e_pot)}')
plt.show()




