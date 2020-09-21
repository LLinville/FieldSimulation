import numpy as np
from math import pi
from matplotlib import pyplot as plt

def energy_kp(field, potential):
    neighbor_average = (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1))/4
    e_pot = np.sum(potential * field * np.conj(field))
    e_kinetic = -2 * np.sum(field*neighbor_avg - field * np.conj(field))

    return e_kinetic, e_pot


def fundamental(nx, ny, width):
    x,y = np.mgrid[0:pi:1j*width,0:pi:1j*width]
    return np.sin(nx * x) * np.sin(ny * y) + (np.random.rand(width,width) - 0.5)/10


width = 150
width2 = width * width
n_eigen = 4
# eigenstates = np.array([
#     fundamental(1,1,width),
#     fundamental(1,2,width),
#     fundamental(2,1,width),
#     fundamental(2,2,width)
# ])
eigenstates = np.array([1*fundamental(x,y,width) for x in range(1,4) for y in range(1,4)])
print(eigenstates)
x,y = np.mgrid[-1:1:1j*width, -1:1:1j*width]
potential = x*x/2+2*y*y

relax = True

for eig_index in range(len(eigenstates)):
    eigenstate = eigenstates[eig_index]
    prev_energy = 0
    #field_mag = np.sum(np.abs(eigenstate))
    field_mag2 = eigenstate * eigenstate
    neighbor_avg = (np.roll(eigenstate, 1, axis=0) + np.roll(eigenstate, -1, axis=0) + np.roll(eigenstate, 1,axis=1) + np.roll(eigenstate, -1, axis=1)) / 4
    e_kinetic, e_pot = energy_kp(eigenstate, potential)
    e_total = e_pot+e_kinetic

    for i in range(20000):
        # Relax
        eigenstate[0,:] = 0
        eigenstate[-1,:] = 0
        eigenstate[:,0] = 0
        eigenstate[:,-1] = 0
        neighbor_avg = (np.roll(eigenstate, 1, axis=0) + np.roll(eigenstate, -1, axis=0) + np.roll(eigenstate, 1, axis=1) + np.roll(eigenstate, -1, axis=1))/4



        print(f'Energy: {e_kinetic + e_pot} = kinetic {e_kinetic} + potential {e_pot}')


        if relax:
            e_kinetic, e_pot = energy_kp(eigenstate, potential)
            e_total = e_pot + e_kinetic
            field_mag2 = np.sum(eigenstate*eigenstate)
            new_eigenstate = neighbor_avg / (1 - 0.5*((e_kinetic + e_pot)/(field_mag2) - potential))

            #mag2_diff = new_eigenstate*new_eigenstate - eigenstate*eigenstate
            eigenstate += 1.0*(new_eigenstate-eigenstate)
            #field_mag2 += mag2_diff
            # eigenstate = new_eigenstate


        if abs(prev_energy - e_total) < e_total * 1e-4:
            pass
            break
        else:
            prev_energy = e_total

        if i%1 == 10:
            # eigenstates[eig_index] = eigenstate / np.sum(np.abs(eigenstate))
            print(f'correcting by {np.sum(eigenstate * eigenstate)}')
            eigenstate /= np.sum(eigenstate * eigenstate)



        for orthogonal_index, orthogonal_state in enumerate(eigenstates):
            if orthogonal_index < eig_index:
                orthogonal_component = orthogonal_state * eigenstate / width2
                #print(f'orthogonal component: {np.sum(orthogonal_component)}')
                eigenstate -= np.sum(orthogonal_component) * orthogonal_state * 1


        if i%10 == 0:
            plt.imshow(eigenstate)
            plt.pause(0.001)
    eigenstates[eig_index] = eigenstate



    print(f'Final Energy: {np.sum(e_total)} = kinetic {np.sum(e_kinetic)} + potential {np.sum(e_pot)}')
plt.show()




