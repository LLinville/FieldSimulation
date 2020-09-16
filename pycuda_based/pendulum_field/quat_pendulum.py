import numpy as np
from itertools import count
from matplotlib import pyplot as plt

real = np.array([1,0,0,0])
i = np.array([0,1,0,0])
j = np.array([0,0,1,0])
k = np.array([0,0,0,1])
ijk = i+j+k

def quat_multiply(quaternion0, quaternion1):
    x0, y0, z0, w0 = np.split(quaternion0, 4, axis=-1)
    x1, y1, z1, w1 = np.split(quaternion1, 4, axis=-1)
    # return np.concatenate(
    #     (x1*w0 + y1*z0 - z1*y0 + w1*x0,
    #      -x1*z0 + y1*w0 + z1*x0 + w1*y0,
    #      x1*y0 - y1*x0 + z1*w0 + w1*z0,
    #      -x1*x0 - y1*y0 - z1*z0 + w1*w0),
    #     axis=-1)
    return np.concatenate((
        x0*x1 - y0*y1 - z0*z1 - w0*w1,
        x0*y1 + x1*y0 + z0*w1 - z1*w0,
        x0*z1 + x1*z0 + y1*w0 - y0*w1,
        x0*w1 + x1*w0 + y0*z1 - y1*z0),
        axis=-1
    )

def quat_rotate(point, axis, mag):
    quat = np.cos(mag) * real + np.sin(mag) * axis
    quat_inv = quat * (real-ijk)
    return quat_multiply(quat, quat_multiply(point, quat_inv))



dt = 0.1


pos = np.array([1,0,0])
vel = np.array([0,0.01,-0])
rotation_axis = np.zeros_like(pos)

p_hist = np.array([pos])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plt.show()

for i in count():
    rotation_axis = np.cross(pos,vel)
    rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)
    rotation_mag = np.linalg.norm(vel)/2.0 * dt

    pos_quat = np.zeros(4)
    axis_quat = np.zeros(4)
    vel_quat = np.zeros(4)

    pos_quat[1:] = pos
    axis_quat[1:] = rotation_axis
    vel_quat[1:] = vel

    pos = quat_rotate(pos_quat, axis_quat, rotation_mag)[1:]
    vel = quat_rotate(vel_quat, axis_quat, rotation_mag)[1:]
    vel[2] -= 0.01 * np.cos(pos[2]*np.pi/2)


    print(pos)
    p_hist = np.concatenate([p_hist, [pos]], axis=0)

    if i%10==0:
        ax.cla()
        ax.plot3D(p_hist[:,0], p_hist[:,1],p_hist[:,2])

        plt.pause(0.001)

