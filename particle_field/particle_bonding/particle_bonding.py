import numpy as np
from scipy.spatial.distance import pdist, squareform


pos = np.array([[0,0],[1,1],[-1,1]])
n_part = len(pos)
eneg = np.ones(n_part)
vel = np.zeros_like(pos)

elec_shared = np.diag(np.ones(n_part))

spread_speed = 0.1

base_spread = (np.ones_like(elec_shared) * spread_speed) / n_part + np.diag(np.ones(n_part) * (1 - spread_speed))

dist = squareform(pdist(pos))

# spread charge to others by dist?

# nonlinear scaling of electronegativity with distance
# particle getting close to another more electronegative one causes an even more imbalanced charge
# polar covalent bond will become more polar with less distance
# potential for interaction at the scale of transferring charge from thermal motion alone
# polar covalent bond pushed closer together will cause loss of negative charge on less electronegative particle
# balancing effect pushing away? What is the attraction that makes it a balanced force?


# distance not involved
# spread by bond strength minus difference in electronegativity
# should be symmetric to preserve charge


# distance involved


print(dist)