import numpy as np
import sympy as sp
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anim
from utils import vehicle_dynamics, base, racing_env
from utils.constants import *
from matplotlib import animation
from matplotlib.collections import LineCollection


plot_lmpc = True
plot_racing_game = False
if plot_lmpc:
    with open("data/simulator/lmpc_racing.obj", "rb") as handle:
        simulator = pickle.load(handle)
if plot_racing_game:
    with open("data/simulator/racing_game_08.obj", "rb") as handle:
        simulator = pickle.load(handle)
track = simulator.track
fig, ax = plt.subplots()
track.plot_track(ax, center_line = False)
ego = simulator.vehicles["ego"]
laps = ego.laps
steps = int(round((ego.times[laps-1][-1] - ego.times[laps-1][0])/ego.timestep))+1
traj_xglob = np.zeros((steps,X_DIM))
for index in range(0, steps):
    traj_xglob[index, :] = ego.xglobs[laps-1][index][:]
colors = traj_xglob[:,0]
points = np.array([traj_xglob[:,4], traj_xglob[:,5]]).T.reshape(-1,1,2)
segements = np.concatenate([points[:-1],points[1:]],axis=1)
lc = LineCollection(segements, cmap = 'viridis')
lc.set_array(colors)
lc.set_linewidth(3)
line = ax.add_collection(lc)
fig.colorbar(line)
#plt.scatter(traj_xglob[:,4], traj_xglob[:,5], c=colors, cmap='viridis', s = 30)
ax.set_xlabel("x [m]", fontsize = 14)
ax.set_ylabel("y [m]", fontsize = 14)
plt.ylim(ymin = -2)
plt.ylim(ymax = 9)
ax.axis("equal")
plt.savefig("media/figures/lmpc_traj.png",format="png",dpi=1000,pad_inches=0)
plt.show()