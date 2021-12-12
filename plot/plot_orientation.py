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
from matplotlib import rcParams


textSize = 35
rcParams['axes.labelsize'] = 35
rcParams['xtick.labelsize'] = textSize
rcParams['ytick.labelsize'] = textSize
with open("data/simulator/lmpc_racing.obj", "rb") as handle:
    simulator_1 = pickle.load(handle)
with open("data/simulator/racing_game_29.obj", "rb") as handle:
    simulator_2 = pickle.load(handle)
figure = plt.figure(figsize=(20.5,6))
ego_1 = simulator_1.vehicles["ego"]
laps_1 = ego_1.laps
steps_1 = int(round((ego_1.times[laps_1-1][-1] - ego_1.times[laps_1-1][0])/ego_1.timestep))+1
traj_xglob_1 = np.zeros((steps_1,X_DIM))
traj_xcurv_1 = np.zeros((steps_1,X_DIM))
for index in range(0, steps_1):
    traj_xglob_1[index, :] = ego_1.xglobs[laps_1-1][index][:]
    traj_xcurv_1[index, :] = ego_1.xcurvs[laps_1-1][index][:]
ego = simulator_2.vehicles["ego"]
car1 = simulator_2.vehicles["car1"]
car2 = simulator_2.vehicles["car2"]
car3 = simulator_2.vehicles["car3"]
laps = ego.laps
steps = int(round((ego.times[laps-1][-1] - ego.times[laps-1][0])/ego.timestep))+1
traj_xglob = np.zeros((steps,X_DIM))
traj_xcurv = np.zeros((steps,X_DIM))
# local info
horizon_planner = 10
vehicles_interest = []
local_traj_xglob = np.zeros((steps, horizon_planner+1, X_DIM))
local_spline_xglob = np.zeros((steps, horizon_planner+1, X_DIM))
traj_xglob_ego = np.zeros((steps, X_DIM))
traj_xglob_car1 = np.zeros((steps, X_DIM))
traj_xglob_car2 = np.zeros((steps, X_DIM))
traj_xglob_car3 = np.zeros((steps, X_DIM))
for index in range(0, steps-1):
    traj_xglob[index, :] = ego.xglobs[laps-1][index][:]
    traj_xcurv[index, :] = ego.xcurvs[laps-1][index][:]
counter = 0
for index in range(0, steps-1):
    traj_xglob_ego[steps - 1 - counter, :] = ego.xglob_log[-1-index][:]
    traj_xglob_car1[steps - 1 - counter, :] = car1.xglob_log[-1-index][:]
    traj_xglob_car2[steps - 1 - counter, :] = car2.xglob_log[-1-index][:]
    traj_xglob_car3[steps - 1 - counter, :] = car3.xglob_log[-1-index][:]
    if ego.local_trajs[-1-index] is None:
        local_traj_xglob[steps - 1 - counter, :, :] = np.zeros((horizon_planner+1, X_DIM))
        local_spline_xglob[steps-1-counter, :, :] = np.zeros((horizon_planner+1, X_DIM))
    else:
        local_traj_xglob[steps-1-counter, :, :] = ego.local_trajs[-1 - index][:, :]
        local_spline_xglob[steps-1-counter, :, :] = ego.splines[-1-index][:, :]
    if ego.vehicles_interest[-1-index] is None:
        vehicles_interest.insert(0, None)
    else:
        vehicles_interest.insert(0, ego.vehicles_interest[-1-index])    
    counter = counter + 1
car1_overtake_time = []
car2_overtake_time = []
car3_overtake_time = []
for index in range(0, steps-1):
    if vehicles_interest[index] is None:
        pass
    else:
        for name in list(vehicles_interest[index]):
            if name == 'car1':
                car1_overtake_time.append(index)
            if name == 'car2':
                car2_overtake_time.append(index)
            if name == 'car3':
                car3_overtake_time.append(index)

plt.plot(traj_xcurv_1[0:-1,4], traj_xcurv_1[0:-1,3], linewidth = 8)
plt.plot(traj_xcurv[0:-1,4], traj_xcurv[0:-1,3], linewidth = 8)
# mark overtake zone
tmp1 = np.zeros((2,2))
tmp1[0,0] = 17.44
tmp1[0,1] = 17.44
tmp1[1,0] = -1
tmp1[1,1] = 1
plt.plot(tmp1[0,:], tmp1[1,:], color = 'red', linewidth = 8)
tmp2 = np.zeros((2,2))
tmp2[0,0] = 27.59
tmp2[0,1] = 27.59
tmp2[1,0] = -1
tmp2[1,1] = 1
plt.plot(tmp2[0,:], tmp2[1,:], color='red', linewidth = 8)
tmp3 = np.zeros((2,2))
tmp4 = np.zeros((2,2))
tmp3[0,0] = 17.44
tmp3[0,1] = 27.59
tmp3[1,0] = 1
tmp3[1,1] = 1
tmp4[0,0] = 17.44
tmp4[0,1] = 27.59
tmp4[1,0] = -1
tmp4[1,1] = -1
plt.plot(tmp3[0,:], tmp3[1,:], color='red', linewidth = 8)
plt.plot(tmp4[0,:], tmp4[1,:], color='red', linewidth = 8)
plt.xticks(fontproperties = 'Times New Roman', size = 35)
plt.yticks(fontproperties = 'Times New Roman', size = 35)
plt.ylabel("${e_\psi}$ [rad]")
plt.xlabel("${s_c}$ [m]")
ax = plt.gca()
ax.spines['right'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)
ax.spines['top'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.set_xlim([-1,51])
ax.yaxis.set_ticks_position('left')
plt.tight_layout()
plt.savefig("media/figures/orientation.png",format="png",dpi=1000,pad_inches=0)
plt.show()