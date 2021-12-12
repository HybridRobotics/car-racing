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


textSize = 14
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = textSize
rcParams['ytick.labelsize'] = textSize
with open("data/simulator/lmpc_racing.obj", "rb") as handle:
    simulator_1 = pickle.load(handle)
with open("data/simulator/racing_game_29.obj", "rb") as handle:
    simulator_2 = pickle.load(handle)
plt.figure(figsize=(8,4))
ax_0=plt.subplot(211)
ax_1=plt.subplot(212)
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

ax_1.plot(traj_xcurv_1[0:-1,4], traj_xcurv_1[0:-1,5], linewidth = 2)
ax_1.plot(traj_xcurv[0:-1,4], traj_xcurv[0:-1,5], linewidth = 2)
# mark overtake zone
tmp1 = np.zeros((2,2))
tmp1[0,0] = 17.44
tmp1[0,1] = 17.44
tmp1[1,0] = -1
tmp1[1,1] = 1
ax_1.plot(tmp1[0,:], tmp1[1,:], color = 'red', linewidth = 2)
tmp2 = np.zeros((2,2))
tmp2[0,0] = 27.59
tmp2[0,1] = 27.59
tmp2[1,0] = -1
tmp2[1,1] = 1
ax_1.plot(tmp2[0,:], tmp2[1,:], color='red', linewidth = 2)
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
ax_1.plot(tmp3[0,:], tmp3[1,:], color='red', linewidth = 2)
ax_1.plot(tmp4[0,:], tmp4[1,:], color='red', linewidth = 2)
ax_0.plot(traj_xcurv_1[0:-1,4], traj_xcurv_1[0:-1,0], linewidth = 2)
ax_0.plot(traj_xcurv[0:-1,4], traj_xcurv[0:-1,0], linewidth = 2)
tmp1 = np.zeros((2,2))
tmp1[0,0] = 17.44
tmp1[0,1] = 17.44
tmp1[1,0] = 1.3
tmp1[1,1] = 5
ax_0.plot(tmp1[0,:], tmp1[1,:], color = 'red', linewidth = 2)
tmp2 = np.zeros((2,2))
tmp2[0,0] = 27.59
tmp2[0,1] = 27.59
tmp2[1,0] = 1.3
tmp2[1,1] = 5
ax_0.plot(tmp2[0,:], tmp2[1,:], color='red', linewidth = 2)
tmp3 = np.zeros((2,2))
tmp4 = np.zeros((2,2))
tmp3[0,0] = 17.44
tmp3[0,1] = 27.59
tmp3[1,0] = 5
tmp3[1,1] = 5
tmp4[0,0] = 17.44
tmp4[0,1] = 27.59
tmp4[1,0] = 1.3
tmp4[1,1] = 1.3
ax_0.plot(tmp3[0,:], tmp3[1,:], color='red', linewidth = 2)
ax_0.plot(tmp4[0,:], tmp4[1,:], color='red', linewidth = 2)
ax_0.set_xlabel("${s_c}$ (m)",fontproperties = 'Times New Roman', size = 14)
ax_0.set_ylabel("${v_x}$ (m/s)",fontproperties = 'Times New Roman', size = 14)
ax_1.set_xlabel("${s_c}$ (m)",fontproperties = 'Times New Roman', size = 14)
ax_1.set_ylabel("${e_y}$ (m)",fontproperties = 'Times New Roman', size = 14)
ax_1.spines['right'].set_linewidth(2)
ax_1.spines['left'].set_linewidth(1)
ax_1.spines['top'].set_linewidth(1)
ax_1.spines['bottom'].set_linewidth(1)
ax_0.spines['right'].set_linewidth(2)
ax_0.spines['left'].set_linewidth(1)
ax_0.spines['top'].set_linewidth(1)
ax_0.spines['bottom'].set_linewidth(1)
plt.tight_layout()
plt.savefig("media/figures/speed_deviation.png",format="png",dpi=1000,pad_inches=0, bbox_inches='tight')
plt.show()