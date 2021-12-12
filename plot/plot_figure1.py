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


with open("data/simulator/racing_game_35.obj", "rb") as handle:
    simulator = pickle.load(handle)
ego = simulator.vehicles["ego"]
car1 = simulator.vehicles["car1"]
car2 = simulator.vehicles["car2"]
laps = ego.laps
steps = int(round((ego.times[laps-1][-1] - ego.times[laps-1][0])/ego.timestep))+1
traj_xglob = np.zeros((steps,X_DIM))
# local info
horizon_planner = 10
vehicles_interest = []
local_traj_xglob = np.zeros((steps, horizon_planner+1, X_DIM))
local_spline_xglob = np.zeros((steps, horizon_planner+1, X_DIM))
traj_xglob_ego = np.zeros((steps, X_DIM))
traj_xglob_car1 = np.zeros((steps, X_DIM))
traj_xglob_car2 = np.zeros((steps, X_DIM))
for index in range(0, steps-1):
    traj_xglob[index, :] = ego.xglobs[laps-1][index][:]
counter = 0
for index in range(0, steps-1):
    traj_xglob_ego[steps - 1 - counter, :] = ego.xglob_log[-1-index][:]
    traj_xglob_car1[steps - 1 - counter, :] = car1.xglob_log[-1-index][:]
    traj_xglob_car2[steps - 1 - counter, :] = car2.xglob_log[-1-index][:]
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
for index in range(0, steps-1):
    if vehicles_interest[index] is None:
        pass
    else:
        for name in list(vehicles_interest[index]):
            if name == 'car1':
                car1_overtake_time.append(index)
            if name == 'car2':
                car2_overtake_time.append(index)
fig = plt.figure(figsize=(9,4.5))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax1 = fig.add_axes([0.12,0.12,0.4,0.4])
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
for j in range(0,8):
    timestep = 37 + j*6
    x_ego, y_ego, dx_ego, dy_ego, alpha_ego = ego.get_vehicle_in_rectangle(traj_xglob_ego[timestep,:])
    ax.add_patch(patches.Rectangle((x_ego, y_ego), dx_ego, dy_ego, alpha_ego, color="red", alpha = 1 - j*0.1))
    x_car2, y_car2, dx_car2, dy_car2, alpha_car2 = car2.get_vehicle_in_rectangle(traj_xglob_car2[timestep,:])
    x_car1, y_car1, dx_car1, dy_car1, alpha_car1 = car1.get_vehicle_in_rectangle(traj_xglob_car1[timestep,:])
    if j ==7 or j ==6 or j ==5 :
        car1_color ='blue'
    else:
        car1_color='green'
    ax.add_patch(patches.Rectangle((x_car1, y_car1), dx_car1, dy_car1, alpha_car1, color=car1_color,alpha = 1 - j*0.12))
    if j ==7 :
        car2_color = 'blue'
    else:
        car2_color = 'green'
    ax.add_patch(patches.Rectangle((x_car2, y_car2), dx_car2, dy_car2, alpha_car2, color=car2_color,alpha = 1 - j*0.1))
track = simulator.track
track.plot_track(ax, center_line = True)
timestep = 37
x_ego, y_ego, dx_ego, dy_ego, alpha_ego = ego.get_vehicle_in_rectangle(traj_xglob_ego[timestep,:])
ax1.add_patch(patches.Rectangle((x_ego, y_ego), dx_ego, dy_ego, alpha_ego, color="red"))
x_car2, y_car2, dx_car2, dy_car2, alpha_car2 = car2.get_vehicle_in_rectangle(traj_xglob_car2[timestep,:])
x_car1, y_car1, dx_car1, dy_car1, alpha_car1 = car1.get_vehicle_in_rectangle(traj_xglob_car1[timestep,:])
car1_color='green'
ax1.add_patch(patches.Rectangle((x_car1, y_car1), dx_car1, dy_car1, alpha_car1, color=car1_color))
car2_color = 'green'
ax1.add_patch(patches.Rectangle((x_car2, y_car2), dx_car2, dy_car2, alpha_car2, color=car2_color))
line1, =ax1.plot(local_traj_xglob[timestep+1, :, 4], local_traj_xglob[timestep+1,:,5], '.-', color = 'orange')
timestep = 35
delta_x = traj_xglob_ego[37,4] - local_spline_xglob[timestep+1, 0, 4]
delta_y = traj_xglob_ego[37,5] - local_spline_xglob[timestep+1, 0, 5]
line2, =ax1.plot(local_spline_xglob[timestep+1, :, 4] + delta_x, local_spline_xglob[timestep+1,:,5]+delta_y,'k--')
ax.axis("equal")
ax.set_xticks([])
ax.set_yticks([])
track.plot_track(ax1, center_line = False)
ax1.axis("equal")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlim([5.5,9.5])
ax1.set_ylim([4.5,8.5])
plt.subplots_adjust(left=0,bottom=0,right=0.1,top=0.1,hspace=0.1,wspace=0.1)
plt.savefig("media/figures/introduction_snap_shot.png",format="png",dpi=1000,pad_inches=0,bbox_inches='tight')
plt.show()