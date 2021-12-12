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


with open("data/simulator/racing_game.obj", "rb") as handle:
    simulator = pickle.load(handle)
ego = simulator.vehicles["ego"]
car1 = simulator.vehicles["car1"]
car2 = simulator.vehicles["car2"]
car3 = simulator.vehicles["car3"]
laps = ego.laps
steps = int(round((ego.times[laps-1][-1] - ego.times[laps-1][0])/ego.timestep))+1
traj_xglob = np.zeros((steps,X_DIM))
# local info
horizon_planner = ego.ctrl_policy.racing_game_param.num_horizon_planner
vehicles_interest = []
local_traj_xglob = np.zeros((steps, horizon_planner+1, X_DIM))
local_spline_xglob = np.zeros((steps, horizon_planner+1, X_DIM))
traj_xglob_ego = np.zeros((steps, X_DIM))
traj_xglob_car1 = np.zeros((steps, X_DIM))
traj_xglob_car2 = np.zeros((steps, X_DIM))
traj_xglob_car3 = np.zeros((steps, X_DIM))
for index in range(0, steps):
    traj_xglob[index, :] = ego.xglobs[laps-1][index][:]
counter = 0
for index in range(0, steps):
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
for index in range(0, steps):
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
fig, ax = plt.subplots(3,2)
fig.set_tight_layout(True)
for j in range(0,6):
    index = j*8
    if j ==5:
        index = 42
    timestep = car2_overtake_time[index]
    if j==0:
        fig_position = ax[0][0]
    if j==1:
        fig_position = ax[0][1]
    if j==2:
        fig_position = ax[1][0]
    if j==3:
        fig_position = ax[1][1]
    if j==4:
        fig_position = ax[2][0]
    if j==5:
        fig_position = ax[2][1]
    x_ego, y_ego, dx_ego, dy_ego, alpha_ego = ego.get_vehicle_in_rectangle(traj_xglob_ego[timestep,:])
    fig_position.add_patch(patches.Rectangle((x_ego, y_ego), dx_ego, dy_ego, alpha_ego, color="red"))
    x_car, y_car, dx_car, dy_car, alpha_car = car2.get_vehicle_in_rectangle(traj_xglob_car2[timestep,:])
    fig_position.add_patch(patches.Rectangle((x_car, y_car), dx_car, dy_car, alpha_car, color="green"))
    line1, =fig_position.plot(local_traj_xglob[timestep+1, :, 4], local_traj_xglob[timestep+1,:,5], '.-', color = 'orange')
    line2, =fig_position.plot(local_spline_xglob[timestep+1, :, 4], local_spline_xglob[timestep+1,:,5],'k--')
    track = simulator.track
    track.plot_track(fig_position, center_line = False)
    fig_position.axis("equal")
    fig_position.set_xticks([])
    fig_position.set_yticks([])
    if j ==0:
        fig_position.set_ylim([-0.5,3.5])
        fig_position.set_xlim([5,9])
    if j ==1:
        fig_position.set_ylim([0.5,4.5])
        fig_position.set_xlim([6,10])
    if j ==2:
        fig_position.set_ylim([1.5,5.5])
        fig_position.set_xlim([6,10])
    if j ==3:
        fig_position.set_ylim([2.5,6.5])
        fig_position.set_xlim([6,10])
    if j ==4:
        fig_position.set_ylim([4.5,8.5])
        fig_position.set_xlim([6,10])
    if j ==5:
        fig_position.set_ylim([4.5,8.5])
        fig_position.set_xlim([6,10])
plt.savefig("media/figures/overtaking_1.png",format="png",dpi=1000,pad_inches=0)
plt.show()