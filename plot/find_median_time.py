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


with open("data/simulator/racing_game.obj", "rb") as handle:
    simulator_1 = pickle.load(handle)
ego_1 = simulator_1.vehicles["ego"]
laps_1 = ego_1.laps
steps_1 = int(round((ego_1.times[laps_1-1][-1] - ego_1.times[laps_1-1][0])/ego_1.timestep))+1
time_solver = []

time_solver = ego_1.solver_time
counter = 0
for index in range(0,steps_1-2):
    if time_solver[index] is None:
        pass
    else:
        num = len(time_solver[index])
        counter = counter+1
times=np.zeros(counter)
tmp = 0
for index in range(0,steps_1-2):
    if time_solver[index] is None:
        pass
    else:
        times[tmp] = max(time_solver[index])
        tmp = tmp + 1
print(np.mean(times))
print(times)