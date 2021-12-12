import numpy as np
import sympy as sp
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anim
from utils import vehicle_dynamics, base, racing_env
from utils.constants import *

times = np.zeros(100)
for iiii in range(1,101):
    file_name = "data/simulator/08_12_single/racing_game" +str(iiii) +".obj"
    with open(file_name, "rb") as handle:
        simulator = pickle.load(handle)
    ego = simulator.vehicles["ego"]
    car1 = simulator.vehicles["car1"]
    vehicles_interest = []
    counter = 0
    laps = ego.laps
    steps = int(round((ego.times[laps-1][-1] - ego.times[laps-1][0])/ego.timestep))+1
    for index in range(0, steps-1):
        if ego.vehicles_interest[-1-index] is None:
            vehicles_interest.insert(0, None)
        else:
            vehicles_interest.insert(0, ego.vehicles_interest[-1-index])    
        counter = counter + 1
    car1_overtake_time = []
    for index in range(0, steps-1):
        if vehicles_interest[index] is None:
            pass
        else:
            for name in list(vehicles_interest[index]):
                if name == 'car1':
                    car1_overtake_time.append(index)
    times[iiii-1] = (max(car1_overtake_time) - min(car1_overtake_time))
print('max',max(times))
print('min',min(times))
print('mean',np.mean(times))
print(times)