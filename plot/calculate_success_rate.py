import numpy as np
import sympy as sp
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anim
from utils import vehicle_dynamics, base, racing_env
from utils.constants import *

counter = 0

for index in range(1,51):
    file_name = "data/simulator/12_16_three/racing_game" +str(index) +".obj"
    with open(file_name, "rb") as handle:
        simulator = pickle.load(handle)
    num_veh=len(simulator.vehicles)-1
    x_ego = simulator.vehicles["ego"].xglob_log[-1][4]
    y_ego = simulator.vehicles["ego"].xglob_log[-1][5]
    psi_ego = simulator.vehicles["ego"].xglob_log[-1][3]
    track = simulator.track
    s_ego, _,_,_ = track.get_local_position(x_ego, y_ego, psi_ego)
    s_ego = track.lap_length + s_ego
    

    
    car_info = np.zeros((num_veh,4))
    Flag = True
    for j in range(0,num_veh):
        car_info[j,0] = simulator.vehicles['car'+str(j+1)].xglob_log[-1][4] # x
        car_info[j,1] = simulator.vehicles['car'+str(j+1)].xglob_log[-1][5] # y
        car_info[j,2] = simulator.vehicles['car'+str(j+1)].xglob_log[-1][3] # psi
        car_info[j,3],_,_,_ = track.get_local_position(car_info[j,0], car_info[j,1], car_info[j,2])
        if car_info[j,3] <=10:
            car_info[j,3] = car_info[j,3]+track.lap_length
        if car_info[j,3] >=s_ego:
            Flag = False
    if Flag==True:
        counter = counter+1
    print(index)
    
print('counter',counter)
