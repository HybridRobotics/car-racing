#!/usr/bin/env python
import numpy as np
import rospy
import pylab
import matplotlib.pyplot as plt
import sys
scripts_realtime_path = sys.path[0]
src_utils_path = scripts_realtime_path+'/../../src/utils'
src_path = scripts_realtime_path+'/../../src'
sys.path.append(src_utils_path)
sys.path.append(src_path)
ellipse_track_spec_path = scripts_realtime_path + '/../../data/track_layout/ellipse.csv'
goggle_track_spec_path = scripts_realtime_path + '/../../data/track_layout/goggle_shaped.csv'
l_track_spec_path = scripts_realtime_path + '/../../data/track_layout/l_shaped.csv'

import realtime, vehicle_dynamics, racing_env
from car_racing.msg import VehicleControl, VehicleState, VehicleStateGlob, VehicleStateCurv, NumVehicle, TrackInfo, VehicleList
from car_racing.srv import AddNewVehicle


def get_msg_xglob(state):
    return VehicleStateGlob(state[0], state[1], state[2], state[3], state[4], state[5])

def get_msg_xcurv(state):
    return VehicleStateCurv(state[0], state[1], state[2], state[3], state[4], state[5])

def get_msg_track_info(track):
    size0,size1 = np.shape(track.point_and_tangent)
    point_and_tangent = np.zeros(size0*size1)
    tmp = 0
    for index_1 in range(size1):
        for index_0 in range(size0):
            point_and_tangent[tmp] = track.point_and_tangent[index_0,index_1]
            tmp = tmp +1
    return TrackInfo(track.lap_length, track.width, size1, point_and_tangent)

def start_simulator(track_layout):
    rospy.init_node("simulator", anonymous = True)
    # get race track
    if track_layout == "ellipse":
        track_spec = np.genfromtxt(ellipse_track_spec_path, delimiter=",")
    elif track_layout == "goggle":
        track_spec = np.genfromtxt(goggle_track_spec_path, delimiter=",")
    elif track_layout =="l_shape":
        track_spec = np.genfromtxt(l_track_spec_path, delimiter=",")
    else:
        pass
    track_width = 1.0
    track = racing_env.ClosedTrack(track_spec, track_width)

    sim = realtime.CarRacingSimRealtime()
    sim.set_track(track)    
    # determin if new vehicle is added
    s = rospy.Service('add_vehicle_simulator',AddNewVehicle,sim.add_vehicle)

    loop_rate = 100
    r = rospy.Rate(loop_rate)  

    sim.__pub_veh_list = rospy.Publisher('vehicle_list', VehicleList, queue_size = 10)
    sim.__pub_veh_num = rospy.Publisher('vehicle_num', NumVehicle, queue_size = 10)    
    sim.__pub_track = rospy.Publisher('track_info', TrackInfo, queue_size = 10)
  
    while not rospy.is_shutdown():
        sim.num_vehicle = 0
        vehicle_list = []
        for name in sim.vehicles:
            sim.vehicles[name].msg_state.state_curv = get_msg_xcurv(sim.vehicles[name].xcurv)
            sim.vehicles[name].msg_state.state_glob = get_msg_xglob(sim.vehicles[name].xglob)
            tmp = 'simulator/' + name +'/state'
            sim.vehicles[name].__pub_state = rospy.Publisher(tmp, VehicleState, queue_size=10)
            sim.vehicles[name].__pub_state.publish(sim.vehicles[name].msg_state)
            vehicle_list.append(name)
            sim.num_vehicle = sim.num_vehicle + 1
        sim.__pub_veh_list.publish(VehicleList(vehicle_list))
        sim.__pub_veh_num.publish(NumVehicle(sim.num_vehicle))
        sim.__pub_track.publish(get_msg_track_info(track))
        r.sleep()
        

if __name__ == '__main__':
    try:
        track_layout = rospy.get_param("track_layout")
        start_simulator(track_layout)
    except rospy.ROSInterruptException:
        pass