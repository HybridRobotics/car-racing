#!/usr/bin/env python
import numpy as np
import rospy
import pylab
import matplotlib.pyplot as plt
from racing import realtime
from utils import racing_env
from car_racing.msg import (
    VehicleControl,
    VehicleState,
    VehicleStateGlob,
    VehicleStateCurv,
    NumVehicle,
    TrackInfo,
    VehicleList,
    OptimalTraj,
)
from car_racing.srv import AddNewVehicle
from utils.constants import *


def get_msg_xglob(state):
    return VehicleStateGlob(state[0], state[1], state[2], state[3], state[4], state[5])


def get_msg_xcurv(state):
    return VehicleStateCurv(state[0], state[1], state[2], state[3], state[4], state[5])


def get_msg_track_info(track, layout):
    size0, size1 = np.shape(track.point_and_tangent)
    point_and_tangent = np.zeros(size0 * size1)
    tmp = 0
    for index_1 in range(size1):
        for index_0 in range(size0):
            point_and_tangent[tmp] = track.point_and_tangent[index_0, index_1]
            tmp = tmp + 1
    return TrackInfo(track.lap_length, track.width, size1, point_and_tangent, layout)


def get_msg_optimal_traj(traj_xglob, traj_xcurv):
    size, _ = np.shape(traj_xglob)
    xglobs = np.zeros(size * X_DIM)
    xcurvs = np.zeros(size * X_DIM)
    tmp = 0
    for index in range(size):
        for index_1 in range(X_DIM):
            xglobs[tmp] = traj_xglob[index, index_1]
            xcurvs[tmp] = traj_xcurv[index, index_1]
            tmp = tmp + 1
    return OptimalTraj(size, xglobs, xcurvs)


def start_simulator(track_layout):
    rospy.init_node("simulator", anonymous=True)
    # get race track
    track_spec = np.genfromtxt("../../data/track_layout/" + track_layout + ".csv", delimiter=",")
    track_width = 0.5
    track = racing_env.ClosedTrack(track_spec, track_width)
    sim = realtime.CarRacingSim()
    sim.set_track(track)
    # determin if new vehicle is added
    s = rospy.Service("add_vehicle_simulator", AddNewVehicle, sim.add_vehicle)
    loop_rate = 100
    r = rospy.Rate(loop_rate)
    sim.__pub_veh_list = rospy.Publisher("vehicle_list", VehicleList, queue_size=10)
    sim.__pub_veh_num = rospy.Publisher("vehicle_num", NumVehicle, queue_size=10)
    sim.__pub_track = rospy.Publisher("track_info", TrackInfo, queue_size=10)
    while not rospy.is_shutdown():
        sim.num_vehicle = 0
        vehicles = []
        for name in sim.vehicles:
            sim.vehicles[name].msg_state.state_curv = get_msg_xcurv(sim.vehicles[name].xcurv)
            sim.vehicles[name].msg_state.state_glob = get_msg_xglob(sim.vehicles[name].xglob)
            tmp = "simulator/" + name + "/state"
            sim.vehicles[name].__pub_state = rospy.Publisher(tmp, VehicleState, queue_size=10)
            sim.vehicles[name].__pub_state.publish(sim.vehicles[name].msg_state)
            vehicles.append(name)
            sim.num_vehicle = sim.num_vehicle + 1
        sim.__pub_veh_list.publish(VehicleList(vehicles))
        sim.__pub_veh_num.publish(NumVehicle(sim.num_vehicle))
        sim.__pub_track.publish(get_msg_track_info(track, track_layout))
        r.sleep()


if __name__ == "__main__":
    try:
        track_layout = rospy.get_param("track_layout")
        start_simulator(track_layout)
    except rospy.ROSInterruptException:
        pass
