#!/usr/bin/env python
import numpy as np
import rospy
import racing_car, policy, vehicle_dynamics, racing_env, racing_sim
from car_racing_dev.msg import VehicleControl, VehicleState, VehicleStateGlob, VehicleStateCurv
import pylab
import matplotlib.pyplot as plt


def get_msg_xglob(state):
    return VehicleStateGlob(state[0], state[1], state[2], state[3], state[4], state[5])

def get_msg_xcurv(state):
    return VehicleStateCurv(state[0], state[1], state[2], state[3], state[4], state[5])

def start_simulator():
    rospy.init_node("simulator", anonymous = True)
    track_spec = np.genfromtxt("/home/suiyihe/catkin_ws/src/car_racing_dev/data/track_spec/default.csv", delimiter=",")
    track_width = 1.0
    track = racing_env.ClosedTrack(track_spec, track_width)
    simulator1 = racing_sim.RealtimeCarRacingSim()
    simulator1.set_track(track)
    simulator1.set_state_glob(np.zeros(6,))
    simulator1.set_state_curv(np.zeros(6,))
    simulator1.set_subscriber()
    simulator1.__pub_state = rospy.Publisher('simulator/vehicle1/state', VehicleState, queue_size = 10)
        
    msg_state = VehicleState()
    msg_state.name = "Vehicle1"
    
    while not rospy.is_shutdown():   
        msg_state.state_curv = get_msg_xcurv(simulator1.vehicle_state_curv)
        msg_state.state_glob = get_msg_xglob(simulator1.vehicle_state_glob)
        simulator1.__pub_state.publish(msg_state)
        

if __name__ == '__main__':
    try:
        start_simulator()
    except rospy.ROSInterruptException:
        pass