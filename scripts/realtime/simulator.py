#!/usr/bin/env python
import numpy as np
import rospy
import pylab
import matplotlib.pyplot as plt
import sys
tests_path = sys.path[0]
src_utils_path = tests_path+'/../../src/utils'
src_path = tests_path+'/../../src'
sys.path.append(src_utils_path)
sys.path.append(src_path)
track_spec_path = tests_path + '/../../data/track_spec/default.csv'
import realtime, vehicle_dynamics, racing_env
from car_racing_sim.msg import VehicleControl, VehicleState, VehicleStateGlob, VehicleStateCurv



def get_msg_xglob(state):
    return VehicleStateGlob(state[0], state[1], state[2], state[3], state[4], state[5])

def get_msg_xcurv(state):
    return VehicleStateCurv(state[0], state[1], state[2], state[3], state[4], state[5])

def start_simulator():
    rospy.init_node("simulator", anonymous = True)
    track_spec = np.genfromtxt(track_spec_path, delimiter=",")
    track_width = 1.0
    track = racing_env.ClosedTrack(track_spec, track_width)
    sim1 = realtime.CarRacingSimRealtime()
    sim1.set_track(track)
    sim1.set_state_glob(np.zeros(6,))
    sim1.set_state_curv(np.zeros(6,))
    sim1.set_subscriber()
    sim1.__pub_state = rospy.Publisher('simulator/vehicle1/state', VehicleState, queue_size = 10)
        
    msg_state = VehicleState()
    msg_state.name = "Vehicle1"

    loop_rate = 100
    r = rospy.Rate(loop_rate)
  
    
    while not rospy.is_shutdown():   
        msg_state.state_curv = get_msg_xcurv(sim1.vehicle_state_curv)
        msg_state.state_glob = get_msg_xglob(sim1.vehicle_state_glob)
        sim1.__pub_state.publish(msg_state)
        r.sleep()
        

if __name__ == '__main__':
    try:
        start_simulator()
    except rospy.ROSInterruptException:
        pass