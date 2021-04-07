#!/usr/bin/env python
import numpy as np
import rospy
import datetime
import racing_car, policy, vehicle_dynamics, racing_env
from car_racing_dev.msg import VehicleControl, VehicleState, VehicleStateCurv, VehicleStateGlob


def get_msg_xglob(state):
    return VehicleStateGlob(state[0], state[1], state[2], state[3], state[4], state[5])

def get_msg_xcurv(state):
    return VehicleStateCurv(state[0], state[1], state[2], state[3], state[4], state[5])

def set_vehicle1():
    rospy.init_node("vehicle1", anonymous = True)

    track_spec = np.genfromtxt("/home/suiyihe/catkin_ws/src/car_racing_dev/data/track_spec/default.csv", delimiter=",")
    track_width = 1.0
    track = racing_env.ClosedTrack(track_spec, track_width)

    vehicle1 = racing_car.RealtimeDynamicBicycleModel(name = "vehicle1",param = racing_car.CarParam())
    vehicle1.set_state_curvilinear(np.zeros((6,)))
    vehicle1.set_state_global(np.zeros((6,)))
    vehicle1.u = np.zeros(2)
    vehicle1.set_subscriber()
    vehicle1.__pub_state = rospy.Publisher('vehicle1/state', VehicleState, queue_size = 10)

    loop_rate = 100
    timestep = 1 / loop_rate
    vehicle1.set_timestep(timestep)

    start_timer = datetime.datetime.now()
    tmp = 0

    msg_state = VehicleState()
    msg_state.name = "Vehicle1"

    while not rospy.is_shutdown():        
        current_time = datetime.datetime.now()
        # update the vehicle's state at 100 Hz
        if (current_time-start_timer).total_seconds() > (tmp+1)*(1/100):
            s = vehicle1.xcurv[4]
            curv = track.get_curvature(s)
            xglob_next, xcurv_next = vehicle_dynamics.vehicle_dynamics(vehicle1.param.dynamics_param, curv, vehicle1.xglob, vehicle1.xcurv, timestep, vehicle1.u)
            msg_state.state_curv = get_msg_xcurv(xcurv_next)
            msg_state.state_glob = get_msg_xglob(xglob_next)
            vehicle1.__pub_state.publish(msg_state)
            vehicle1.set_state_curvilinear(xcurv_next)
            vehicle1.set_state_global(xglob_next)
            tmp = tmp + 1
        else:
            pass
       
   
if __name__ == '__main__':
    try:
        set_vehicle1()
    except rospy.ROSInterruptException:
        pass