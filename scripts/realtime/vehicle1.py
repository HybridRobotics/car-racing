#!/usr/bin/env python
import rospy
import numpy as np
import datetime
import sys
tests_path = sys.path[0]
src_utils_path = tests_path+'/../../src/utils'
src_path = tests_path+'/../../src'
sys.path.append(src_utils_path)
sys.path.append(src_path)
track_spec_path = tests_path + '/../../data/track_spec/default.csv'
from car_racing_sim.msg import VehicleControl, VehicleState, VehicleStateCurv, VehicleStateGlob
import vehicle_dynamics, racing_env, base, realtime
                            
def get_msg_xglob(state):
    return VehicleStateGlob(state[0], state[1], state[2], state[3], state[4], state[5])


def get_msg_xcurv(state):
    return VehicleStateCurv(state[0], state[1], state[2], state[3], state[4], state[5])


def set_vehicle1():
    rospy.init_node("vehicle1", anonymous=True)
    track_spec = np.genfromtxt(track_spec_path, delimiter=",")
    track_width = 1.0
    track = racing_env.ClosedTrack(track_spec, track_width)

    veh1 = realtime.DynamicBicycleModelRealtime(
        name="vehicle1", param=base.CarParam())
    veh1.set_state_curvilinear(np.zeros((6,)))
    veh1.set_state_global(np.zeros((6,)))
    veh1.u = np.zeros(2)
    veh1.set_subscriber()
    veh1.__pub_state = rospy.Publisher(
        'vehicle1/state', VehicleState, queue_size=10)

    loop_rate = 100
    timestep = 1 / loop_rate
    veh1.set_timestep(timestep)

    r = rospy.Rate(loop_rate) # 100Hz

    start_timer = datetime.datetime.now()
    tmp = 0

    msg_state = VehicleState()
    msg_state.name = "Vehicle1"

    while not rospy.is_shutdown():
        current_time = datetime.datetime.now()
        # update the vehicle's state at 100 Hz
        if (current_time-start_timer).total_seconds() > (tmp+1)*(1/100):
            s = veh1.xcurv[4]
            curv = track.get_curvature(s)
            xglob_next, xcurv_next = vehicle_dynamics.vehicle_dynamics(
                veh1.param.dynamics_param, curv, veh1.xglob, veh1.xcurv, timestep, veh1.u)
            msg_state.state_curv = get_msg_xcurv(xcurv_next)
            msg_state.state_glob = get_msg_xglob(xglob_next)
            veh1.__pub_state.publish(msg_state)
            veh1.set_state_curvilinear(xcurv_next)
            veh1.set_state_global(xglob_next)
            tmp = tmp + 1
        else:
            pass
        r.sleep()


if __name__ == '__main__':
    try:
        set_vehicle1()
    except rospy.ROSInterruptException:
        pass
