#!/usr/bin/env python
import rospy
import numpy as np
import sys
tests_path = sys.path[0]
src_utils_path = tests_path+'/../../src/utils'
src_path = tests_path+'/../../src'
track_spec_path = tests_path + '/../../data/track_spec/default.csv'
matrix_A_path = tests_path + '/../../data/sys/LTI/matrix_A.csv'
matrix_B_path = tests_path + '/../../data/sys/LTI/matrix_B.csv'
sys.path.append(src_utils_path)
sys.path.append(src_path)
from car_racing_sim.msg import VehicleControl, VehicleState
import racing_env, realtime


def set_controller1(ctrl_policy):
    rospy.init_node("controller1", anonymous=True)
    track_spec = np.genfromtxt(track_spec_path, delimiter=",")
    track_width = 1.0
    track = racing_env.ClosedTrack(track_spec, track_width)

    loop_rate = 20
    timestep = 1 / loop_rate

    matrix_A = np.genfromtxt(matrix_A_path, delimiter=",")
    matrix_B = np.genfromtxt(matrix_B_path, delimiter=",")
    matrix_Q = np.diag([10.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    matrix_R = np.diag([0.1, 0.1])
    if ctrl_policy=="mpc-lti":
        ctrl1 = realtime.MPCTrackingRealtime(
            matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.8)
    else:
        ctrl1 = realtime.PIDTrackingRealtime(vt = 0.8)
    ctrl1.set_timestep(timestep)
    ctrl1.set_state(np.zeros(6,))
    ctrl1.u = np.zeros(2)
    ctrl1.set_subscriber()
    ctrl1.__pub_input = rospy.Publisher(
        'vehicle1/input', VehicleControl, queue_size=10)

    while not rospy.is_shutdown():
        ctrl1.calc_input()
        u = ctrl1.get_input()
        ctrl1.__pub_input.publish(VehicleControl(u[1], u[0]))


if __name__ == '__main__':
    try:
        ctrl_policy = rospy.get_param("ctrl_policy")
        set_controller1(ctrl_policy)
    except rospy.ROSInterruptException:
        pass
