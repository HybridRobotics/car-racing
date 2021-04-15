#!/usr/bin/env python
import rospy
import numpy as np
import sys
import time
scripts_realtime_path = sys.path[0]
src_utils_path = scripts_realtime_path+'/../../src/utils'
src_path = scripts_realtime_path+'/../../src'
matrix_A_path = scripts_realtime_path + '/../../data/sys/LTI/matrix_A.csv'
matrix_B_path = scripts_realtime_path + '/../../data/sys/LTI/matrix_B.csv'
sys.path.append(src_utils_path)
sys.path.append(src_path)
from car_racing.msg import VehicleControl, VehicleState
import racing_env, realtime


def set_controller(args):

    veh_name = args["veh_name"]
    ctrl_policy = args["ctrl_policy"]
    
    node_name = 'controller_' + veh_name
    rospy.init_node(node_name, anonymous=True)

    loop_rate = 20
    timestep = 1 / loop_rate

    matrix_A = np.genfromtxt(matrix_A_path, delimiter=",")
    matrix_B = np.genfromtxt(matrix_B_path, delimiter=",")
    matrix_Q = np.diag([10.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    matrix_R = np.diag([0.1, 0.1])
    if ctrl_policy == "mpc-lti":
        ctrl = realtime.MPCTrackingRealtime(
            matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.8)
    elif ctrl_policy == "pid":
        ctrl = realtime.PIDTrackingRealtime(vt=0.8)
    elif ctrl_policy == "mpc-cbf":
        ctrl = realtime.MPCCBFRacingRealtime(
            matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.8)
        ctrl.agent_name = veh_name
        ctrl.set_subscriber_track()
        ctrl.set_subscriber_veh()
    else:
        pass
    ctrl.set_timestep(timestep)
    ctrl.set_state(np.zeros(6,))
    ctrl.u = np.zeros(2)
    ctrl.set_subscriber_state(veh_name)
    veh_input_topic = veh_name + '/input'
    ctrl.__pub_input = rospy.Publisher(
        veh_input_topic, VehicleControl, queue_size=10)

    while not rospy.is_shutdown():
        ctrl.calc_input()
        u = ctrl.get_input()
        ctrl.__pub_input.publish(VehicleControl(u[1], u[0]))


if __name__ == '__main__':
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--veh-name", type=str)
        parser.add_argument("--ctrl-policy", type=str)
        args = vars(parser.parse_args())
        set_controller(args)

    except rospy.ROSInterruptException:
        pass
