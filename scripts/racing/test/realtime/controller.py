#!/usr/bin/env python
import rospy
import numpy as np
import time
import datetime
from car_racing.msg import VehicleControl, VehicleState
from scripts.utils import base
from scripts.control import lmpc_helper
from scripts.utils.constants import *
from scripts.racing import realtime


def set_controller(args):
    veh_name = args["veh_name"]
    ctrl_policy = args["ctrl_policy"]
    node_name = "controller_" + veh_name
    rospy.init_node(node_name, anonymous=True)
    loop_rate = 10
    timestep = 1 / loop_rate
    matrix_A = np.genfromtxt("data/sys/LTI/matrix_A.csv", delimiter=",")
    matrix_B = np.genfromtxt("data/sys/LTI/matrix_B.csv", delimiter=",")
    matrix_Q = np.diag([10.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    matrix_R = np.diag([0.1, 0.1])
    mpc_lti_param = base.MPCTrackingParam(matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.8)
    if ctrl_policy == "mpc-lti":
        ctrl = realtime.MPCTracking(mpc_lti_param)
        ctrl.set_subscriber_track()
    elif ctrl_policy == "pid":
        ctrl = realtime.PIDTracking(vt=0.8)
        ctrl.set_subscriber_track()
    elif ctrl_policy == "mpc-cbf":
        mpc_cbf_param = base.MPCCBFRacingParam(matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.8)
        ctrl = realtime.MPCCBFRacing(mpc_cbf_param)
        ctrl.agent_name = veh_name
        ctrl.set_subscriber_track()
        ctrl.set_subscriber_veh()
    elif ctrl_policy == "lmpc":
        laps_number = 100
        N = 12
        num_ss_it = 2
        num_ss_points = 32 + N
        shift = 0
        points_lmpc = 5000
        time_lmpc = points_lmpc * timestep
        # Cost on the slack variable for the terminal constraint
        matrix_Qslack = 5 * np.diag([10, 1, 1, 1, 10, 1])
        # State cost x = [vx, vy, wz, epsi, s, ey]
        matrix_Q_LMPC = 0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Input cost u = [delta, a]
        matrix_R_LMPC = 1 * np.diag([1.0, 1.0])
        # Input rate cost u
        matrix_dR_LMPC = 5 * np.diag([1.0, 1.0])
        lmpc_param = base.LMPCRacingParam(
            num_ss_points,
            num_ss_it,
            N,
            matrix_Qslack,
            matrix_Q_LMPC,
            matrix_R_LMPC,
            matrix_dR_LMPC,
            shift,
            timestep,
            laps_number,
            time_lmpc,
        )

        lmpc_ctrl = realtime.LMPCRacingGame(lmpc_param)
        lmpc_ctrl.openloop_prediction_lmpc = lmpc_helper.lmpc_prediction(
            N, points_lmpc, num_ss_points, laps_number
        )
        lmpc_ctrl.set_subscriber_track()
        lmpc_ctrl.set_timestep(timestep)
        lmpc_ctrl.u = np.zeros(2)
        pid_ctrl = realtime.PIDTracking(vt=0.8)
        pid_ctrl.set_subscriber_track()
        pid_ctrl.set_timestep(timestep)
        pid_ctrl.u = np.zeros(2)
        mpc_lti_ctrl = realtime.MPCTracking(mpc_lti_param)
        mpc_lti_ctrl.set_subscriber_track()
        mpc_lti_ctrl.set_timestep(timestep)
        mpc_lti_ctrl.u = np.zeros(2)
        veh_input_topic = veh_name + "/input"
        lmpc_ctrl.set_vehicles_track()
        pid_ctrl.__pub_input = rospy.Publisher(veh_input_topic, VehicleControl, queue_size=10)
        mpc_lti_ctrl.__pub_input = rospy.Publisher(veh_input_topic, VehicleControl, queue_size=10)
        lmpc_ctrl.__pub_input = rospy.Publisher(veh_input_topic, VehicleControl, queue_size=10)
        current_lap = 0
    else:
        pass
    if ctrl_policy == "lmpc":
        pid_ctrl.set_subscriber_state(veh_name)
        lap_start_timer = datetime.datetime.now()
    else:
        ctrl.set_subscriber_optimal_traj()
        ctrl.set_timestep(timestep)
        ctrl.set_state(
            np.zeros((X_DIM,)),
            np.zeros((X_DIM,)),
        )
        ctrl.u = np.zeros((U_DIM,))
        ctrl.set_subscriber_state(veh_name)
        veh_input_topic = veh_name + "/input"
        ctrl.__pub_input = rospy.Publisher(veh_input_topic, VehicleControl, queue_size=10)
    r = rospy.Rate(loop_rate)
    start_timer = datetime.datetime.now()
    tmp = 0

    while not rospy.is_shutdown():
        current_time = datetime.datetime.now()
        if (current_time - start_timer).total_seconds() >= (tmp + 1) * (1 / loop_rate):
            if ctrl_policy == "lmpc":
                # for lmpc, for the first lap, the ego vehicle will use pid to track the center line and store the data
                if current_lap == 0:
                    # determine, if the first lap is finished
                    pid_ctrl.calc_input()
                    u = pid_ctrl.get_input()
                    pid_ctrl.__pub_input.publish(VehicleControl(u[1], u[0]))
                    time.sleep(0.05)
                    pid_ctrl.update_memory(current_lap)
                    if pid_ctrl.x[4] >= (current_lap + 1) * pid_ctrl.lap_length:
                        lmpc_ctrl.add_trajectory(
                            pid_ctrl.times,
                            pid_ctrl.timestep,
                            pid_ctrl.xcurvs,
                            pid_ctrl.xglobs,
                            pid_ctrl.inputs,
                            0,
                        )
                        end_timer = datetime.datetime.now()
                        delta_time = (end_timer - lap_start_timer).total_seconds()
                        print(
                            "lap:",
                            current_lap + 1,
                            ",lap time: {}".format(delta_time),
                        )

                        lap_start_timer = datetime.datetime.now()
                        mpc_lti_ctrl.set_subscriber_state(veh_name)
                        # time.sleep(0.01)
                        current_lap = current_lap + 1
                    else:
                        pass
                elif current_lap == 1:
                    mpc_lti_ctrl.calc_input()
                    u = mpc_lti_ctrl.get_input()
                    mpc_lti_ctrl.__pub_input.publish(VehicleControl(u[1], u[0]))
                    time.sleep(0.05)
                    mpc_lti_ctrl.update_memory(current_lap)
                    if mpc_lti_ctrl.x[4] >= (current_lap + 1) * mpc_lti_ctrl.lap_length:
                        lmpc_ctrl.add_trajectory(
                            mpc_lti_ctrl.times,
                            mpc_lti_ctrl.timestep,
                            mpc_lti_ctrl.xcurvs,
                            mpc_lti_ctrl.xglobs,
                            mpc_lti_ctrl.inputs,
                            0,
                        )
                        end_timer = datetime.datetime.now()
                        delta_time = (end_timer - lap_start_timer).total_seconds()
                        print(
                            "lap:",
                            current_lap + 1,
                            ",lap time: {}".format(delta_time),
                        )
                        lap_start_timer = datetime.datetime.now()
                        lmpc_ctrl.set_subscriber_state(veh_name)
                        # time.sleep(0.01)
                        print("start lmpc")
                        current_lap = current_lap + 1
                    else:
                        pass
                else:
                    lmpc_ctrl.calc_input()
                    u = lmpc_ctrl.get_input()
                    lmpc_ctrl.__pub_input.publish(VehicleControl(u[1], u[0]))
                    time.sleep(0.02)
                    lmpc_ctrl.update_memory(current_lap)
                    if lmpc_ctrl.x[4] >= (current_lap + 1) * lmpc_ctrl.lap_length:
                        lmpc_ctrl.add_trajectory(
                            lmpc_ctrl.times,
                            lmpc_ctrl.timestep,
                            lmpc_ctrl.xcurvs,
                            lmpc_ctrl.xglobs,
                            lmpc_ctrl.inputs,
                            current_lap - 2,
                        )
                        end_timer = datetime.datetime.now()
                        delta_time = (end_timer - lap_start_timer).total_seconds()
                        print(
                            "lap:",
                            current_lap + 1,
                            ",lap time: {}".format(delta_time),
                        )
                        lap_start_timer = datetime.datetime.now()
                        current_lap = current_lap + 1
                    else:
                        pass
            else:
                ctrl.calc_input()
                u = ctrl.get_input()
                ctrl.__pub_input.publish(VehicleControl(u[1], u[0]))
            tmp = tmp + 1
        else:
            pass
        r.sleep()


if __name__ == "__main__":
    try:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--veh-name", type=str)
        parser.add_argument("--ctrl-policy", type=str)
        args = vars(parser.parse_args())
        set_controller(args)
    except rospy.ROSInterruptException:
        pass
