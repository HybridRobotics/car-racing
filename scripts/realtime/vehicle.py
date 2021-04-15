#!/usr/bin/env python
import rospy
import numpy as np
import datetime
import sys
import time
scripts_realtime_path = sys.path[0]
src_utils_path = scripts_realtime_path+'/../../src/utils'
src_path = scripts_realtime_path+'/../../src'
sys.path.append(src_utils_path)
sys.path.append(src_path)
from car_racing.msg import VehicleControl, VehicleState, VehicleStateCurv, VehicleStateGlob, TrackInfo
import vehicle_dynamics, racing_env, base, realtime
from car_racing.srv import AddNewVehicle


def get_msg_xglob(state):
    return VehicleStateGlob(state[0], state[1], state[2], state[3], state[4], state[5])


def get_msg_xcurv(state):
    return VehicleStateCurv(state[0], state[1], state[2], state[3], state[4], state[5])

# call ros service to add the vehicle in simulator node
def add_vehicle_client_simulator(veh_name, veh_color):
    rospy.wait_for_service('add_vehicle_simulator')
    try:
        add_vehicle = rospy.ServiceProxy(
            'add_vehicle_simulator', AddNewVehicle)
        completed_flag = add_vehicle(veh_name, veh_color)
        return completed_flag
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


# call ros service to add the vehicle in visualization node
def add_vehicle_client_visualization(veh_name, veh_color):
    rospy.wait_for_service('add_vehicle_visualization')
    try:
        add_vehicle = rospy.ServiceProxy(
            'add_vehicle_visualization', AddNewVehicle)
        completed_flag = add_vehicle(veh_name, veh_color)
        return completed_flag
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def set_vehicle(args):
    veh_name = args["veh_name"]
    initial_xcurv = args["vx"], args["vy"], args["wz"], args["epsi"], args["s"], args["ey"]
    veh_color = args["color"]
    # initial a ros node for vehicle
    rospy.init_node(veh_name, anonymous=True)
    # call ros service to add vehicle in simulator and visualization node
    add_vehicle_client_simulator(veh_name, veh_color)
    add_vehicle_client_visualization(veh_name, veh_color)
    veh = realtime.DynamicBicycleModelRealtime(
        name=veh_name, param=base.CarParam())
    veh.set_subscriber_track()
    veh.u = np.zeros(2)
    veh.set_subscriber_input(veh_name)
    veh_state_topic = veh_name + '/state'
    veh.__pub_state = rospy.Publisher(
        veh_state_topic, VehicleState, queue_size=10)
    # get initial state in Frenet and X-Y coordinates
    veh.set_state_curvilinear(initial_xcurv)
    s0 = initial_xcurv[4]
    ey0 = initial_xcurv[5]
    veh.closedloop_xcurv.append(initial_xcurv)
    xglob0 = np.zeros(6)
    xglob0[0:3] = initial_xcurv[0:3]
    psi0 = racing_env.get_orientation(
        veh.lap_length, veh.lap_width, veh.point_and_tangent, s0, ey0)
    x0, y0 = racing_env.get_global_position(
        veh.lap_length, veh.lap_width, veh.point_and_tangent, s0, ey0)
    xglob0[3] = psi0
    xglob0[4] = x0
    xglob0[5] = y0
    veh.set_state_global(xglob0)
    veh.closedloop_xglob.append(xglob0)
    loop_rate = 100
    timestep = 1 / loop_rate
    veh.set_timestep(timestep)
    r = rospy.Rate(loop_rate)  # 100Hz
    start_timer = datetime.datetime.now()
    tmp = 0
    msg_state = VehicleState()
    msg_state.name = veh_name

    while not rospy.is_shutdown():
        current_time = datetime.datetime.now()
        # update the vehicle's state at 100 Hz
        if (current_time-start_timer).total_seconds() > (tmp+1)*(1/100):
            s = veh.xcurv[4]
            curv = racing_env.get_curvature(
                veh.lap_length, veh.lap_width, veh.point_and_tangent, s)
            xglob_next, xcurv_next = vehicle_dynamics.vehicle_dynamics(
                veh.param.dynamics_param, curv, veh.xglob, veh.xcurv, timestep, veh.u)
            veh.closedloop_xcurv.append(xcurv_next)
            veh.closedloop_xglob.append(xglob_next)
            msg_state.state_curv = get_msg_xcurv(xcurv_next)
            msg_state.state_glob = get_msg_xglob(xglob_next)
            veh.__pub_state.publish(msg_state)
            veh.set_state_curvilinear(xcurv_next)
            veh.set_state_global(xglob_next)
            tmp = tmp + 1
        else:
            pass
        r.sleep()


if __name__ == '__main__':
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--veh-name", type=str)
        parser.add_argument("--color", type=str)
        # get vehicle initial state in Frenet coordinate
        parser.add_argument("--vx", type=float)
        parser.add_argument("--vy", type=float)
        parser.add_argument("--wz", type=float)
        parser.add_argument("--epsi", type=float)
        parser.add_argument("--s", type=float)
        parser.add_argument("--ey", type=float)
        args = vars(parser.parse_args())
        set_vehicle(args)

    except rospy.ROSInterruptException:
        pass
