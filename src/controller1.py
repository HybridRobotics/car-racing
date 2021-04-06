#!/usr/bin/env python
import numpy as np
import rospy
import racing_car, policy, vehicle_dynamics, racing_env
from car_racing_dev.msg import VehicleControl, VehicleState


def set_controller1():
    rospy.init_node("controller1", anonymous = True)
    track_spec = np.genfromtxt("/home/suiyihe/catkin_ws/src/car_racing_dev/data/track_spec/default.csv", delimiter=",")
    track_width = 1.0
    track = racing_env.ClosedTrack(track_spec, track_width)

    loop_rate = 20
    timestep = 1 / loop_rate

    matrix_A = np.genfromtxt("/home/suiyihe/catkin_ws/src/car_racing_dev/data/sys/LTI/matrix_A.csv", delimiter=",")
    matrix_B = np.genfromtxt("/home/suiyihe/catkin_ws/src/car_racing_dev/data/sys/LTI/matrix_B.csv", delimiter=",")
    matrix_Q = np.diag([10.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    matrix_R = np.diag([0.1, 0.1])
    controller1 = policy.RealtimeMPCTracking(matrix_A, matrix_B, matrix_Q, matrix_R, vt = 0.8)
    #controller1 = policy.RealtimePIDTracking(vt = 0.8)
    controller1.set_timestep(timestep)
    controller1.set_state(np.zeros(6,))
    controller1.u = np.zeros(2)   
    controller1.set_subscriber()
    controller1.__pub_input = rospy.Publisher('vehicle1/input', VehicleControl, queue_size = 10)

    while not rospy.is_shutdown():
        controller1.calc_input()
        u = controller1.get_input()
        controller1.__pub_input.publish(VehicleControl(u[1],u[0]))


if __name__ == '__main__':
    try:
        set_controller1()
    except rospy.ROSInterruptException:
        pass