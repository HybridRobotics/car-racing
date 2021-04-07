#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import racing_env, visulization
import rospy
from car_racing_dev.msg import VehicleControl, VehicleState


def start_visulization():
    rospy.init_node("visulization", anonymous = True)
    track_spec = np.genfromtxt("/home/suiyihe/catkin_ws/src/car_racing_dev/data/track_spec/default.csv", delimiter=",")
    track_width = 1.0
    track = racing_env.ClosedTrack(track_spec, track_width)
    visulization1 = visulization.Visulization()
    visulization1.set_track(track)
    visulization1.set_state_glob(np.zeros(6))
    visulization1.set_state_curv(np.zeros(6))
    visulization1.set_subscriber()

    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.set_axis_off()
    ax.set_xlim(-6,6)
    ax.set_ylim(-8,4)

    visulization1.ax = ax

    x_car, y_car, width_car, height_car, angle_car = visulization1.get_vehicle_in_rectangle()       
    visulization1.patch  = patches.Rectangle((x_car,y_car),width_car, height_car,angle_car,color='blue')

    while not rospy.is_shutdown():
        ani = animation.FuncAnimation(fig, visulization1.update, init_func=visulization1.init)    
        plt.show()


if __name__ == '__main__':
    try:
        start_visulization()
    except rospy.ROSInterruptException:
        pass