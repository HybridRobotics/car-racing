#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import sys
tests_path = sys.path[0]
src_utils_path = tests_path+'/../../src/utils'
src_path = tests_path+'/../../src'
track_spec_path = tests_path + '/../../data/track_spec/default.csv'
sys.path.append(src_utils_path)
sys.path.append(src_path)
import racing_env, realtime
from car_racing_sim.msg import VehicleControl, VehicleState

def start_visualization():
    rospy.init_node("visualization", anonymous=True)
    track_spec = np.genfromtxt(track_spec_path, delimiter=",")
    track_width = 1.0
    track = racing_env.ClosedTrack(track_spec, track_width)
    visual1 = realtime.VisualizationRealtime()
    visual1.set_track(track)
    visual1.set_state_glob(np.zeros(6))
    visual1.set_state_curv(np.zeros(6))
    visual1.set_subscriber()

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_axis_off()
    ax.set_xlim(-6, 6)
    ax.set_ylim(-8, 4)

    visual1.ax = ax

    x_car, y_car, width_car, height_car, angle_car = visual1.get_vehicle_in_rectangle()
    visual1.patch = patches.Rectangle(
        (x_car, y_car), width_car, height_car, angle_car, color='blue')

    while not rospy.is_shutdown():
        ani = animation.FuncAnimation(
            fig, visual1.update, init_func=visual1.init)
        plt.show()


if __name__ == '__main__':
    try:
        start_visualization()
    except rospy.ROSInterruptException:
        pass
