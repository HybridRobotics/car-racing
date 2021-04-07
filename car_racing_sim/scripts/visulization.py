import numpy as np
import racing_env
import rospy
from car_racing_dev.msg import VehicleState


class Visulization:
    def __init__(self):
        self.__sub_state = None
        self.vehicle_state_glob = None
        self.vehicle_state_curv = None
        self.track = None
        self.patch = None
        self.ax = None

    def init(self):
        self.plot_track(self.ax)
        self.ax.add_patch(self.patch)
        return self.patch, 

    def update(self,i):
        x_car, y_car, width_car, height_car, angle_car = self.get_vehicle_in_rectangle()
        self.patch.set_xy([x_car, y_car])
        self.patch.angle = angle_car
        return self.patch,

    def set_track(self, track):
        self.track = track

    def set_state_glob(self, state):
        self.vehicle_state_glob = state
    
    def set_state_curv(self, state):
        self.vehicle_state_curv = state

    def __state_cb(self, msg):
        self.vehicle_state_curv[0] = msg.state_curv.vx
        self.vehicle_state_curv[1] = msg.state_curv.vy
        self.vehicle_state_curv[2] = msg.state_curv.wz
        self.vehicle_state_curv[3] = msg.state_curv.epsi
        self.vehicle_state_curv[4] = msg.state_curv.s
        self.vehicle_state_curv[5] = msg.state_curv.ey

        self.vehicle_state_glob[0] = msg.state_glob.vx
        self.vehicle_state_glob[1] = msg.state_glob.vy
        self.vehicle_state_glob[2] = msg.state_glob.wz
        self.vehicle_state_glob[3] = msg.state_glob.psi
        self.vehicle_state_glob[4] = msg.state_glob.x
        self.vehicle_state_glob[5] = msg.state_glob.y

    def set_subscriber(self):
        self.__sub_state = rospy.Subscriber('simulator/vehicle1/state', VehicleState, self.__state_cb)

    def plot_track(self, ax):
        num_sampling_per_meter = 100
        num_track_points = int(np.floor(num_sampling_per_meter * self.track.lap_length))
        points_out = np.zeros((num_track_points, 2))
        points_center = np.zeros((num_track_points, 2))
        points_in = np.zeros((num_track_points, 2))
        for i in range(0, num_track_points):
            points_out[i, :] = self.track.get_global_position(i / float(num_sampling_per_meter), self.track.width)
            points_center[i, :] = self.track.get_global_position(i / float(num_sampling_per_meter), 0.0)
            points_in[i, :] = self.track.get_global_position(i / float(num_sampling_per_meter), -self.track.width)
        ax.plot(points_center[:, 0], points_center[:, 1], "--r")
        ax.plot(points_in[:, 0], points_in[:, 1], "-b")
        ax.plot(points_out[:, 0], points_out[:, 1], "-b")

    def get_vehicle_in_rectangle(self):
        car_dx = 0.5 * 0.4
        car_dy = 0.5 * 0.2
        car_xs_origin = [car_dx, car_dx, -car_dx, -car_dx, car_dx]
        car_ys_origin = [car_dy, -car_dy, -car_dy, car_dy, car_dy]
        car_frame = np.vstack((np.array(car_xs_origin), np.array(car_ys_origin)))
        x = self.vehicle_state_glob[4]
        y = self.vehicle_state_glob[5]
        R = np.matrix([[np.cos(self.vehicle_state_glob[3]), -np.sin(self.vehicle_state_glob[3])], [np.sin(self.vehicle_state_glob[3]), np.cos(self.vehicle_state_glob[3])]])
        rotated_car_frame = R * car_frame
        return x+rotated_car_frame[0,2], y+rotated_car_frame[1,2],2*car_dx,2*car_dy, self.vehicle_state_glob[3]*180/3.14