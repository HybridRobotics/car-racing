import rospy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anim
from car_racing_sim.msg import VehicleControl, VehicleState
import vehicle_dynamics, base, racing_env


# real-time controller
class ControlRealtimeBase:
    def __init__(self):
        self.__sub_state_curv = None
        self.__pub_input = None

    def __state_cb(self, msg):
        self.x[0] = msg.state_curv.vx
        self.x[1] = msg.state_curv.vy
        self.x[2] = msg.state_curv.wz
        self.x[3] = msg.state_curv.epsi
        self.x[4] = msg.state_curv.s
        self.x[5] = msg.state_curv.ey

    def set_subscriber(self):
        self.__sub_state = rospy.Subscriber(
            'simulator/vehicle1/state', VehicleState, self.__state_cb)


class PIDTrackingRealtime(base.PIDTracking, ControlRealtimeBase):
    def __init__(self, vt=0.6, eyt=0.0):
        base.PIDTracking.__init__(self, vt, eyt)
        ControlRealtimeBase.__init__(self)


class MPCTrackingRealtime(base.MPCTracking, ControlRealtimeBase):
    def __init__(self, matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6, eyt=0.0):
        base.MPCTracking.__init__(
            self, matrix_A, matrix_B, matrix_Q, matrix_R, vt, eyt)
        ControlRealtimeBase.__init__(self)


class MPCCBFRacingRealtime(base.MPCCBFRacing, ControlRealtimeBase):
    def __init__(self, matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6, eyt=0.0):
        base.MPCCBFRacing.__init__(
            self, matrix_A, matrix_B, matrix_Q, matrix_R, vt, eyt)
        ControlRealtimeBase.__init__(self)


# real-time dynamic model
class ModelRealtimeBase:
    def __init__(self):
        self.__sub_input = None
        self.__pub_state = None

    def __input_cb(self, msg):
        self.u[1] = msg.acc
        self.u[0] = msg.delta

    def set_subscriber(self):
        self.__sub_input = rospy.Subscriber(
            'vehicle1/input', VehicleControl, self.__input_cb)


class DynamicBicycleModelRealtime(base.DynamicBicycleModel, ModelRealtimeBase):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None):
        base.DynamicBicycleModel.__init__(
            self, name=name, param=param)
        ModelRealtimeBase.__init__(self)


# real-time simulator
class CarRacingSimRealtime(base.CarRacingSim):
    def __init__(self):
        base.CarRacingSim.__init__(self)
        self.__sub_state = None
        self.__pub_state = None
        self.vehicle_state_glob = None
        self.vehicle_state_curv = None

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
        self.__sub_state = rospy.Subscriber(
            'vehicle1/state', VehicleState, self.__state_cb)


# real-time visualization
class VisualizationRealtime:
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

    def update(self, i):
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
        self.__sub_state = rospy.Subscriber(
            'simulator/vehicle1/state', VehicleState, self.__state_cb)

    def plot_track(self, ax):
        num_sampling_per_meter = 100
        num_track_points = int(
            np.floor(num_sampling_per_meter * self.track.lap_length))
        points_out = np.zeros((num_track_points, 2))
        points_center = np.zeros((num_track_points, 2))
        points_in = np.zeros((num_track_points, 2))
        for i in range(0, num_track_points):
            points_out[i, :] = self.track.get_global_position(
                i / float(num_sampling_per_meter), self.track.width)
            points_center[i, :] = self.track.get_global_position(
                i / float(num_sampling_per_meter), 0.0)
            points_in[i, :] = self.track.get_global_position(
                i / float(num_sampling_per_meter), -self.track.width)
        ax.plot(points_center[:, 0], points_center[:, 1], "--r")
        ax.plot(points_in[:, 0], points_in[:, 1], "-b")
        ax.plot(points_out[:, 0], points_out[:, 1], "-b")

    def get_vehicle_in_rectangle(self):
        car_dx = 0.5 * 0.4
        car_dy = 0.5 * 0.2
        car_xs_origin = [car_dx, car_dx, -car_dx, -car_dx, car_dx]
        car_ys_origin = [car_dy, -car_dy, -car_dy, car_dy, car_dy]
        car_frame = np.vstack(
            (np.array(car_xs_origin), np.array(car_ys_origin)))
        x = self.vehicle_state_glob[4]
        y = self.vehicle_state_glob[5]
        R = np.matrix([[np.cos(self.vehicle_state_glob[3]), -np.sin(self.vehicle_state_glob[3])],
                      [np.sin(self.vehicle_state_glob[3]), np.cos(self.vehicle_state_glob[3])]])
        rotated_car_frame = R * car_frame
        return x+rotated_car_frame[0, 2], y+rotated_car_frame[1, 2], 2*car_dx, 2*car_dy, self.vehicle_state_glob[3]*180/3.14
