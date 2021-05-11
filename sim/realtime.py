import rospy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import time
from car_racing.msg import VehicleControl, VehicleState, VehicleList, NumVehicle, TrackInfo, OptimalTraj
from utils import vehicle_dynamics, base, ctrl, racing_env
import copy


# real-time dynamic model
class ModelRealtimeBase:
    def __init__(self):
        # track relevant attributes
        self.lap_length = None
        self.lap_width = None
        self.point_and_tangent = None
        self.track_layout = None
        # visualization relevant attributes
        self.ax = None
        self.ani = None
        self.patch = None
        # subscirber, get (other) vehicle's state, input and track information
        self.__sub_input = None
        self.__sub_state = None
        self.__sub_track = None
        # publisher and msg, publish the vehicle's state
        self.__pub_state = None
        self.msg_state = VehicleState()

    # track call back function, get track information
    def __track_cb(self, msg):
        size1 = msg.size
        size0 = int(np.size(msg.point_and_tangent)/size1)
        self.point_and_tangent = np.zeros((size0, size1))
        tmp = 0
        # get matrix from 1D array
        for index_1 in range(size1):
            for index_0 in range(size0):
                self.point_and_tangent[index_0,
                                       index_1] = msg.point_and_tangent[tmp]
                tmp = tmp + 1
        self.lap_length = msg.length
        self.lap_width = msg.width
        self.track_layout = msg.track_layout

    def set_subscriber_track(self):
        self.__sub_track = rospy.Subscriber(
            'track_info', TrackInfo, self.__track_cb)
        # wait for the track info
        if self.lap_length == None:
            time.sleep(0.2)
        else:
            pass

    # vehicle input call back function, update the vehicle's input
    def __input_cb(self, msg):
        if self.u is None:
            self.u = np.zeros(2)
        self.u[1] = msg.acc
        self.u[0] = msg.delta

    def set_subscriber_input(self, veh_name):
        tmp = veh_name+'/input'
        self.__sub_input = rospy.Subscriber(
            tmp, VehicleControl, self.__input_cb)


class DynamicBicycleModelRealtime(base.DynamicBicycleModel, ModelRealtimeBase):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None):
        base.DynamicBicycleModel.__init__(
            self, name=name, param=param)
        ModelRealtimeBase.__init__(self)

    # vehicle state call back function, get the vehicle's state
    def __state_cb(self, msg):
        self.xcurv[0] = msg.state_curv.vx
        self.xcurv[1] = msg.state_curv.vy
        self.xcurv[2] = msg.state_curv.wz
        self.xcurv[3] = msg.state_curv.epsi
        self.xcurv[4] = msg.state_curv.s
        self.xcurv[5] = msg.state_curv.ey

        self.xglob[0] = msg.state_glob.vx
        self.xglob[1] = msg.state_glob.vy
        self.xglob[2] = msg.state_glob.wz
        self.xglob[3] = msg.state_glob.psi
        self.xglob[4] = msg.state_glob.x
        self.xglob[5] = msg.state_glob.y

    # get vehicle's input from controller
    def set_subscriber_ctrl(self, veh_name):
        tmp = 'simulator/'+veh_name + '/state'
        self.__sub_state = rospy.Subscriber(
            tmp, VehicleState, self.__state_cb)

    # get vehicle's state from simulator
    def set_subscriber_sim(self, veh_name):
        tmp = veh_name + '/state'
        self.__sub_state = rospy.Subscriber(
            tmp, VehicleState, self.__state_cb)

    # initialization function for animation
    def init(self):
        self.ax.add_patch(self.patch)
        return self.patch,

    # update function for animation, the rectangle's parameters are updated through call back function
    def update(self, i):
        return self.patch,

    # update the vehicle's state in visualizaiton node
    def __state_cb_visual(self, msg):
        # check if the information of the vehicle is update
        if (self.xglob[4] == msg.state_glob.x) and (self.xglob[5] == msg.state_glob.y) and (self.xglob[3] == msg.state_glob.psi):
            self.patch.set_width(0)
            self.patch.set_height(0)
            rospy.logerr('No update information for %s', msg.name)
        else:
            self.xglob[0] = msg.state_glob.vx
            self.xglob[1] = msg.state_glob.vy
            self.xglob[2] = msg.state_glob.wz
            self.xglob[3] = msg.state_glob.psi
            self.xglob[4] = msg.state_glob.x
            self.xglob[5] = msg.state_glob.y
            # update the rectangle information of animation
            x_car, y_car, width_car, height_car, angle_car = self.get_vehicle_in_rectangle(
                self.xglob, self.param)
            self.patch.set_xy([x_car, y_car])
            self.patch.angle = angle_car
            self.patch.set_width(width_car)
            self.patch.set_height(height_car)

    def set_subscriber_visual(self, veh_name):
        tmp = 'simulator/'+veh_name + '/state'
        self.__sub_state = rospy.Subscriber(
            tmp, VehicleState, self.__state_cb_visual)

    # show the vehicle with a rectangle
    def get_vehicle_in_rectangle(self, vehicle_state_glob, veh_param):
        car_length = veh_param.length
        car_width = veh_param.width
        car_dx = 0.5 * car_length
        car_dy = 0.5 * car_width
        car_xs_origin = [car_dx, car_dx, -car_dx, -car_dx, car_dx]
        car_ys_origin = [car_dy, -car_dy, -car_dy, car_dy, car_dy]
        car_frame = np.vstack(
            (np.array(car_xs_origin), np.array(car_ys_origin)))
        x = vehicle_state_glob[4]
        y = vehicle_state_glob[5]
        R = np.matrix([[np.cos(vehicle_state_glob[3]), -np.sin(vehicle_state_glob[3])],
                      [np.sin(vehicle_state_glob[3]), np.cos(vehicle_state_glob[3])]])
        rotated_car_frame = R * car_frame
        return x+rotated_car_frame[0, 2], y+rotated_car_frame[1, 2], 2*car_dx, 2*car_dy, vehicle_state_glob[3]*180/3.14

    # in this estimation, the vehicles is assumed to move with input is equal to zero
    def get_estimation(self, xglob, xcurv):
        curv = racing_env.get_curvature(
            self.lap_length, self.point_and_tangent, xcurv[4])
        xcurv_est = np.zeros(6)
        xglob_est = np.zeros(6)
        xcurv_est[0:3] = xcurv[0:3]
        xcurv_est[3] = xcurv[3] + self.timestep * (xcurv[2]-(xcurv[0]*np.cos(
            xcurv[3])-xcurv[1]*np.sin(xcurv[3]))/(1-curv*xcurv[5])*curv)
        xcurv_est[4] = xcurv[4] + self.timestep * \
            ((xcurv[0]*np.cos(xcurv[3])-xcurv[1]
             * np.sin(xcurv[3]))/(1-curv*xcurv[5]))
        xcurv_est[5] = xcurv[5] + self.timestep * \
            (xcurv[0]*np.sin(xcurv[3])+xcurv[1]*np.cos(xcurv[3]))
        xglob_est[0:3] = xglob[0:3]
        xglob_est[3] = xglob[3] + self.timestep * (xglob[2])
        xglob_est[4] = xglob[4] + self.timestep * \
            (xglob[0]*np.cos(xglob[3])-xglob[1]*np.sin(xglob[3]))
        xglob_est[4] = xglob[4] + self.timestep * \
            (xglob[0]*np.sin(xglob[3])+xglob[1]*np.cos(xglob[3]))

        return xcurv_est, xglob_est

    # get prediction for mpc-cbf controller
    def get_trajectory_nsteps(self, n):
        xcurv_nsteps = np.zeros((6, n))
        xglob_nsteps = np.zeros((6, n))
        for index in range(n):
            if index == 0:
                xcurv_est, xglob_est = self.get_estimation(
                    self.xglob, self.xcurv)
            else:
                xcurv_est, xglob_est = self.get_estimation(
                    xglob_nsteps[:, index-1], xcurv_nsteps[:, index-1])
            xcurv_nsteps[:, index] = xcurv_est
            xglob_nsteps[:, index] = xglob_est
        return xcurv_nsteps, xglob_nsteps


# real-time controller
class ControlRealtimeBase:
    def __init__(self):
        # track information
        self.lap_length = None
        self.lap_width = None
        self.point_and_tangent = None
        self.track_layout = None
        # optimal trajectory
        self.opti_traj_xcurv = None
        self.opti_size = None
        # subscriber for vehicle's state (in Frenet coordinate) and track
        self.__sub_state_curv = None
        self.__sub_track = None
        self.__sub_optimal_traj = None
        # publisher for vehicle's input
        self.__pub_input = None
        # indicate the realtime simulator
        self.realtime_flag = True
        self.xglob = np.zeros(6)

    def __track_cb(self, msg):
        size1 = msg.size
        size0 = int(np.size(msg.point_and_tangent)/size1)
        self.point_and_tangent = np.zeros((size0, size1))
        tmp = 0
        for index_1 in range(size1):
            for index_0 in range(size0):
                self.point_and_tangent[index_0,
                                       index_1] = msg.point_and_tangent[tmp]
                tmp = tmp + 1
        self.lap_length = msg.length
        self.lap_width = msg.width
        self.track_layout = msg.track_layout

    def set_subscriber_track(self):
        self.__sub_track = rospy.Subscriber(
            'track_info', TrackInfo, self.__track_cb)
        if self.lap_length == None:
            time.sleep(0.1)
        else:
            pass

    def __optimal_traj_cb(self, msg):
        size = msg.size
        self.opti_size = msg.size
        self.opti_traj_xcurv = np.zeros((size, 6))
        tmp = 0
        for index in range(size):
            for index_1 in range(6):
                self.opti_traj_xcurv[index, index_1] = msg.list_xcurv[tmp]
                tmp = tmp + 1

    def set_subscriber_optimal_traj(self):
        self.__sub_optimal_traj = rospy.Subscriber('optimal_traj', OptimalTraj, self.__optimal_traj_cb)
        if self.opti_traj_xcurv == None:
            time.sleep(0.1)
        else:
            pass

    def __track_cb(self, msg):
        size1 = msg.size
        size0 = int(np.size(msg.point_and_tangent)/size1)
        self.point_and_tangent = np.zeros((size0, size1))
        tmp = 0
        for index_1 in range(size1):
            for index_0 in range(size0):
                self.point_and_tangent[index_0,
                                       index_1] = msg.point_and_tangent[tmp]
                tmp = tmp + 1
        self.lap_length = msg.length
        self.lap_width = msg.width
        self.track_layout = msg.track_layout

    def set_subscriber_track(self):
        self.__sub_track = rospy.Subscriber(
            'track_info', TrackInfo, self.__track_cb)
        if self.lap_length == None:
            time.sleep(0.1)
        else:
            pass

    def __optimal_traj_cb(self, msg):
        size = msg.size
        self.opti_size = msg.size
        self.opti_traj_xcurv = np.zeros((size, 6))
        tmp = 0
        for index in range(size):
            for index_1 in range(6):
                self.opti_traj_xcurv[index, index_1] = msg.list_xcurv[tmp]
                tmp = tmp + 1

    def set_subscriber_optimal_traj(self):
        self.__sub_optimal_traj = rospy.Subscriber('optimal_traj', OptimalTraj, self.__optimal_traj_cb)
        if self.opti_traj_xcurv == None:
            time.sleep(0.1)
        else:
            pass

    def __track_cb(self, msg):
        size1 = msg.size
        size0 = int(np.size(msg.point_and_tangent)/size1)
        self.point_and_tangent = np.zeros((size0, size1))
        tmp = 0
        for index_1 in range(size1):
            for index_0 in range(size0):
                self.point_and_tangent[index_0,
                                       index_1] = msg.point_and_tangent[tmp]
                tmp = tmp + 1
        self.lap_length = msg.length
        self.lap_width = msg.width

    def set_subscriber_track(self):
        self.__sub_track = rospy.Subscriber(
            'track_info', TrackInfo, self.__track_cb)
        if self.lap_length == None:
            time.sleep(0.1)
        else:
            pass

    def __state_cb(self, msg):
        if self.x is None:
            self.x = np.zeros(6,)
            self.xglob = np.zeros(6,)
            self.x[0] = msg.state_curv.vx
            self.x[1] = msg.state_curv.vy
            self.x[2] = msg.state_curv.wz
            self.x[3] = msg.state_curv.epsi
            self.x[4] = msg.state_curv.s
            self.x[5] = msg.state_curv.ey
            self.xglob[0] = msg.state_glob.vx
            self.xglob[1] = msg.state_glob.vy
            self.xglob[2] = msg.state_glob.wz
            self.xglob[3] = msg.state_glob.psi
            self.xglob[4] = msg.state_glob.x
            self.xglob[5] = msg.state_glob.y
            # for initial state, it should be at the first lap for the corresponding controller
            x = copy.deepcopy(self.x)
            while x[4]>self.lap_length:
                x[4] = x[4] - self.lap_length
            self.traj_xcurv.append(x)
            self.traj_xglob.append(self.xglob)
        else:
            self.x[0] = msg.state_curv.vx
            self.x[1] = msg.state_curv.vy
            self.x[2] = msg.state_curv.wz
            self.x[3] = msg.state_curv.epsi
            self.x[4] = msg.state_curv.s
            self.x[5] = msg.state_curv.ey
            self.xglob[0] = msg.state_glob.vx
            self.xglob[1] = msg.state_glob.vy
            self.xglob[2] = msg.state_glob.wz
            self.xglob[3] = msg.state_glob.psi
            self.xglob[4] = msg.state_glob.x
            self.xglob[5] = msg.state_glob.y
            
    def set_subscriber_state(self, veh_name):
        tmp = 'simulator/'+veh_name+'/state'
        self.__sub_state = rospy.Subscriber(
            tmp, VehicleState, self.__state_cb)

class PIDTrackingRealtime(base.PIDTracking, ControlRealtimeBase):
    def __init__(self, vt=0.6, eyt=0.0):
        base.PIDTracking.__init__(self, vt, eyt)
        ControlRealtimeBase.__init__(self)

class MPCTrackingRealtime(base.MPCTracking, ControlRealtimeBase):
    def __init__(self, matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6, eyt=0.0):
        base.MPCTracking.__init__(
            self, matrix_A, matrix_B, matrix_Q, matrix_R, vt, eyt)
        ControlRealtimeBase.__init__(self)

class LMPCRacingRealtime(base.LMPCRacing, ControlRealtimeBase):
    def __init__(self, num_ss_points, num_ss_it, N, matrix_Qslack, matrix_Q_LMPC, matrix_R_LMPC, matrix_dR_LMPC, xdim, udim, shift, timestep, laps, time_lmpc):
        base.LMPCRacing.__init__(self, num_ss_points, num_ss_it, N, matrix_Qslack, matrix_Q_LMPC, matrix_R_LMPC, matrix_dR_LMPC, xdim, udim, shift, timestep, laps, time_lmpc)
        ControlRealtimeBase.__init__(self)

class MPCCBFRacingRealtime(base.MPCCBFRacing, ControlRealtimeBase):
    def __init__(self, matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6, eyt=0.0):
        base.MPCCBFRacing.__init__(
            self, matrix_A, matrix_B, matrix_Q, matrix_R, vt, eyt)
        ControlRealtimeBase.__init__(self)
        # vehicle's list
        self.vehicles = {}
        self.num_veh = None
        # get vehicle's list from simulator node
        self.__sub_num_veh = None
        self.__sub_veh_list = None

    def __sub_num_veh_cb(self, msg):
        self.num_veh = msg.num

    def __sub_veh_list_cb(self, msg):
        if self.num_veh == None:
            pass
        else:
            for index in range(self.num_veh):
                name = msg.vehicle_list[index]
                # ego vehicle
                if name == self.agent_name:
                    self.vehicles[name] = DynamicBicycleModelRealtime(
                        name=name, param=base.CarParam())
                    self.vehicles[name].name = self.agent_name
                    self.vehicles[name].xglob = np.zeros(6)
                    self.vehicles[name].xcurv = np.zeros(6)
                # other vehicle
                else:
                    self.vehicles[name] = DynamicBicycleModelRealtime(
                        name=name, param=base.CarParam())
                    self.vehicles[name].xglob = np.zeros(6)
                    self.vehicles[name].xcurv = np.zeros(6)
                    self.vehicles[name].timestep = self.timestep
                    self.vehicles[name].lap_length = self.lap_length
                    self.vehicles[name].lap_width = self.lap_width
                    self.vehicles[name].point_and_tangent = self.point_and_tangent
                    self.vehicles[name].set_subscriber_ctrl(name)

    def set_subscriber_veh(self):
        self.__sub_num_veh = rospy.Subscriber(
            'vehicle_num', NumVehicle, self.__sub_num_veh_cb)
        self.__sub_veh_list = rospy.Subscriber(
            'vehicle_list', VehicleList, self.__sub_veh_list_cb)

# real-time simulator
class CarRacingSimRealtime(base.CarRacingSim):
    def __init__(self):
        base.CarRacingSim.__init__(self)
        self.num_vehicle = 0
        # publisher for vehicle's list and track information
        self.__pub_veh_list = None
        self.__pub_veh_num = None
        self.__pub_track = None
        self.__pub_optimal_traj = None

    # add new vehicle in vehicle list
    def add_vehicle(self, req):
        self.vehicles[req.name] = DynamicBicycleModelRealtime(
            name=req.name, param=base.CarParam())
        self.vehicles[req.name].xglob = np.zeros(6)
        self.vehicles[req.name].xcurv = np.zeros(6)
        self.vehicles[req.name].msg_state.name = req.name
        self.vehicles[req.name].set_subscriber_sim(req.name)
        self.num_vehicle = self.num_vehicle + 1
        return 1

# real-time visualization
class VisualizationRealtime:
    def __init__(self):
        # visualization relevant attributes
        self.patch = None
        self.ax = None
        self.fig = None
        # vehicle list
        self.num_vehicle = 0
        self.vehicles = {}
        # track information
        self.lap_length = None
        self.lap_width = None
        self.point_and_tangent = None
        self.track_layout = None
        # optimal trajectory
        self.opti_traj_xglob = None
        # subscriber for track and optimal trajectory
        self.__sub_track = None
        self.__sub_optimal_traj = None

    def __track_cb(self, msg):
        size1 = msg.size
        size0 = int(np.size(msg.point_and_tangent)/size1)
        self.point_and_tangent = np.zeros((size0, size1))
        tmp = 0
        for index_1 in range(size1):
            for index_0 in range(size0):
                self.point_and_tangent[index_0,
                                       index_1] = msg.point_and_tangent[tmp]
                tmp = tmp + 1
        self.lap_length = msg.length
        self.lap_width = msg.width
        self.track_layout = msg.track_layout

    def set_subscriber_track(self):
        self.__sub_track = rospy.Subscriber(
            'track_info', TrackInfo, self.__track_cb)
        if self.lap_length == None:
            time.sleep(0.1)
        else:
            pass

    def __optimal_traj_cb(self, msg):
        size = msg.size
        self.opti_traj_xglob = np.zeros((size, 6))
        tmp = 0
        for index  in range(size):
            for index_1 in range(6):
                self.opti_traj_xglob[index, index_1] = msg.list_xglob[tmp]
                tmp = tmp + 1
    
    def set_subscriber_optimal_traj(self):
        self.__sub_optimal_traj = rospy.Subscriber('optimal_traj', OptimalTraj, self.__optimal_traj_cb)
        if self.opti_traj_xglob == None:
            time.sleep(0.1)
        else:
            pass

    def set_ax(self, ax):
        self.ax = ax

    def add_vehicle(self, req):
        self.vehicles[req.name] = DynamicBicycleModelRealtime(
            name=req.name, param=base.CarParam())
        self.vehicles[req.name].ax = self.ax
        self.vehicles[req.name].xglob = np.zeros(6)
        self.vehicles[req.name].xcurv = np.zeros(6)
        self.num_vehicle = self.num_vehicle + 1
        x_car, y_car, width_car, height_car, angle_car = self.get_vehicle_in_rectangle(
            self.vehicles[req.name].xglob, self.vehicles[req.name].param)
        self.vehicles[req.name].patch = patches.Rectangle(
            (x_car, y_car), width_car, height_car, angle_car, color=req.color)
        self.vehicles[req.name].set_subscriber_visual(req.name)
        self.vehicles[req.name].ani = animation.FuncAnimation(
            self.fig, self.vehicles[req.name].update, init_func=self.vehicles[req.name].init)
        return 1

    def init(self):
        self.plot_track(self.ax)

    def update(self, i):
        pass

    def plot_track(self, ax):
        num_sampling_per_meter = 100
        num_track_points = int(
            np.floor(num_sampling_per_meter * self.lap_length))
        points_out = np.zeros((num_track_points, 2))
        points_center = np.zeros((num_track_points, 2))
        points_in = np.zeros((num_track_points, 2))
        for i in range(0, num_track_points):
            points_out[i, :] = racing_env.get_global_position(self.lap_length, self.lap_width, self.point_and_tangent,
                                                              i / float(num_sampling_per_meter), self.lap_width)
            points_center[i, :] = racing_env.get_global_position(self.lap_length, self.lap_width, self.point_and_tangent,
                                                                 i / float(num_sampling_per_meter), 0.0)
            points_in[i, :] = racing_env.get_global_position(self.lap_length, self.lap_width, self.point_and_tangent,
                                                             i / float(num_sampling_per_meter), -self.lap_width)
        ax.plot(points_center[:, 0], points_center[:, 1], "--r")
        ax.plot(points_in[:, 0], points_in[:, 1], "-b")
        ax.plot(points_out[:, 0], points_out[:, 1], "-b")

    def get_vehicle_in_rectangle(self, vehicle_state_glob, veh_param):
        car_length = veh_param.length
        car_width = veh_param.width
        car_dx = 0.5 * car_length
        car_dy = 0.5 * car_width
        car_xs_origin = [car_dx, car_dx, -car_dx, -car_dx, car_dx]
        car_ys_origin = [car_dy, -car_dy, -car_dy, car_dy, car_dy]
        car_frame = np.vstack(
            (np.array(car_xs_origin), np.array(car_ys_origin)))
        x = vehicle_state_glob[4]
        y = vehicle_state_glob[5]
        R = np.matrix([[np.cos(vehicle_state_glob[3]), -np.sin(vehicle_state_glob[3])],
                      [np.sin(vehicle_state_glob[3]), np.cos(vehicle_state_glob[3])]])
        rotated_car_frame = R * car_frame
        return x+rotated_car_frame[0, 2], y+rotated_car_frame[1, 2], 2*car_dx, 2*car_dy, vehicle_state_glob[3]*180/3.14