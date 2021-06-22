import datetime
import numpy as np
import casadi as ca
from utils import lmpc_helper, racing_env
from casadi import *
from scipy import sparse
from scipy.sparse import vstack
from scipy.interpolate import interp1d
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class OvertakePlanner:
    def __init__(self, racing_game_param):
        self.racing_game_param = racing_game_param
        self.vehicles = None
        self.agent_name = None
        self.track = None
        self.opti_traj_xcurv = None

    def get_overtake_flag(self, xcurv_ego):
        overtake_flag = False
        for name in list(self.vehicles):
            if name != self.agent_name:
                x_other = copy.deepcopy(self.vehicles[name].xcurv)
                while x_other[4] > self.track.lap_length:
                    x_other[4] = x_other[4] - self.track.lap_length
                if abs(x_other[4] - xcurv_ego[4]) <= self.racing_game_param.planning_prediction_factor*self.vehicles[name].xcurv[0] or abs(x_other[4] + self.track.lap_length - xcurv_ego[4]) <= self.racing_game_param.planning_prediction_factor*self.vehicles[name].xcurv[0] or abs(x_other[4] - self.track.lap_length - xcurv_ego[4]) <= self.racing_game_param.planning_prediction_factor*self.vehicles[name].xcurv[0]:
                    overtake_flag = True
        return overtake_flag

    def get_local_path(self, xcurv_ego, time):
        start_timer = datetime.datetime.now()
        planning_prediction_factor = self.racing_game_param.planning_prediction_factor
        num_horizon_planner = self.racing_game_param.num_horizon_planner
        alpha = self.racing_game_param.alpha
        vehicles = self.vehicles
        track = self.track
        agent_name = self.agent_name
        optimal_traj_xcurv = self.opti_traj_xcurv
        bezier_order = self.racing_game_param.bezier_order
        safety_factor = self.racing_game_param.safety_factor
        obs_traj_list = []
        # the maximum and minimum ey's value of the obstacle's predicted trajectory
        max_ey_obs = None
        min_ey_obs = None
        num_veh = 0
        # save the obstacle's planned trajectory
        overtaken_name = None
        for name in list(vehicles):
            if name != agent_name:
                s_other = copy.deepcopy(vehicles[name].xcurv[4])
                while s_other > track.lap_length:
                    s_other = s_other - track.lap_length
                if abs(s_other - xcurv_ego[4]) <= planning_prediction_factor*vehicles[name].xcurv[0] or abs(s_other + track.lap_length - xcurv_ego[4]) <= planning_prediction_factor*vehicles[name].xcurv[0] or abs(s_other - track.lap_length - xcurv_ego[4]) <= planning_prediction_factor*vehicles[name].xcurv[0]:
                    s_veh = s_other
                    ey_veh = vehicles[name].xcurv[5]
                    if vehicles[name].no_dynamics:
                        obs_traj, _ = vehicles[name].get_trajectory_nsteps(
                            time, self.racing_game_param.timestep, num_horizon_planner+1)
                    else:
                        obs_traj, _ = vehicles[name].get_trajectory_nsteps(
                            num_horizon_planner+1)
                    obs_traj = obs_traj.T
                    obs_traj_list.append(obs_traj)
                    num_veh = num_veh + 1
                    if min_ey_obs is None:
                        min_ey_obs = obs_traj[0, 5]
                    if max_ey_obs is None:
                        max_ey_obs = obs_traj[0, 5]
                    for index in range(np.size(obs_traj, 0)):
                        if obs_traj[index, 5] <= min_ey_obs:
                            min_ey_obs = obs_traj[index, 5]
                        if obs_traj[index, 5] >= max_ey_obs:
                            max_ey_obs = obs_traj[index, 5]
                else:
                    pass
        func_optimal_ey = interp1d(
            optimal_traj_xcurv[:, 4], optimal_traj_xcurv[:, 5])
        # points for cubic Bezier curves, _l or _r indicates the point is on left or right, seperately
        s0 = s_veh - planning_prediction_factor*obs_traj[0, 0]
        # calculate the ey of first point, when the ego vehicle has just crossed the start line, s0 could be negative
        if s0 < 0:
            ey0 = func_optimal_ey(s0+track.lap_length)
        elif s0 < optimal_traj_xcurv[0, 4]:
            ey0 = optimal_traj_xcurv[0, 5]
        else:
            ey0 = func_optimal_ey(s0)
        # second point
        s1_l = s_veh - (planning_prediction_factor/bezier_order)*obs_traj[0, 0]
        ey1_l = track.width
        s1_r = s_veh - (planning_prediction_factor/bezier_order)*obs_traj[0, 0]
        ey1_r = -track.width
        # third point
        s2_l = s_veh + (planning_prediction_factor/bezier_order)*obs_traj[0, 0]
        ey2_l = track.width
        s2_r = s_veh + (planning_prediction_factor/bezier_order)*obs_traj[0, 0]
        ey2_r = -track.width
        # fourth point, it might be greater than lap length
        s3 = s_veh + planning_prediction_factor*obs_traj[0, 0]
        while s3 > track.lap_length:
            s3 = s3-track.lap_length
        if s3 <= optimal_traj_xcurv[0, 4]:
            s3 = optimal_traj_xcurv[0, 4]
        ey3 = func_optimal_ey(s3)
        s3 = s_veh + planning_prediction_factor*obs_traj[0, 0]
        # initialize two optimization problem
        opti_l = ca.Opti()
        eyvar_l = opti_l.variable(num_horizon_planner+1)
        opti_r = ca.Opti()
        eyvar_r = opti_r.variable(num_horizon_planner+1)
        cost_l = 0
        cost_r = 0
        bezier_xcurv_l = np.zeros((num_horizon_planner+1, 2))
        bezier_xcurv_r = np.zeros((num_horizon_planner+1, 2))
        bezier_xglob_l = np.zeros((num_horizon_planner+1, 2))
        bezier_xglob_r = np.zeros((num_horizon_planner+1, 2))
        for i in range(num_horizon_planner+1):
            t = i*(1.0/num_horizon_planner)
            bezier_xcurv_l[i, 0] = func_bezier_s(t, s0, s1_l, s2_l, s3)
            bezier_xcurv_l[i, 1] = func_bezier_ey(t, ey0, ey1_l, ey2_l, ey3)
            bezier_xcurv_r[i, 0] = func_bezier_s(t, s0, s1_r, s2_r, s3)
            bezier_xcurv_r[i, 1] = func_bezier_ey(t, ey0, ey1_r, ey2_r, ey3)
        func_bezier_l_ey = interp1d(bezier_xcurv_l[:, 0], bezier_xcurv_l[:, 1])
        func_bezier_r_ey = interp1d(bezier_xcurv_r[:, 0], bezier_xcurv_r[:, 1])
        # get the front and rear range of the obstacle vehicle
        vehicle_range_f = obs_traj[-1, 4]+safety_factor*vehicles["ego"].param.length
        vehicle_range_r = obs_traj[0, 4]-safety_factor*vehicles["ego"].param.length
        while vehicle_range_f > track.lap_length:
            vehicle_range_f = vehicle_range_f - track.lap_length
        while vehicle_range_r > track.lap_length:
            vehicle_range_r = vehicle_range_r - track.lap_length
        # construct the optimization problem
        for i in range(num_horizon_planner+1):
            s_ego = copy.deepcopy(xcurv_ego[4])
            # if the leading vehicle has crossed the start line, the ego vehilce is behind the start line
            if s_ego >= s_veh and s_veh <= track.lap_length/3 and s_ego >= 2*track.lap_length/3:
                s_veh = s_veh + track.lap_length
            # if the ego vehicle has crossed the start line, the other vehicle is behind the start line
            if s_ego <= s_veh and s_ego <= track.lap_length/3 and s_veh >= 2*track.lap_length/3:
                s_ego = s_ego + track.lap_length
            # traveling distance for the ith point
            s_tmp = s_ego+(s_veh+planning_prediction_factor *
                           obs_traj[0, 0]-s_ego)*i/num_horizon_planner
            if bezier_xcurv_l[0, 0] < 0:
                if s_tmp - track.lap_length <= bezier_xcurv_l[-1, 0] and s_tmp - track.lap_length >= bezier_xcurv_l[0, 0]:
                    s_tmp = s_tmp-track.lap_length
            if s_tmp > bezier_xcurv_l[-1, 0]:
                s_tmp = bezier_xcurv_l[-1, 0]
            if s_tmp < bezier_xcurv_l[0, 0]:
                s_tmp = bezier_xcurv_l[0, 0]
            cost_l += alpha * \
                ((eyvar_l[i]-func_bezier_l_ey(s_tmp))**2)
            cost_r += alpha * \
                ((eyvar_r[i]-func_bezier_r_ey(s_tmp))**2)
            # add constraint to make the trajectory avoid obstacle
            s_l = s_ego+(s_veh+planning_prediction_factor *
                         obs_traj[0, 0]-s_ego)*i/num_horizon_planner
            s_r = s_ego+(s_veh+planning_prediction_factor *
                         obs_traj[0, 0]-s_ego)*i/num_horizon_planner
            if (s_l < vehicle_range_r) or (s_l > vehicle_range_f):
                pass
            else:
                if i == 0 and xcurv_ego[5] <= (max_ey_obs+safety_factor*vehicles[agent_name].param.width):
                    pass
                else:
                    opti_l.subject_to(eyvar_l[i] > (
                        max_ey_obs+safety_factor*vehicles[agent_name].param.width))
            if (s_r < vehicle_range_r) or (s_r > vehicle_range_f):
                pass
            else:
                if i == 0 and xcurv_ego[5] >= (min_ey_obs-safety_factor*vehicles[agent_name].param.width):
                    pass
                else:
                    opti_r.subject_to(eyvar_r[i] <= (
                        min_ey_obs-safety_factor*vehicles[agent_name].param.width))
            while s_l >= track.lap_length:
                s_l = s_l - track.lap_length
            while s_r >= track.lap_length:
                s_r = s_r - track.lap_length
            if s_l <= optimal_traj_xcurv[0, 4]:
                s_l = optimal_traj_xcurv[0, 4]
            if s_r <= optimal_traj_xcurv[0, 4]:
                s_r = optimal_traj_xcurv[0, 4]
            # deviation from optimal trajectory
            cost_l += (1-alpha)*((eyvar_l[i] - func_optimal_ey(s_l))**2)
            cost_r += (1-alpha)*((eyvar_r[i] - func_optimal_ey(s_r))**2)
            # change rate
            if i >= 1:
                cost_l += 100*((eyvar_l[i] - eyvar_l[i-1])**2)
                cost_r += 100*((eyvar_r[i] - eyvar_r[i-1])**2)
            else:
                pass
            # add constraint to make the trajectory at inside of track
            opti_l.subject_to(eyvar_l[i] <= track.width)
            opti_l.subject_to(eyvar_l[i] >= -track.width)
            opti_r.subject_to(eyvar_r[i] >= -track.width)
            opti_r.subject_to(eyvar_r[i] <= track.width)
        opti_l.subject_to(eyvar_l[0] == xcurv_ego[5])
        opti_l.subject_to(eyvar_l[num_horizon_planner] == ey3)
        opti_r.subject_to(eyvar_r[0] == xcurv_ego[5])
        opti_r.subject_to(eyvar_r[num_horizon_planner] == ey3)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti_l.minimize(cost_l)
        opti_l.solver("ipopt", option)
        opti_r.minimize(cost_r)
        opti_r.solver("ipopt", option)
        # calculate two optimal trajectories
        try:
            sol_l = opti_l.solve()
            cost_l = sol_l.value(cost_l)
            ey_pred_l = sol_l.value(eyvar_l)
        except RuntimeError:
            cost_l = 1000000
        try:
            sol_r = opti_r.solve()
            cost_r = sol_r.value(cost_r)
            ey_pred_r = sol_r.value(eyvar_r)
        except RuntimeError:
            cost_r = 1000000
        direction_flag = None
        if cost_l < cost_r:
            ey_pred = ey_pred_l
            print("left")
            direction_flag = 1
        else:
            ey_pred = ey_pred_r
            print("right")
            direction_flag = -1
        target_traj = np.zeros((num_horizon_planner+1, 6))
        for i in range(num_horizon_planner+1):
            t = i*(1.0/num_horizon_planner)
            target_traj[i, 4] = s_ego + \
                (s_veh+planning_prediction_factor *
                 obs_traj[0, 0]-s_ego)*i/num_horizon_planner
            target_traj[i, 5] = ey_pred[i]
        end_timer = datetime.datetime.now()
        solver_time = (end_timer - start_timer).total_seconds()
        print("local planner solver time: {}".format(solver_time))
        target_traj_xglob = np.zeros((num_horizon_planner+1, 6))
        for index in range(num_horizon_planner+1):
            s_i = copy.deepcopy(target_traj[index, 4])
            while s_i > track.lap_length:
                s_i = s_i - track.lap_length
            target_traj_xglob[index, 4], target_traj_xglob[index, 5] = racing_env.get_global_position(
                track.lap_length, track.width, track.point_and_tangent, s_i, target_traj[index, 5])
        #debug_plot(track, vehicles,target_traj_xglob)
        return target_traj, target_traj_xglob, direction_flag


def func_bezier_s(t, s0, s1, s2, s3):
    func_bezier_s = s0*((1-t)**3) + 3*s1*t*((1-t)**2)+3*s2*(t**2)*(1-t)+s3*(t**3)
    return func_bezier_s


def func_bezier_ey(t, ey0, ey1, ey2, ey3):
    func_bezier_ey = ey0*((1-t)**3) + 3*ey1*t*((1-t)**2) + \
        3*ey2*(t**2)*(1-t)+ey3*(t**3)
    return func_bezier_ey


def debug_plot(track, vehicles, target_traj_xglob):
    fig, ax = plt.subplots()
    num_sampling_per_meter = 100
    num_track_points = int(
        np.floor(num_sampling_per_meter * track.lap_length))
    points_out = np.zeros((num_track_points, 2))
    points_center = np.zeros((num_track_points, 2))
    points_in = np.zeros((num_track_points, 2))
    for i in range(0, num_track_points):
        points_out[i, :] = track.get_global_position(
            i / float(num_sampling_per_meter), track.width)
        points_center[i, :] = track.get_global_position(
            i / float(num_sampling_per_meter), 0.0)
        points_in[i, :] = track.get_global_position(
            i / float(num_sampling_per_meter), -track.width)
    ax.plot(points_center[:, 0], points_center[:, 1], "--r")
    ax.plot(points_in[:, 0], points_in[:, 1], "-b")
    ax.plot(points_out[:, 0], points_out[:, 1], "-b")
    x_ego, y_ego, dx_ego, dy_ego, alpha_ego = get_vehicle_in_rectangle(
        vehicles["ego"].xglob, vehicles["ego"].param)
    ax.add_patch(patches.Rectangle((x_ego, y_ego),
                 dx_ego, dy_ego, alpha_ego, color='red'))
    for name in list(vehicles):
        x_car, y_car, dx_car, dy_car, alpha_car = get_vehicle_in_rectangle(
            vehicles[name].xglob, vehicles[name].param)
        ax.add_patch(patches.Rectangle((x_car, y_car),
                                       dx_car, dy_car, alpha_car, color='blue'))
    ax.plot(target_traj_xglob[:, 4], target_traj_xglob[:, 5])
    ax.axis('equal')
    plt.show()


def get_vehicle_in_rectangle(vehicle_state_glob, veh_param):
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