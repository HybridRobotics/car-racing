import datetime
import numpy as np
import casadi as ca
from utils import lmpc_helper, racing_env
from utils.racing_game_helper import *
from casadi import *
from scipy import sparse
from scipy.sparse import vstack
from scipy.interpolate import interp1d
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.constants import *


class OvertakePlanner:
    def __init__(self, racing_game_param):
        self.racing_game_param = racing_game_param
        self.vehicles = None
        self.agent_name = None
        self.track = None
        self.opti_traj_xcurv = None

    def get_overtake_flag(self, xcurv_ego):
        overtake_flag = False
        overtake_list = {}
        for name in list(self.vehicles):
            if name != self.agent_name:
                x_other = copy.deepcopy(self.vehicles[name].xcurv)
                while x_other[4] > self.track.lap_length:
                    x_other[4] = x_other[4] - self.track.lap_length
                delta_v = abs(xcurv_ego[0] - self.vehicles[name].xcurv[0])
                if (
                    (
                        (
                            x_other[4] - xcurv_ego[4]
                            <= self.racing_game_param.safety_factor
                            * self.vehicles[name].param.length
                            + self.racing_game_param.planning_prediction_factor
                            * delta_v
                        )
                        and (x_other[4] >= xcurv_ego[4])
                    )
                    or (
                        (
                            x_other[4] + self.track.lap_length - xcurv_ego[4]
                            <= self.racing_game_param.safety_factor
                            * self.vehicles[name].param.length
                            + self.racing_game_param.planning_prediction_factor
                            * delta_v
                        )
                        and (x_other[4] + self.track.lap_length >= xcurv_ego[4])
                    )
                    or (
                        (
                            -x_other[4] + xcurv_ego[4]
                            <= 0.0
                            * self.racing_game_param.safety_factor
                            * self.vehicles[name].param.length
                            + self.racing_game_param.planning_prediction_factor
                            * delta_v
                        )
                        and (x_other[4] <= xcurv_ego[4])
                    )
                    or (
                        (
                            -x_other[4] + xcurv_ego[4] + self.track.lap_length
                            <= 0.0
                            * self.racing_game_param.safety_factor
                            * self.vehicles[name].param.length
                            + self.racing_game_param.planning_prediction_factor
                            * delta_v
                        )
                        and (x_other[4] <= xcurv_ego[4] + self.track.lap_length)
                    )
                ):
                    overtake_flag = True
                    overtake_list[name] = self.vehicles[name]
        return overtake_flag, overtake_list

    def get_local_path(self, xcurv_ego, time, overtake_list):
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
        veh_length = vehicles["ego"].param.length
        veh_width = vehicles["ego"].param.width
        # the maximum and minimum ey's value of the obstacle's predicted trajectory
        num_veh = len(overtake_list)
        num = 0
        veh_name_list = []
        # in this list, the vehicle with biggest ey (left side) will be the first
        for name in list(overtake_list):
            if num == 0:
                veh_name_list.append(name)
            elif (
                overtake_list[name].xcurv[5] >= overtake_list[veh_name_list[0]].xcurv[5]
            ):
                veh_name_list.insert(0, name)
            elif (
                overtake_list[name].xcurv[5] <= overtake_list[veh_name_list[0]].xcurv[5]
            ):
                veh_name_list.append(name)
            num += 1
        veh_info_list = np.zeros((num_veh, 3))
        min_s_obs = None
        max_s_obs = None
        min_v_obs = None
        max_v_obs = None

        max_delta_v_obs = None
        min_delta_v_obs = None

        for index in range(num_veh):
            name = veh_name_list[index]
            s_veh = copy.deepcopy(vehicles[name].xcurv[4])
            while s_veh > track.lap_length:
                s_veh = s_veh - track.lap_length
            if vehicles[name].no_dynamics:
                obs_traj, _ = vehicles[name].get_trajectory_nsteps(
                    time,
                    self.racing_game_param.timestep,
                    num_horizon_planner + 1,
                )
            else:
                obs_traj, _ = vehicles[name].get_trajectory_nsteps(
                    num_horizon_planner + 1
                )
            obs_traj = obs_traj.T
            # find the minimum and maximum deviation for this obstacle
            max_ey_obs = None
            min_ey_obs = None
            if min_ey_obs is None:
                min_ey_obs = obs_traj[0, 5]
            if max_ey_obs is None:
                max_ey_obs = obs_traj[0, 5]
            for i in range(np.size(obs_traj, 0)):
                if obs_traj[i, 5] <= min_ey_obs:
                    min_ey_obs = obs_traj[i, 5]
                if obs_traj[i, 5] >= max_ey_obs:
                    max_ey_obs = obs_traj[i, 5]
            # find the minimum and maximum traveling distance from all obstacles
            if min_s_obs is None:
                min_s_obs = s_veh
            # determine if the obstacles are at different sides of start line
            elif (
                min_s_obs <= track.lap_length / 3 and s_veh >= 2 * track.lap_length / 3
            ):
                min_s_obs = s_veh
            elif (
                min_s_obs >= 2 * track.lap_length / 3 and s_veh <= track.lap_length / 3
            ):
                pass
            elif s_veh <= min_s_obs:
                min_s_obs = s_veh
            else:
                pass
            if max_s_obs is None:
                max_s_obs = s_veh
            elif (
                max_s_obs >= 2 * track.lap_length / 3 and s_veh <= track.lap_length / 3
            ):
                max_s_obs = s_veh
            elif (
                max_s_obs <= track.lap_length / 3 and s_veh >= 2 * track.lap_length / 3
            ):
                pass
            elif s_veh >= max_s_obs:
                max_s_obs = s_veh
            else:
                pass
            # find the minimum and maximum velocity from all obstacles
            if min_v_obs is None:
                min_v_obs = vehicles[name].xcurv[0]
            elif vehicles[name].xcurv[0] <= min_v_obs:
                min_v_obs = vehicles[name].xcurv[0]
            else:
                pass

            if min_delta_v_obs is None:
                min_delta_v_obs = abs(xcurv_ego[0] - vehicles[name].xcurv[0])
            elif abs(xcurv_ego[0] - vehicles[name].xcurv[0]) <= min_delta_v_obs:
                min_delta_v_obs = abs(xcurv_ego[0] - vehicles[name].xcurv[0])
            else:
                pass

            if max_delta_v_obs is None:
                max_delta_v_obs = abs(xcurv_ego[0] - vehicles[name].xcurv[0])
            elif abs(xcurv_ego[0] - vehicles[name].xcurv[0]) >= max_delta_v_obs:
                max_delta_v_obs = abs(xcurv_ego[0] - vehicles[name].xcurv[0])
            else:
                pass

            if max_v_obs is None:
                max_v_obs = vehicles[name].xcurv[0]
            elif vehicles[name].xcurv[0] >= max_v_obs:
                max_v_obs = vehicles[name].xcurv[0]
            else:
                pass
            veh_info_list[index, 0] = s_veh
            veh_info_list[index, 1] = max_ey_obs
            veh_info_list[index, 2] = min_ey_obs
        func_optimal_ey = interp1d(optimal_traj_xcurv[:, 4], optimal_traj_xcurv[:, 5])
        bezier_control_point = get_bezier_control_points(num_veh, bezier_order, planning_prediction_factor, safety_factor, track, veh_length, veh_width, optimal_traj_xcurv, max_s_obs, min_s_obs, max_delta_v_obs)
        # initialize optimization problems
        opti_list = []
        opti_var_list = []
        cost_list = []
        bezier_xcurv_list = np.zeros((num_veh + 1, num_horizon_planner + 1, 2))
        bezier_func_list = []
        vehicle_range_f_list = np.zeros((num_veh, 1))
        vehicle_range_r_list = np.zeros((num_veh, 1))
        for index in range(num_veh + 1):
            opti_list.append(ca.Opti())
            opti_var_list.append(opti_list[index].variable(num_horizon_planner + 1))
            cost_list.append(0)
        for index in range(num_horizon_planner + 1):
            t = index * (1.0 / num_horizon_planner)
            for j in range(num_veh + 1):
                bezier_xcurv_list[j, index, 0] = func_bezier_s(
                    t,
                    bezier_control_point[j, 0, 0],
                    bezier_control_point[j, 1, 0],
                    bezier_control_point[j, 2, 0],
                    bezier_control_point[j, 3, 0],
                )
                bezier_xcurv_list[j, index, 1] = func_bezier_ey(
                    t,
                    bezier_control_point[j, 0, 1],
                    bezier_control_point[j, 1, 1],
                    bezier_control_point[j, 2, 1],
                    bezier_control_point[j, 3, 1],
                )
        for index in range(num_veh + 1):
            bezier_func_list.append(
                interp1d(
                    bezier_xcurv_list[index, :, 0],
                    bezier_xcurv_list[index, :, 1],
                )
            )
        for index in range(num_veh):
            vehicle_range_f_list[index, 0] = (
                veh_info_list[index, 0] + 1.2 * safety_factor * veh_length
            )
            vehicle_range_r_list[index, 0] = (
                veh_info_list[index, 0] - 1.2 * safety_factor * veh_length
            )
            while vehicle_range_f_list[index, 0] > track.lap_length:
                vehicle_range_f_list[index, 0] = (
                    vehicle_range_f_list[index, 0] - track.lap_length
                )
            while vehicle_range_r_list[index, 0] > track.lap_length:
                vehicle_range_r_list[index, 0] = (
                    vehicle_range_r_list[index, 0] - track.lap_length
                )
        # construct the optimization problem, calculate cost
        for i in range(num_horizon_planner + 1):
            s_ego = copy.deepcopy(xcurv_ego[4])
            if (
                s_ego <= min_s_obs
                and s_ego <= track.lap_length / 3
                and min_s_obs >= 2 * track.lap_length / 3
            ):
                s_ego = s_ego + track.lap_length
            s_tmp = (
                s_ego
                + (
                    max_s_obs
                    + safety_factor * veh_length
                    + planning_prediction_factor * max_delta_v_obs
                    - s_ego
                )
                * i
                / num_horizon_planner
            )
            for index in range(num_veh + 1):
                s_now = copy.deepcopy(s_tmp)
                while s_now >= track.lap_length:
                    s_now = s_now - track.lap_length
                if s_now <= optimal_traj_xcurv[0, 4]:
                    s_now = optimal_traj_xcurv[0, 4]
                cost_list[index] += (1 - alpha) * (
                    (opti_var_list[index][i] - func_optimal_ey(s_now)) ** 2
                )
            if bezier_xcurv_list[0, 0, 0] < 0:
                if (
                    s_tmp - track.lap_length <= bezier_xcurv_list[0, -1, 0]
                    and s_tmp - track.lap_length >= bezier_xcurv_list[0, 0, 0]
                ):
                    s_tmp = s_tmp - track.lap_length
            if s_tmp > bezier_xcurv_list[0, -1, 0]:
                s_tmp = bezier_xcurv_list[0, -1, 0]
            if s_tmp < bezier_xcurv_list[0, 0, 0]:
                s_tmp = bezier_xcurv_list[0, 0, 0]
            for index in range(num_veh + 1):
                cost_list[index] += alpha * (
                    (opti_var_list[index][i] - bezier_func_list[index](s_tmp)) ** 2
                )
                if i >= 1:
                    cost_list[index] += 100 * (
                        (opti_var_list[index][i] - opti_var_list[index][i - 1]) ** 2
                    )
                else:
                    pass
        # construct the optimization problem, add constraint
        for i in range(num_horizon_planner + 1):
            s_ego = copy.deepcopy(xcurv_ego[4])
            if (
                s_ego <= min_s_obs
                and s_ego <= track.lap_length / 3
                and min_s_obs >= 2 * track.lap_length / 3
            ):
                s_ego = s_ego + track.lap_length
            s_tmp = (
                s_ego
                + (
                    max_s_obs
                    + safety_factor * veh_length
                    + planning_prediction_factor * max_v_obs
                    - s_ego
                )
                * i
                / num_horizon_planner
            )
            for index in range(num_veh + 1):
                if i == 0:
                    opti_list[index].subject_to(opti_var_list[index][0] == xcurv_ego[5])
                if i == num_horizon_planner:
                    opti_list[index].subject_to(
                        opti_var_list[index][num_horizon_planner]
                        == bezier_control_point[index, 3, 1]
                    )
                opti_list[index].subject_to(opti_var_list[index][i] <= track.width)
                opti_list[index].subject_to(opti_var_list[index][i] >= -track.width)
                # constraint on the left, first line is the track boundary
                if index == 0:
                    pass
                else:
                    if (s_tmp < vehicle_range_r_list[index - 1, 0]) or (
                        s_tmp > vehicle_range_f_list[index - 1, 0]
                    ):
                        pass
                    else:
                        if i == 0 and xcurv_ego[5] >= (
                            veh_info_list[index - 1, 2] - safety_factor * veh_width
                        ):
                            pass
                        else:
                            opti_list[index].subject_to(
                                opti_var_list[index][i]
                                < (
                                    veh_info_list[index - 1, 2]
                                    - safety_factor * veh_width
                                )
                            )
                # constraint on the right, last line is the track boundary
                if index == num_veh:
                    pass
                else:
                    if (s_tmp < vehicle_range_r_list[index, 0]) or (
                        s_tmp > vehicle_range_f_list[index, 0]
                    ):
                        pass
                    else:
                        if i == 0 and xcurv_ego[5] <= (
                            veh_info_list[index, 1] + safety_factor * veh_width
                        ):
                            pass
                        else:
                            opti_list[index].subject_to(
                                opti_var_list[index][i]
                                > (veh_info_list[index, 1] + safety_factor * veh_width)
                            )
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        solution_ey = np.zeros((num_horizon_planner + 1, num_veh + 1))
        for index in range(num_veh + 1):
            opti_list[index].minimize(cost_list[index])
            opti_list[index].solver("ipopt", option)
            try:
                sol = opti_list[index].solve()
                solution_ey[:, index] = sol.value(opti_var_list[index])
                cost_list[index] = sol.value(cost_list[index])
            except RuntimeError:
                cost_list[index] = float("inf")
        direction_flag = None
        min_cost = None
        best_ey = None
        for index in range(num_veh + 1):
            if min_cost is None:
                min_cost = cost_list[index]
                best_ey = solution_ey[:, index]
                direction_flag = index
            else:
                if cost_list[index] > min_cost:
                    pass
                else:
                    min_cost = cost_list[index]
                    best_ey = solution_ey[:, index]
                    direction_flag = index
        target_traj_xcurv = np.zeros((num_horizon_planner + 1, X_DIM))
        target_traj_xglob = np.zeros((num_horizon_planner + 1, X_DIM))
        bezier_xglob = np.zeros((num_horizon_planner + 1, X_DIM))
        for index in range(num_horizon_planner + 1):
            t = i * (1.0 / num_horizon_planner)
            if max_s_obs <= track.lap_length / 3 and s_ego >= 2 * track.lap_length / 3:
                target_traj_xcurv[index, 4] = (
                    s_ego
                    + (
                        max_s_obs
                        + track.lap_length
                        + safety_factor * veh_length
                        + planning_prediction_factor * max_delta_v_obs
                        - s_ego
                    )
                    * index
                    / num_horizon_planner
                )
            else:
                target_traj_xcurv[index, 4] = (
                    s_ego
                    + (
                        max_s_obs
                        + safety_factor * veh_length
                        + planning_prediction_factor * max_delta_v_obs
                        - s_ego
                    )
                    * index
                    / num_horizon_planner
                )
            target_traj_xcurv[index, 5] = best_ey[index]
        end_timer = datetime.datetime.now()
        solver_time = (end_timer - start_timer).total_seconds()
        print("local planner solver time: {}".format(solver_time))
        for index in range(num_horizon_planner + 1):
            s_i = copy.deepcopy(target_traj_xcurv[index, 4])
            while s_i > track.lap_length:
                s_i = s_i - track.lap_length
            (
                target_traj_xglob[index, 4],
                target_traj_xglob[index, 5],
            ) = racing_env.get_global_position(
                track.lap_length,
                track.width,
                track.point_and_tangent,
                s_i,
                target_traj_xcurv[index, 5],
            )
            (
                bezier_xglob[index, 4],
                bezier_xglob[index, 5],
            ) = racing_env.get_global_position(
                track.lap_length,
                track.width,
                track.point_and_tangent,
                bezier_xcurv_list[direction_flag, index, 0],
                bezier_xcurv_list[direction_flag, index, 1],
            )
        # debug_plot(track, vehicles, target_traj_xglob)
        return (
            target_traj_xcurv,
            target_traj_xglob,
            direction_flag,
            veh_name_list,
            bezier_xglob,
        )


