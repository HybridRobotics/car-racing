import copy
import datetime
from multiprocess import Process, Manager

import casadi as ca
import numpy as np
import scipy.interpolate

from planner.base import AgentInfo, check_ego_agent_distance, get_traj_xglob, RacingGameParam
from racing_env.params import X_DIM, U_DIM

# ---------------- HELPER FUNCTIONS FOR BEZIER CURVES ----------------

def _get_bezier_control_points(
    vehicles_interest,
    veh_info_list,
    agent_info,
    racing_game_param,
    track,
    optimal_traj_xcurv,
    sorted_vehicles,
    xcurv_ego,
):
    num_veh = len(vehicles_interest)
    veh_length = vehicles_interest[(list(vehicles_interest)[0])].param.length
    veh_width = vehicles_interest[(list(vehicles_interest)[0])].param.width
    prediction_factor = racing_game_param.planning_prediction_factor
    safety_factor = racing_game_param.safety_factor
    bezier_control_point = np.zeros(
        (num_veh + 1, racing_game_param.bezier_order + 1, 2)
    )  # for each point, coordinate in (s, ey)
    func_optimal_ey = scipy.interpolate.interp1d(optimal_traj_xcurv[:, 4], optimal_traj_xcurv[:, 5])
    for index in range(num_veh + 1):
        # s0
        bezier_control_point[index, 0, 0] = xcurv_ego[4]
        # s3
        bezier_control_point[index, 3, 0] = (
            xcurv_ego[4] + prediction_factor * agent_info.max_delta_v + 4
        )
        # when the s3 is ahead start line, s0 is behind start line
        if bezier_control_point[index, 0, 0] > bezier_control_point[index, 3, 0]:
            # s1
            bezier_control_point[index, 1, 0] = (
                bezier_control_point[index, 3, 0]
                + track.lap_length
                - bezier_control_point[index, 0, 0]
            ) / 3.0 + bezier_control_point[index, 0, 0]
            # s2
            bezier_control_point[index, 2, 0] = (
                2.0
                * (
                    bezier_control_point[index, 3, 0]
                    + track.lap_length
                    - bezier_control_point[index, 0, 0]
                )
                / 3.0
                + bezier_control_point[index, 0, 0]
            )
            # new s3 value, add lap length
            bezier_control_point[index, 3, 0] = bezier_control_point[index, 3, 0] + track.lap_length
        # when s0 and s3 is in the same side of start line
        else:
            # s1
            bezier_control_point[index, 1, 0] = (
                bezier_control_point[index, 3, 0] - bezier_control_point[index, 0, 0]
            ) / 3.0 + bezier_control_point[index, 0, 0]
            # s2
            bezier_control_point[index, 2, 0] = (
                2.0 * (bezier_control_point[index, 3, 0] - bezier_control_point[index, 0, 0]) / 3.0
                + bezier_control_point[index, 0, 0]
            )
        # ey0
        if bezier_control_point[index, 0, 0] < 0:
            bezier_control_point[index, 0, 1] = func_optimal_ey(
                bezier_control_point[index, 0, 0] + track.lap_length
            )
        elif bezier_control_point[index, 0, 0] < optimal_traj_xcurv[0, 4]:
            bezier_control_point[index, 0, 1] = optimal_traj_xcurv[0, 5]
        else:
            bezier_control_point[index, 0, 1] = func_optimal_ey(bezier_control_point[index, 0, 0])
        bezier_control_point[index, 0, 1] = xcurv_ego[5]
        # ey1 and ey2
        # the first curve
        if index == 0:
            bezier_control_point[index, 1, 1] = (
                0.8 * track.width - (-veh_info_list[index, 1] - 0.5 * veh_width) * 0.2
            )
            bezier_control_point[index, 2, 1] = (
                0.8 * track.width - (-veh_info_list[index, 1] - 0.5 * veh_width) * 0.2
            )
        # the last curve
        elif index == num_veh:
            bezier_control_point[index, 1, 1] = (
                -0.8 * track.width + ((veh_info_list[index - 1, 1] - 0.5 * veh_width)) * 0.2
            )
            bezier_control_point[index, 2, 1] = (
                -0.8 * track.width + ((veh_info_list[index - 1, 1] - 0.5 * veh_width)) * 0.2
            )
        else:
            bezier_control_point[index, 1, 1] = 0.7 * (
                veh_info_list[index, 1] + 0.5 * veh_width
            ) + 0.3 * (veh_info_list[index - 1, 1] - 0.5 * veh_width)
            bezier_control_point[index, 2, 1] = 0.7 * (
                veh_info_list[index, 1] + 0.5 * veh_width
            ) + 0.3 * (veh_info_list[index - 1, 1] - 0.5 * veh_width)
        # ey3
        if bezier_control_point[index, 3, 0] >= track.lap_length:
            if bezier_control_point[index, 3, 0] - track.lap_length <= optimal_traj_xcurv[0, 4]:
                bezier_control_point[index, 3, 1] = optimal_traj_xcurv[0, 5]
            else:
                bezier_control_point[index, 3, 1] = func_optimal_ey(
                    bezier_control_point[index, 3, 0] - track.lap_length
                )
        else:
            if bezier_control_point[index, 3, 0] <= optimal_traj_xcurv[0, 4]:
                bezier_control_point[index, 3, 1] = optimal_traj_xcurv[0, 5]
            else:
                bezier_control_point[index, 3, 1] = func_optimal_ey(
                    bezier_control_point[index, 3, 0]
                )
    return bezier_control_point


def _get_bezier_curve(bezier_control_point, t):
    s0, s1, s2, s3 = bezier_control_point[:, 0]
    ey0, ey1, ey2, ey3 = bezier_control_point[:, 1]
    bezier_curve_s = (
        s0 * ((1 - t) ** 3)
        + 3 * s1 * t * ((1 - t) ** 2)
        + 3 * s2 * (t ** 2) * (1 - t)
        + s3 * (t ** 3)
    )
    bezier_curve_ey = (
        ey0 * ((1 - t) ** 3)
        + 3 * ey1 * t * ((1 - t) ** 2)
        + 3 * ey2 * (t ** 2) * (1 - t)
        + ey3 * (t ** 3)
    )
    return [bezier_curve_s, bezier_curve_ey]

# ---------------- END OF HELPER FUNCTIONS ----------------

class OvertakePathPlanner:
    def __init__(self, racing_game_param: RacingGameParam):
        self.racing_game_param = racing_game_param
        self.vehicles = None
        self.agent_name = None
        self.track = None
        self.opti_traj_xcurv = None

    def get_overtake_flag(self, xcurv_ego):
        overtake_flag = False
        vehicles_interest = {}
        for name in list(self.vehicles):
            if name != self.agent_name:
                if check_ego_agent_distance(
                    self.vehicles[self.agent_name],
                    self.vehicles[name],
                    self.racing_game_param,
                    self.track.lap_length,
                ):
                    overtake_flag = True
                    vehicles_interest[name] = self.vehicles[name]
        return overtake_flag, vehicles_interest

    def get_local_path(self, xcurv_ego, time, vehicles_interest):
        start_timer = datetime.datetime.now()
        num_horizon = self.racing_game_param.num_horizon_planner
        vehicles = self.vehicles
        track = self.track
        optimal_traj_xcurv = self.opti_traj_xcurv
        ego = self.vehicles[self.agent_name]
        # the maximum and minimum ey's value of the obstacle's predicted trajectory
        num_veh = len(vehicles_interest)
        num = 0
        sorted_vehicles = []
        obs_traj_infos = np.zeros((num_veh, 3))
        # in this list, the vehicle with biggest ey (left side) will be the first
        for name in list(vehicles_interest):
            if num == 0:
                sorted_vehicles.append(name)
            elif vehicles_interest[name].xcurv[5] >= vehicles_interest[sorted_vehicles[0]].xcurv[5]:
                sorted_vehicles.insert(0, name)
            elif vehicles_interest[name].xcurv[5] <= vehicles_interest[sorted_vehicles[0]].xcurv[5]:
                sorted_vehicles.append(name)
            num += 1
        # get maximum and minimum infos of agent(s)
        agent_infos = AgentInfo.get_agent_info(vehicles, sorted_vehicles, track)
        for index in range(num_veh):
            name = sorted_vehicles[index]
            if vehicles[name].no_dynamics:
                obs_traj, _ = vehicles[name].get_trajectory_nsteps(
                    time,
                    self.racing_game_param.timestep,
                    num_horizon + 1,
                )
            else:
                obs_traj, _ = vehicles[name].get_trajectory_nsteps(num_horizon + 1)
            # save the position information of other agent
            obs_traj_infos[index, :] = (
                vehicles[name].xcurv[4],
                max(obs_traj.T[:, 5]),
                min(obs_traj.T[:, 5]),
            )
        func_optimal_ey = scipy.interpolate.interp1d(optimal_traj_xcurv[:, 4], optimal_traj_xcurv[:, 5])
        func_optimal_vx = scipy.interpolate.interp1d(optimal_traj_xcurv[:, 4], optimal_traj_xcurv[:, 0])
        bezier_control_points = _get_bezier_control_points(
            vehicles_interest,
            obs_traj_infos,
            agent_infos,
            self.racing_game_param,
            track,
            self.opti_traj_xcurv,
            sorted_vehicles,
            xcurv_ego,
        )
        bezier_xcurvs = np.zeros((num_veh + 1, num_horizon + 1, 2))
        bezier_funcs = []
        for index in range(num_veh + 1):
            for j in range(num_horizon + 1):
                t = j * (1.0 / num_horizon)
                # s and ey for each point
                bezier_xcurvs[index, j, :] = _get_bezier_curve(bezier_control_points[index, :, :], t)
            bezier_funcs.append(
                scipy.interpolate.interp1d(
                    bezier_xcurvs[index, :, 0],
                    bezier_xcurvs[index, :, 1],
                )
            )
        agents_front_ranges, agents_rear_ranges = self.get_agents_range(num_veh, obs_traj_infos)
        best_ey, direction_flag = self.solve_optimization_problem(
            sorted_vehicles,
            bezier_xcurvs,
            bezier_funcs,
            agent_infos,
            obs_traj_infos,
            agents_front_ranges,
            agents_rear_ranges,
            func_optimal_ey,
            bezier_control_points,
        )
        target_traj_xcurv = np.zeros((num_horizon + 1, X_DIM))
        bezier_xglob = np.zeros((num_horizon + 1, X_DIM))
        for index in range(num_horizon + 1):
            t = index * (1.0 / num_horizon)
            target_traj_xcurv[index, 4] = (
                ego.xcurv[4]
                + (
                    agent_infos.max_s
                    + self.racing_game_param.safety_factor * vehicles["ego"].param.length
                    + self.racing_game_param.planning_prediction_factor * agent_infos.max_delta_v
                    - ego.xcurv[4]
                )
                * index
                / num_horizon
            )
            target_traj_xcurv[index, 5] = best_ey[index]
        if target_traj_xcurv[-1, 4] >= track.lap_length:
            if target_traj_xcurv[-1, 4] - track.lap_length < optimal_traj_xcurv[0, 4]:
                vx_target = func_optimal_vx(optimal_traj_xcurv[0, 4])
            else:
                vx_target = func_optimal_vx(target_traj_xcurv[-1, 4] - track.lap_length)
        elif target_traj_xcurv[-1, 4] < optimal_traj_xcurv[0, 4]:
            vx_target = func_optimal_vx(optimal_traj_xcurv[0, 4])
        else:
            vx_target = func_optimal_vx(target_traj_xcurv[-1, 4])
        delta_t = 2 * (target_traj_xcurv[-1, 4] - xcurv_ego[4]) / (vx_target + xcurv_ego[0])
        a_target = (vx_target - xcurv_ego[0]) / delta_t
        # the desired acc should under constraints
        a_target = np.clip(a_target, -1.5, 1.5)
        # refine the path, add vx information
        target_traj_xcurv = self.get_speed_info(target_traj_xcurv, xcurv_ego, a_target)
        end_timer = datetime.datetime.now()
        solver_time = (end_timer - start_timer).total_seconds()
        print("local planner solver time: {}".format(solver_time))
        target_traj_xglob = get_traj_xglob(target_traj_xcurv, track)
        bezier_line_xcurv = np.zeros((num_horizon + 1, X_DIM))
        bezier_line_xcurv[:, 4:6] = bezier_xcurvs[direction_flag, :, :]
        bezier_xglob = get_traj_xglob(bezier_line_xcurv, track)

        all_bezier_xcurv = np.zeros((num_veh + 1, num_horizon + 1, X_DIM))
        all_local_traj_xcurv = np.zeros((num_veh + 1, num_horizon + 1, X_DIM))
        all_bezier_xcurv[:, :, 4:6] = bezier_xcurvs[:, :, :]

        all_local_traj_xglob = np.zeros((num_veh + 1, num_horizon + 1, X_DIM))
        all_bezier_xglob = np.zeros((num_veh + 1, num_horizon + 1, X_DIM))
        for index in range(num_veh + 1):
            all_bezier_xglob[index, :, :] = get_traj_xglob(all_bezier_xcurv[index, :, :], track)

        # debug_plot(track, vehicles, target_traj_xglob)
        return (
            target_traj_xcurv,
            target_traj_xglob,
            direction_flag,
            sorted_vehicles,
            bezier_xglob,
            solver_time,
            all_bezier_xglob,
            all_local_traj_xglob,
        )

    def get_speed_info(self, target_traj_xcurv, xcurv, a_target):
        num_horizon = self.racing_game_param.num_horizon_planner
        traj_xcurv = np.zeros((num_horizon + 1, 6))
        traj_xcurv = target_traj_xcurv
        traj_xcurv[0, :] = xcurv
        for index in range(num_horizon):
            traj_xcurv[index, 0] = (
                xcurv[0] ** 2 + 2 * a_target * (traj_xcurv[index, 4] - xcurv[4])
            ) ** 0.5
        return traj_xcurv

    def get_agents_range(self, num_veh, obs_traj_infos):
        veh_length = self.vehicles[self.agent_name].param.length
        track = self.track
        safety_factor = self.racing_game_param.safety_factor
        agents_front_ranges = np.zeros((num_veh, 1))
        agents_rear_ranges = np.zeros((num_veh, 1))
        for index in range(num_veh):
            agents_front_ranges[index, 0] = obs_traj_infos[index, 0] + safety_factor * veh_length
            agents_rear_ranges[index, 0] = obs_traj_infos[index, 0] - safety_factor * veh_length
            while agents_front_ranges[index, 0] > track.lap_length:
                agents_front_ranges[index, 0] = agents_front_ranges[index, 0] - track.lap_length
            while agents_rear_ranges[index, 0] > track.lap_length:
                agents_rear_ranges[index, 0] = agents_rear_ranges[index, 0] - track.lap_length
        return agents_front_ranges, agents_rear_ranges

    def solve_optimization_problem(
        self,
        sorted_vehicles,
        bezier_xucrvs,
        bezier_funcs,
        agent_infos,
        obs_traj_infos,
        agents_front_ranges,
        agents_rear_ranges,
        func_optimal_ey,
        bezier_control_points,
    ):
        num_horizon = self.racing_game_param.num_horizon_planner
        num_veh = len(sorted_vehicles)
        ego = self.vehicles[self.agent_name]
        veh_length = ego.param.length
        veh_width = ego.param.width
        track = self.track
        safety_factor = self.racing_game_param.safety_factor
        alpha = self.racing_game_param.alpha
        opti_traj_xcurv = self.opti_traj_xcurv
        # initialize optimization problem
        optis = []
        opti_vars = []
        costs = []
        for index in range(num_veh + 1):
            optis.append(ca.Opti())
            opti_vars.append(optis[index].variable(num_horizon + 1))
            costs.append(0)
        # construct optimization problem
        for index in range(num_horizon + 1):
            s_tmp = (
                ego.xcurv[4]
                + (
                    agent_infos.max_s
                    + safety_factor * veh_length
                    + self.racing_game_param.planning_prediction_factor * agent_infos.max_delta_v
                    - ego.xcurv[4]
                )
                * index
                / num_horizon
            )
            while s_tmp >= track.lap_length:
                s_tmp = s_tmp - track.lap_length
            if s_tmp <= opti_traj_xcurv[0, 4]:
                s_tmp = opti_traj_xcurv[0, 4]
            s_tmp = np.clip(s_tmp, bezier_xucrvs[0, 0, 0], bezier_xucrvs[0, -1, 0])
            # add cost
            for j in range(num_veh + 1):
                # compared with optimal trajectory
                costs[j] += (1 - alpha) * ((opti_vars[j][index] - func_optimal_ey(s_tmp)) ** 2)
                # compared with reference Bezier curve
                costs[j] += alpha * ((opti_vars[j][index] - bezier_funcs[j](s_tmp)) ** 2)
                # changing rate
                if index >= 1:
                    costs[j] += 100 * ((opti_vars[j][index] - opti_vars[j][index - 1]) ** 2)
            # constraint for optimization
            for j in range(num_veh + 1):
                # initial state constraint
                if index == 0:
                    optis[j].subject_to(opti_vars[j][0] == ego.xcurv[5])
                # final state constraint
                if index == num_horizon:
                    optis[j].subject_to(opti_vars[j][num_horizon] == bezier_control_points[j, 3, 1])
                # track boundary constraint
                optis[j].subject_to(opti_vars[j][index] <= track.width)
                optis[j].subject_to(opti_vars[j][index] >= -track.width)
                # constraint on the left, first line is the track boundary
                if j == 0:
                    pass
                else:
                    if (s_tmp < agents_rear_ranges[j - 1, 0]) or (
                        s_tmp > agents_front_ranges[j - 1, 0]
                    ):
                        pass
                    else:
                        if index == 0 and ego.xcurv[5] >= (
                            obs_traj_infos[j - 1, 2] - safety_factor * veh_width
                        ):
                            pass
                        else:
                            optis[j].subject_to(
                                opti_vars[j][index]
                                < (obs_traj_infos[j - 1, 2] - safety_factor * veh_width)
                            )
                # constraint on the right, last line is the track boundary
                if j == num_veh:
                    pass
                else:
                    if (s_tmp < agents_rear_ranges[j, 0]) or (s_tmp > agents_front_ranges[j, 0]):
                        pass
                    else:
                        if index == 0 and ego.xcurv[5] <= (
                            obs_traj_infos[j, 1] + safety_factor * veh_width
                        ):
                            pass
                        else:
                            optis[j].subject_to(
                                opti_vars[j][index]
                                > (obs_traj_infos[j, 1] + safety_factor * veh_width)
                            )
        # get the optimized solution
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        solution_ey = np.zeros((num_horizon + 1, num_veh + 1))
        for index in range(num_veh + 1):
            optis[index].minimize(costs[index])
            optis[index].solver("ipopt", option)
            try:
                sol = optis[index].solve()
                solution_ey[:, index] = sol.value(opti_vars[index])
                costs[index] = sol.value(costs[index])
            except RuntimeError:
                solution_ey[:, index] = optis[index].debug.value(opti_vars[index])
                costs[index] = float("inf")
        direction_flag = costs.index(min(costs))
        min_cost = min(costs)
        best_ey = solution_ey[:, direction_flag]
        if min_cost == float('inf'):
            print("path planner failed")
        return best_ey, direction_flag


class OvertakeTrajPlanner:
    def __init__(self, racing_game_param: RacingGameParam):
        self.racing_game_param = racing_game_param
        self.vehicles = None
        self.agent_name = None
        self.track = None
        self.opti_traj_xcurv = None
        self.matrix_Atv = None
        self.matrix_Btv = None
        self.matrix_Ctv = None
        self.sorted_vehicles = None
        self.obs_infos = None
        self.old_ey = None
        self.old_direction_flag = None
        self.bezier_xcurvs = None
        self.bezier_funcs = None
        self.xcurv_ego = None

    def get_overtake_flag(self, xcurv_ego):
        overtake_flag = False
        vehicles_interest = {}
        for name in list(self.vehicles):
            if name != self.agent_name:
                if check_ego_agent_distance(
                    self.vehicles[self.agent_name],
                    self.vehicles[name],
                    self.racing_game_param,
                    self.track.lap_length,
                ):
                    overtake_flag = True
                    vehicles_interest[name] = self.vehicles[name]
        return overtake_flag, vehicles_interest

    def get_local_traj(
        self,
        xcurv_ego,
        time,
        vehicles_interest,
        matrix_Atv,
        matrix_Btv,
        matrix_Ctv,
        old_ey,
        old_direction_flag,
    ):
        self.matrix_Atv = matrix_Atv
        self.matrix_Btv = matrix_Btv
        self.matrix_Ctv = matrix_Ctv
        start_timer = datetime.datetime.now()
        num_horizon_planner = self.racing_game_param.num_horizon_planner
        vehicles = self.vehicles
        track = self.track
        obs_trajs = []
        veh_length = vehicles["ego"].param.length
        veh_width = vehicles["ego"].param.width
        num_veh = len(vehicles_interest)
        num = 0
        sorted_vehicles = []
        obs_infos = {}
        veh_infos = np.zeros((num_veh, 3))
        for name in list(vehicles_interest):
            if num == 0:
                sorted_vehicles.append(name)
            elif vehicles_interest[name].xcurv[5] >= vehicles_interest[sorted_vehicles[0]].xcurv[5]:
                sorted_vehicles.insert(0, name)
            elif vehicles_interest[name].xcurv[5] <= vehicles_interest[sorted_vehicles[0]].xcurv[5]:
                sorted_vehicles.append(name)
            if vehicles[name].no_dynamics:
                obs_traj, _ = vehicles[name].get_trajectory_nsteps(
                    time,
                    self.racing_game_param.timestep,
                    num_horizon_planner + 1,
                )
            else:
                obs_traj, _ = vehicles[name].get_trajectory_nsteps(num_horizon_planner + 1)
            obs_infos[name] = obs_traj
            # save the position information of other agent
            veh_infos[num, :] = (
                vehicles[name].xcurv[4],
                max(obs_traj.T[:, 5]),
                min(obs_traj.T[:, 5]),
            )
            num += 1
        # get agents infos and reference Bezier curve
        agent_info = AgentInfo.get_agent_info(vehicles, sorted_vehicles, track)
        bezier_control_point = _get_bezier_control_points(
            vehicles_interest,
            veh_infos,
            agent_info,
            self.racing_game_param,
            track,
            self.opti_traj_xcurv,
            sorted_vehicles,
            xcurv_ego,
        )
        bezier_xcurvs = np.zeros((num_veh + 1, num_horizon_planner + 1, 2))
        bezier_funcs = []
        for index in range(num_veh + 1):
            for j in range(num_horizon_planner + 1):
                t = j * (1.0 / num_horizon_planner)
                # s and ey for each point
                bezier_xcurvs[index, j, :] = _get_bezier_curve(bezier_control_point[index, :, :], t)
            bezier_funcs.append(
                scipy.interpolate.interp1d(
                    bezier_xcurvs[index, :, 0],
                    bezier_xcurvs[index, :, 1],
                )
            )
        self.sorted_vehicles = sorted_vehicles
        self.obs_infos = obs_infos
        self.old_ey = old_ey
        self.old_direction_flag = old_direction_flag
        self.bezier_xcurvs = bezier_xcurvs
        self.bezier_funcs = bezier_funcs
        self.xcurv_ego = xcurv_ego
        (
            target_traj_xcurv,
            direction_flag,
            solve_time,
            solution_xvar,
        ) = self.solve_optimization_problem()
        end_timer = datetime.datetime.now()
        solver_time = (end_timer - start_timer).total_seconds()
        print("local planner solver time: {}".format(solver_time))
        target_traj_xglob = get_traj_xglob(target_traj_xcurv, track)
        bezier_line_xcurv = np.zeros((num_horizon_planner + 1, X_DIM))
        bezier_line_xcurv[:, 4:6] = bezier_xcurvs[direction_flag, :, :]
        bezier_xglob = get_traj_xglob(bezier_line_xcurv, track)
        all_bezier_xcurv = np.zeros((num_veh + 1, num_horizon_planner + 1, X_DIM))
        all_local_traj_xcurv = np.zeros((num_veh + 1, num_horizon_planner + 1, X_DIM))
        all_bezier_xcurv[:, :, 4:6] = bezier_xcurvs[:, :, :]
        all_local_traj_xglob = np.zeros((num_veh + 1, num_horizon_planner + 1, X_DIM))
        all_bezier_xglob = np.zeros((num_veh + 1, num_horizon_planner + 1, X_DIM))
        for index in range(num_veh + 1):
            all_local_traj_xcurv[index, :, :] = solution_xvar[index, :, :].T
        for index in range(num_veh + 1):
            all_local_traj_xglob[index, :, :] = get_traj_xglob(
                all_local_traj_xcurv[index, :, :], track
            )
            all_bezier_xglob[index, :, :] = get_traj_xglob(all_bezier_xcurv[index, :, :], track)
        # debug_plot(track, vehicles, target_traj_xglob)
        return (
            target_traj_xcurv,
            target_traj_xglob,
            direction_flag,
            sorted_vehicles,
            bezier_xglob,
            solve_time,
            all_bezier_xglob,
            all_local_traj_xglob,
        )

    def solve_optimization_problem(self):
        sorted_vehicles = self.sorted_vehicles
        obs_infos = self.obs_infos
        old_ey = self.old_ey
        old_direction_flag = self.old_direction_flag
        bezier_xcurvs = self.bezier_xcurvs
        bezier_funcs = self.bezier_funcs
        xcurv_ego = self.xcurv_ego
        num_horizon = self.racing_game_param.num_horizon_planner
        num_veh = len(self.sorted_vehicles)
        ego = self.vehicles[self.agent_name]
        veh_length = ego.param.length
        veh_width = ego.param.width
        track = self.track
        safety_margin = 0.15
        manager = Manager()
        dict_traj = manager.dict()
        dict_solve_time = manager.dict()
        dict_cost = manager.dict()
        list_opti = []
        for index in range(num_veh+1):
            list_opti.append(
                Process(
                    target=self.generate_traj_per_region,
                    args=(
                        index,
                        dict_traj,
                        dict_solve_time,
                        dict_cost,
                    ),
                )
            )
        for index in range(num_veh+1):
            list_opti[index].start()
        for index in range(num_veh+1):
            list_opti[index].join()
        costs = []
        solution_xvar = np.zeros((num_veh + 1, X_DIM, num_horizon + 1))
        solve_time = np.zeros(num_veh + 1)
        for index in range(num_veh+1):
            solution_xvar[index, :, :] = dict_traj[index]
            costs.append(dict_cost[index])
            solve_time[index] = dict_solve_time[index]
        cost_selection = []
        for index in range(num_veh + 1):
            cost_selection.append(0)
        for index in range(num_veh + 1):
            cost_selection[index] = -10 * (solution_xvar[index, 4, -1] - solution_xvar[index, 4, 0])
            if index == 0:
                pass
            else:
                name = sorted_vehicles[index - 1]
                obs_traj = obs_infos[name]
                for j in range(num_horizon + 1):
                    while obs_traj[4, j] > track.lap_length:
                        obs_traj[4, j] = obs_traj[4, j] - track.lap_length
                    diffs = solution_xvar[index, 4, j] - obs_traj[4, j]
                    diffey = solution_xvar[index, 5, j] - obs_traj[5, j]
                    if diffs ** 2 + diffey ** 2 - veh_length ** 2 - veh_width ** 2 >= 0:
                        cost_selection[index] += 0
                    else:
                        cost_selection[index] += 100
            if index == num_veh:
                pass
            else:
                name = sorted_vehicles[index]
                obs_traj = obs_infos[name]
                for j in range(num_horizon + 1):
                    while obs_traj[4, j] > track.lap_length:
                        obs_traj[4, j] = obs_traj[4, j] - track.lap_length
                    diffs = solution_xvar[index, 4, j] - obs_traj[4, j]
                    diffey = solution_xvar[index, 5, j] - obs_traj[5, j]
                    if diffs ** 2 + diffey ** 2 - veh_length ** 2 - veh_width ** 2 >= 0:
                        cost_selection[index] += 0
                    else:
                        cost_selection[index] += 100
            if old_direction_flag is None:
                pass
            elif old_direction_flag == index:
                pass
            else:
                cost_selection[index] += 100
        direction_flag = cost_selection.index(min(cost_selection))
        traj_xcurv = solution_xvar[direction_flag, :, :].T
        return traj_xcurv, direction_flag, solve_time, solution_xvar

    def generate_traj_per_region(self, pos_index, dict_traj, dict_solve_time, dict_cost):
        sorted_vehicles = self.sorted_vehicles
        obs_infos = self.obs_infos
        old_ey = self.old_ey
        old_direction_flag = self.old_direction_flag
        bezier_xcurvs = self.bezier_xcurvs
        bezier_funcs = self.bezier_funcs
        xcurv_ego = self.xcurv_ego
        num_horizon = self.racing_game_param.num_horizon_planner
        num_veh = len(self.sorted_vehicles)
        ego = self.vehicles[self.agent_name]
        veh_length = ego.param.length
        veh_width = ego.param.width
        track = self.track
        safety_margin = 0.15
        opti = ca.Opti()
        opti_xvar = opti.variable(X_DIM, num_horizon + 1)
        opti_uvar = opti.variable(U_DIM, num_horizon)
        opti.subject_to(opti_xvar[:, 0] == ego.xcurv)
        cost = 0
        for index in range(num_horizon):
            # dynamic state update constraint
            opti.subject_to(
                opti_xvar[:, index + 1]
                == ca.mtimes(self.racing_game_param.matrix_A, opti_xvar[:, index])
                + ca.mtimes(self.racing_game_param.matrix_B, opti_uvar[:, index])
            )
            # min and max of vx, ey
            opti.subject_to(opti_xvar[0, index + 1] <= 5.0)
            opti.subject_to(opti_xvar[5, index] <= track.width - 0.5 * veh_width)
            opti.subject_to(opti_xvar[5, index] >= -track.width + 0.5 * veh_width)
            # min and max of delta
            opti.subject_to(opti_uvar[0, index] <= 0.5)
            opti.subject_to(opti_uvar[0, index] >= -0.5)
            # min and max of a
            opti.subject_to(opti_uvar[1, index] <= 1.5)
            opti.subject_to(opti_uvar[1, index] >= -1.5)
            # constraint on the left, first line is the track boundary
            if pos_index == 0:
                pass
            else:
                name = sorted_vehicles[pos_index - 1]
                obs_traj = obs_infos[name]
                while obs_traj[4, index] > track.lap_length:
                    obs_traj[4, index] = obs_traj[4, index] - track.lap_length
                diffs = opti_xvar[4, index] - obs_traj[4, index]
                diffey = opti_xvar[5, index] - obs_traj[5, index]
                if (
                    xcurv_ego[4] + index * 0.1 * xcurv_ego[0]
                    >= obs_traj[4, index] - veh_length - safety_margin
                ) & (
                    xcurv_ego[4] + index * 0.1 * xcurv_ego[0]
                    <= obs_traj[4, index] + veh_length + safety_margin
                ):
                    opti.subject_to(diffey >= veh_width + safety_margin)
                else:
                    pass
            # constraint on the right, last line is the track boundary
            if pos_index == num_veh:
                pass
            else:
                name = sorted_vehicles[pos_index]
                obs_traj = obs_infos[name]
                while obs_traj[4, index] > track.lap_length:
                    obs_traj[4, index] = obs_traj[4, index] - track.lap_length
                diffs = opti_xvar[4, index] - obs_traj[4, index]
                diffey = opti_xvar[5, index] - obs_traj[5, index]
                if (
                    xcurv_ego[4] + index * 0.1 * xcurv_ego[0]
                    >= obs_traj[4, index] - veh_length - safety_margin
                ) & (
                    xcurv_ego[4] + index * 0.1 * xcurv_ego[0]
                    <= obs_traj[4, index] + veh_length + safety_margin
                ):
                    opti.subject_to(diffey >= veh_width + safety_margin)
                else:
                    pass
        for index in range(num_horizon):
            if index > 1:
                cost += 30 * ((opti_xvar[5, index] - opti_xvar[5, index - 1]) ** 2)
        cost += -200 * (opti_xvar[4, -1] - opti_xvar[4, 0])  # 500
        for j in range(num_horizon + 1):
            s_tmp = ego.xcurv[4] + 1.0 * j * ego.xcurv[0] * 0.1
            s_tmp = np.clip(s_tmp, bezier_xcurvs[pos_index, 0, 0], bezier_xcurvs[pos_index, -1, 0])
            ey_bezier = bezier_funcs[pos_index](s_tmp)
            cost += 20 * (opti_xvar[5, j] - ey_bezier) ** 2  # 40
            cost += 20 * (opti_xvar[4, j] - s_tmp) ** 2  # 40
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        solution_xvar = np.zeros((X_DIM, num_horizon + 1))
        for j in range(num_horizon):
            s_j = j * ego.xcurv[0] * 0.1 + ego.xcurv[4]
            # set initial value of s
            opti.set_initial(opti_xvar[4, j], s_j)
            # when the reference Bezier curve is across the start line and ego's position is on the next lap
            if (
                bezier_xcurvs[pos_index, -1, 0] > track.lap_length
                and s_j < bezier_xcurvs[pos_index, 0, 0]
            ):
                s_j = s_j + track.lap_length
            # when the reference Bezier curve is accross the start line and ego's position is on the previous lap
            if (
                bezier_xcurvs[pos_index, 0, 0] < 0
                and s_j - track.lap_length >= bezier_xcurvs[pos_index, 0, 0]
            ):
                s_j = s_j - track.lap_length
            s_j = np.clip(s_j, bezier_xcurvs[pos_index, 0, 0], bezier_xcurvs[pos_index, -1, 0])
            # set initial value of ey
            ey_j = bezier_funcs[pos_index](s_j)
            opti.set_initial(opti_xvar[5, j], ey_j)
            opti.set_initial(opti_xvar[0, j], ego.xcurv[0])
        start_time = datetime.datetime.now()
        opti.minimize(cost)
        opti.solver("ipopt", option)
        try:
            sol = opti.solve()
            solution_xvar = sol.value(opti_xvar)
            cost = sol.value(cost)
        except RuntimeError:
            for j in range(0, num_horizon + 1):
                stmp = xcurv_ego[4] + 1.1 * j * 0.1 * xcurv_ego[0]
                solution_xvar[0, j] = 1.1 * xcurv_ego[0]
                solution_xvar[4, j] = stmp
                stmp = np.clip(
                    stmp, bezier_xcurvs[pos_index, 0, 0], bezier_xcurvs[pos_index, -1, 0]
                )
                solution_xvar[5, j] = bezier_funcs[pos_index](stmp)
            cost = float("inf")
        end_time = datetime.datetime.now()
        solve_time = (end_time - start_time).total_seconds()
        dict_traj[pos_index] = solution_xvar
        dict_solve_time[pos_index] = solve_time
        dict_cost[pos_index] = cost
