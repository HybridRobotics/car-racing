import datetime
import numpy as np
import casadi as ca
from scripts.utils import racing_env
from scripts.control import lmpc_helper
from scripts.planning.planner_helper import *
from casadi import *
from scipy import sparse
from scipy.sparse import vstack
from scipy.interpolate import interp1d
from scripts.utils.constants import *


class OvertakePathPlanner:
    def __init__(self, racing_game_param):
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
                if check_ego_agent_distance(self.vehicles[self.agent_name], self.vehicles[name], self.racing_game_param, self.track.lap_length):
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
            elif (
                vehicles_interest[name].xcurv[5] >= vehicles_interest[sorted_vehicles[0]].xcurv[5]
            ):
                sorted_vehicles.insert(0, name)
            elif (
                vehicles_interest[name].xcurv[5] <= vehicles_interest[sorted_vehicles[0]].xcurv[5]
            ):
                sorted_vehicles.append(name)
            num += 1
        # get maximum and minimum infos of agent(s)
        agent_infos = get_agent_info(vehicles, sorted_vehicles, track)
        for index in range(num_veh):
            name = sorted_vehicles[index]
            if vehicles[name].no_dynamics:
                obs_traj, _ = vehicles[name].get_trajectory_nsteps(
                    time,
                    self.racing_game_param.timestep,
                    num_horizon + 1,
                )
            else:
                obs_traj, _ = vehicles[name].get_trajectory_nsteps(
                    num_horizon + 1
                )
            # save the position information of other agent
            obs_traj_infos[index, :] = vehicles[name].xcurv[4], max(obs_traj.T[:, 5]), min(obs_traj.T[:, 5])
        func_optimal_ey = interp1d(optimal_traj_xcurv[:, 4], optimal_traj_xcurv[:, 5])
        func_optimal_vx = interp1d(optimal_traj_xcurv[:, 4], optimal_traj_xcurv[:, 0])
        bezier_control_points = get_bezier_control_points(vehicles_interest, obs_traj_infos, agent_infos, self.racing_game_param, track, self.opti_traj_xcurv,sorted_vehicles, xcurv_ego)
        bezier_xcurvs = np.zeros((num_veh + 1, num_horizon + 1, 2))
        bezier_funcs = []
        for index in range(num_veh + 1):
            for j in range(num_horizon + 1):
                t = j * (1.0 / num_horizon)
                # s and ey for each point
                bezier_xcurvs[index, j, :] = get_bezier_curve(bezier_control_points[index, :, :], t)
            bezier_funcs.append(
                interp1d(
                    bezier_xcurvs[index, :, 0],
                    bezier_xcurvs[index, :, 1],
                )
            )
        agents_front_ranges, agents_rear_ranges = self.get_agents_range(num_veh, obs_traj_infos)
        best_ey, direction_flag = self.solve_optimization_problem(sorted_vehicles, bezier_xcurvs, bezier_funcs, agent_infos, obs_traj_infos, agents_front_ranges, agents_rear_ranges, func_optimal_ey, bezier_control_points)
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
        if target_traj_xcurv[-1, 4] >=track.lap_length:
            if target_traj_xcurv[-1, 4] - track.lap_length <optimal_traj_xcurv[0, 4]:
                vx_target = func_optimal_vx(optimal_traj_xcurv[0, 4])
            else:    
                vx_target = func_optimal_vx(target_traj_xcurv[-1, 4] - track.lap_length)
        elif target_traj_xcurv[-1, 4] <optimal_traj_xcurv[0, 4]:
            vx_target = func_optimal_vx(optimal_traj_xcurv[0, 4])
        else:
            vx_target = func_optimal_vx(target_traj_xcurv[-1, 4])
        delta_t = 2*(target_traj_xcurv[-1, 4] - xcurv_ego[4])/(vx_target + xcurv_ego[0])
        a_target = (vx_target - xcurv_ego[0])/delta_t
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
        

        all_bezier_xcurv = np.zeros((num_veh+1, num_horizon+1,X_DIM))
        all_local_traj_xcurv = np.zeros((num_veh+1, num_horizon+1, X_DIM))
        all_bezier_xcurv[:,:,4:6] = bezier_xcurvs[:,:,:]
        

        all_local_traj_xglob = np.zeros((num_veh+1, num_horizon+1, X_DIM))
        all_bezier_xglob = np.zeros((num_veh+1,num_horizon+1, X_DIM))        
        for index in range(num_veh+1):
            all_bezier_xglob[index,:,:] = get_traj_xglob(all_bezier_xcurv[index,:,:],track)

        # debug_plot(track, vehicles, target_traj_xglob)
        return (
            target_traj_xcurv,
            target_traj_xglob,
            direction_flag,
            sorted_vehicles,
            bezier_xglob,
            solver_time,
            all_bezier_xglob,
            all_local_traj_xglob
        )

    def get_speed_info(self, target_traj_xcurv, xcurv, a_target):        
        num_horizon = self.racing_game_param.num_horizon_planner
        traj_xcurv = np.zeros((num_horizon + 1, 6))
        traj_xcurv = target_traj_xcurv
        traj_xcurv[0,:] = xcurv
        for index in range (num_horizon):
            traj_xcurv[index, 0] = (xcurv[0]**2 + 2*a_target*(traj_xcurv[index, 4] - xcurv[4]))**0.5
        return traj_xcurv

    def get_agents_range(self, num_veh, obs_traj_infos):
        veh_length = self.vehicles[self.agent_name].param.length
        track = self.track
        safety_factor = self.racing_game_param.safety_factor
        agents_front_ranges = np.zeros((num_veh, 1))
        agents_rear_ranges = np.zeros((num_veh, 1))
        for index in range(num_veh):
            agents_front_ranges[index, 0] = (
                obs_traj_infos[index, 0] + safety_factor * veh_length
            )
            agents_rear_ranges[index, 0] = (
                obs_traj_infos[index, 0] - safety_factor * veh_length
            )
            while agents_front_ranges[index, 0] > track.lap_length:
                agents_front_ranges[index, 0] = (
                    agents_front_ranges[index, 0] - track.lap_length
                )
            while agents_rear_ranges[index, 0] > track.lap_length:
                agents_rear_ranges[index, 0] = (
                    agents_rear_ranges[index, 0] - track.lap_length
                )
        return agents_front_ranges, agents_rear_ranges
    
    def solve_optimization_problem(self, sorted_vehicles, bezier_xucrvs, bezier_funcs, agent_infos, obs_traj_infos, agents_front_ranges, agents_rear_ranges, func_optimal_ey, bezier_control_points):
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
                costs[j] += (1 - alpha) * (
                    (opti_vars[j][index] - func_optimal_ey(s_tmp)) ** 2
                )
                # compared with reference Bezier curve
                costs[j] += alpha * (
                    (opti_vars[j][index] - bezier_funcs[j](s_tmp)) ** 2
                )
                # changing rate
                if index >= 1:
                    costs[j] += 100 * (
                        (opti_vars[j][index] - opti_vars[j][index - 1]) ** 2
                    )
            # constraint for optimization
            for j in range(num_veh + 1):
                # initial state constraint
                if index == 0:
                    optis[j].subject_to(opti_vars[j][0] == ego.xcurv[5])
                # final state constraint
                if index == num_horizon:
                    optis[j].subject_to(
                        opti_vars[j][num_horizon]
                        == bezier_control_points[j, 3, 1]
                    )
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
                                < (
                                    obs_traj_infos[j - 1, 2]
                                    - safety_factor * veh_width
                                )
                            )
                # constraint on the right, last line is the track boundary
                if j == num_veh:
                    pass
                else:
                    if (s_tmp < agents_rear_ranges[j, 0]) or (
                        s_tmp > agents_front_ranges[j, 0]
                    ):
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
        if min_cost == float(inf):
            print("path planner failed")
        return best_ey, direction_flag