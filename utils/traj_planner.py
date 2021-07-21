import datetime
import numpy as np
import casadi as ca
from utils import lmpc_helper, vehicle_dynamics
from utils.racing_game_helper import *
from casadi import *
from scipy import sparse
from scipy.sparse import vstack
from scipy.interpolate import interp1d
import copy


class OvertakeTrajPlanner:
    def __init__(self, racing_game_param):
        self.racing_game_param = racing_game_param
        self.vehicles = None
        self.agent_name = None
        self.track = None
        self.opti_traj_xcurv = None
        self.matrix_Atv = None
        self.matrix_Btv = None
        self.matrix_Ctv = None

    def get_overtake_flag(self, xcurv_ego):
        overtake_flag = False
        vehicles_interest = {}
        for name in list(self.vehicles):
            if name != self.agent_name:
                if check_ego_agent_distance(self.vehicles[self.agent_name], self.vehicles[name], self.racing_game_param, self.track.lap_length):
                    overtake_flag = True
                    vehicles_interest[name] = self.vehicles[name]
        return overtake_flag, vehicles_interest

    def get_local_traj(self, xcurv_ego, time, vehicles_interest, matrix_Atv, matrix_Btv, matrix_Ctv, old_ey):
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
            elif (
                vehicles_interest[name].xcurv[5] >= vehicles_interest[sorted_vehicles[0]].xcurv[5]
            ):
                sorted_vehicles.insert(0, name)
            elif (
                vehicles_interest[name].xcurv[5] <= vehicles_interest[sorted_vehicles[0]].xcurv[5]
            ):
                sorted_vehicles.append(name)
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
            obs_infos[name] = obs_traj
            # save the position information of other agent
            veh_infos[num, :] = vehicles[name].xcurv[4], max(obs_traj.T[:, 5]), min(obs_traj.T[:, 5])
            num += 1
        # get agents infos and reference Bezier curve
        agent_info = get_agent_info(vehicles, sorted_vehicles, track)
        bezier_control_point = get_bezier_control_points(vehicles_interest, veh_infos, agent_info, self.racing_game_param, track, self.opti_traj_xcurv)
        bezier_xcurvs = np.zeros((num_veh + 1, num_horizon_planner + 1, 2))
        bezier_funcs = []
        for index in range(num_veh + 1):
            for j in range(num_horizon_planner + 1):
                t = j * (1.0 / num_horizon_planner)
                # s and ey for each point
                bezier_xcurvs[index, j, :] = get_bezier_curve(bezier_control_point[index, :, :], t)
            bezier_funcs.append(
                interp1d(
                    bezier_xcurvs[index, :, 0],
                    bezier_xcurvs[index, :, 1],
                )
            )
        target_traj_xcurv, direction_flag = self.solve_optimization_problem(sorted_vehicles, obs_infos, old_ey, bezier_xcurvs, bezier_funcs)
        end_timer = datetime.datetime.now()
        solver_time = (end_timer - start_timer).total_seconds()
        print("local planner solver time: {}".format(solver_time))
        target_traj_xglob = get_traj_xglob(target_traj_xcurv, track)
        bezier_line_xcurv = np.zeros((num_horizon_planner+1, X_DIM))
        bezier_line_xcurv[:, 4:6] = bezier_xcurvs[direction_flag, :, :]
        bezier_xglob = get_traj_xglob(bezier_line_xcurv, track)
        #debug_plot(track, vehicles, target_traj_xglob)
        return (
            target_traj_xcurv,
            target_traj_xglob,
            direction_flag,
            sorted_vehicles,
            bezier_xglob
        )

    def solve_optimization_problem(self, sorted_vehicles, obs_infos, old_ey, bezier_xcurvs, bezier_funcs):
        num_horizon = self.racing_game_param.num_horizon_planner
        num_veh = len(sorted_vehicles)
        ego = self.vehicles[self.agent_name]
        veh_length = ego.param.length
        veh_width = ego.param.width
        track = self.track
        safety_margin = 0.15
        optis = []
        opti_xvars = []
        opti_uvars = []
        costs = []
        for index in range(num_veh+1):
            optis.append(ca.Opti())
            opti_xvars.append(optis[index].variable(X_DIM, num_horizon+1))
            opti_uvars.append(optis[index].variable(U_DIM, num_horizon))
            costs.append(0)
            # initial state constraint
            optis[index].subject_to(opti_xvars[index][:, 0] == ego.xcurv)
        for index in range(num_horizon):
            for j in range(num_veh+1):
                # dynamic state update constraint
                optis[j].subject_to(opti_xvars[j][:, index + 1] == mtimes(self.matrix_Atv[index], opti_xvars[j][:, index]) + mtimes(self.matrix_Btv[index], opti_uvars[j][:, index]) + self.matrix_Ctv[index])
                # min and max of vx, ey
                optis[j].subject_to(opti_xvars[j][0, index + 1] <= 5.0)
                optis[j].subject_to(opti_xvars[j][5, index] <= track.width)
                optis[j].subject_to(opti_xvars[j][5, index] >= -track.width)
                # min and max of delta
                optis[j].subject_to(opti_uvars[j][0, index]<=0.5) 
                optis[j].subject_to(opti_uvars[j][0, index]>=-0.5)
                # min and max of a
                optis[j].subject_to(opti_uvars[j][1, index]<=1.5)
                optis[j].subject_to(opti_uvars[j][1, index]>=-1.5)
                # constraint on the left, first line is the track boundary
                if j == 0:
                    pass
                else:
                    name = sorted_vehicles[j - 1]
                    obs_traj = obs_infos[name]
                    while obs_traj[4, index] > track.lap_length:
                        obs_traj[4, index] = obs_traj[4, index] - track.lap_length
                    diffs = opti_xvars[j][4, index] - obs_traj[4, index]
                    diffey = opti_xvars[j][5, index] - obs_traj[5, index]
                    optis[j].subject_to(diffs**2 + diffey**2 >= veh_length**2 + veh_width**2+safety_margin)
                # constraint on the right, last line is the track boundary
                if j == num_veh:
                    pass
                else:
                    name = sorted_vehicles[j]
                    obs_traj = obs_infos[name]
                    while obs_traj[4, index] > track.lap_length:
                        obs_traj[4, index] = obs_traj[4, index] - track.lap_length
                    diffs = opti_xvars[j][4, index] - obs_traj[4, index]
                    diffey = opti_xvars[j][5, index] - obs_traj[5, index]
                    optis[j].subject_to(diffs**2 + diffey**2  >= veh_length**2 + veh_width**2+0.4*veh_length)
                if index >1:
                    costs[j] += 100*((opti_xvars[j][5, index] - opti_xvars[j][5, index - 1])**2)
                    costs[j] += ca.mtimes((opti_uvars[j][:, index] - opti_uvars[j][:, index -1]).T, ca.mtimes(self.racing_game_param.matrix_dR_planner, opti_uvars[j][:, index] - opti_uvars[j][:, index-1]))
        for index in range(num_veh+1):
            costs[index] += -5*(opti_xvars[index][4, -1] - opti_xvars[index][4, 0])
            if old_ey is None:
                pass
            else:
                costs[index] += 30*((opti_xvars[index][5, -1] - old_ey)**2) 
        for index in range(num_veh+1):
            for j in range(num_horizon+1):
                s_tmp = ego.xcurv[4] + j*ego.xcurv[0]*0.1
                s_tmp = np.clip(s_tmp, bezier_xcurvs[index, 0, 0], bezier_xcurvs[index, -1, 0])
                ey_bezier = bezier_funcs[index](s_tmp)
                costs[index] += 20*(opti_xvars[index][5, j] - ey_bezier)**2
                costs[index] += 20*(opti_xvars[index][4, j] - s_tmp)**2
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        solution_xvar = np.zeros((num_veh+1, 6, num_horizon + 1))
        for index in range(num_veh+1):
            for j in range (num_horizon):
                s_j = j*ego.xcurv[0]*0.1 + ego.xcurv[4]
                # set initial value of s
                optis[index].set_initial(opti_xvars[index][4, j], s_j)
                # when the reference Bezier curve is across the start line and ego's position is on the next lap
                if (bezier_xcurvs[index, -1, 0]>track.lap_length and s_j < bezier_xcurvs[index, 0, 0]):
                    s_j = s_j + track.lap_length
                # when the reference Bezier curve is accross the start line and ego's position is on the previous lap
                if (bezier_xcurvs[index, 0, 0] < 0 and s_j - track.lap_length >= bezier_xcurvs[index, 0, 0]):
                    s_j = s_j - track.lap_length               
                s_j = np.clip(s_j, bezier_xcurvs[index, 0, 0], bezier_xcurvs[index, -1, 0])
                # set initial value of ey
                ey_j = bezier_funcs[index](s_j)
                optis[index].set_initial(opti_xvars[index][5, j], ey_j)
                optis[index].set_initial(opti_xvars[index][0, j], ego.xcurv[0])
        for index in range(num_veh+1):
            optis[index].minimize(costs[index])
            optis[index].solver("ipopt", option)
            try:
                sol = optis[index].solve()
                solution_xvar[index, :, :] = sol.value(opti_xvars[index])
                costs[index] = sol.value(costs[index])
            except RuntimeError:
                solution_xvar[index, :, :] = optis[index].debug.value(opti_xvars[index])
                costs[index] = float("inf")
        direction_flag = costs.index(min(costs))
        traj_xcurv = solution_xvar[direction_flag, :, :].T
        cost = min(costs)
        if cost == float(inf):
            print("trajectory planner failed")
        return traj_xcurv, direction_flag
  