import datetime
import numpy as np
import sympy as sp
from scripts.utils.constants import *
from scripts.utils import racing_env
from scripts.system import vehicle_dynamics
from scripts.control import control, lmpc_helper
from scripts.planning import overtake_path_planner, overtake_traj_planner
from scipy.interpolate import interp1d
from pathos.multiprocessing import ProcessingPool as Pool
from cvxopt.solvers import qp
from cvxopt import spmatrix, matrix, solvers
import copy


# Base Controller
class ControlBase:
    def __init__(self):
        self.agent_name = None
        self.time = 0.0
        self.timestep = None
        self.x = None
        self.xglob = None
        self.u = None
        self.realtime_flag = False
        # store the information (e.g. states, inputs) of current lap
        self.lap_times, self.lap_xcurvs, self.lap_xglobs, self.lap_inputs = [], [], [], []
        self.lap_times.append(self.time)
        # store the information (e.g. state, inputs) of the whole simulation
        self.times, self.xglobs, self.xcurvs, self.inputs = [], [], [], []
        self.laps = 0
        self.track = None
        self.opti_traj_xcurv = None
        self.opti_traj_xglob = None

    def set_track(self, track):
        self.track = track
        self.lap_length = track.lap_length
        self.point_and_tangent = track.point_and_tangent
        self.lap_width = track.width

    def set_opti_traj(self, opti_traj_xcurv, opti_traj_xglob):
        self.opti_traj_xcurv = opti_traj_xcurv
        self.opti_traj_xglob = opti_traj_xglob

    def set_racing_sim(self, racing_sim):
        self.racing_sim = racing_sim

    def set_timestep(self, timestep):
        self.timestep = timestep

    def set_target_speed(self, vt):
        self.vt = vt

    def set_target_deviation(self, eyt):
        self.eyt = eyt

    def set_state(self, xcurv, xglob):
        self.x = xcurv
        self.xglob = xglob

    def calc_input(self):
        pass

    def get_input(self):
        return self.u

    def update_memory(self, current_lap):
        xcurv = copy.deepcopy(self.x)
        xglob = copy.deepcopy(self.xglob)
        time = copy.deepcopy(self.time)
        if xcurv[4] > self.lap_length * (current_lap + 1):
            self.lap_xglobs.append(xglob)
            self.lap_times.append(time)
            xcurv[4] = xcurv[4] - current_lap * self.lap_length
            self.lap_xcurvs.append(xcurv)
            self.lap_inputs.append(self.u)
            self.xglobs.append(self.lap_xglobs)
            self.times.append(self.lap_times)
            self.xcurvs.append(self.lap_xcurvs)
            self.inputs.append(self.lap_inputs)
            x = copy.deepcopy(self.x)
            x[4] = x[4] - self.lap_length * (current_lap + 1)
            self.laps = self.laps + 1
            self.lap_xglobs, self.lap_xcurvs, self.lap_inputs, self.lap_times = [], [], [], []
            self.lap_xglobs.append(xglob)
            self.lap_times.append(time)
            self.lap_xcurvs.append(x)
        else:
            xcurv[4] = xcurv[4] - current_lap * self.lap_length
            self.lap_xglobs.append(xglob)
            self.lap_times.append(time)
            self.lap_xcurvs.append(xcurv)
            self.lap_inputs.append(self.u)


class PidTrackingBase(ControlBase):
    def __init__(self, vt=0.6, eyt=0.0):
        ControlBase.__init__(self)
        self.set_target_speed(vt)
        self.set_target_deviation(eyt)

    def calc_input(self):
        xtarget = np.array([self.vt, 0, 0, 0, 0, self.eyt]).reshape(X_DIM, 1)
        self.u = control.pid(self.x, xtarget)
        if self.agent_name == "ego":
            if self.realtime_flag == False:
                vehicles = self.racing_sim.vehicles
            else:
                vehicles = self.vehicles
            vehicles["ego"].local_trajs.append(None)
            vehicles["ego"].vehicles_interest.append(None)
            vehicles["ego"].splines.append(None)
            vehicles["ego"].all_splines.append(None)
            vehicles["ego"].all_local_trajs.append(None)
            vehicles["ego"].lmpc_prediction.append(None)
            vehicles["ego"].mpc_cbf_prediction.append(None)
        self.time += self.timestep


class MpcTrackingParam:
    def __init__(
        self,
        matrix_A=np.genfromtxt("data/sys/LTI/matrix_A.csv", delimiter=","),
        matrix_B=np.genfromtxt("data/sys/LTI/matrix_B.csv", delimiter=","),
        matrix_Q=np.diag([10.0, 0.0, 0.0, 4.0, 0.0, 40.0]),
        matrix_R=np.diag([0.1, 0.1]),
        vt=0.6,
        eyt=0.0,
        num_horizon=10,
    ):
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.vt = vt
        self.eyt = eyt
        self.num_horizon = num_horizon


class MpcTrackingBase(ControlBase):
    def __init__(self, mpc_lti_param, system_param):
        ControlBase.__init__(self)
        self.set_target_speed(mpc_lti_param.vt)
        self.set_target_deviation(mpc_lti_param.eyt)
        self.mpc_lti_param = mpc_lti_param
        self.system_param = system_param

    def calc_input(self):
        xtarget = np.array([self.vt, 0, 0, 0, 0, self.eyt]).reshape(X_DIM, 1)
        self.u = control.mpc_lti(
            self.x, xtarget, self.mpc_lti_param, self.system_param, self.track)
        if self.agent_name == "ego":
            if self.realtime_flag == False:
                vehicles = self.racing_sim.vehicles
            else:
                vehicles = self.vehicles
            vehicles["ego"].local_trajs.append(None)
            vehicles["ego"].vehicles_interest.append(None)
            vehicles["ego"].splines.append(None)
            vehicles["ego"].all_splines.append(None)
            vehicles["ego"].all_local_trajs.append(None)
            vehicles["ego"].lmpc_prediction.append(None)
            vehicles["ego"].mpc_cbf_prediction.append(None)
        self.time += self.timestep


class MpcCbfRacingParam:
    def __init__(
        self,
        matrix_A=np.genfromtxt("data/sys/LTI/matrix_A.csv", delimiter=","),
        matrix_B=np.genfromtxt("data/sys/LTI/matrix_B.csv", delimiter=","),
        matrix_Q=np.diag([10.0, 0.0, 0.0, 4.0, 0.0, 40.0]),
        matrix_R=np.diag([0.1, 0.1]),
        vt=0.6,
        eyt=0.0,
        num_horizon=10,
        alpha=0.6,
    ):
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.vt = vt
        self.eyt = eyt
        self.num_horizon = num_horizon
        self.alpha = alpha


class MpcCbfRacingBase(ControlBase):
    def __init__(self, mpc_cbf_param, system_param):
        ControlBase.__init__(self)
        self.set_target_speed(mpc_cbf_param.vt)
        self.set_target_deviation(mpc_cbf_param.eyt)
        self.realtime_flag = None
        self.mpc_cbf_param = mpc_cbf_param
        self.system_param = system_param

    def calc_input(self):
        xtarget = np.array([self.vt, 0, 0, 0, 0, self.eyt]).reshape(X_DIM, 1)
        # determine if it is a real-time simulator
        if self.realtime_flag == False:
            self.u = control.mpccbf(
                self.x,
                xtarget,
                self.mpc_cbf_param,
                self.racing_sim.vehicles,
                self.agent_name,
                self.racing_sim.track.lap_length,
                self.time,
                self.timestep,
                self.realtime_flag,
                self.track,
                self.system_param
            )
        elif self.realtime_flag == True:
            self.u = control.mpccbf(
                self.x,
                xtarget,
                self.mpc_cbf_param,
                self.vehicles,
                self.agent_name,
                self.lap_length,
                self.time,
                self.timestep,
                self.realtime_flag,
                self.track,
                self.system_param
            )
        else:
            pass
        if self.agent_name == "ego":
            if self.realtime_flag == False:
                vehicles = self.racing_sim.vehicles
            else:
                vehicles = self.vehicles
            vehicles["ego"].local_trajs.append(None)
            vehicles["ego"].vehicles_interest.append(None)
            vehicles["ego"].splines.append(None)
            vehicles["ego"].all_splines.append(None)
            vehicles["ego"].all_local_trajs.append(None)
            vehicles["ego"].lmpc_prediction.append(None)
            vehicles["ego"].mpc_cbf_prediction.append(None)
        self.time += self.timestep


class LmpcRacingParam:
    def __init__(
        self,
        matrix_Q=0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        matrix_R=1 * np.diag([1.0, 0.25]),
        matrix_Qslack=5 * np.diag([10, 0, 0, 1, 10, 0]),
        matrix_dR=5 * np.diag([0.8, 0.0]),
        num_ss_points=32 + 12,
        num_ss_iter=2,
        num_horizon=12,
        shift=0,
        timestep=None,
        lap_number=None,
        time_lmpc=None,
    ):
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.matrix_Qslack = matrix_Qslack
        self.matrix_dR = matrix_dR
        self.num_ss_points = num_ss_points
        self.num_ss_iter = num_ss_iter
        self.num_horizon = num_horizon
        self.shift = shift
        self.timestep = timestep
        self.lap_number = lap_number
        self.time_lmpc = time_lmpc


class RacingGameParam:
    def __init__(
        self,
        matrix_A=np.genfromtxt("data/sys/LTI/matrix_A.csv", delimiter=","),
        matrix_B=np.genfromtxt("data/sys/LTI/matrix_B.csv", delimiter=","),
        matrix_Q=np.diag([10.0, 0.0, 0.0, 5.0, 0.0, 50.0]),
        matrix_R=np.diag([0.1, 0.1]),
        matrix_R_planner=1 * np.diag([5, 0.10]),
        matrix_dR_planner=5 * np.diag([1.8, 0.0]),
        bezier_order=3,
        safety_factor=4.5,
        num_horizon_ctrl=10,
        num_horizon_planner=10,
        planning_prediction_factor=0.5,  # 2.0,
        alpha=0.98,
        timestep=None,
    ):
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.matrix_R_planner = matrix_R_planner
        self.matrix_dR_planner = matrix_dR_planner
        self.num_horizon_ctrl = num_horizon_ctrl
        self.num_horizon_planner = num_horizon_planner
        self.planning_prediction_factor = planning_prediction_factor
        self.alpha = alpha
        self.timestep = timestep
        self.bezier_order = bezier_order
        self.safety_factor = safety_factor


class LmpcRacingGameBase(ControlBase):
    def __init__(self, lmpc_param, racing_game_param=None, system_param=None):
        ControlBase.__init__(self)
        self.path_planner = False
        self.lmpc_param = lmpc_param
        self.racing_game_param = racing_game_param
        self.system_param = system_param
        if self.path_planner:
            self.overtake_planner = overtake_path_planner.OvertakePathPlanner(
                racing_game_param)
        else:
            self.overtake_planner = overtake_traj_planner.OvertakeTrajPlanner(
                racing_game_param)
        self.x_pred = None
        self.u_pred = None
        self.lin_points = None
        self.lin_input = None
        self.ss_point_selected_tot = None
        self.Qfun_selected_tot = None
        num_points = int(lmpc_param.time_lmpc / lmpc_param.timestep) + 1
        # Time at which each j-th iteration is completed
        self.time_ss = 10000 * np.ones(lmpc_param.lap_number).astype(int)
        self.ss_xcurv = 10000 * np.ones(
            (num_points, X_DIM, lmpc_param.lap_number)
        )  # Sampled Safe SS
        # Input associated with the points in SS
        self.u_ss = 10000 * np.ones((num_points, U_DIM, lmpc_param.lap_number))
        # Qfun: cost-to-go from each point in SS
        self.Qfun = 0 * np.ones((num_points, lmpc_param.lap_number))
        # SS in global (X-Y) used for plotting
        self.ss_glob = 10000 * \
            np.ones((num_points, X_DIM, lmpc_param.lap_number))
        # Initialize the controller iteration
        self.iter = 0
        self.time_in_iter = 0
        self.p = Pool(4)
        self.openloop_prediction = None
        self.old_ey = None
        self.old_direction_flag = None

    def set_vehicles_track(self):
        if self.realtime_flag == False:
            vehicles = self.racing_sim.vehicles
            self.overtake_planner.track = self.track
        else:
            vehicles = self.vehicles
        self.overtake_planner.vehicles = vehicles

    def calc_input(self):
        self.overtake_planner.agent_name = self.agent_name
        self.overtake_planner.opti_traj_xcurv = self.opti_traj_xcurv
        matrix_Atv, matrix_Btv, matrix_Ctv, _ = self.estimate_ABC()
        x = copy.deepcopy(self.x)
        while x[4] > self.lap_length:
            x[4] = x[4] - self.lap_length
        if self.u_pred is None:
            u_old = np.zeros((1, 2))
        else:
            u_old = copy.deepcopy(self.u_pred[0, :])
        overtake_flag, vehicles_interest = self.overtake_planner.get_overtake_flag(
            x)
        if overtake_flag == False:
            (
                self.u_pred,
                self.x_pred,
                self.ss_point_selected_tot,
                self.Qfun_selected_tot,
                self.lin_points,
                self.lin_input,
            ) = control.lmpc(
                x,
                self.lmpc_param,
                matrix_Atv,
                matrix_Btv,
                matrix_Ctv,
                self.ss_xcurv,
                self.Qfun,
                self.iter,
                self.lap_length,
                self.lap_width,
                u_old,
                self.system_param
            )
            self.u = self.u_pred[0, :]
            self.old_ey = None
            self.old_direction_flag = None
            iter = self.iter
            self.openloop_prediction.predicted_xcurv[:,
                                                     :, self.time_in_iter, iter] = self.x_pred
            self.openloop_prediction.predicted_u[:, :,
                                                 self.time_in_iter, iter] = self.u_pred
            self.openloop_prediction.ss_used[
                :, :, self.time_in_iter, iter
            ] = self.ss_point_selected_tot
            self.openloop_prediction.Qfun_used[:,
                                               self.time_in_iter, iter] = self.Qfun_selected_tot
            self.add_point(self.x, self.u, self.time_in_iter)
            self.time_in_iter = self.time_in_iter + 1
            x_pred_xglob = np.zeros((self.lmpc_param.num_horizon + 1, X_DIM))
            for jjj in range(0, self.lmpc_param.num_horizon + 1):
                xxx, yyy = self.track.get_global_position(
                    self.x_pred[jjj, 4], self.x_pred[jjj, 5])
                psipsi = self.track.get_orientation(
                    self.x_pred[jjj, 4], self.x_pred[jjj, 5])
                x_pred_xglob[jjj, 0:3] = self.x_pred[jjj, 0:3]
                x_pred_xglob[jjj, 3] = psipsi
                x_pred_xglob[jjj, 4] = xxx
                x_pred_xglob[jjj, 5] = yyy
            self.overtake_planner.vehicles["ego"].local_trajs.append(None)
            self.overtake_planner.vehicles["ego"].vehicles_interest.append(
                None)
            self.overtake_planner.vehicles["ego"].splines.append(None)
            self.overtake_planner.vehicles["ego"].solver_time.append(None)
            self.overtake_planner.vehicles["ego"].all_splines.append(None)
            self.overtake_planner.vehicles["ego"].all_local_trajs.append(None)
            self.overtake_planner.vehicles["ego"].lmpc_prediction.append(
                x_pred_xglob)
            self.overtake_planner.vehicles["ego"].mpc_cbf_prediction.append(
                None)
        else:
            if self.path_planner:
                (
                    overtake_traj_xcurv,
                    overtake_traj_xglob,
                    direction_flag,
                    sorted_vehicles,
                    bezier_xglob,
                    solve_time,
                    all_bezier_xglob,
                    all_traj_xglob,
                ) = self.overtake_planner.get_local_path(x, self.time, vehicles_interest)
            else:
                (
                    overtake_traj_xcurv,
                    overtake_traj_xglob,
                    direction_flag,
                    sorted_vehicles,
                    bezier_xglob,
                    solve_time,
                    all_bezier_xglob,
                    all_traj_xglob,
                ) = self.overtake_planner.get_local_traj(
                    x,
                    self.time,
                    vehicles_interest,
                    matrix_Atv,
                    matrix_Btv,
                    matrix_Ctv,
                    self.old_ey,
                    self.old_direction_flag,
                )
            self.old_ey = overtake_traj_xcurv[-1, 5]
            self.old_direction_flag = direction_flag
            self.overtake_planner.vehicles["ego"].local_trajs.append(
                overtake_traj_xglob)
            self.overtake_planner.vehicles["ego"].vehicles_interest.append(
                vehicles_interest)
            self.overtake_planner.vehicles["ego"].splines.append(bezier_xglob)
            self.overtake_planner.vehicles["ego"].solver_time.append(
                solve_time)
            self.overtake_planner.vehicles["ego"].all_splines.append(
                all_bezier_xglob)
            self.overtake_planner.vehicles["ego"].all_local_trajs.append(
                all_traj_xglob)
            self.u, x_pred = control.mpc_multi_agents(
                x,
                self.racing_game_param,
                self.track,
                matrix_Atv,
                matrix_Btv,
                matrix_Ctv,
                self.system_param,
                target_traj_xcurv=overtake_traj_xcurv,
                vehicles=self.overtake_planner.vehicles,
                agent_name=self.agent_name,
                direction_flag=direction_flag,
                target_traj_xglob=overtake_traj_xglob,
                sorted_vehicles=sorted_vehicles,
            )
            x_pred_xglob = np.zeros((self.racing_game_param.num_horizon_planner + 1, X_DIM))
            for jjj in range(0, self.racing_game_param.num_horizon_planner + 1):
                xxx, yyy = self.track.get_global_position(
                    x_pred[jjj, 4], x_pred[jjj, 5])
                psipsi = self.track.get_orientation(
                    x_pred[jjj, 4], x_pred[jjj, 5])
                x_pred_xglob[jjj, 0:3] = x_pred[jjj, 0:3]
                x_pred_xglob[jjj, 3] = psipsi
                x_pred_xglob[jjj, 4] = xxx
                x_pred_xglob[jjj, 5] = yyy
            self.overtake_planner.vehicles["ego"].lmpc_prediction.append(None)
            self.overtake_planner.vehicles["ego"].mpc_cbf_prediction.append(
                x_pred_xglob)
        self.time += self.timestep

    def estimate_ABC(self):
        lin_points = self.lin_points
        lin_input = self.lin_input
        num_horizon = self.lmpc_param.num_horizon
        ss_xcurv = self.ss_xcurv
        u_ss = self.u_ss
        time_ss = self.time_ss
        point_and_tangent = self.point_and_tangent
        timestep = self.timestep
        iter = self.iter
        p = self.p
        Atv = []
        Btv = []
        Ctv = []
        index_used_list = []
        lap_used_for_linearization = self.lmpc_param.num_ss_iter
        used_iter = range(iter - lap_used_for_linearization, iter)
        max_num_point = 40
        for i in range(0, num_horizon):
            (Ai, Bi, Ci, index_selected,) = lmpc_helper.regression_and_linearization(
                lin_points,
                lin_input,
                used_iter,
                ss_xcurv,
                u_ss,
                time_ss,
                max_num_point,
                qp,
                matrix,
                point_and_tangent,
                timestep,
                i,
            )
            Atv.append(Ai)
            Btv.append(Bi)
            Ctv.append(Ci)
            index_used_list.append(index_selected)
        return Atv, Btv, Ctv, index_used_list

    def add_point(self, x, u, i):
        counter = self.time_ss[self.iter - 1]
        self.ss_xcurv[counter + i + 1, :, self.iter - 1] = x + np.array(
            [0, 0, 0, 0, self.lap_length, 0]
        )
        self.u_ss[counter + i + 1, :, self.iter - 1] = u[:]
        if self.Qfun[counter + i + 1, self.iter - 1] == 0:
            self.Qfun[counter + i + 1, self.iter -
                      1] == self.Qfun[counter + i, self.iter - 1] - 1

    def add_trajectory(self, ego, lap_number):

        iter = self.iter
        end_iter = int(
            round((ego.times[lap_number][-1] - ego.times[lap_number][0]) / ego.timestep))
        times = np.stack(ego.times[lap_number], axis=0)
        self.time_ss[iter] = end_iter
        xcurvs = np.stack(ego.xcurvs[lap_number], axis=0)
        self.ss_xcurv[0: (end_iter + 1), :,
                      iter] = xcurvs[0: (end_iter + 1), :]
        xglobs = np.stack(ego.xglobs[lap_number], axis=0)
        self.ss_glob[0: (end_iter + 1), :, iter] = xglobs[0: (end_iter + 1), :]
        inputs = np.stack(ego.inputs[lap_number], axis=0)
        self.u_ss[0:end_iter, :, iter] = inputs[0:end_iter, :]
        self.Qfun[0: (end_iter + 1), iter] = lmpc_helper.compute_cost(
            xcurvs[0: (end_iter + 1), :],
            inputs[0:(end_iter), :],
            self.lap_length,
        )
        for i in np.arange(0, self.Qfun.shape[0]):
            if self.Qfun[i, iter] == 0:
                self.Qfun[i, iter] = self.Qfun[i - 1, iter] - 1
        if self.iter == 0:
            self.lin_points = self.ss_xcurv[1:
                                            self.lmpc_param.num_horizon + 2, :, iter]
            self.lin_input = self.u_ss[1: self.lmpc_param.num_horizon + 1, :, iter]
        self.iter = self.iter + 1
        self.time_in_iter = 0


# Base Racing Car
class BicycleDynamicsParam:
    def __init__(
        self,
        m=1.98,
        lf=0.125,
        lr=0.125,
        Iz=0.024,
        Df=0.8 * 1.98 * 9.81 / 2.0,
        Cf=1.25,
        Bf=1.0,
        Dr=0.8 * 1.98 * 9.81 / 2.0,
        Cr=1.25,
        Br=1.0,
    ):
        self.m = m
        self.lf = lf
        self.lr = lr
        self.Iz = Iz
        self.Df = Df
        self.Cf = Cf
        self.Bf = Bf
        self.Dr = Dr
        self.Cr = Cr
        self.Br = Br

    def get_params(self):
        return (
            self.m,
            self.lf,
            self.lr,
            self.Iz,
            self.Df,
            self.Cf,
            self.Bf,
            self.Dr,
            self.Cr,
            self.Br,
        )


class CarParam:
    def __init__(self, length=0.4, width=0.2, facecolor="None", edgecolor="black"):
        self.length = length
        self.width = width
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dynamics_param = BicycleDynamicsParam()


class SystemParam:
    def __init__(self, delta_max=0.5, a_max=1.0, v_max=10, v_min=0):
        self.delta_max = delta_max
        self.a_max = a_max
        self.v_max = v_max
        self.v_min = v_min


class ModelBase:
    def __init__(self, name=None, param=None, no_dynamics=False, system_param=None):
        self.name = name
        self.param = param
        self.system_param = system_param
        self.no_dynamics = False
        self.time = 0.0
        self.timestep = None
        self.xcurv = None
        self.xglob = None
        self.u = None
        self.zero_noise_flag = False
        self.lap_times = []
        if self.no_dynamics:
            pass
        else:
            self.lap_times.append(self.time)
        self.lap_xcurvs, self.lap_xglobs, self.lap_inputs = [], [], []
        self.times, self.xglobs, self.xcurvs, self.inputs = [], [], [], []
        self.laps = 0
        self.realtime_flag = False
        self.xglob_log = []
        self.xcurv_log = []
        self.local_trajs = []
        self.vehicles_interest = []
        self.splines = []
        self.solver_time = []
        self.all_splines = []
        self.all_local_trajs = []
        self.lmpc_prediction = []
        self.mpc_cbf_prediction = []

    def set_zero_noise(self):
        self.zero_noise_flag = True

    def set_timestep(self, dt):
        self.timestep = dt

    def set_state_curvilinear(self, xcurv):
        self.xcurv = xcurv

    def set_state_global(self, xglob):
        self.xglob = xglob

    def start_logging(self):
        self.lap_xcurvs, self.lap_xglobs, self.lap_inputs = [], [], []
        self.lap_xcurvs.append(self.xcurv)
        self.lap_xglobs.append(self.xglob)

    def set_track(self, track):
        self.track = track
        self.lap_length = track.lap_length
        self.point_and_tangent = track.point_and_tangent
        self.lap_width = track.width

    def set_ctrl_policy(self, ctrl_policy):
        self.ctrl_policy = ctrl_policy
        self.ctrl_policy.agent_name = self.name

    def calc_ctrl_input(self):
        self.ctrl_policy.set_state(self.xcurv, self.xglob)
        self.ctrl_policy.calc_input()
        self.u = self.ctrl_policy.get_input()

    def forward_dynamics(self):
        pass

    def forward_one_step(self, realtime_flag):
        if self.no_dynamics:
            self.forward_dynamics()
            self.update_memory()
        elif realtime_flag == False:
            self.calc_ctrl_input()
            self.forward_dynamics(realtime_flag)
            self.ctrl_policy.set_state(self.xcurv, self.xglob)
            self.update_memory()
        elif realtime_flag == True:
            self.forward_dynamics(realtime_flag)

    def update_memory(self):
        xcurv = copy.deepcopy(self.xcurv)
        xglob = copy.deepcopy(self.xglob)
        self.xglob_log.append(self.xglob)
        self.xcurv_log.append(self.xcurv)
        if xcurv[4] > self.lap_length:
            self.lap_xglobs.append(self.xglob)
            self.lap_times.append(self.time)
            self.lap_xcurvs.append(xcurv)
            self.lap_inputs.append(self.u)
            self.xglobs.append(self.lap_xglobs)
            self.times.append(self.lap_times)
            self.xcurvs.append(self.lap_xcurvs)
            self.inputs.append(self.lap_inputs)
            self.xcurv[4] = self.xcurv[4] - self.lap_length
            self.laps = self.laps + 1
            self.lap_xglobs, self.lap_xcurvs, self.lap_inputs, self.lap_times = [], [], [], []
            self.lap_xglobs.append(self.xglob)
            self.lap_times.append(self.time)
            self.lap_xcurvs.append(self.xcurv)
        else:
            self.lap_xglobs.append(self.xglob)
            self.lap_times.append(self.time)
            self.lap_xcurvs.append(self.xcurv)
            self.lap_inputs.append(self.u)

    def get_vehicle_in_rectangle(self, vehicle_xglob):
        car_length = self.param.length
        car_width = self.param.width
        car_dx = 0.5 * car_length
        car_dy = 0.5 * car_width
        car_xs_origin = [car_dx, car_dx, -car_dx, -car_dx, car_dx]
        car_ys_origin = [car_dy, -car_dy, -car_dy, car_dy, car_dy]
        car_frame = np.vstack(
            (np.array(car_xs_origin), np.array(car_ys_origin)))
        x = vehicle_xglob[4]
        y = vehicle_xglob[5]
        R = np.matrix(
            [
                [np.cos(vehicle_xglob[3]), -np.sin(vehicle_xglob[3])],
                [np.sin(vehicle_xglob[3]), np.cos(vehicle_xglob[3])],
            ]
        )
        rotated_car_frame = R * car_frame
        return (
            x + rotated_car_frame[0, 2],
            y + rotated_car_frame[1, 2],
            2 * car_dx,
            2 * car_dy,
            vehicle_xglob[3] * 180 / 3.14,
        )


class NoDynamicsModelBase(ModelBase):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None):
        ModelBase.__init__(self, name=name, param=param)
        self.no_dynamics = True

    def set_state_curvilinear_func(self, t_symbol, s_func, ey_func):
        self.t_symbol = t_symbol
        self.s_func = s_func
        self.ey_func = ey_func
        self.xcurv = np.zeros((X_DIM,))
        self.xglob = np.zeros((X_DIM,))
        self.xcurv, self.xglob = self.get_estimation(0)

    def get_estimation(self, t0):
        # position estimation in curvilinear coordinates
        xcurv_est = np.zeros((X_DIM,))
        xcurv_est[0] = sp.diff(
            self.s_func, self.t_symbol).subs(self.t_symbol, t0)
        xcurv_est[1] = sp.diff(
            self.ey_func, self.t_symbol).subs(self.t_symbol, t0)
        xcurv_est[2] = 0
        xcurv_est[3] = 0
        xcurv_est[4] = self.s_func.subs(self.t_symbol, t0)
        xcurv_est[5] = self.ey_func.subs(self.t_symbol, t0)
        # position estimation in global coordinates
        X, Y = self.track.get_global_position(xcurv_est[4], xcurv_est[5])
        psi = self.track.get_orientation(xcurv_est[4], xcurv_est[5])
        xglob_est = np.zeros((X_DIM,))
        xglob_est[0:3] = xcurv_est[0:3]
        xglob_est[3] = psi
        xglob_est[4] = X
        xglob_est[5] = Y
        return xcurv_est, xglob_est

    def get_trajectory_nsteps(self, t0, delta_t, n):
        xcurv_est_nsteps = np.zeros((X_DIM, n))
        xglob_est_nsteps = np.zeros((X_DIM, n))
        for index in range(n):
            xcurv_est, xglob_est = self.get_estimation(
                self.time + index * delta_t)
            xcurv_est_nsteps[:, index] = xcurv_est
            xglob_est_nsteps[:, index] = xglob_est
        return xcurv_est_nsteps, xglob_est_nsteps

    def forward_dynamics(self):
        self.time += self.timestep
        self.xcurv, self.xglob = self.get_estimation(self.time)


class DynamicBicycleModelBase(ModelBase):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None, system_param=None):
        ModelBase.__init__(self, name=name, param=param,
                           system_param=system_param)

    def forward_dynamics(self, realtime_flag):
        # This function computes the system evolution. Note that the discretization is delta_t and therefore is needed that
        # dt <= delta_t and ( dt / delta_t) = integer value
        # Discretization Parameters
        delta_t = 0.001
        xglob_next = np.zeros((X_DIM,))
        xcurv_next = np.zeros((X_DIM,))
        xglob_next = self.xglob
        xcurv_next = self.xcurv
        vehicle_param = CarParam()
        # Initialize counter
        i = 0
        while (i + 1) * delta_t <= self.timestep:
            s = xcurv_next[4]
            if realtime_flag == False:
                curv = self.track.get_curvature(s)
            else:
                curv = racing_env.get_curvature(
                    self.lap_length, self.point_and_tangent, s)
            # for realtime simulation, if no controller is set up, the model won't update
            if self.u is None:
                pass
            else:
                xglob_next, xcurv_next = vehicle_dynamics.vehicle_dynamics(
                    vehicle_param.dynamics_param,
                    curv,
                    xglob_next,
                    xcurv_next,
                    delta_t,
                    self.u,
                )
            # Increment counter
            i = i + 1
        # Noises
        noise_vx = np.maximum(-0.05,
                              np.minimum(np.random.randn() * 0.01, 0.05))
        noise_vy = np.maximum(-0.1, np.minimum(np.random.randn() * 0.01, 0.1))
        noise_wz = np.maximum(-0.05,
                              np.minimum(np.random.randn() * 0.005, 0.05))
        if ((realtime_flag == True) and (self.u is None)) or self.zero_noise_flag:
            # for realtime simulation, if no controller is set up, no update for state
            pass
        else:
            xcurv_next[0] = xcurv_next[0] + 0.5 * noise_vx
            xcurv_next[1] = xcurv_next[1] + 0.5 * noise_vy
            xcurv_next[2] = xcurv_next[2] + 0.5 * noise_wz
        self.xcurv = xcurv_next
        self.xglob = xglob_next
        self.time += self.timestep


# Base Simulator
class CarRacingSimBase:
    def __init__(self):
        self.track = None
        self.vehicles = {}
        self.opti_traj_xglob = None

    def set_timestep(self, dt):
        self.timestep = dt

    def set_track(self, track):
        self.track = track

    def set_opti_traj(self, opti_traj_xglob):
        self.opti_traj_xglob = opti_traj_xglob
