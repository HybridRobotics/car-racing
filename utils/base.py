import datetime
import numpy as np
import sympy as sp
from utils import vehicle_dynamics, ctrl, lmpc_helper, racing_env
from scipy.interpolate import interp1d
from pathos.multiprocessing import ProcessingPool as Pool
from cvxopt.solvers import qp
from cvxopt import spmatrix, matrix, solvers
import copy


# Base Controller
class ControlBase:
    def __init__(self):
        self.agent_name = None
        self.xdim = 6
        self.udim = 2
        self.time = 0.0
        self.timestep = None
        self.x = None
        self.xglob = None
        self.u = None
        self.realtime_flag = False

        self.traj_time = []
        self.traj_time.append(self.time)
        self.traj_xcurv = []
        self.traj_xglob = []
        self.traj_u = []

        self.time_list = []
        self.xglob_list = []
        self.xcurv_list = []
        self.u_list = []
        self.laps = 0
        self.track = None
    
    def set_track(self, track):
        self.track = track
        self.lap_length = track.lap_length
        self.point_and_tangent = track.point_and_tangent
        self.lap_width = track.width

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
        if xcurv[4]>self.lap_length*(current_lap+1):
            self.traj_xglob.append(xglob)
            self.traj_time.append(time)
            xcurv[4] = xcurv[4]- current_lap*self.lap_length
            self.traj_xcurv.append(xcurv)
            self.traj_u.append(self.u)
            self.xglob_list.append(self.traj_xglob)
            self.time_list.append(self.traj_time)
            self.xcurv_list.append(self.traj_xcurv)
            self.u_list.append(self.traj_u)
            x = copy.deepcopy(self.x)
            x[4] = x[4] - self.lap_length*(current_lap+1)
            self.laps = self.laps + 1
            self.traj_xglob = []
            self.traj_xcurv = []
            self.traj_u = []
            self.traj_time = []
            self.traj_xglob.append(xglob)
            self.traj_time.append(self.time)
            self.traj_xcurv.append(x)
        else:
            xcurv[4] = xcurv[4] -current_lap*self.lap_length
            self.traj_xglob.append(xglob)
            self.traj_time.append(self.time)
            self.traj_xcurv.append(xcurv)
            self.traj_u.append(self.u)


class PIDTracking(ControlBase):
    def __init__(self, vt=0.6, eyt=0.0):
        ControlBase.__init__(self)
        self.set_target_speed(vt)
        self.set_target_deviation(eyt)

    def calc_input(self):
        xtarget = np.array([self.vt, 0, 0, 0, 0, self.eyt]).reshape(self.xdim, 1)
        self.u = ctrl.pid(self.x, xtarget, self.udim)
        self.time += self.timestep


class MPCTracking(ControlBase):
    def __init__(self, matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6, eyt=0.0):
        ControlBase.__init__(self)
        self.set_target_speed(vt)
        self.set_target_deviation(eyt)
        self.num_of_horizon = 10
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R

    def calc_input(self):
        xtarget = np.array([self.vt, 0, 0, 0, 0, self.eyt]).reshape(self.xdim, 1)
        self.u = ctrl.mpc(self.x, xtarget, self.udim, self.num_of_horizon,
                            self.matrix_A, self.matrix_B, self.matrix_Q, self.matrix_R)
        self.time += self.timestep


class MPCCBFRacing(ControlBase):
    def __init__(self, matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6, eyt=0.0):
        ControlBase.__init__(self)
        self.set_target_speed(vt)
        self.set_target_deviation(eyt)
        self.num_of_horizon = 10
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.alpha = 0.6
        self.realtime_flag = None

    def calc_input(self):
        xtarget = np.array([self.vt, 0, 0, 0, 0, self.eyt]).reshape(self.xdim, 1)
        # determine if it is a real-time simulator
        if self.realtime_flag == False:
            self.u = ctrl.mpccbf(self.x, xtarget, self.udim, self.num_of_horizon, self.matrix_A, self.matrix_B, self.matrix_Q, self.matrix_R,
                               self.racing_sim.vehicles, self.agent_name, self.racing_sim.track.lap_length, self.time, self.timestep, self.alpha, self.realtime_flag)
        elif self.realtime_flag == True:
             self.u = ctrl.mpccbf(self.x, xtarget, self.udim, self.num_of_horizon, self.matrix_A, self.matrix_B, self.matrix_Q, self.matrix_R,
                               self.vehicles, self.agent_name, self.lap_length, self.time, self.timestep, self.alpha, self.realtime_flag)
        else:
            pass
        self.time += self.timestep


class LMPCRacing(ControlBase):
    def __init__(self, num_ss_points, num_ss_iter, N, matrix_Qslack, matrix_Q_LMPC, matrix_R_LMPC, matrix_dR_LMPC, xdim, udim, shift, timestep, lap_number, time_lmpc):
        ControlBase.__init__(self)
        self.num_ss_points = num_ss_points
        self.num_ss_iter = num_ss_iter
        self.N = N
        self.matrix_Qslack = matrix_Qslack
        self.matrix_Q_LMPC = matrix_Q_LMPC
        self.matrix_R_LMPC = matrix_R_LMPC
        self.matrix_dR_LMPC = matrix_dR_LMPC
        self.xdim = xdim
        self.udim = udim
        self.shift = shift
        self.timestep = timestep
        self.lap_number = lap_number
        self.x_pred = None
        self.u_pred = None
        self.lin_points = None
        self.lin_input = None
        self.ss_point_selected_tot = None
        self.Qfun_selected_tot = None
        num_points = int(time_lmpc/timestep) + 1
        # Time at which each j-th iteration is completed
        self.time_ss = 10000 * np.ones(lap_number).astype(int)
        self.ss_xcurv = 10000 * \
            np.ones((num_points, xdim, lap_number))  # Sampled Safe SS
        # Input associated with the points in SS
        self.u_ss = 10000 * np.ones((num_points, udim, lap_number))
        # Qfun: cost-to-go from each point in SS
        self.Qfun = 0 * np.ones((num_points, lap_number))
        # SS in global (X-Y) used for plotting
        self.ss_glob = 10000 * np.ones((num_points, xdim, lap_number))
        # Initialize the controller iteration
        self.iter = 0
        self.time_in_iter = 0
        self.p = Pool(4)
        self.openloop_prediction_lmpc = None

    def calc_input(self):
        matrix_Atv, matrix_Btv, matrix_Ctv, _ = self.estimate_ABC()
        x = copy.deepcopy(self.x)
        while x[4] > self.lap_length:
            x[4] = x[4] - self.lap_length
        if self.u_pred is None:
            u_old = np.zeros((1, 2))
        else:
            u_old = copy.deepcopy(self.u_pred[0, :])
        self.u_pred, self.x_pred, self.ss_point_selected_tot, self.Qfun_selected_tot, self.lin_points, self.lin_input = ctrl.lmpc(
            x, matrix_Atv, matrix_Btv, matrix_Ctv, self.matrix_R_LMPC, self.matrix_Q_LMPC, self.matrix_dR_LMPC, self.matrix_Qslack, self.xdim, self.udim, self.N, self.num_ss_iter, self.ss_xcurv, self.Qfun, self.iter, self.num_ss_points, self.shift, self.lap_length, self.lap_width, u_old)
        self.u = self.u_pred[0, :]
        iter = self.iter
        self.openloop_prediction_lmpc.predicted_xcurv[:, :, self.time_in_iter,
                                                      iter] = self.x_pred
        self.openloop_prediction_lmpc.predicted_u[:, :, self.time_in_iter,
                                                  iter] = self.u_pred
        self.openloop_prediction_lmpc.ss_used[:, :, self.time_in_iter,
                                              iter] = self.ss_point_selected_tot
        self.openloop_prediction_lmpc.Qfun_used[:, self.time_in_iter,
                                                iter] = self.Qfun_selected_tot
        self.add_point(self.x, self.u, self.time_in_iter)
        self.time_in_iter = self.time_in_iter + 1
        self.time += self.timestep

    def estimate_ABC(self):
        lin_points = self.lin_points
        lin_input = self.lin_input
        N = self.N
        xdim = self.xdim
        udim = self.udim
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
        lap_used_for_linearization = 2  # 2
        used_iter = range(iter-lap_used_for_linearization, iter)
        max_num_point = 40
        for i in range(0, N):
            Ai, Bi, Ci, index_selected = lmpc_helper.regression_and_linearization(lin_points, lin_input, used_iter, ss_xcurv, u_ss, time_ss,
                                                                                    max_num_point, qp, xdim, udim, matrix, point_and_tangent, timestep, i)
            Atv.append(Ai)
            Btv.append(Bi)
            Ctv.append(Ci)
            index_used_list.append(index_selected)
        return Atv, Btv, Ctv, index_used_list

    def add_point(self, x, u, i):
        counter = self.time_ss[self.iter - 1]
        self.ss_xcurv[counter + i + 1, :, self.iter - 1] = x + \
            np.array([0, 0, 0, 0, self.lap_length, 0])
        self.u_ss[counter + i + 1, :, self.iter - 1] = u
        if self.Qfun[counter + i + 1, self.iter - 1] == 0:
            self.Qfun[counter+i+1, self.iter -
                      1] == self.Qfun[counter+i, self.iter - 1] - 1

    def add_trajectory(self, time_list, timestep, xcurv_list, xglob_list, u_list, lap_number):
        iter = self.iter
        end_iter = int(round((time_list[lap_number][-1] - time_list[lap_number][0])/timestep))
        time_list = np.stack(time_list[lap_number], axis=0)
        self.time_ss[iter] = end_iter
        xcurv_list = np.stack(xcurv_list[lap_number], axis=0)
        self.ss_xcurv[0:(end_iter + 1),:, iter] = xcurv_list[0:(end_iter+1),:]
        xglob_list = np.stack(xglob_list[lap_number], axis = 0)
        self.ss_glob[0:(end_iter + 1),:, iter] = xglob_list[0:(end_iter+1),:]
        u_list = np.stack(u_list[lap_number],axis = 0)
        self.u_ss[0:end_iter, :, iter] = u_list[0:end_iter, :]
        self.Qfun[0:(end_iter + 1), iter] = lmpc_helper.compute_cost(xcurv_list[0:(end_iter + 1), :],u_list[0:(end_iter), :], self.lap_length)
        for i in np.arange(0, self.Qfun.shape[0]):
            if self.Qfun[i, iter] == 0:
                self.Qfun[i, iter] = self.Qfun[i - 1, iter] - 1
        if self.iter == 0:
            self.lin_points = self.ss_xcurv[1:self.N + 2, :, iter]
            self.lin_input = self.u_ss[1:self.N + 1, :, iter]
        self.iter = self.iter + 1
        self.time_in_iter = 0


# Base Racing Car
class BicycleDynamicsParam:
    def __init__(self, m=1.98, lf=0.125, lr=0.125, Iz=0.024, Df=0.8 * 1.98 * 9.81 / 2.0, Cf=1.25, Bf=1.0, Dr=0.8 * 1.98 * 9.81 / 2.0, Cr=1.25, Br=1.0):
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
        return self.m, self.lf, self.lr, self.Iz, self.Df, self.Cf, self.Bf, self.Dr, self.Cr, self.Br


class CarParam:
    def __init__(self, length=0.4, width=0.2, facecolor="None", edgecolor="black"):
        self.length = length
        self.width = width
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dynamics_param = BicycleDynamicsParam()


class ModelBase:
    def __init__(self, name=None, param=None, no_dynamics = False):
        self.name = name
        self.param = param
        self.no_dynamics = False
        self.xdim = 6
        self.udim = 2
        self.time = 0.0
        self.timestep = None
        self.xcurv = None
        self.xglob = None
        self.u = None

        self.traj_time = []
        if self.no_dynamics:
            pass
        else:
            self.traj_time.append(self.time)
        self.traj_xcurv = []
        self.traj_xglob = []
        self.traj_u = []
        self.time_list = []
        self.xglob_list = []
        self.xcurv_list = []
        self.u_list = []
        self.laps = 0
        self.realtime_flag = False

    def set_timestep(self, dt):
        self.timestep = dt

    def set_state_curvilinear(self, xcurv):
        self.xcurv = xcurv
        self.traj_xcurv = []
        self.traj_xcurv.append(xcurv)

    def set_state_global(self, xglob):
        self.xglob = xglob
        self.traj_xglob = []
        self.traj_u = []
        self.traj_xglob.append(xglob)

    def set_track(self, track):
        self.track = track
        self.lap_length = track.lap_length

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
        if xcurv[4]>self.lap_length:
            self.traj_xglob.append(self.xglob)
            self.traj_time.append(self.time)
            self.traj_xcurv.append(xcurv)
            self.traj_u.append(self.u)
            self.xglob_list.append(self.traj_xglob)
            self.time_list.append(self.traj_time)
            self.xcurv_list.append(self.traj_xcurv)
            self.u_list.append(self.traj_u)
            self.xcurv[4] = self.xcurv[4] - self.lap_length
            self.laps = self.laps + 1           
            self.traj_xglob = []
            self.traj_xcurv = []
            self.traj_u = []
            self.traj_time = []
            self.traj_xglob.append(self.xglob)
            self.traj_time.append(self.time)
            self.traj_xcurv.append(self.xcurv)
        else:
            self.traj_xglob.append(self.xglob)
            self.traj_time.append(self.time)
            self.traj_xcurv.append(self.xcurv)
            self.traj_u.append(self.u)
        

class NoDynamicsModel(ModelBase):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None):
        ModelBase.__init__(self, name=name, param=param)
        self.no_dynamics = True

    def set_state_curvilinear_func(self, t_symbol, s_func, ey_func):
        self.t_symbol = t_symbol
        self.s_func = s_func
        self.ey_func = ey_func
        xcurv = np.zeros(self.xdim)
        xglob = np.zeros(self.xdim)
        xcurv, xglob = self.get_estimation(0)
        self.traj_xcurv.append(xcurv)
        self.traj_xglob.append(xglob)

    def get_estimation(self, t0):
        # position estimation in curvilinear coordinates
        xcurv_est = np.zeros(self.xdim)
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
        xglob_est = np.zeros(self.xdim)
        xglob_est[0:3] = xcurv_est[0:3]
        xglob_est[3] = psi
        xglob_est[4] = X
        xglob_est[5] = Y
        return xcurv_est, xglob_est

    def get_trajectory_nsteps(self, t0, delta_t, n):
        xcurv_est_nsteps = np.zeros((self.xdim, n))
        xglob_est_nsteps = np.zeros((self.xdim, n))
        for index in range(n):
            xcurv_est, xglob_est = self.get_estimation(
                self.time + index * delta_t)
            xcurv_est_nsteps[:, index] = xcurv_est
            xglob_est_nsteps[:, index] = xglob_est
        return xcurv_est_nsteps, xglob_est_nsteps

    def forward_dynamics(self):
        self.time += self.timestep
        self.xcurv, self.xglob = self.get_estimation(self.time)


class DynamicBicycleModel(ModelBase):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None):
        ModelBase.__init__(self, name=name, param=param)

    def forward_dynamics(self, realtime_flag):
        # This function computes the system evolution. Note that the discretization is delta_t and therefore is needed that
        # dt <= delta_t and ( dt / delta_t) = integer value

        # Discretization Parameters
        delta_t = 0.001
        xglob_next = np.zeros(self.xdim)
        xcurv_next = np.zeros(self.xdim)
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
                curv = racing_env.get_curvature(self.lap_length, self.point_and_tangent,s)
            # for realtime simulation, if no controller is set up, the model won't update
            if self.u is None:
                pass
            else:
                xglob_next, xcurv_next = vehicle_dynamics.vehicle_dynamics(
                    vehicle_param.dynamics_param, curv, xglob_next, xcurv_next, delta_t, self.u)
            
            if s < 0:
                pass
                # Don't need this checker as curvature can be calculated even s < 0
                # print("Start Point: ", self.x, " Input: ", self.ctrl_policy.u)
                # print("x_next: ", x_next)
            # Increment counter
            i = i + 1
        # Noises
        noise_vx = np.maximum(-0.05,
                              np.minimum(np.random.randn() * 0.01, 0.05))
        noise_vy = np.maximum(-0.1, np.minimum(np.random.randn() * 0.01, 0.1))
        noise_wz = np.maximum(-0.05,
                              np.minimum(np.random.randn() * 0.005, 0.05))
        if (realtime_flag == True) and (self.u is None):
            # for realtime simulation, if no controller is set up, no update for state
            pass
        else:
            xcurv_next[0] = xcurv_next[0] + 0.1 * noise_vx
            xcurv_next[1] = xcurv_next[1] + 0.1 * noise_vy
            xcurv_next[2] = xcurv_next[2] + 0.1 * noise_wz
        self.xcurv = xcurv_next
        self.xglob = xglob_next
        self.time += self.timestep


# Base Simulator
class CarRacingSim:
    def __init__(self):
        self.track = None
        self.vehicles = {}

    def set_timestep(self, dt):
        self.timestep = dt

    def set_track(self, track):
        self.track = track
